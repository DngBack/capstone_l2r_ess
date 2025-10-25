# src/train/gse_constrained_plugin.py
"""
GSE-Constrained Plugin: Constrained optimization approach for balanced risk minimization
with coverage and fairness constraints.

This implements the Lagrangian formulation:
L(α, μ, t, λ, ν) = (1/K) Σ e_k + λ(τ - (1/K) Σ cov_k) + Σ ν_k(e_k - δ)

Where:
- e_k: per-group error when accepting
- cov_k: per-group coverage
- τ: minimum average coverage constraint
- δ: maximum per-group error constraint
- λ: coverage multiplier
- ν_k: fairness multipliers per group
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torchvision

# Import our custom modules
from src.models.argse import AR_GSE
from src.data.groups import get_class_to_group_by_threshold
from src.data.datasets import get_cifar100_lt_counts

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- CONFIGURATION ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'grouping': {
        'threshold': 20,  # classes with >threshold samples are head
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/',
    },
    'constrained_params': {
        'T': 50,  # outer iterations
        'tau': 0.65,  # minimum average coverage constraint
        'delta_multiplier': 1.3,  # δ = delta_multiplier * average_error
        'eta_dual': 0.05,  # dual step size
        'eta_primal': 0.01,  # primal step size  
        'lambda_grid': [round(x, 2) for x in np.linspace(-2.0, 2.0, 21)],
        'alpha_min': 0.75,
        'alpha_max': 1.35,
        'patience': 8,  # early stopping
        'convergence_tol': 1e-4,
        'adaptive_delta': True,  # adaptively adjust δ based on progress
        'warmup_iters': 10,  # warmup iterations before enforcing constraints
    },
    'output': {
        'checkpoints_dir': './checkpoints/argse_constrained_plugin/',
    },
    'seed': 42
}

@torch.no_grad()
def cache_eta_mix(gse_model, loader, class_to_group):
    """Cache mixture posteriors η̃(x) for a split."""
    gse_model.eval()
    etas, labels = [], []
    
    for logits, y in tqdm(loader, desc="Caching η̃"):
        logits = logits.to(DEVICE)
        
        # Get mixture posterior
        expert_posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
        
        # Get gating weights
        gating_features = gse_model.feature_builder(logits)
        w = torch.softmax(gse_model.gating_net(gating_features), dim=1)  # [B, E]
        
        # Mixture: η̃_y(x) = Σ_e w^(e)(x) * p^(e)(y|x)
        eta = torch.einsum('be,bec->bc', w, expert_posteriors)  # [B, C]
        
        etas.append(eta.cpu())
        labels.append(y.cpu())
    
    return torch.cat(etas), torch.cat(labels)

def compute_raw_margin(eta, alpha, mu, class_to_group):
    """Compute raw margin score without threshold."""
    device = eta.device
    alpha = alpha.to(device)
    mu = mu.to(device)
    class_to_group = class_to_group.to(device)
    
    # score = max_y α_{g(y)} * η̃_y
    score = (alpha[class_to_group] * eta).max(dim=1).values  # [N]
    # threshold = Σ_y (α_{g(y)} - μ_{g(y)}) * η̃_y
    coeff = alpha[class_to_group] - mu[class_to_group]
    threshold = (coeff.unsqueeze(0) * eta).sum(dim=1)        # [N]
    return score - threshold

def compute_group_metrics(eta, y, alpha, mu, t, class_to_group, K):
    """
    Compute per-group error and coverage.
    
    Args:
        eta: mixture posteriors [N, C]
        y: ground truth labels [N]
        alpha, mu: per-group parameters [K]
        t: threshold (scalar or per-group [K])
        class_to_group: class to group mapping [C]
        K: number of groups
        
    Returns:
        e_k: per-group error rates [K]
        cov_k: per-group coverage rates [K]
        accepted: acceptance mask [N]
        preds: predictions [N]
    """
    device = eta.device
    
    # Compute margins
    raw_margins = compute_raw_margin(eta, alpha, mu, class_to_group)
    
    # Handle per-group or global threshold
    if isinstance(t, torch.Tensor) and t.numel() == K:
        # Per-group thresholds
        y_groups = class_to_group[y]
        thresholds = t[y_groups]
    else:
        # Global threshold
        thresholds = t
    
    # Acceptance and predictions
    accepted = (raw_margins >= thresholds)
    alpha_per_class = alpha[class_to_group]
    preds = (alpha_per_class * eta).argmax(dim=1)
    
    # Per-group metrics
    y_groups = class_to_group[y]
    e_k = torch.zeros(K, device=device)
    cov_k = torch.zeros(K, device=device)
    
    for k in range(K):
        group_mask = (y_groups == k)
        n_k = group_mask.sum().float()
        
        if n_k > 0:
            # Coverage: fraction of group samples accepted
            cov_k[k] = (accepted & group_mask).sum().float() / n_k
            
            # Error: fraction of accepted group samples that are wrong
            accepted_group = accepted & group_mask
            if accepted_group.sum() > 0:
                correct_group = (preds == y) & accepted_group
                e_k[k] = 1.0 - (correct_group.sum().float() / accepted_group.sum().float())
            else:
                e_k[k] = 1.0  # No accepted samples = worst error
        else:
            cov_k[k] = 0.0
            e_k[k] = 1.0
    
    return e_k, cov_k, accepted, preds

def project_alpha(alpha, alpha_min=0.75, alpha_max=1.35):
    """Project alpha to valid range with geometric mean normalization."""
    alpha = alpha.clamp_min(alpha_min)
    log_alpha = alpha.log()
    alpha = torch.exp(log_alpha - log_alpha.mean())
    return alpha.clamp(min=alpha_min, max=alpha_max)

def gse_constrained_plugin(eta_S1, y_S1, eta_S2, y_S2, class_to_group, K, config):
    """
    Main constrained optimization algorithm.
    
    Args:
        eta_S1, y_S1: tuning split for threshold fitting
        eta_S2, y_S2: validation split for optimization
        class_to_group: class to group mapping
        K: number of groups
        config: configuration dictionary
        
    Returns:
        best_alpha, best_mu, best_t: optimal parameters
        history: optimization history
    """
    device = eta_S1.device
    y_S1 = y_S1.to(device)
    y_S2 = y_S2.to(device)
    class_to_group = class_to_group.to(device)
    
    # Extract config
    T = config['T']
    tau = config['tau']
    delta_multiplier = config['delta_multiplier']
    eta_dual = config['eta_dual']
    eta_primal = config['eta_primal']
    lambda_grid = config['lambda_grid']
    alpha_min = config['alpha_min']
    alpha_max = config['alpha_max']
    patience = config['patience']
    warmup_iters = config['warmup_iters']
    
    print("=== GSE Constrained Plugin ===")
    print(f"Coverage constraint: τ ≥ {tau:.2f}")
    print(f"Outer iterations: {T}")
    print(f"Dual step size: {eta_dual}, Primal: {eta_primal}")
    
    # Initialize parameters
    alpha = torch.ones(K, device=device)
    mu = torch.zeros(K, device=device)
    t = 0.0  # Global threshold
    
    # Initialize dual variables
    lambda_cov = 0.0
    nu = torch.zeros(K, device=device)
    
    # History tracking
    history = {
        'balanced_error': [],
        'worst_error': [],
        'coverage': [],
        'lagrangian': [],
        'lambda_cov': [],
        'nu': [],
        'delta': []
    }
    
    best_objective = float('inf')
    best_params = None
    no_improve_count = 0
    
    # Initial error estimate for δ
    e_k_init, cov_k_init, _, _ = compute_group_metrics(eta_S2, y_S2, alpha, mu, t, class_to_group, K)
    delta = delta_multiplier * e_k_init.mean().item()
    print(f"Initial δ = {delta:.3f} ({delta_multiplier}× avg error)")
    
    for outer_iter in range(T):
        # Fit threshold on S1 for current (α, μ)
        raw_S1 = compute_raw_margin(eta_S1, alpha, mu, class_to_group)
        t_new = torch.quantile(raw_S1, 1.0 - tau).item()  # Fit for target coverage
        
        # Smooth threshold update
        t = 0.7 * t + 0.3 * t_new
        
        # Compute metrics on S2
        e_k, cov_k, accepted, preds = compute_group_metrics(eta_S2, y_S2, alpha, mu, t, class_to_group, K)
        
        # Objective components
        balanced_error = e_k.mean()
        worst_error = e_k.max()
        avg_coverage = cov_k.mean()
        
        # Lagrangian
        coverage_violation = tau - avg_coverage
        fairness_violations = e_k - delta
        
        lagrangian = balanced_error.item()
        if outer_iter >= warmup_iters:  # Apply constraints after warmup
            lagrangian += lambda_cov * coverage_violation.item()
            lagrangian += (nu * torch.clamp(fairness_violations, min=0)).sum().item()
        
        # Track history
        history['balanced_error'].append(balanced_error.item())
        history['worst_error'].append(worst_error.item())
        history['coverage'].append(avg_coverage.item())
        history['lagrangian'].append(lagrangian)
        history['lambda_cov'].append(lambda_cov)
        history['nu'].append(nu.clone().cpu().numpy().tolist())
        history['delta'].append(delta)
        
        # Logging
        constraint_info = ""
        if outer_iter >= warmup_iters:
            constraint_info = f", λ={lambda_cov:.3f}, ν_max={nu.max():.3f}"
        
        print(f"[{outer_iter+1:2d}] bal={balanced_error:.4f}, worst={worst_error:.4f}, "
              f"cov={avg_coverage:.3f}, L={lagrangian:.4f}{constraint_info}")
        
        # Early stopping based on Lagrangian
        if lagrangian < best_objective:
            best_objective = lagrangian
            best_params = (alpha.clone(), mu.clone(), t)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at iteration {outer_iter+1}")
                break
        
        # Update primal variables (α, μ) via grid search
        if outer_iter < T - 1:  # Don't update on last iteration
            best_alpha_new = alpha.clone()
            best_mu_new = mu.clone()
            best_loss = lagrangian
            
            # Grid search over μ (via λ grid)
            for lam in lambda_grid:
                if K == 2:
                    mu_candidate = torch.tensor([+lam/2, -lam/2], device=device)
                else:
                    continue  # Skip for K > 2
                
                # Fixed-point updates for α
                alpha_candidate = alpha.clone()
                for _ in range(3):  # Few fixed-point steps
                    # Compute acceptance rates per group
                    raw_margins = compute_raw_margin(eta_S1, alpha_candidate, mu_candidate, class_to_group)
                    accepted_fp = (raw_margins >= t)
                    y_groups_S1 = class_to_group[y_S1]
                    
                    alpha_new = torch.zeros(K, device=device)
                    for k in range(K):
                        group_mask = (y_groups_S1 == k)
                        if group_mask.sum() > 0:
                            acceptance_rate = (accepted_fp & group_mask).sum().float() / group_mask.sum().float()
                            alpha_new[k] = acceptance_rate + 1e-3  # Smooth
                        else:
                            alpha_new[k] = 1.0
                    
                    # EMA and project
                    alpha_candidate = 0.7 * alpha_candidate + 0.3 * alpha_new
                    alpha_candidate = project_alpha(alpha_candidate, alpha_min, alpha_max)
                
                # Evaluate candidate
                e_k_cand, cov_k_cand, _, _ = compute_group_metrics(eta_S2, y_S2, alpha_candidate, mu_candidate, t, class_to_group, K)
                
                loss_cand = e_k_cand.mean().item()
                if outer_iter >= warmup_iters:
                    cov_violation = tau - cov_k_cand.mean()
                    fair_violations = e_k_cand - delta
                    loss_cand += lambda_cov * cov_violation.item()
                    loss_cand += (nu * torch.clamp(fair_violations, min=0)).sum().item()
                
                if loss_cand < best_loss:
                    best_loss = loss_cand
                    best_alpha_new = alpha_candidate.clone()
                    best_mu_new = mu_candidate.clone()
            
            # Update primal with momentum
            alpha = 0.8 * alpha + 0.2 * best_alpha_new
            mu = 0.8 * mu + 0.2 * best_mu_new
            alpha = project_alpha(alpha, alpha_min, alpha_max)
        
        # Update dual variables (after warmup)
        if outer_iter >= warmup_iters:
            # Coverage multiplier
            lambda_cov = max(0.0, lambda_cov + eta_dual * coverage_violation.item())
            
            # Fairness multipliers
            nu = torch.clamp(nu + eta_dual * fairness_violations, min=0.0)
        
        # Adaptive δ adjustment
        if config.get('adaptive_delta', False) and outer_iter > 0 and outer_iter % 10 == 0:
            current_avg_error = e_k.mean().item()
            if current_avg_error > 0:
                delta = max(delta * 0.95, delta_multiplier * current_avg_error)
                print(f"  📊 Adjusted δ = {delta:.4f}")
    
    # Return best parameters
    if best_params is not None:
        best_alpha, best_mu, best_t = best_params
        print(f"\n✅ Best solution: L={best_objective:.4f}")
        return best_alpha.cpu(), best_mu.cpu(), best_t, history
    else:
        return alpha.cpu(), mu.cpu(), t, history

def load_data_from_logits(config):
    """Load pre-computed logits for tuneV (S1) and val_lt (S2) splits."""
    logits_root = Path(config['experts']['logits_dir']) / config['dataset']['name']
    splits_dir = Path(config['dataset']['splits_dir'])
    expert_names = config['experts']['names']
    num_experts = len(expert_names)
    num_classes = config['dataset']['num_classes']
    
    dataloaders = {}
    
    # Base datasets
    cifar_train_full = torchvision.datasets.CIFAR100(root='./data', train=True, download=False)
    cifar_test_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    
    # Use tuneV (S1) and val_lt (S2) splits
    splits_config = [
        {'split_name': 'tuneV', 'base_dataset': cifar_train_full, 'indices_file': 'tuneV_indices.json'},
        {'split_name': 'val_lt', 'base_dataset': cifar_test_full, 'indices_file': 'val_lt_indices.json'}
    ]
    
    for split in splits_config:
        split_name = split['split_name']
        base_dataset = split['base_dataset']
        indices_path = splits_dir / split['indices_file']
        print(f"Loading data for split: {split_name}")
        
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing indices file: {indices_path}")
        indices = json.loads(indices_path.read_text())

        # Stack expert logits
        stacked_logits = torch.zeros(len(indices), num_experts, num_classes)
        for i, expert_name in enumerate(expert_names):
            logits_path = logits_root / expert_name / f"{split_name}_logits.pt"
            if not logits_path.exists():
                raise FileNotFoundError(f"Missing logits file: {logits_path}")
            stacked_logits[:, i, :] = torch.load(logits_path, map_location='cpu')

        labels = torch.tensor(np.array(base_dataset.targets)[indices])
        dataset = TensorDataset(stacked_logits, labels)
        dataloaders[split_name] = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    return dataloaders['tuneV'], dataloaders['val_lt']

def main():
    """Main constrained plugin training."""
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    print("=== GSE-Constrained Plugin Training ===")
    
    # 1) Load data
    S1_loader, S2_loader = load_data_from_logits(CONFIG)
    print(f"✅ Loaded S1 (tuneV): {len(S1_loader)} batches")
    print(f"✅ Loaded S2 (val_lt): {len(S2_loader)} batches")
    
    # 2) Set up grouping
    class_counts = get_cifar100_lt_counts(imb_factor=100)
    class_to_group = get_class_to_group_by_threshold(class_counts, threshold=CONFIG['grouping']['threshold'])
    num_groups = class_to_group.max().item() + 1
    head = (class_to_group == 0).sum().item()
    tail = (class_to_group == 1).sum().item()
    print(f"✅ Groups: {head} head classes, {tail} tail classes")
    
    # 3) Load GSE model
    num_experts = len(CONFIG['experts']['names'])
    
    # Dynamic gating feature dimension
    with torch.no_grad():
        dummy_logits = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
        del temp_model
    
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)
    
    # Load pre-trained gating if available
    gating_ckpt_path = Path('./checkpoints/gating_pretrained/') / CONFIG['dataset']['name'] / 'gating_pretrained.ckpt'
    if gating_ckpt_path.exists():
        try:
            gating_ckpt = torch.load(gating_ckpt_path, map_location=DEVICE, weights_only=False)
            model.gating_net.load_state_dict(gating_ckpt['gating_net_state_dict'])
            print(f"✅ Loaded pre-trained gating from {gating_ckpt_path}")
        except Exception as e:
            print(f"⚠️ Failed to load gating checkpoint: {e}")
    else:
        print("⚠️ No pre-trained gating found. Using random initialization.")
    
    # Initialize model parameters
    with torch.no_grad():
        model.alpha.fill_(1.0)
        model.mu.fill_(0.0)
    
    # 4) Cache mixture posteriors
    print("\n=== Caching mixture posteriors ===")
    eta_S1, y_S1 = cache_eta_mix(model, S1_loader, class_to_group)
    eta_S2, y_S2 = cache_eta_mix(model, S2_loader, class_to_group)
    
    print(f"✅ Cached η̃_S1: {eta_S1.shape}, y_S1: {y_S1.shape}")
    print(f"✅ Cached η̃_S2: {eta_S2.shape}, y_S2: {y_S2.shape}")
    
    # 5) Run constrained optimization
    print("\n=== Running Constrained Plugin Optimization ===")
    alpha_star, mu_star, t_star, history = gse_constrained_plugin(
        eta_S1=eta_S1.to(DEVICE),
        y_S1=y_S1.to(DEVICE),
        eta_S2=eta_S2.to(DEVICE),
        y_S2=y_S2.to(DEVICE),
        class_to_group=class_to_group.to(DEVICE),
        K=num_groups,
        config=CONFIG['constrained_params']
    )
    
    # 6) Final evaluation
    print("\n=== Final Results ===")
    final_e_k, final_cov_k, _, _ = compute_group_metrics(
        eta_S2.to(DEVICE), y_S2.to(DEVICE), alpha_star.to(DEVICE), 
        mu_star.to(DEVICE), t_star, class_to_group.to(DEVICE), num_groups
    )
    
    balanced_error = final_e_k.mean().item()
    worst_error = final_e_k.max().item()
    avg_coverage = final_cov_k.mean().item()
    
    print(f"α* = [{alpha_star[0]:.4f}, {alpha_star[1]:.4f}]")
    print(f"μ* = [{mu_star[0]:.4f}, {mu_star[1]:.4f}]")
    print(f"t* = {t_star:.4f}")
    print(f"Balanced error = {balanced_error:.4f}")
    print(f"Worst error = {worst_error:.4f}")
    print(f"Coverage = {avg_coverage:.3f}")
    print(f"Per-group errors: {final_e_k.cpu().numpy()}")
    print(f"Per-group coverage: {final_cov_k.cpu().numpy()}")
    
    # 7) Save results
    output_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'alpha': alpha_star,
        'mu': mu_star,
        'threshold': t_star,
        'class_to_group': class_to_group,
        'num_groups': num_groups,
        'balanced_error': balanced_error,
        'worst_error': worst_error,
        'coverage': avg_coverage,
        'per_group_errors': final_e_k.cpu().numpy().tolist(),
        'per_group_coverage': final_cov_k.cpu().numpy().tolist(),
        'optimization_history': history,
        'config': CONFIG,
        'gating_net_state_dict': model.gating_net.state_dict(),
        'source': 'constrained_plugin'
    }
    
    ckpt_path = output_dir / 'gse_constrained_plugin.ckpt'
    torch.save(checkpoint, ckpt_path)
    print(f"💾 Saved checkpoint to {ckpt_path}")
    
    # 8) Save optimization plots
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Errors over time
        axes[0,0].plot(history['balanced_error'], label='Balanced Error', color='blue')
        axes[0,0].plot(history['worst_error'], label='Worst Error', color='red')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Error Rate')
        axes[0,0].set_title('Error Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot 2: Coverage over time
        axes[0,1].plot(history['coverage'], label='Coverage', color='green')
        axes[0,1].axhline(y=CONFIG['constrained_params']['tau'], color='red', linestyle='--', label=f'Target τ={CONFIG["constrained_params"]["tau"]:.2f}')
        axes[0,1].set_xlabel('Iteration')
        axes[0,1].set_ylabel('Coverage')
        axes[0,1].set_title('Coverage Evolution')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot 3: Lagrangian over time
        axes[1,0].plot(history['lagrangian'], color='purple')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Lagrangian')
        axes[1,0].set_title('Objective Evolution')
        axes[1,0].grid(True)
        
        # Plot 4: Dual variables
        axes[1,1].plot(history['lambda_cov'], label='λ (coverage)', color='orange')
        nu_history = np.array(history['nu'])  # [T, K]
        for k in range(num_groups):
            axes[1,1].plot(nu_history[:, k], label=f'ν_{k} (group {k})', linestyle='--')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Dual Variables')
        axes[1,1].set_title('Dual Variables Evolution')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plot_path = output_dir / 'constrained_optimization_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved optimization plots to {plot_path}")
        
    except ImportError:
        print("⚠️ matplotlib not available. Skipping plot generation.")

if __name__ == '__main__':
    main()