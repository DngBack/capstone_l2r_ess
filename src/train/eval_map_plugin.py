"""
MAP Plugin Evaluation Script
=============================

Comprehensive evaluation với:
- Group-wise errors & coverage
- RC curves & AURC
- Calibration metrics
- Reweighting cho balanced test set

Usage:
    python eval_map_plugin.py
    python eval_map_plugin.py --no_reweight  # Disable reweighting
    python eval_map_plugin.py --visualize    # Generate plots
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.gating_network_map import GatingNetwork, compute_uncertainty_for_map
from src.models.map_selector import MAPSelector, MAPConfig, compute_selective_metrics
from src.models.map_optimization import RCCurveComputer


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
        'num_classes': 100,
        'num_groups': 2,
        'group_boundaries': [50],
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'gating': {
        'checkpoint': './checkpoints/gating_map/cifar100_lt_if100/best_gating.pth',
    },
    'map': {
        'parameters': './checkpoints/map_plugin/cifar100_lt_if100/map_parameters.json',
        'uncertainty_coeff_a': 1.0,
        'uncertainty_coeff_b': 1.0,
        'uncertainty_coeff_d': 1.0,
    },
    'evaluation': {
        'cost_grid': list(np.linspace(-2.0, 2.0, 100)),
        'use_reweighting': True,
        'splits': ['val', 'test'],  # Evaluate on both
    },
    'output': {
        'results_dir': './results/map_plugin/',
    },
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# DATA LOADING (reuse from train_map_plugin.py)
# ============================================================================

def load_expert_logits(expert_names, logits_dir, split_name, device='cpu'):
    """Load expert logits."""
    logits_list = []
    
    for expert_name in expert_names:
        logits_path = Path(logits_dir) / expert_name / f"{split_name}_logits.pt"
        
        if not logits_path.exists():
            raise FileNotFoundError(f"Logits not found: {logits_path}")
        
        logits_e = torch.load(logits_path, map_location=device).float()
        logits_list.append(logits_e)
    
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    return logits


def load_labels(splits_dir, split_name, device='cpu'):
    """Load labels."""
    import json
    
    indices_file = f"{split_name}_indices.json"
    indices_path = Path(splits_dir) / indices_file
    
    if not indices_path.exists():
        raise FileNotFoundError(f"Indices not found: {indices_path}")
    
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    
    if split_name in ['gating', 'expert', 'train']:
        cifar_train = True
    else:
        cifar_train = False
    
    dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=cifar_train,
        download=False
    )
    
    labels = torch.tensor([dataset.targets[i] for i in indices], device=device)
    return labels


def load_class_weights(splits_dir, device='cpu'):
    """Load class weights from training distribution."""
    weights_path = Path(splits_dir) / 'class_weights.json'
    
    if not weights_path.exists():
        print("⚠️  class_weights.json not found, using uniform weights")
        return torch.ones(100, device=device) / 100
    
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
    
    if isinstance(weights_data, list):
        weights = torch.tensor(weights_data, device=device, dtype=torch.float32)
    elif isinstance(weights_data, dict):
        weights = torch.tensor([weights_data[str(i)] for i in range(100)], 
                              device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected format: {type(weights_data)}")
    
    return weights


def compute_sample_weights(labels, class_weights):
    """Convert class weights to per-sample weights."""
    return class_weights[labels]


def generate_mixture_posteriors(gating, expert_logits, device='cpu'):
    """Generate mixture posteriors và uncertainty."""
    gating.eval()
    
    posteriors = torch.softmax(expert_logits, dim=-1)
    
    with torch.no_grad():
        weights, aux = gating(posteriors)
    
    mixture = gating.get_mixture_posterior(posteriors, weights)
    
    uncertainty = compute_uncertainty_for_map(
        posteriors, weights, mixture,
        coeffs={
            'a': CONFIG['map']['uncertainty_coeff_a'],
            'b': CONFIG['map']['uncertainty_coeff_b'],
            'd': CONFIG['map']['uncertainty_coeff_d']
        }
    )
    
    return {
        'mixture_posteriors': mixture,
        'uncertainties': uncertainty,
        'gating_weights': weights,
        'expert_posteriors': posteriors
    }


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_classification_metrics(
    posteriors: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Returns:
        dict với accuracy, top5_accuracy, nll, ece
    """
    preds = torch.argmax(posteriors, dim=-1)
    
    if sample_weights is None:
        sample_weights = torch.ones_like(labels, dtype=torch.float32)
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum()
    
    # Accuracy
    correct = (preds == labels).float()
    accuracy = (correct * sample_weights).sum().item()
    
    # Top-5
    _, top5_preds = torch.topk(posteriors, k=5, dim=-1)
    top5_correct = torch.any(top5_preds == labels.unsqueeze(-1), dim=-1).float()
    top5_accuracy = (top5_correct * sample_weights).sum().item()
    
    # NLL
    log_probs = torch.log(posteriors + 1e-10)
    nll_per_sample = F.nll_loss(log_probs, labels, reduction='none')
    nll = (nll_per_sample * sample_weights).sum().item()
    
    # ECE (10 bins)
    ece = compute_ece(posteriors, labels, sample_weights, n_bins=10)
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'nll': nll,
        'ece': ece
    }


def compute_ece(posteriors, labels, sample_weights, n_bins=10):
    """Expected Calibration Error."""
    confidences, preds = torch.max(posteriors, dim=-1)
    correct = (preds == labels).float()
    
    bins = torch.linspace(0, 1, n_bins + 1, device=posteriors.device)
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        
        if bin_mask.sum() > 0:
            bin_weights = sample_weights[bin_mask]
            bin_conf = confidences[bin_mask]
            bin_correct = correct[bin_mask]
            
            # Weighted averages
            avg_conf = (bin_conf * bin_weights).sum() / bin_weights.sum()
            avg_acc = (bin_correct * bin_weights).sum() / bin_weights.sum()
            
            bin_weight = bin_weights.sum()
            ece += bin_weight * torch.abs(avg_conf - avg_acc)
    
    return ece.item()


def compute_group_metrics(
    posteriors: torch.Tensor,
    labels: torch.Tensor,
    group_boundaries: List[int],
    sample_weights: Optional[torch.Tensor] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics per group.
    
    Args:
        group_boundaries: e.g., [50] → group 0: [0,50), group 1: [50,100)
    
    Returns:
        dict[group_id] → metrics dict
    """
    num_groups = len(group_boundaries) + 1
    
    # Assign groups
    groups = torch.zeros_like(labels, dtype=torch.long)
    for g_id, boundary in enumerate(group_boundaries):
        groups[labels >= boundary] = g_id + 1
    
    if sample_weights is None:
        sample_weights = torch.ones_like(labels, dtype=torch.float32)
    
    group_metrics = {}
    
    for g_id in range(num_groups):
        mask = (groups == g_id)
        
        if mask.sum() == 0:
            continue
        
        g_posteriors = posteriors[mask]
        g_labels = labels[mask]
        g_weights = sample_weights[mask]
        
        metrics = compute_classification_metrics(g_posteriors, g_labels, g_weights)
        metrics['count'] = mask.sum().item()
        metrics['weight_sum'] = g_weights.sum().item()
        
        group_metrics[g_id] = metrics
    
    return group_metrics


# ============================================================================
# RC CURVE VISUALIZATION
# ============================================================================

def plot_rc_curve(
    rc_data: Dict,
    save_path: Optional[Path] = None,
    title: str = "Risk-Coverage Curve"
):
    """Plot RC curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    rejection_rates = rc_data['rejection_rates']
    selective_errors = rc_data['selective_errors']
    aurc = rc_data['aurc']
    
    # Main curve
    ax.plot(rejection_rates, selective_errors, 'b-', linewidth=2, label='RC Curve')
    
    # AUC
    ax.fill_between(rejection_rates, 0, selective_errors, alpha=0.2)
    
    # Oracle (minimum error at each coverage)
    ax.axhline(y=selective_errors.min(), color='g', linestyle='--', 
               linewidth=1, label='Oracle (min error)')
    
    ax.set_xlabel('Rejection Rate', fontsize=12)
    ax.set_ylabel('Selective Error', fontsize=12)
    ax.set_title(f"{title}\nAURC = {aurc:.4f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to: {save_path}")
    
    plt.close()


def plot_group_rc_curves(
    rc_data: Dict,
    save_path: Optional[Path] = None,
    title: str = "Group-wise RC Curves"
):
    """Plot RC curves per group."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    rejection_rates = rc_data['rejection_rates']
    group_errors_list = rc_data['group_errors_list']
    
    num_groups = len(group_errors_list[0])
    colors = ['blue', 'red', 'green', 'orange']
    
    for g_id in range(num_groups):
        errors = np.array([ge[g_id] for ge in group_errors_list])
        ax.plot(rejection_rates, errors, 
                color=colors[g_id], linewidth=2, 
                label=f'Group {g_id}')
    
    # Overall
    selective_errors = rc_data['selective_errors']
    ax.plot(rejection_rates, selective_errors, 
            'k--', linewidth=2, label='Overall')
    
    ax.set_xlabel('Rejection Rate', fontsize=12)
    ax.set_ylabel('Selective Error', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to: {save_path}")
    
    plt.close()


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_map_plugin(
    split_name: str = 'test',
    use_reweighting: bool = True,
    visualize: bool = False
) -> Dict:
    """
    Evaluate MAP plugin on a split.
    
    Args:
        split_name: 'val' or 'test'
        use_reweighting: reweight balanced test set
        visualize: generate plots
    
    Returns:
        dict với all metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {split_name.upper()}")
    print(f"{'='*70}")
    
    # ========================================================================
    # 1. LOAD GATING
    # ========================================================================
    print("\n1. Loading gating network...")
    
    num_experts = len(CONFIG['experts']['names'])
    num_classes = CONFIG['dataset']['num_classes']
    
    gating = GatingNetwork(
        num_experts=num_experts,
        num_classes=num_classes,
        routing='dense'
    ).to(DEVICE)
    
    gating_checkpoint_path = Path(CONFIG['gating']['checkpoint'])
    checkpoint = torch.load(gating_checkpoint_path, map_location=DEVICE, weights_only=False)
    gating.load_state_dict(checkpoint['model_state_dict'])
    gating.eval()
    
    print(f"  ✓ Gating loaded")
    
    # ========================================================================
    # 2. LOAD DATA
    # ========================================================================
    print("\n2. Loading data...")
    
    logits = load_expert_logits(
        CONFIG['experts']['names'],
        CONFIG['experts']['logits_dir'],
        split_name,
        DEVICE
    )
    labels = load_labels(
        CONFIG['dataset']['splits_dir'],
        split_name,
        DEVICE
    )
    
    print(f"  ✓ {logits.shape[0]:,} samples")
    
    # Generate mixture posteriors
    data = generate_mixture_posteriors(gating, logits, DEVICE)
    
    # Sample weights
    if use_reweighting:
        class_weights = load_class_weights(CONFIG['dataset']['splits_dir'], DEVICE)
        sample_weights = compute_sample_weights(labels, class_weights)
        print(f"  ✓ Reweighting enabled")
    else:
        sample_weights = None
        print(f"  ✓ No reweighting (uniform)")
    
    # ========================================================================
    # 3. LOAD MAP PARAMETERS
    # ========================================================================
    print("\n3. Loading MAP parameters...")
    
    params_path = Path(CONFIG['map']['parameters'])
    if not params_path.exists():
        raise FileNotFoundError(f"MAP parameters not found: {params_path}")
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Initialize selector
    map_config = MAPConfig(
        num_classes=CONFIG['dataset']['num_classes'],
        num_groups=CONFIG['dataset']['num_groups'],
        group_boundaries=CONFIG['dataset']['group_boundaries'],
        objective=params.get('objective', 'balanced')
    )
    
    selector = MAPSelector(map_config).to(DEVICE)
    
    # Set parameters
    alpha = torch.tensor(params['alpha'], device=DEVICE, dtype=torch.float32)
    mu = torch.tensor(params['mu'], device=DEVICE, dtype=torch.float32)
    gamma = params['gamma']
    
    selector.set_parameters(alpha=alpha, mu=mu, gamma=gamma, cost=0.0)
    
    print(f"  ✓ MAP parameters loaded")
    print(f"     λ = {params['lambda']:.3f}")
    print(f"     γ = {gamma:.3f}")
    print(f"     ν = {params['nu']:.3f}")
    
    # ========================================================================
    # 4. CLASSIFICATION METRICS (without rejection)
    # ========================================================================
    print("\n4. Classification metrics (full test set)...")
    
    class_metrics = compute_classification_metrics(
        data['mixture_posteriors'],
        labels,
        sample_weights
    )
    
    print(f"  Overall:")
    print(f"    Accuracy: {class_metrics['accuracy']:.4f}")
    print(f"    Top-5 Acc: {class_metrics['top5_accuracy']:.4f}")
    print(f"    NLL: {class_metrics['nll']:.4f}")
    print(f"    ECE: {class_metrics['ece']:.4f}")
    
    # Group-wise
    group_metrics = compute_group_metrics(
        data['mixture_posteriors'],
        labels,
        CONFIG['dataset']['group_boundaries'],
        sample_weights
    )
    
    print(f"\n  Group-wise:")
    for g_id, metrics in group_metrics.items():
        print(f"    Group {g_id}: Acc={metrics['accuracy']:.4f}, "
              f"Count={metrics['count']}, Weight={metrics['weight_sum']:.4f}")
    
    # ========================================================================
    # 5. RC CURVE
    # ========================================================================
    print("\n5. Computing RC curve...")
    
    rc_computer = RCCurveComputer(map_config)
    
    rc_data = rc_computer.compute_rc_curve(
        selector,
        data['mixture_posteriors'],
        data['uncertainties'],
        labels,
        alpha=alpha,
        mu=mu,
        gamma=gamma,
        cost_grid=np.array(CONFIG['evaluation']['cost_grid']),
        sample_weights=sample_weights
    )
    
    print(f"  ✓ AURC: {rc_data['aurc']:.4f}")
    
    # Operating points
    print(f"\n  Operating points:")
    for target_rej in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        idx = np.argmin(np.abs(rc_data['rejection_rates'] - target_rej))
        actual_rej = rc_data['rejection_rates'][idx]
        error = rc_data['selective_errors'][idx]
        group_errors = rc_data['group_errors_list'][idx]
        
        print(f"    Rej≈{target_rej:.1f}: error={error:.4f}, "
              f"head={group_errors[0]:.4f}, tail={group_errors[1]:.4f}")
    
    # ========================================================================
    # 6. VISUALIZATION
    # ========================================================================
    if visualize:
        print("\n6. Generating visualizations...")
        
        results_dir = Path(CONFIG['output']['results_dir']) / CONFIG['dataset']['name']
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # RC curve
        plot_rc_curve(
            rc_data,
            save_path=results_dir / f'rc_curve_{split_name}.png',
            title=f'Risk-Coverage Curve ({split_name})'
        )
        
        # Group-wise RC curves
        plot_group_rc_curves(
            rc_data,
            save_path=results_dir / f'rc_curve_groups_{split_name}.png',
            title=f'Group-wise RC Curves ({split_name})'
        )
        
        print(f"  ✓ Visualizations saved")
    
    # ========================================================================
    # 7. COLLECT RESULTS
    # ========================================================================
    results = {
        'split': split_name,
        'use_reweighting': use_reweighting,
        'classification_metrics': class_metrics,
        'group_metrics': {int(k): v for k, v in group_metrics.items()},
        'rc_curve': {
            'aurc': float(rc_data['aurc']),
            'rejection_rates': rc_data['rejection_rates'].tolist(),
            'selective_errors': rc_data['selective_errors'].tolist(),
            'group_errors': [g.tolist() for g in rc_data['group_errors_list']]
        },
        'map_parameters': {
            'lambda': params['lambda'],
            'gamma': params['gamma'],
            'nu': params['nu'],
            'alpha': params['alpha'],
            'mu': params['mu']
        }
    }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate MAP Plugin')
    parser.add_argument('--split', type=str, default='test',
                       choices=['val', 'test'],
                       help='Split to evaluate on')
    parser.add_argument('--no_reweight', action='store_true',
                       help='Disable reweighting')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Update config
    if args.no_reweight:
        CONFIG['evaluation']['use_reweighting'] = False
    
    # Evaluate
    print("="*70)
    print("🔍 MAP PLUGIN EVALUATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    results = evaluate_map_plugin(
        split_name=args.split,
        use_reweighting=CONFIG['evaluation']['use_reweighting'],
        visualize=args.visualize
    )
    
    # Save results
    results_dir = Path(CONFIG['output']['results_dir']) / CONFIG['dataset']['name']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f'evaluation_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Split: {args.split}")
    print(f"Accuracy: {results['classification_metrics']['accuracy']:.4f}")
    print(f"AURC: {results['rc_curve']['aurc']:.4f}")
    print(f"Head Acc: {results['group_metrics'][0]['accuracy']:.4f}")
    print(f"Tail Acc: {results['group_metrics'][1]['accuracy']:.4f}")
    
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
