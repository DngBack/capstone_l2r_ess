"""
Complete MAP Plugin Training Script
====================================

Pipeline:
1. Load gating weights & generate mixture posteriors
2. S1: Fixed-point α optimization
3. S2: Grid search (μ, γ, ν)
4. (Optional) EG-outer for worst-group
5. Evaluation với RC curve & AURC
6. Reweighting cho balanced test set

Usage:
    python train_map_plugin.py --objective balanced
    python train_map_plugin.py --objective worst --eg_outer
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, Optional
import torchvision

from src.models.gating_network_map import GatingNetwork, compute_uncertainty_for_map
from src.models.map_selector import MAPSelector, MAPConfig
from src.models.map_optimization import (
    GridSearchOptimizer,
    EGOuterOptimizer,
    RCCurveComputer
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
        'num_classes': 100,
        'num_groups': 2,
        'group_boundaries': [50],  # head: 0-49, tail: 50-99
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',
    },
    'gating': {
        'checkpoint': './checkpoints/gating_map/cifar100_lt_if100/best_gating.pth',
    },
    'map': {
        # Grid ranges (L2R defaults)
        'lambda_grid': list(np.linspace(-3.0, 3.0, 13)),  # μ_1 - μ_2
        'gamma_grid': [0.0, 0.5, 1.0, 2.0],  # Uncertainty penalty
        'nu_grid': [2.0, 5.0, 10.0],  # Sigmoid slope
        
        # Fixed-point
        'fp_iterations': 10,
        'fp_ema': 0.7,
        'fp_alpha_min': 0.1,
        
        # EG-outer
        'eg_iterations': 10,
        'eg_xi': 0.1,
        
        # Uncertainty coefficients
        'uncertainty_coeff_a': 1.0,  # H(w)
        'uncertainty_coeff_b': 1.0,  # Disagreement
        'uncertainty_coeff_d': 1.0,  # H(η̃)
    },
    'evaluation': {
        'cost_grid': list(np.linspace(-2.0, 2.0, 50)),  # For RC curve
        'use_reweighting': True,  # Reweight balanced test theo train distribution
    },
    'output': {
        'checkpoints_dir': './checkpoints/map_plugin/',
        'results_dir': './results/map_plugin/',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# DATA LOADING
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
    
    # Stack: [N, E, C]
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    
    return logits


def load_labels(splits_dir, split_name, device='cpu'):
    """Load labels."""
    import json
    
    # Load indices
    indices_file = f"{split_name}_indices.json"
    indices_path = Path(splits_dir) / indices_file
    
    if not indices_path.exists():
        raise FileNotFoundError(f"Indices not found: {indices_path}")
    
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    
    # Determine train/test
    if split_name in ['gating', 'expert', 'train']:
        cifar_train = True
    else:
        cifar_train = False
    
    # Load CIFAR-100
    dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=cifar_train,
        download=False
    )
    
    # Extract labels
    labels = torch.tensor([dataset.targets[i] for i in indices], device=device)
    
    return labels


def load_class_weights(splits_dir, device='cpu'):
    """
    Load class weights from training distribution.
    
    Dùng để reweight balanced test set → simulate long-tail performance.
    """
    weights_path = Path(splits_dir) / 'class_weights.json'
    
    if not weights_path.exists():
        print("⚠️  class_weights.json not found, using uniform weights")
        return torch.ones(100, device=device) / 100
    
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
    
    # Convert to tensor
    if isinstance(weights_data, list):
        weights = torch.tensor(weights_data, device=device, dtype=torch.float32)
    elif isinstance(weights_data, dict):
        weights = torch.tensor([weights_data[str(i)] for i in range(100)], 
                              device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected format: {type(weights_data)}")
    
    return weights


def compute_sample_weights(labels, class_weights):
    """
    Convert class weights to per-sample weights.
    
    Args:
        labels: [N] class labels
        class_weights: [C] class weights
    
    Returns:
        sample_weights: [N]
    """
    return class_weights[labels]


# ============================================================================
# MIXTURE POSTERIOR GENERATION
# ============================================================================

def generate_mixture_posteriors(
    gating: GatingNetwork,
    expert_logits: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Generate mixture posteriors và uncertainty từ gating.
    
    Args:
        gating: trained GatingNetwork
        expert_logits: [N, E, C]
    
    Returns:
        dict với 'mixture_posteriors', 'uncertainties', 'gating_weights'
    """
    gating.eval()
    
    # Convert logits to posteriors
    posteriors = torch.softmax(expert_logits, dim=-1)  # [N, E, C]
    
    # Forward gating
    with torch.no_grad():
        weights, aux = gating(posteriors)  # [N, E]
    
    # Mixture posterior
    mixture = gating.get_mixture_posterior(posteriors, weights)  # [N, C]
    
    # Uncertainty for MAP
    uncertainty = compute_uncertainty_for_map(
        posteriors, weights, mixture,
        coeffs={
            'a': CONFIG['map']['uncertainty_coeff_a'],
            'b': CONFIG['map']['uncertainty_coeff_b'],
            'd': CONFIG['map']['uncertainty_coeff_d']
        }
    )  # [N]
    
    return {
        'mixture_posteriors': mixture,
        'uncertainties': uncertainty,
        'gating_weights': weights,
        'expert_posteriors': posteriors
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_map_plugin(
    objective: str = 'balanced',
    use_eg_outer: bool = False,
    verbose: bool = True
):
    """
    Main training function.
    
    Args:
        objective: 'balanced' or 'worst'
        use_eg_outer: use EG-outer for worst-group
    """
    print("="*70)
    print("🚀 MAP PLUGIN TRAINING")
    print("="*70)
    print(f"Objective: {objective}")
    print(f"EG-Outer: {use_eg_outer}")
    print(f"Device: {DEVICE}")
    
    # Setup
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # ========================================================================
    # 1. LOAD GATING
    # ========================================================================
    print("\n" + "="*70)
    print("1. LOADING GATING NETWORK")
    print("="*70)
    
    num_experts = len(CONFIG['experts']['names'])
    num_classes = CONFIG['dataset']['num_classes']
    
    gating = GatingNetwork(
        num_experts=num_experts,
        num_classes=num_classes,
        routing='dense'
    ).to(DEVICE)
    
    gating_checkpoint_path = Path(CONFIG['gating']['checkpoint'])
    if not gating_checkpoint_path.exists():
        raise FileNotFoundError(f"Gating checkpoint not found: {gating_checkpoint_path}")
    
    checkpoint = torch.load(gating_checkpoint_path, map_location=DEVICE, weights_only=False)
    gating.load_state_dict(checkpoint['model_state_dict'])
    gating.eval()
    
    print(f"✅ Loaded gating from: {gating_checkpoint_path}")
    
    # ========================================================================
    # 2. LOAD DATA & GENERATE MIXTURE POSTERIORS
    # ========================================================================
    print("\n" + "="*70)
    print("2. LOADING DATA & GENERATING MIXTURE POSTERIORS")
    print("="*70)
    
    expert_names = CONFIG['experts']['names']
    logits_dir = CONFIG['experts']['logits_dir']
    splits_dir = CONFIG['dataset']['splits_dir']
    
    # S1: tunev (for fixed-point α)
    print("\nS1 (tunev):")
    s1_logits = load_expert_logits(expert_names, logits_dir, 'tunev', DEVICE)
    s1_labels = load_labels(splits_dir, 'tunev', DEVICE)
    s1_data = generate_mixture_posteriors(gating, s1_logits, DEVICE)
    print(f"  ✓ {s1_logits.shape[0]:,} samples")
    
    # S2: val (for model selection)
    print("\nS2 (val):")
    s2_logits = load_expert_logits(expert_names, logits_dir, 'val', DEVICE)
    s2_labels = load_labels(splits_dir, 'val', DEVICE)
    s2_data = generate_mixture_posteriors(gating, s2_logits, DEVICE)
    print(f"  ✓ {s2_logits.shape[0]:,} samples")
    
    # Test: balanced test set
    print("\nTest (balanced):")
    test_logits = load_expert_logits(expert_names, logits_dir, 'test', DEVICE)
    test_labels = load_labels(splits_dir, 'test', DEVICE)
    test_data = generate_mixture_posteriors(gating, test_logits, DEVICE)
    print(f"  ✓ {test_logits.shape[0]:,} samples")
    
    # Load class weights for reweighting
    if CONFIG['evaluation']['use_reweighting']:
        print("\nLoading class weights for reweighting...")
        class_weights = load_class_weights(splits_dir, DEVICE)
        
        # Compute sample weights
        s2_weights = compute_sample_weights(s2_labels, class_weights)
        test_weights = compute_sample_weights(test_labels, class_weights)
        
        print(f"  ✓ Class weights range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
        print(f"  ✓ Sample weights will simulate long-tail distribution on balanced test")
    else:
        s2_weights = None
        test_weights = None
    
    # ========================================================================
    # 3. INITIALIZE MAP SELECTOR
    # ========================================================================
    print("\n" + "="*70)
    print("3. INITIALIZING MAP SELECTOR")
    print("="*70)
    
    map_config = MAPConfig(
        num_classes=CONFIG['dataset']['num_classes'],
        num_groups=CONFIG['dataset']['num_groups'],
        group_boundaries=CONFIG['dataset']['group_boundaries'],
        lambda_grid=CONFIG['map']['lambda_grid'],
        gamma_grid=CONFIG['map']['gamma_grid'],
        nu_grid=CONFIG['map']['nu_grid'],
        fp_iterations=CONFIG['map']['fp_iterations'],
        fp_ema=CONFIG['map']['fp_ema'],
        fp_alpha_min=CONFIG['map']['fp_alpha_min'],
        objective=objective,
        eg_iterations=CONFIG['map']['eg_iterations'],
        eg_xi=CONFIG['map']['eg_xi']
    )
    
    selector = MAPSelector(map_config).to(DEVICE)
    print(f"✅ MAP Selector initialized")
    print(f"   Groups: {map_config.num_groups}")
    print(f"   Boundaries: {map_config.group_boundaries}")
    
    # ========================================================================
    # 4. OPTIMIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("4. OPTIMIZATION")
    print("="*70)
    
    if use_eg_outer and objective == 'worst':
        # EG-Outer for worst-group
        print("\nUsing EG-Outer for worst-group...")
        eg_opt = EGOuterOptimizer(map_config)
        
        best_result, best_beta = eg_opt.optimize(
            selector,
            s1_data['mixture_posteriors'],
            s1_data['uncertainties'],
            s1_labels,
            s2_data['mixture_posteriors'],
            s2_data['uncertainties'],
            s2_labels,
            cost=0.0,
            s2_weights=s2_weights,
            verbose=True
        )
    else:
        # Standard grid search
        print("\nGrid search...")
        grid_opt = GridSearchOptimizer(map_config)
        
        best_result = grid_opt.search(
            selector,
            s1_data['mixture_posteriors'],
            s1_data['uncertainties'],
            s1_labels,
            s2_data['mixture_posteriors'],
            s2_data['uncertainties'],
            s2_labels,
            cost=0.0,
            beta=None,
            s2_weights=s2_weights,
            verbose=True
        )
    
    # Set best parameters
    selector.set_parameters(
        alpha=best_result.alpha.to(DEVICE),
        mu=best_result.mu.to(DEVICE),
        gamma=best_result.gamma,
        cost=0.0  # Will sweep this for RC curve
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"   Best λ: {best_result.lambda_val:.3f}")
    print(f"   Best γ: {best_result.gamma:.3f}")
    print(f"   Best ν: {best_result.nu:.3f}")
    
    # ========================================================================
    # 5. EVALUATION ON TEST SET
    # ========================================================================
    print("\n" + "="*70)
    print("5. EVALUATION ON TEST SET")
    print("="*70)
    
    # RC Curve
    print("\nComputing RC curve...")
    rc_computer = RCCurveComputer(map_config)
    
    rc_data = rc_computer.compute_rc_curve(
        selector,
        test_data['mixture_posteriors'],
        test_data['uncertainties'],
        test_labels,
        alpha=best_result.alpha.to(DEVICE),
        mu=best_result.mu.to(DEVICE),
        gamma=best_result.gamma,
        cost_grid=np.array(CONFIG['evaluation']['cost_grid']),
        sample_weights=test_weights
    )
    
    print(f"✅ RC curve computed")
    print(f"   AURC: {rc_data['aurc']:.4f}")
    print(f"   Points: {len(rc_data['rejection_rates'])}")
    
    # Specific operating points
    print("\nOperating points:")
    for i, rej_rate in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        # Find closest point
        idx = np.argmin(np.abs(rc_data['rejection_rates'] - rej_rate))
        actual_rej = rc_data['rejection_rates'][idx]
        error = rc_data['selective_errors'][idx]
        group_errors = rc_data['group_errors_list'][idx]
        
        print(f"  Rejection≈{rej_rate:.1f}: error={error:.4f}, "
              f"head={group_errors[0]:.4f}, tail={group_errors[1]:.4f}")
    
    # ========================================================================
    # 6. SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("6. SAVING RESULTS")
    print("="*70)
    
    # Create output directories
    checkpoint_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    results_dir = Path(CONFIG['output']['results_dir']) / CONFIG['dataset']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save MAP selector parameters
    save_dict = {
        'alpha': best_result.alpha.cpu().numpy().tolist(),
        'mu': best_result.mu.cpu().numpy().tolist(),
        'gamma': best_result.gamma,
        'nu': best_result.nu,
        'lambda': best_result.lambda_val,
        'objective': objective,
        'use_eg_outer': use_eg_outer,
        'config': CONFIG
    }
    
    with open(checkpoint_dir / 'map_parameters.json', 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"✅ Saved parameters to: {checkpoint_dir / 'map_parameters.json'}")
    
    # Save RC curve data
    rc_save_dict = {
        'rejection_rates': rc_data['rejection_rates'].tolist(),
        'selective_errors': rc_data['selective_errors'].tolist(),
        'aurc': float(rc_data['aurc']),
        'cost_grid': rc_data['cost_grid'].tolist(),
        'group_errors': [g.tolist() for g in rc_data['group_errors_list']]
    }
    
    with open(results_dir / 'rc_curve.json', 'w') as f:
        json.dump(rc_save_dict, f, indent=2)
    
    print(f"✅ Saved RC curve to: {results_dir / 'rc_curve.json'}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"AURC: {rc_data['aurc']:.4f}")
    print(f"Best configuration:")
    print(f"  λ = {best_result.lambda_val:.3f}")
    print(f"  γ = {best_result.gamma:.3f}")
    print(f"  ν = {best_result.nu:.3f}")
    
    return selector, best_result, rc_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MAP Plugin')
    parser.add_argument('--objective', type=str, default='balanced',
                       choices=['balanced', 'worst'],
                       help='Optimization objective')
    parser.add_argument('--eg_outer', action='store_true',
                       help='Use EG-outer for worst-group')
    parser.add_argument('--no_reweight', action='store_true',
                       help='Disable reweighting on test set')
    
    args = parser.parse_args()
    
    # Update config
    if args.no_reweight:
        CONFIG['evaluation']['use_reweighting'] = False
    
    # Train
    selector, result, rc_data = train_map_plugin(
        objective=args.objective,
        use_eg_outer=args.eg_outer,
        verbose=True
    )
    
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
