#!/usr/bin/env python3
"""
LtR Plugin Training Script - Paper Compliant
============================================

Trains LtR plugin using the paper-compliant CE expert.
Implements Algorithm 1 (Power Iteration) from the paper.

Paper Reference: "Learning to Reject Meets Long-Tail Learning" (ICLR 2024)
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Import our LtR plugin implementation
from src.models.ltr_plugin import LtRPlugin, LtRPluginConfig, LtRPowerIterOptimizer
from src.metrics.selective_metrics import calculate_selective_errors

# ============================================================================
# PAPER CONFIGURATION
# ============================================================================

PAPER_CONFIG = {
    'dataset': {
        'name': 'cifar100_lt',
        'num_classes': 100,
        'num_groups': 2,  # Head/Tail
        'threshold': 20,  # Classes with <=20 samples are tail
    },
    'expert': {
        'name': 'ce_expert',
        'logits_dir': './outputs/logits_paper',
    },
    'ltr': {
        'param_mode': 'group',  # Group-level parameters
        'alpha_grid': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        'mu_grid': [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        'cost_grid': [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.91, 0.95, 0.97, 0.99],
        # Cost sweep for RC curve (Paper Method)
        'cost_sweep': [
            0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4,
            0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7,
            0.72, 0.74, 0.76, 0.78, 0.8
        ],
    },
    'optimizer': {
        'type': 'power_iter',  # Algorithm 1 from paper
        'num_iters': 10,       # M in Algorithm 1
        'alpha_init_mode': 'prior',
    },
    'output': {
        'results_dir': './results/ltr_plugin_paper',
        'plots_dir': './results/ltr_plugin_paper/plots',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# DATA LOADING
# ============================================================================

def get_cifar100_lt_counts(imb_factor=100, num_classes=100):
    """Generate CIFAR-100-LT class counts using exponential profile."""
    img_max = 500.0
    counts = []
    for cls_idx in range(num_classes):
        num = img_max * (imb_factor ** (-cls_idx / (num_classes - 1.0)))
        counts.append(max(1, int(num)))
    return counts

def get_class_to_group_by_threshold(class_counts, threshold=20):
    """Map classes to groups based on sample count threshold."""
    num_classes = len(class_counts)
    class_to_group = torch.zeros(num_classes, dtype=torch.long)
    
    for class_idx, count in enumerate(class_counts):
        if count > threshold:
            class_to_group[class_idx] = 0  # Head group
        else:
            class_to_group[class_idx] = 1  # Tail group
    
    return class_to_group

def load_expert_logits(logits_dir, split='test'):
    """Load expert logits and targets."""
    # Try paper-compliant format first
    paper_logits_path = Path(logits_dir) / f"ce_expert_{split}_logits.pth"
    
    if paper_logits_path.exists():
        print(f"Loading from paper-compliant format: {paper_logits_path}")
        data = torch.load(paper_logits_path, map_location='cpu')
        logits = data['logits'].float()
        targets = data['targets'].long()
    else:
        # Fallback to existing format
        existing_logits_path = Path('./outputs/logits/cifar100_lt_if100/ce_baseline') / f"{split}_logits.pt"
        
        if not existing_logits_path.exists():
            raise FileNotFoundError(f"Logits file not found in either format:\n"
                                  f"  Paper format: {paper_logits_path}\n"
                                  f"  Existing format: {existing_logits_path}\n"
                                  f"Please run 'python train_ce_expert_paper.py' first.")
        
        print(f"Loading from existing format: {existing_logits_path}")
        logits = torch.load(existing_logits_path, map_location='cpu').float()
        
        # Load targets from test indices
        with open('data/cifar100_lt_if100_splits_fixed/test_indices.json', 'r') as f:
            test_indices = json.load(f)
        
        import torchvision
        test_dataset = torchvision.datasets.CIFAR100(root='data', train=False, download=False)
        targets = torch.tensor([test_dataset.targets[i] for i in test_indices])
    
    print(f"Loaded {split} logits: {logits.shape}")
    print(f"Loaded {split} targets: {targets.shape}")
    
    return logits, targets

# ============================================================================
# PLUGIN TRAINING
# ============================================================================

def train_single_cost(plugin, train_logits, train_targets, test_logits, test_targets, 
                     class_to_group, cost, objective='balanced', verbose=True):
    """Train plugin for a single cost value."""
    
    # Create optimizer
    config = LtRPluginConfig(
        num_classes=PAPER_CONFIG['dataset']['num_classes'],
        num_groups=PAPER_CONFIG['dataset']['num_groups'],
        class_to_group=class_to_group,
        param_mode=PAPER_CONFIG['ltr']['param_mode'],
        alpha_grid=PAPER_CONFIG['ltr']['alpha_grid'],
        mu_grid=PAPER_CONFIG['ltr']['mu_grid'],
        cost_grid=[cost],  # Single cost
        objective=objective
    )
    
    optimizer = LtRPowerIterOptimizer(
        config=config,
        num_iters=PAPER_CONFIG['optimizer']['num_iters'],
        alpha_init_mode=PAPER_CONFIG['optimizer']['alpha_init_mode']
    )
    
    # Convert logits to probabilities
    train_probs = torch.softmax(train_logits, dim=1)
    test_probs = torch.softmax(test_logits, dim=1)
    
    # Search for optimal parameters
    result = optimizer.search(
        plugin=plugin,
        mixture_posterior=train_probs,
        labels=train_targets,
        verbose=verbose
    )
    
    # Evaluate on test set
    plugin.set_parameters(
        alpha=torch.tensor(result.alpha, dtype=torch.float32, device=DEVICE),
        mu=torch.tensor(result.mu, dtype=torch.float32, device=DEVICE),
        cost=cost
    )
    
    with torch.no_grad():
        test_preds = plugin.predict_class(test_probs.to(DEVICE))
        test_reject = plugin.predict_reject(test_probs.to(DEVICE))
    
    # Compute metrics
    metrics = calculate_selective_errors(
        test_preds.cpu(), test_targets, test_reject.cpu(), class_to_group
    )
    
    return {
        'cost': cost,
        'alpha': result.alpha,
        'mu': result.mu,
        'selective_error': metrics['selective_error'],
        'coverage': metrics['coverage'],
        'group_errors': metrics['group_errors'],
        'worst_group_error': metrics['worst_group_error'],
        'balanced_error': np.mean(metrics['group_errors']),
        'objective_value': result.objective_value
    }

def train_with_cost_sweep(plugin, train_logits, train_targets, test_logits, test_targets,
                         class_to_group, objective='balanced'):
    """Train plugin with cost sweep for RC curve."""
    
    print(f"\nTraining LtR Plugin with cost sweep...")
    print(f"Objective: {objective}")
    print(f"Cost grid: {len(PAPER_CONFIG['ltr']['cost_sweep'])} points")
    
    results = []
    
    for i, cost in enumerate(tqdm(PAPER_CONFIG['ltr']['cost_sweep'], desc="Cost sweep")):
        verbose = (i == 0 or i == len(PAPER_CONFIG['ltr']['cost_sweep']) - 1)
        
        result = train_single_cost(
            plugin=plugin,
            train_logits=train_logits,
            train_targets=train_targets,
            test_logits=test_logits,
            test_targets=test_targets,
            class_to_group=class_to_group,
            cost=cost,
            objective=objective,
            verbose=verbose
        )
        
        results.append(result)
        
        if verbose:
            print(f"Cost {cost:.3f}: Coverage={result['coverage']:.3f}, "
                  f"Balanced Error={result['balanced_error']:.4f}, "
                  f"Worst Error={result['worst_group_error']:.4f}")
    
    return results

# ============================================================================
# PLOTTING
# ============================================================================

def plot_rc_curves(results, objective, output_dir):
    """Plot Risk-Coverage curves."""
    
    # Extract data
    costs = [r['cost'] for r in results]
    coverages = [r['coverage'] for r in results]
    balanced_errors = [r['balanced_error'] for r in results]
    worst_errors = [r['worst_group_error'] for r in results]
    
    # Calculate AURC
    aurc_balanced = np.trapz(balanced_errors, coverages)
    aurc_worst = np.trapz(worst_errors, coverages)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Balanced Error
    ax1.plot(coverages, balanced_errors, 'o-', linewidth=3, markersize=6,
             color='green', markerfacecolor='lightgreen', 
             markeredgecolor='darkgreen', markeredgewidth=1,
             label=f'Plug-in [Balanced] (AURC={aurc_balanced:.4f})')
    
    ax1.set_xlabel('Proportion of Rejections')
    ax1.set_ylabel('Balanced Error')
    ax1.set_title(f'Balanced Error vs Rejection Rate ({objective.capitalize()})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # Plot 2: Worst Error
    ax2.plot(coverages, worst_errors, 'o-', linewidth=3, markersize=6,
             color='red', markerfacecolor='lightcoral',
             markeredgecolor='darkred', markeredgewidth=1,
             label=f'Plug-in [Worst] (AURC={aurc_worst:.4f})')
    
    ax2.set_xlabel('Proportion of Rejections')
    ax2.set_ylabel('Worst-group Error')
    ax2.set_title(f'Worst-group Error vs Rejection Rate ({objective.capitalize()})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f'ltr_rc_curves_{objective}_paper.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"RC curves saved to: {output_path}")
    
    return aurc_balanced, aurc_worst

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LtR Plugin')
    parser.add_argument('--objective', type=str, default='balanced',
                       choices=['balanced', 'worst'], help='Objective function')
    parser.add_argument('--cost_sweep', action='store_true',
                       help='Run cost sweep for RC curve')
    parser.add_argument('--single_cost', type=float, default=0.5,
                       help='Single cost value (if not cost_sweep)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LTR PLUGIN TRAINING - PAPER COMPLIANT")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Objective: {args.objective}")
    print(f"Mode: {'Cost sweep' if args.cost_sweep else f'Single cost ({args.single_cost})'}")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(PAPER_CONFIG['seed'])
    np.random.seed(PAPER_CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(PAPER_CONFIG['seed'])
    
    # Create output directories
    results_dir = Path(PAPER_CONFIG['output']['results_dir'])
    plots_dir = Path(PAPER_CONFIG['output']['plots_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load expert logits
    print("\nLoading expert logits...")
    train_logits, train_targets = load_expert_logits(PAPER_CONFIG['expert']['logits_dir'], 'train')
    test_logits, test_targets = load_expert_logits(PAPER_CONFIG['expert']['logits_dir'], 'test')
    
    # Create group mapping
    print("\nCreating group mapping...")
    class_counts = get_cifar100_lt_counts()
    class_to_group = get_class_to_group_by_threshold(class_counts, PAPER_CONFIG['dataset']['threshold'])
    
    head_classes = (class_to_group == 0).sum().item()
    tail_classes = (class_to_group == 1).sum().item()
    print(f"Head classes (>20 samples): {head_classes}")
    print(f"Tail classes (<=20 samples): {tail_classes}")
    
    # Create plugin
    print("\nCreating LtR Plugin...")
    plugin = LtRPlugin(
        num_classes=PAPER_CONFIG['dataset']['num_classes'],
        num_groups=PAPER_CONFIG['dataset']['num_groups'],
        class_to_group=class_to_group.to(DEVICE)
    ).to(DEVICE)
    
    # Training
    if args.cost_sweep:
        # Cost sweep for RC curve
        results = train_with_cost_sweep(
            plugin=plugin,
            train_logits=train_logits,
            train_targets=train_targets,
            test_logits=test_logits,
            test_targets=test_targets,
            class_to_group=class_to_group,
            objective=args.objective
        )
        
        # Save results
        results_path = results_dir / f'ltr_plugin_{args.objective}_paper.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot RC curves
        aurc_balanced, aurc_worst = plot_rc_curves(results, args.objective, plots_dir)
        
        print(f"\n" + "=" * 80)
        print("COST SWEEP COMPLETED")
        print("=" * 80)
        print(f"Results saved to: {results_path}")
        print(f"AURC (Balanced): {aurc_balanced:.4f}")
        print(f"AURC (Worst): {aurc_worst:.4f}")
        
    else:
        # Single cost training
        result = train_single_cost(
            plugin=plugin,
            train_logits=train_logits,
            train_targets=train_targets,
            test_logits=test_logits,
            test_targets=test_targets,
            class_to_group=class_to_group,
            cost=args.single_cost,
            objective=args.objective,
            verbose=True
        )
        
        print(f"\n" + "=" * 80)
        print("SINGLE COST TRAINING COMPLETED")
        print("=" * 80)
        print(f"Cost: {result['cost']:.3f}")
        print(f"Coverage: {result['coverage']:.3f}")
        print(f"Balanced Error: {result['balanced_error']:.4f}")
        print(f"Worst Error: {result['worst_group_error']:.4f}")
        print(f"Alpha: {result['alpha']}")
        print(f"Mu: {result['mu']}")
    
    print(f"\n" + "=" * 80)
    print("LTR PLUGIN TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
