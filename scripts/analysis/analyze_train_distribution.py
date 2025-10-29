#!/usr/bin/env python3
"""
Analyze training distribution and importance weighting for CIFAR-100 long-tail learning.
"""

import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CFG = type('Config', (), {
    'splits_dir': 'data/cifar100_lt_if100_splits_fixed',
    'num_classes': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})()

def load_train_class_counts():
    """Load training class counts from splits."""
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]
    
    return np.array(class_counts, dtype=np.int64)

def load_class_to_group():
    """Load class-to-group mapping."""
    groups_path = Path(CFG.splits_dir) / "group_config.json"
    with open(groups_path, "r", encoding="utf-8") as f:
        group_config = json.load(f)
    
    # Create class-to-group mapping
    class_to_group = np.zeros(CFG.num_classes, dtype=np.int32)
    for group_idx, classes in enumerate(group_config['groups']):
        for class_idx in classes:
            class_to_group[class_idx] = group_idx
    
    return class_to_group

def analyze_training_distribution():
    """Analyze training distribution in detail."""
    print("=" * 80)
    print("CIFAR-100 LONG-TAIL TRAINING DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Load data
    class_counts = load_train_class_counts()
    class_to_group = load_class_to_group()
    
    # Basic statistics
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    num_groups = len(np.unique(class_to_group))
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total training samples: {total_samples:,}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Number of groups: {num_groups}")
    
    # Group analysis
    group_counts = np.zeros(num_groups, dtype=np.int64)
    for g in range(num_groups):
        group_mask = (class_to_group == g)
        group_counts[g] = class_counts[group_mask].sum()
    
    print(f"\n📈 GROUP ANALYSIS:")
    for g in range(num_groups):
        group_classes = np.where(class_to_group == g)[0]
        print(f"   Group {g}: {group_counts[g]:,} samples from {len(group_classes)} classes")
        print(f"      Classes: {group_classes[:5]}{'...' if len(group_classes) > 5 else ''}")
    
    # Class distribution analysis
    print(f"\n📋 CLASS DISTRIBUTION ANALYSIS:")
    print(f"   Head class (most samples): {class_counts.max():,} samples")
    print(f"   Tail class (least samples): {class_counts.min():,} samples")
    print(f"   Imbalance ratio: {class_counts.max() / class_counts.min():.1f}x")
    
    # Calculate importance weights
    train_probs = class_counts / total_samples
    test_probs = np.ones(num_classes) / num_classes  # Balanced test set
    importance_weights = train_probs  # Simplified as discussed
    
    print(f"\n⚖️  IMPORTANCE WEIGHTING:")
    print(f"   Training distribution range: {train_probs.min():.6f} to {train_probs.max():.6f}")
    print(f"   Test distribution (balanced): {test_probs[0]:.6f}")
    print(f"   Importance weights range: {importance_weights.min():.6f} to {importance_weights.max():.6f}")
    print(f"   Weight ratio (head/tail): {importance_weights.max() / importance_weights.min():.1f}x")
    
    # Detailed class analysis
    print(f"\n📝 DETAILED CLASS ANALYSIS:")
    print(f"{'Class':<6} {'Group':<6} {'Samples':<8} {'Train Prob':<12} {'Test Prob':<12} {'Weight':<12} {'Ratio':<8}")
    print("-" * 80)
    
    # Sort by sample count (descending)
    sorted_indices = np.argsort(class_counts)[::-1]
    
    for i, class_idx in enumerate(sorted_indices):
        group = class_to_group[class_idx]
        samples = class_counts[class_idx]
        train_prob = train_probs[class_idx]
        test_prob = test_probs[class_idx]
        weight = importance_weights[class_idx]
        ratio = weight / importance_weights.min()
        
        print(f"{class_idx:<6} {group:<6} {samples:<8,} {train_prob:<12.6f} {test_prob:<12.6f} {weight:<12.6f} {ratio:<8.1f}")
        
        # Show only first 20 and last 20 classes
        if i == 19:
            print("   ... (middle classes omitted) ...")
            break
    
    # Show last 20 classes
    print("\nLast 20 classes (least samples):")
    for i, class_idx in enumerate(sorted_indices[-20:]):
        group = class_to_group[class_idx]
        samples = class_counts[class_idx]
        train_prob = train_probs[class_idx]
        test_prob = test_probs[class_idx]
        weight = importance_weights[class_idx]
        ratio = weight / importance_weights.min()
        
        print(f"{class_idx:<6} {group:<6} {samples:<8,} {train_prob:<12.6f} {test_prob:<12.6f} {weight:<12.6f} {ratio:<8.1f}")
    
    # Group-wise statistics
    print(f"\n📊 GROUP-WISE STATISTICS:")
    for g in range(num_groups):
        group_mask = (class_to_group == g)
        group_class_counts = class_counts[group_mask]
        group_train_probs = train_probs[group_mask]
        group_weights = importance_weights[group_mask]
        
        print(f"\n   Group {g}:")
        print(f"      Classes: {np.sum(group_mask)}")
        print(f"      Total samples: {group_class_counts.sum():,}")
        print(f"      Avg samples per class: {group_class_counts.mean():.1f}")
        print(f"      Min samples: {group_class_counts.min():,}")
        print(f"      Max samples: {group_class_counts.max():,}")
        print(f"      Avg weight: {group_weights.mean():.6f}")
        print(f"      Weight range: {group_weights.min():.6f} to {group_weights.max():.6f}")
    
    # Save detailed results
    results = {
        'total_samples': int(total_samples),
        'num_classes': int(num_classes),
        'num_groups': int(num_groups),
        'class_counts': class_counts.tolist(),
        'class_to_group': class_to_group.tolist(),
        'train_probs': train_probs.tolist(),
        'test_probs': test_probs.tolist(),
        'importance_weights': importance_weights.tolist(),
        'group_counts': group_counts.tolist(),
        'imbalance_ratio': float(class_counts.max() / class_counts.min()),
        'weight_ratio': float(importance_weights.max() / importance_weights.min())
    }
    
    with open('train_distribution_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: train_distribution_analysis.json")
    
    # Create visualization
    create_distribution_plots(class_counts, class_to_group, train_probs, importance_weights)
    
    return results

def create_distribution_plots(class_counts, class_to_group, train_probs, importance_weights):
    """Create visualization plots for the distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Class sample counts
    axes[0, 0].bar(range(len(class_counts)), class_counts)
    axes[0, 0].set_title('Training Samples per Class')
    axes[0, 0].set_xlabel('Class Index')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_yscale('log')
    
    # 2. Group distribution
    group_counts = np.zeros(len(np.unique(class_to_group)), dtype=np.int64)
    for g in range(len(group_counts)):
        group_mask = (class_to_group == g)
        group_counts[g] = class_counts[group_mask].sum()
    
    axes[0, 1].bar(range(len(group_counts)), group_counts)
    axes[0, 1].set_title('Training Samples per Group')
    axes[0, 1].set_xlabel('Group Index')
    axes[0, 1].set_ylabel('Number of Samples')
    
    # 3. Training probabilities
    axes[1, 0].bar(range(len(train_probs)), train_probs)
    axes[1, 0].set_title('Training Class Probabilities')
    axes[1, 0].set_xlabel('Class Index')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_yscale('log')
    
    # 4. Importance weights
    axes[1, 1].bar(range(len(importance_weights)), importance_weights)
    axes[1, 1].set_title('Importance Weights')
    axes[1, 1].set_xlabel('Class Index')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('train_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: train_distribution_analysis.png")

if __name__ == "__main__":
    try:
        results = analyze_training_distribution()
        print(f"\n✅ Analysis completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
