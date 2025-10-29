#!/usr/bin/env python3
"""
Quick analysis of training distribution and importance weighting.
"""

import json
import numpy as np

# Load data
with open('data/cifar100_lt_if100_splits_fixed/train_class_counts.json', 'r') as f:
    class_counts = json.load(f)

with open('data/cifar100_lt_if100_splits_fixed/group_config.json', 'r') as f:
    group_config = json.load(f)

# Convert to numpy arrays
class_counts = np.array(class_counts)
num_classes = len(class_counts)
total_samples = class_counts.sum()

# Create class-to-group mapping
class_to_group = np.zeros(num_classes, dtype=int)
for group_idx, classes in enumerate(group_config['groups']):
    for class_idx in classes:
        class_to_group[class_idx] = group_idx

# Calculate importance weights
train_probs = class_counts / total_samples
test_probs = np.ones(num_classes) / num_classes  # Balanced test set
importance_weights = train_probs  # Simplified as discussed

print("=" * 80)
print("CIFAR-100 LONG-TAIL TRAINING DISTRIBUTION ANALYSIS")
print("=" * 80)

# Basic statistics
print(f"\n📊 BASIC STATISTICS:")
print(f"   Total training samples: {total_samples:,}")
print(f"   Number of classes: {num_classes}")
print(f"   Number of groups: {len(group_config['groups'])}")

# Group analysis
group_counts = np.zeros(len(group_config['groups']), dtype=int)
for g in range(len(group_config['groups'])):
    group_mask = (class_to_group == g)
    group_counts[g] = class_counts[group_mask].sum()

print(f"\n📈 GROUP ANALYSIS:")
for g in range(len(group_config['groups'])):
    group_classes = np.where(class_to_group == g)[0]
    print(f"   Group {g} ({group_config['group_names'][g]}): {group_counts[g]:,} samples from {len(group_classes)} classes")
    print(f"      Classes: {group_classes[:10]}{'...' if len(group_classes) > 10 else ''}")

# Class distribution analysis
print(f"\n📋 CLASS DISTRIBUTION ANALYSIS:")
print(f"   Head class (most samples): {class_counts.max():,} samples")
print(f"   Tail class (least samples): {class_counts.min():,} samples")
print(f"   Imbalance ratio: {class_counts.max() / class_counts.min():.1f}x")

# Importance weighting
print(f"\n⚖️  IMPORTANCE WEIGHTING:")
print(f"   Training distribution range: {train_probs.min():.6f} to {train_probs.max():.6f}")
print(f"   Test distribution (balanced): {test_probs[0]:.6f}")
print(f"   Importance weights range: {importance_weights.min():.6f} to {importance_weights.max():.6f}")
print(f"   Weight ratio (head/tail): {importance_weights.max() / importance_weights.min():.1f}x")

# Detailed class analysis - first 20 classes
print(f"\n📝 FIRST 20 CLASSES (most samples):")
print(f"{'Class':<6} {'Group':<6} {'Samples':<8} {'Train Prob':<12} {'Weight':<12} {'Ratio':<8}")
print("-" * 70)

sorted_indices = np.argsort(class_counts)[::-1]
for i, class_idx in enumerate(sorted_indices[:20]):
    group = class_to_group[class_idx]
    samples = class_counts[class_idx]
    train_prob = train_probs[class_idx]
    weight = importance_weights[class_idx]
    ratio = weight / importance_weights.min()
    
    print(f"{class_idx:<6} {group:<6} {samples:<8,} {train_prob:<12.6f} {weight:<12.6f} {ratio:<8.1f}")

# Last 20 classes
print(f"\n📝 LAST 20 CLASSES (least samples):")
print(f"{'Class':<6} {'Group':<6} {'Samples':<8} {'Train Prob':<12} {'Weight':<12} {'Ratio':<8}")
print("-" * 70)

for i, class_idx in enumerate(sorted_indices[-20:]):
    group = class_to_group[class_idx]
    samples = class_counts[class_idx]
    train_prob = train_probs[class_idx]
    weight = importance_weights[class_idx]
    ratio = weight / importance_weights.min()
    
    print(f"{class_idx:<6} {group:<6} {samples:<8,} {train_prob:<12.6f} {weight:<12.6f} {ratio:<8.1f}")

# Group-wise statistics
print(f"\n📊 GROUP-WISE STATISTICS:")
for g in range(len(group_config['groups'])):
    group_mask = (class_to_group == g)
    group_class_counts = class_counts[group_mask]
    group_train_probs = train_probs[group_mask]
    group_weights = importance_weights[group_mask]
    
    print(f"\n   Group {g} ({group_config['group_names'][g]}):")
    print(f"      Classes: {np.sum(group_mask)}")
    print(f"      Total samples: {group_class_counts.sum():,}")
    print(f"      Avg samples per class: {group_class_counts.mean():.1f}")
    print(f"      Min samples: {group_class_counts.min():,}")
    print(f"      Max samples: {group_class_counts.max():,}")
    print(f"      Avg weight: {group_weights.mean():.6f}")
    print(f"      Weight range: {group_weights.min():.6f} to {group_weights.max():.6f}")

print(f"\n✅ Analysis completed!")
