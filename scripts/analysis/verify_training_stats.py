#!/usr/bin/env python3
"""
Verify training statistics with code calculation.
"""

import json
import numpy as np

# Load data
with open('data/cifar100_lt_if100_splits_fixed/train_class_counts.json', 'r') as f:
    class_counts = json.load(f)

with open('data/cifar100_lt_if100_splits_fixed/group_config.json', 'r') as f:
    group_config = json.load(f)

# Convert to numpy array
class_counts = np.array(class_counts, dtype=np.int64)

# Calculate statistics
total_samples = class_counts.sum()
num_classes = len(class_counts)
max_samples = class_counts.max()
min_samples = class_counts.min()
imbalance_ratio = max_samples / min_samples

# Calculate training probabilities
train_probs = class_counts / total_samples

# Create class-to-group mapping
class_to_group = np.zeros(num_classes, dtype=int)
for group_idx, classes in enumerate(group_config['groups']):
    for class_idx in classes:
        class_to_group[class_idx] = group_idx

# Group statistics
group_counts = np.zeros(len(group_config['groups']), dtype=np.int64)
for g in range(len(group_config['groups'])):
    group_mask = (class_to_group == g)
    group_counts[g] = class_counts[group_mask].sum()

print("=" * 80)
print("VERIFIED CIFAR-100 LONG-TAIL TRAINING STATISTICS")
print("=" * 80)

print(f"\n📊 BASIC STATISTICS:")
print(f"   Total training samples: {total_samples:,}")
print(f"   Number of classes: {num_classes}")
print(f"   Number of groups: {len(group_config['groups'])}")

print(f"\n📈 CLASS DISTRIBUTION:")
print(f"   Head class (most samples): {max_samples:,} samples")
print(f"   Tail class (least samples): {min_samples:,} samples")
print(f"   Imbalance ratio: {imbalance_ratio:.1f}x")

print(f"\n📋 GROUP ANALYSIS:")
for g in range(len(group_config['groups'])):
    group_classes = np.where(class_to_group == g)[0]
    print(f"   Group {g} ({group_config['group_names'][g]}): {group_counts[g]:,} samples from {len(group_classes)} classes")
    print(f"      Classes: {group_classes[:10]}{'...' if len(group_classes) > 10 else ''}")

print(f"\n⚖️  TRAINING PROBABILITIES:")
print(f"   Head class (class 0): {train_probs[0]:.6f}")
print(f"   Tail class (class 99): {train_probs[-1]:.6f}")
print(f"   Ratio (head/tail): {train_probs[0] / train_probs[-1]:.1f}x")

print(f"\n📝 DETAILED CLASS ANALYSIS (first 10 classes):")
print(f"{'Class':<6} {'Group':<6} {'Samples':<8} {'Train Prob':<12} {'Weight':<12}")
print("-" * 60)

for i in range(10):
    group = class_to_group[i]
    samples = class_counts[i]
    train_prob = train_probs[i]
    weight = train_prob  # Importance weight = training probability
    
    print(f"{i:<6} {group:<6} {samples:<8,} {train_prob:<12.6f} {weight:<12.6f}")

print(f"\n📝 DETAILED CLASS ANALYSIS (last 10 classes):")
print(f"{'Class':<6} {'Group':<6} {'Samples':<8} {'Train Prob':<12} {'Weight':<12}")
print("-" * 60)

for i in range(-10, 0):
    class_idx = num_classes + i
    group = class_to_group[class_idx]
    samples = class_counts[class_idx]
    train_prob = train_probs[class_idx]
    weight = train_prob  # Importance weight = training probability
    
    print(f"{class_idx:<6} {group:<6} {samples:<8,} {train_prob:<12.6f} {weight:<12.6f}")

# Verify against the log output
print(f"\n🔍 VERIFICATION AGAINST LOG OUTPUT:")
print(f"   Expected from log: head=0.046096, tail=0.000461")
print(f"   Calculated: head={train_probs[0]:.6f}, tail={train_probs[-1]:.6f}")
print(f"   Match: {'✅ YES' if abs(train_probs[0] - 0.046096) < 1e-6 and abs(train_probs[-1] - 0.000461) < 1e-6 else '❌ NO'}")

print(f"\n✅ VERIFICATION COMPLETED!")
