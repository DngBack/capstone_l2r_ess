#!/usr/bin/env python3
"""
Test different importance weighting approaches.
"""

import json
import numpy as np

# Load training data
with open('data/cifar100_lt_if100_splits_fixed/train_class_counts.json', 'r') as f:
    class_counts = json.load(f)

class_counts = np.array(class_counts, dtype=np.float64)
total_train = class_counts.sum()
num_classes = len(class_counts)

# Calculate different weighting approaches
train_probs = class_counts / total_train
test_probs = np.ones(num_classes) / num_classes  # Balanced test set

print("=" * 80)
print("IMPORTANCE WEIGHTING COMPARISON")
print("=" * 80)

print(f"\n📊 TRAINING DATA:")
print(f"   Total samples: {total_train:,}")
print(f"   Head class (0): {class_counts[0]:,} samples")
print(f"   Tail class (99): {class_counts[99]:,} samples")
print(f"   Imbalance ratio: {class_counts[0] / class_counts[99]:.1f}x")

print(f"\n📈 DISTRIBUTIONS:")
print(f"   Training distribution - head: {train_probs[0]:.6f}")
print(f"   Training distribution - tail: {train_probs[99]:.6f}")
print(f"   Test distribution (balanced): {test_probs[0]:.6f}")

print(f"\n⚖️  WEIGHTING APPROACHES:")

# Approach 1: Direct training distribution
weights1 = train_probs
print(f"\n1. Direct training distribution:")
print(f"   Head weight: {weights1[0]:.6f}")
print(f"   Tail weight: {weights1[99]:.6f}")
print(f"   Weight ratio: {weights1[0] / weights1[99]:.1f}x")

# Approach 2: Inverse weights (train_probs / test_probs)
weights2 = train_probs / test_probs
print(f"\n2. Inverse weights (train_probs / test_probs):")
print(f"   Head weight: {weights2[0]:.6f}")
print(f"   Tail weight: {weights2[99]:.6f}")
print(f"   Weight ratio: {weights2[0] / weights2[99]:.1f}x")

# Approach 3: Normalized inverse weights
weights3 = weights2 / weights2.sum() * num_classes
print(f"\n3. Normalized inverse weights:")
print(f"   Head weight: {weights3[0]:.6f}")
print(f"   Tail weight: {weights3[99]:.6f}")
print(f"   Weight ratio: {weights3[0] / weights3[99]:.1f}x")

print(f"\n🤔 ANALYSIS:")
print(f"   Approach 1: Head samples get LOWER weight than tail samples")
print(f"   Approach 2: Head samples get HIGHER weight than tail samples")
print(f"   Approach 3: Normalized version of approach 2")

print(f"\n💡 RECOMMENDATION:")
print(f"   For re-weighting balanced test set to training distribution:")
print(f"   - Use Approach 2 (inverse weights)")
print(f"   - This up-weights head classes and down-weights tail classes")
print(f"   - Makes test set appear as if it has training distribution")
