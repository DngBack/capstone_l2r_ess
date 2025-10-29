#!/usr/bin/env python3
"""
Calculate total training samples correctly.
"""

import json

# Load class counts
with open('data/cifar100_lt_if100_splits_fixed/train_class_counts.json', 'r') as f:
    class_counts = json.load(f)

# Calculate total
total_samples = sum(class_counts)
num_classes = len(class_counts)

print(f"Number of classes: {num_classes}")
print(f"Total training samples: {total_samples:,}")

# Show first few and last few classes
print(f"\nFirst 5 classes:")
for i in range(5):
    print(f"  Class {i}: {class_counts[i]} samples")

print(f"\nLast 5 classes:")
for i in range(-5, 0):
    print(f"  Class {num_classes + i}: {class_counts[i]} samples")

# Calculate imbalance ratio
max_samples = max(class_counts)
min_samples = min(class_counts)
imbalance_ratio = max_samples / min_samples

print(f"\nImbalance ratio: {imbalance_ratio:.1f}x")
print(f"Max samples: {max_samples}")
print(f"Min samples: {min_samples}")

# Calculate training probabilities
train_probs = [count / total_samples for count in class_counts]
print(f"\nTraining probabilities:")
print(f"  Head class (class 0): {train_probs[0]:.6f}")
print(f"  Tail class (class 99): {train_probs[-1]:.6f}")
print(f"  Ratio: {train_probs[0] / train_probs[-1]:.1f}x")
