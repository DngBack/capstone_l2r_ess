#!/usr/bin/env python3
"""
Manual calculation of training statistics.
"""

# Load class counts directly
class_counts = [
    500, 477, 455, 434, 415, 396, 378, 361, 344, 328,
    314, 299, 286, 273, 260, 248, 237, 226, 216, 206,
    197, 188, 179, 171, 163, 156, 149, 142, 135, 129,
    123, 118, 112, 107, 102, 98, 93, 89, 85, 81,
    77, 74, 70, 67, 64, 61, 58, 56, 53, 51,
    48, 46, 44, 42, 40, 38, 36, 35, 33, 32,
    30, 29, 27, 26, 25, 24, 23, 22, 21, 20,
    19, 18, 17, 16, 15, 15, 14, 13, 13, 12,
    12, 11, 11, 10, 10, 9, 9, 8, 8, 7,
    7, 7, 6, 6, 6, 6, 5, 5, 5, 5
]

# Calculate total
total_samples = sum(class_counts)
num_classes = len(class_counts)

print(f"Total training samples: {total_samples:,}")
print(f"Number of classes: {num_classes}")

# Calculate training probabilities
train_probs = [count / total_samples for count in class_counts]

print(f"\nHead class (class 0): {train_probs[0]:.6f}")
print(f"Tail class (class 99): {train_probs[-1]:.6f}")
print(f"Imbalance ratio: {class_counts[0] / class_counts[-1]:.1f}x")

# Verify against log output
print(f"\nVerification:")
print(f"Expected from log: head=0.046096, tail=0.000461")
print(f"Calculated: head={train_probs[0]:.6f}, tail={train_probs[-1]:.6f}")

# Check if they match
head_match = abs(train_probs[0] - 0.046096) < 1e-6
tail_match = abs(train_probs[-1] - 0.000461) < 1e-6

print(f"Head match: {'✅ YES' if head_match else '❌ NO'}")
print(f"Tail match: {'✅ YES' if tail_match else '❌ NO'}")
print(f"Overall match: {'✅ YES' if head_match and tail_match else '❌ NO'}")
