#!/usr/bin/env python3
"""
Find classes with duplicate sample counts in training set.
"""

import json
from collections import Counter

# Load training class counts
with open('data/cifar100_lt_if100_splits_fixed/train_class_counts.json', 'r') as f:
    class_counts = json.load(f)

# Find duplicates
count_frequency = Counter(class_counts)
duplicates = {count: classes for count, classes in count_frequency.items() if classes > 1}

print("=" * 80)
print("CLASSES WITH DUPLICATE SAMPLE COUNTS")
print("=" * 80)

print(f"\nTotal classes: {len(class_counts)}")
print(f"Unique sample counts: {len(count_frequency)}")
print(f"Duplicate counts: {len(duplicates)}")

print(f"\n📊 DUPLICATE SAMPLE COUNTS:")
for count in sorted(duplicates.keys(), reverse=True):
    # Find which classes have this count
    classes_with_count = [i for i, c in enumerate(class_counts) if c == count]
    print(f"   {count} samples: Classes {classes_with_count} ({len(classes_with_count)} classes)")

print(f"\n📋 DETAILED BREAKDOWN:")
for count in sorted(duplicates.keys(), reverse=True):
    classes_with_count = [i for i, c in enumerate(class_counts) if c == count]
    print(f"\n   {count} samples ({duplicates[count]} classes):")
    for i, class_idx in enumerate(classes_with_count):
        print(f"      Class {class_idx:2d}: {count} samples")
        if i < len(classes_with_count) - 1:
            print(f"      Class {class_idx:2d}: {count} samples")

print(f"\n✅ VERIFICATION:")
print(f"   Total classes with duplicates: {sum(duplicates.values()) - len(duplicates)}")
print(f"   Classes with unique counts: {len(class_counts) - (sum(duplicates.values()) - len(duplicates))}")
