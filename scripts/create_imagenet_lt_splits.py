#!/usr/bin/env python3
"""
Quick script to generate ImageNet-LT splits from label files.
Usage: python scripts/create_imagenet_lt_splits.py
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.imagenet_lt_splits import create_imagenet_lt_splits


def main():
    parser = argparse.ArgumentParser(
        description="Generate ImageNet-LT dataset splits from label files"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/imagenet_lt',
        help='Directory containing images and label files (default: data/imagenet_lt)'
    )
    parser.add_argument(
        '--train-label-file',
        type=str,
        default='ImageNet_LT_train.txt',
        help='Train label file name (default: ImageNet_LT_train.txt)'
    )
    parser.add_argument(
        '--val-label-file',
        type=str,
        default='ImageNet_LT_test.txt',
        help='Val label file name (default: ImageNet_LT_test.txt)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/imagenet_lt_splits',
        help='Output directory for splits (default: data/imagenet_lt_splits)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--expert-ratio',
        type=float,
        default=0.9,
        help='Expert split ratio (default: 0.9)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file. If provided, all output will be saved to this file (default: None)'
    )
    
    args = parser.parse_args()
    
    print("Generating ImageNet-LT splits...")
    
    splits, class_weights = create_imagenet_lt_splits(
        data_dir=args.data_dir,
        train_label_file=args.train_label_file,
        val_label_file=args.val_label_file,
        output_dir=args.output_dir,
        seed=args.seed,
        expert_ratio=args.expert_ratio,
        log_file=args.log_file
    )
    
    print("\n✓ All splits generated successfully!")


if __name__ == "__main__":
    main()

