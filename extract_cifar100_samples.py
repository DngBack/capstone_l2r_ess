#!/usr/bin/env python3
"""
Extract one sample image per class from CIFAR-100 dataset and save to organized folders.

This script:
1. Loads CIFAR-100 dataset from pickle files
2. Determines head/tail classification based on train class counts
3. Extracts one sample per class
4. Saves images to infer_samples/Cifar100/head/ or infer_samples/Cifar100/tail/
   with naming format: {group}_{class_idx}_{class_name}.png
"""

import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import json
import argparse


def load_cifar100_pickle(data_dir: Path, split: str = "test"):
    """Load CIFAR-100 data from pickle file."""
    pickle_file = data_dir / "cifar-100-python" / split
    
    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    
    # Decode bytes keys
    data = data_dict[b'data']
    fine_labels = data_dict[b'fine_labels']
    
    # Reshape data: [N, 3072] -> [N, 32, 32, 3]
    # CIFAR-100 stores as RRR...GGG...BBB, need to reshape and transpose
    data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
    
    return data, fine_labels


def load_class_to_group(splits_dir: Path, num_classes: int = 100, tail_threshold: int = 20):
    """Load class-to-group mapping (head=0, tail=1)."""
    counts_path = splits_dir / "train_class_counts.json"
    
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(num_classes)]
    
    counts = np.array(class_counts)
    tail_mask = counts <= tail_threshold
    class_to_group = np.zeros(num_classes, dtype=np.int64)
    class_to_group[tail_mask] = 1  # 0=head, 1=tail
    
    return class_to_group


def get_class_names(data_dir: Path):
    """Load CIFAR-100 class names from meta file."""
    meta_file = data_dir / "cifar-100-python" / "meta"
    
    with open(meta_file, 'rb') as f:
        meta_dict = pickle.load(f, encoding='bytes')
    
    # Decode fine_label_names
    fine_label_names = meta_dict[b'fine_label_names']
    class_names = [name.decode('utf-8') for name in fine_label_names]
    
    return class_names


def extract_samples(
    data_dir: Path = Path("./data"),
    splits_dir: Path = Path("./data/cifar100_lt_if100_splits_fixed"),
    output_dir: Path = Path("./infer_samples/Cifar100"),
    split: str = "test",
    tail_threshold: int = 20
):
    """Extract one sample per class and save to organized folders."""
    
    print("="*70)
    print("Extracting CIFAR-100 samples for inference")
    print("="*70)
    
    # Create output directories
    head_dir = output_dir / "head"
    tail_dir = output_dir / "tail"
    head_dir.mkdir(parents=True, exist_ok=True)
    tail_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n1. Loading CIFAR-100 {split} data...")
    data, labels = load_cifar100_pickle(data_dir, split)
    print(f"   Loaded {len(data)} samples")
    
    # Load class names
    print(f"\n2. Loading class names...")
    class_names = get_class_names(data_dir)
    print(f"   Found {len(class_names)} classes")
    
    # Load class-to-group mapping
    print(f"\n3. Loading class-to-group mapping...")
    class_to_group = load_class_to_group(splits_dir, num_classes=100, tail_threshold=tail_threshold)
    num_head = (class_to_group == 0).sum()
    num_tail = (class_to_group == 1).sum()
    print(f"   Head classes: {num_head}, Tail classes: {num_tail}")
    
    # Extract one sample per class
    print(f"\n4. Extracting samples...")
    extracted = {0: 0, 1: 0}  # Count extracted per group
    
    for class_idx in range(100):
        # Find first sample of this class
        class_indices = np.where(np.array(labels) == class_idx)[0]
        
        if len(class_indices) == 0:
            print(f"   ⚠️  Warning: No samples found for class {class_idx}")
            continue
        
        # Take first sample
        sample_idx = class_indices[0]
        image_data = data[sample_idx]  # [32, 32, 3]
        
        # Determine group
        group = class_to_group[class_idx]
        group_name = "head" if group == 0 else "tail"
        
        # Get class name
        class_name = class_names[class_idx]
        
        # Create filename: {group}_{class_idx}_{class_name}.png
        filename = f"{group_name}_{class_idx}_{class_name}.png"
        output_path = (head_dir if group == 0 else tail_dir) / filename
        
        # Convert to PIL Image and save
        image = Image.fromarray(image_data.astype(np.uint8))
        image.save(output_path)
        
        extracted[group] += 1
        if (class_idx + 1) % 20 == 0:
            print(f"   Processed {class_idx + 1}/100 classes...")
    
    print(f"\n✅ Extraction complete!")
    print(f"   Head samples: {extracted[0]}/{num_head}")
    print(f"   Tail samples: {extracted[1]}/{num_tail}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Head images: {head_dir}")
    print(f"   Tail images: {tail_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract CIFAR-100 samples for inference")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing CIFAR-100 data (default: ./data)"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="./data/cifar100_lt_if100_splits_fixed",
        help="Directory containing train_class_counts.json (default: ./data/cifar100_lt_if100_splits_fixed)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./infer_samples/Cifar100",
        help="Output directory (default: ./infer_samples/Cifar100)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to extract from (default: test)"
    )
    parser.add_argument(
        "--tail-threshold",
        type=int,
        default=20,
        help="Train count threshold for tail classes (default: 20)"
    )
    
    args = parser.parse_args()
    
    extract_samples(
        data_dir=Path(args.data_dir),
        splits_dir=Path(args.splits_dir),
        output_dir=Path(args.output_dir),
        split=args.split,
        tail_threshold=args.tail_threshold
    )


if __name__ == "__main__":
    main()

