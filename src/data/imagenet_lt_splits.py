#!/usr/bin/env python3
"""
ImageNet-LT Dataset Splits Generator
===================================

Generate splits for ImageNet-LT dataset from label files.
Following same methodology as CIFAR-100-LT and iNaturalist 2018

Key Features:
- Dataset already has long-tail distribution (no need to create)
- Uses label files: ImageNet_LT_train.txt, ImageNet_LT_test.txt
- Format: "path/to/image class_id"
- Threshold: 20 samples to distinguish head/tail classes

Split Strategy:
- Train set (from ImageNet_LT_train.txt):
  → Expert split: 90% (for expert training)
  → Gating split: 10% (for gating network training)
  
- Test set (from ImageNet_LT_test.txt):
  → Test split: 80% (main test set)
  → Val split: 10% (validation set)
  → TuneV split: 10% (tuning/validation set)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from PIL import Image
import sys
from datetime import datetime


class ImageNetLTDataset:
    """Wrapper for ImageNet-LT dataset from label files."""
    
    def __init__(self, data_dir: str, label_file: str, transform=None):
        """
        Args:
            data_dir: Directory containing images (e.g., 'data/imagenet_lt')
            label_file: Path to label file (e.g., 'ImageNet_LT_train.txt')
            transform: Optional transforms
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load label file
        label_path = Path(label_file)
        if not label_path.is_absolute():
            label_path = self.data_dir / label_file
        
        print(f"Loading labels from: {label_path}")
        self.samples = []
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: "path/to/image class_id"
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                image_path_str = parts[0]
                class_id = int(parts[1])
                
                # Construct full image path
                image_path = self.data_dir / image_path_str
                
                self.samples.append((image_path, class_id))
        
        print(f"  Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_imagenet_lt_transforms():
    """Get ImageNet-LT transforms (ImageNet-style preprocessing)."""
    
    # Training transforms (RandomResizedCrop + RandomHorizontalFlip)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Evaluation transforms (Resize + CenterCrop)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, eval_transform


def analyze_imagenet_lt_distribution(
    train_dataset,
    val_dataset,
    output_dir: Optional[Path] = None
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Analyze ImageNet-LT distribution from datasets.
    
    Args:
        train_dataset: Dataset loaded from ImageNet_LT_train.txt
        val_dataset: Dataset loaded from ImageNet_LT_test.txt (will be split into Test/Val/TuneV)
        output_dir: Optional output directory for visualizations
    
    Returns:
        Tuple of (train_counts, test_counts, total_counts) dicts
        Note: test_counts represents counts from ImageNet_LT_test.txt (before splitting)
    """
    print("\n" + "="*80)
    print("ANALYZING ImageNet-LT DISTRIBUTION")
    print("="*80)
    
    # Count samples per class
    train_counts = Counter()
    test_counts = Counter()  # Actually test set from ImageNet_LT_test.txt
    
    print("\nCounting train samples (from ImageNet_LT_train.txt)...")
    for _, label in train_dataset.samples:
        train_counts[label] += 1
    
    print("Counting test samples (from ImageNet_LT_test.txt, will be split into Test/Val/TuneV)...")
    for _, label in val_dataset.samples:
        test_counts[label] += 1
    
    # Rename for compatibility with existing code
    val_counts = test_counts
    
    # Combine counts
    all_classes = set(train_counts.keys()) | set(val_counts.keys())
    total_counts = {cls: train_counts.get(cls, 0) + val_counts.get(cls, 0) 
                    for cls in all_classes}
    
    # Statistics - Show both train and total stats
    num_classes = len(all_classes)
    
    # Train statistics
    train_sample_counts = list(train_counts.values())
    train_min = min(train_sample_counts) if train_sample_counts else 0
    train_max = max(train_sample_counts) if train_sample_counts else 0
    train_mean = np.mean(train_sample_counts) if train_sample_counts else 0
    train_median = np.median(train_sample_counts) if train_sample_counts else 0
    train_std = np.std(train_sample_counts) if train_sample_counts else 0
    train_sorted = sorted(train_sample_counts)
    train_q25 = np.percentile(train_sorted, 25) if train_sorted else 0
    train_q75 = np.percentile(train_sorted, 75) if train_sorted else 0
    train_imbalance = train_max / train_min if train_min > 0 else float('inf')
    
    # Total statistics (train + val)
    total_sample_counts = list(total_counts.values())
    total_min = min(total_sample_counts) if total_sample_counts else 0
    total_max = max(total_sample_counts) if total_sample_counts else 0
    total_mean = np.mean(total_sample_counts) if total_sample_counts else 0
    total_median = np.median(total_sample_counts) if total_sample_counts else 0
    total_std = np.std(total_sample_counts) if total_sample_counts else 0
    total_sorted = sorted(total_sample_counts)
    total_q25 = np.percentile(total_sorted, 25) if total_sorted else 0
    total_q75 = np.percentile(total_sorted, 75) if total_sorted else 0
    total_imbalance = total_max / total_min if total_min > 0 else float('inf')
    
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total classes: {num_classes:,}")
    print(f"Train samples (from ImageNet_LT_train.txt): {len(train_dataset):,}")
    print(f"Test samples (from ImageNet_LT_test.txt): {len(val_dataset):,}")
    print(f"  Note: Test set will be split into Test (80%) + Val (10%) + TuneV (10%)")
    print(f"Total samples: {len(train_dataset) + len(val_dataset):,}")
    
    print(f"\n{'='*80}")
    print("TRAIN SET STATISTICS (used for head/tail classification)")
    print(f"{'='*80}")
    print(f"Samples per class (train only):")
    print(f"  Min:          {train_min}")
    print(f"  Max:          {train_max:,}")
    print(f"  Mean:         {train_mean:.1f}")
    print(f"  Median:       {train_median:.1f}")
    print(f"  Std Dev:      {train_std:.1f}")
    print(f"  Q25:          {train_q25:.1f}")
    print(f"  Q75:          {train_q75:.1f}")
    print(f"  Imbalance Ratio (max/min): {train_imbalance:,.1f}x")
    
    print(f"\n{'='*80}")
    print("TOTAL STATISTICS (train + val combined)")
    print(f"{'='*80}")
    print(f"Samples per class (train + val):")
    print(f"  Min:          {total_min}")
    print(f"  Max:          {total_max:,}")
    print(f"  Mean:         {total_mean:.1f}")
    print(f"  Median:       {total_median:.1f}")
    print(f"  Std Dev:      {total_std:.1f}")
    print(f"  Q25:          {total_q25:.1f}")
    print(f"  Q75:          {total_q75:.1f}")
    print(f"  Imbalance Ratio (max/min): {total_imbalance:,.1f}x")
    
    # Head/Tail analysis - Use TRAIN counts (not total) as per paper
    threshold = 20
    head_classes = [cls for cls, count in train_counts.items() if count > threshold]
    tail_classes = [cls for cls, count in train_counts.items() if count <= threshold]
    
    print(f"\n{'='*80}")
    print(f"HEAD/TAIL CLASSIFICATION (threshold = {threshold} samples in TRAIN set)")
    print(f"{'='*80}")
    print(f"Head classes (> {threshold} train samples): {len(head_classes):,} ({100*len(head_classes)/num_classes:.1f}%)")
    print(f"Tail classes (≤ {threshold} train samples): {len(tail_classes):,} ({100*len(tail_classes)/num_classes:.1f}%)")
    
    if len(head_classes) > 0:
        head_train_counts = [train_counts[cls] for cls in head_classes]
        print(f"\nHead class stats (train samples):")
        print(f"  Min: {min(head_train_counts)}, Max: {max(head_train_counts):,}, Mean: {np.mean(head_train_counts):.1f}")
    
    if len(tail_classes) > 0:
        tail_train_counts = [train_counts[cls] for cls in tail_classes]
        print(f"Tail class stats (train samples):")
        print(f"  Min: {min(tail_train_counts)}, Max: {max(tail_train_counts)}, Mean: {np.mean(tail_train_counts):.1f}")
    else:
        print(f"\n⚠️  Warning: No tail classes found (all classes have > {threshold} train samples)")
        print(f"   This might indicate the dataset is not a proper long-tail distribution.")
    
    return train_counts, val_counts, total_counts


def visualize_tail_proportion(train_counts: Dict[int, int], threshold: int, output_dir: Path):
    """
    Visualize tail proportion to verify long-tail distribution.
    
    Args:
        train_counts: Dict mapping class_id to train count
        threshold: Threshold for head/tail classification
        output_dir: Directory to save visualization
    """
    print("\n" + "="*80)
    print("COMPUTING TAIL PROPORTION FOR VERIFICATION")
    print("="*80)
    
    # Count head/tail classes and samples
    tail_classes = [cls for cls, count in train_counts.items() if count <= threshold]
    head_classes = [cls for cls, count in train_counts.items() if count > threshold]
    
    num_tail_classes = len(tail_classes)
    num_head_classes = len(head_classes)
    num_total_classes = len(train_counts)
    
    # Calculate samples for head/tail classes
    tail_samples = sum(train_counts[cls] for cls in tail_classes)
    head_samples = sum(train_counts[cls] for cls in head_classes)
    total_samples = sum(train_counts.values())
    
    # Tail proportion: proportion of SAMPLES (not classes) that are in tail classes
    tail_prop = tail_samples / total_samples if total_samples > 0 else 0
    head_prop = head_samples / total_samples if total_samples > 0 else 0
    
    # Class proportions (for reference)
    tail_class_prop = num_tail_classes / num_total_classes
    head_class_prop = num_head_classes / num_total_classes
    
    print(f"Total classes: {num_total_classes:,}")
    print(f"Head classes (> {threshold} samples): {num_head_classes:,} ({head_class_prop*100:.2f}% of classes)")
    print(f"Tail classes (≤ {threshold} samples): {num_tail_classes:,} ({tail_class_prop*100:.2f}% of classes)")
    print(f"\nTotal samples: {total_samples:,}")
    print(f"Head samples: {head_samples:,} ({head_prop*100:.2f}% of samples)")
    print(f"Tail samples: {tail_samples:,} ({tail_prop*100:.2f}% of samples)")
    print(f"\n✓ Tail proportion (samples): {tail_prop:.4f} (Proportion of SAMPLES in tail classes)")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Class distribution by count bins
    sorted_counts = sorted(train_counts.values())
    bins = np.logspace(np.log10(1), np.log10(max(sorted_counts)), 50)
    ax1.hist(sorted_counts, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Number of Samples per Class', fontsize=11)
    ax1.set_ylabel('Number of Classes', fontsize=11)
    ax1.set_title('Class Distribution (Log Scale)', fontsize=13, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # Plot 2: Head vs Tail pie chart (by SAMPLES, not classes)
    sizes = [head_samples, tail_samples]
    labels = [f'Head ({head_samples:,} samples)', f'Tail ({tail_samples:,} samples)']
    colors = ['lightblue', 'coral']
    explode = (0.05, 0.1)
    ax2.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title(f'Head/Tail Sample Proportions\n(Threshold: {threshold} samples per class)', 
                  fontsize=13, fontweight='bold')
    
    # Plot 3: Cumulative distribution
    sorted_counts_arr = np.array(sorted_counts)
    cumulative = np.arange(1, len(sorted_counts_arr) + 1)
    ax3.plot(sorted_counts_arr, cumulative, linewidth=2, color='steelblue')
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    ax3.axhline(y=num_head_classes, color='green', linestyle=':', linewidth=1.5,
                label=f'Head classes ({num_head_classes:,})')
    ax3.set_xlabel('Number of Samples per Class', fontsize=11)
    ax3.set_ylabel('Cumulative Number of Classes', fontsize=11)
    ax3.set_title('Cumulative Class Distribution', fontsize=13, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Samples per class (sorted, for overview)
    ax4.bar(range(len(sorted_counts)), sorted_counts, color='steelblue', alpha=0.7)
    ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold})')
    ax4.set_xlabel('Class Index (sorted by sample count)', fontsize=11)
    ax4.set_ylabel('Number of Samples', fontsize=11)
    ax4.set_title('All Classes Distribution (Sorted)', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_dir / "tail_proportion_analysis.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved tail proportion analysis to: {viz_path}")
    plt.close()
    
    return tail_prop


def visualize_all_splits_comprehensive(
    train_counts: List[int],
    expert_counts: List[int],
    gating_counts: List[int],
    val_counts: List[int],
    test_counts: List[int],
    tunev_counts: List[int],
    output_dir: Path,
    threshold: int = 20
):
    """
    Create comprehensive visualization with 6 subplots showing all splits.
    
    IMPORTANT: Head/tail classification is ALWAYS based on TRAIN counts, not on individual split counts.
    This ensures consistency: a class that is head on train is head on val/test/tunev as well.
    
    Args:
        train_counts: Train split counts per class (used for head/tail definition)
        expert_counts: Expert split counts per class
        gating_counts: Gating split counts per class
        val_counts: Val split counts per class
        test_counts: Test split counts per class
        tunev_counts: TuneV split counts per class
        output_dir: Directory to save visualization
        threshold: Threshold for head/tail classification (based on TRAIN counts)
    """
    print(f"\n{'='*80}")
    print("CREATING COMPREHENSIVE SPLITS VISUALIZATION")
    print(f"{'='*80}")
    print(f"IMPORTANT: Head/tail classification is based on TRAIN counts (threshold = {threshold})")
    print(f"  This ensures consistency across all splits (val/test/tunev use same head/tail definition)")
    
    num_classes = len(train_counts)
    
    # Define head/tail classes based on TRAIN counts (not individual split counts)
    # This ensures consistency: class is head/tail on all splits based on train distribution
    train_counts_array = np.array(train_counts)
    head_classes_mask = train_counts_array > threshold  # Head classes (based on train)
    tail_classes_mask = ~head_classes_mask  # Tail classes (based on train)
    head_class_indices = np.where(head_classes_mask)[0]
    tail_class_indices = np.where(tail_classes_mask)[0]
    
    num_head_classes = len(head_class_indices)
    num_tail_classes = len(tail_class_indices)
    
    # Calculate statistics for each split
    # Head/tail classification is based on train, but sample counts come from the split itself
    def calc_stats(counts, name, train_head_mask, train_tail_mask, threshold=20):
        """Calculate statistics for a split.
        
        Args:
            counts: Sample counts for this split
            name: Split name
            train_head_mask: Boolean mask for head classes (based on train)
            train_tail_mask: Boolean mask for tail classes (based on train)
            threshold: Threshold used (for reference)
        """
        counts_array = np.array(counts)
        total_samples = counts_array.sum()
        
        # Use train-based head/tail classification
        head_samples = counts_array[train_head_mask].sum()
        tail_samples = counts_array[train_tail_mask].sum()
        
        num_head = train_head_mask.sum()  # Number of head classes (from train)
        num_tail = train_tail_mask.sum()  # Number of tail classes (from train)
        
        head_pct = (num_head / num_classes * 100) if num_classes > 0 else 0
        tail_pct = (num_tail / num_classes * 100) if num_classes > 0 else 0
        head_sample_pct = (head_samples / total_samples * 100) if total_samples > 0 else 0
        tail_sample_pct = (tail_samples / total_samples * 100) if total_samples > 0 else 0
        
        max_count = counts_array.max() if len(counts_array) > 0 else 0
        min_count = counts_array[counts_array > 0].min() if (counts_array > 0).any() else 0
        imbalance_factor = max_count / min_count if min_count > 0 else float('inf')
        
        mean_count = counts_array.mean()
        median_count = np.median(counts_array)
        
        return {
            'total': total_samples,
            'num_head': num_head,
            'num_tail': num_tail,
            'head_pct': head_pct,
            'tail_pct': tail_pct,
            'head_sample_pct': head_sample_pct,
            'tail_sample_pct': tail_sample_pct,
            'imbalance_factor': imbalance_factor,
            'max': max_count,
            'min': min_count,
            'mean': mean_count,
            'median': median_count
        }
    
    train_stats = calc_stats(train_counts, "Train", head_classes_mask, tail_classes_mask, threshold)
    expert_stats = calc_stats(expert_counts, "Expert", head_classes_mask, tail_classes_mask, threshold)
    gating_stats = calc_stats(gating_counts, "Gating", head_classes_mask, tail_classes_mask, threshold)
    val_stats = calc_stats(val_counts, "Val", head_classes_mask, tail_classes_mask, threshold)
    test_stats = calc_stats(test_counts, "Test", head_classes_mask, tail_classes_mask, threshold)
    tunev_stats = calc_stats(tunev_counts, "TuneV", head_classes_mask, tail_classes_mask, threshold)
    
    # Create 3x2 subplot grid
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    splits_data = [
        ("Train", train_counts, train_stats, 'steelblue'),
        ("Expert", expert_counts, expert_stats, 'forestgreen'),
        ("Gating", gating_counts, gating_stats, 'coral'),
        ("Val", val_counts, val_stats, 'mediumpurple'),
        ("Test", test_counts, test_stats, 'gold'),
        ("TuneV", tunev_counts, tunev_stats, 'orange')
    ]
    
    for idx, (name, counts, stats, color) in enumerate(splits_data):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Sort counts for visualization (ascending order to show long-tail distribution)
        sorted_counts = sorted(counts)
        class_indices = np.arange(len(sorted_counts))
        
        # Plot bars
        bars = ax.bar(class_indices, sorted_counts, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Head/Tail threshold ({threshold})')
        
        # Calculate and show head/tail boundary based on train classification
        # Since counts are sorted ascending, we need to find where tail classes appear
        # Tail classes are those with train_count <= threshold
        tail_count = num_tail_classes  # Based on train, not this split
        if tail_count > 0 and tail_count < num_classes:
            # Find the position in sorted counts where tail classes end
            # Since we're sorting counts ascending, tail classes (with lower counts) appear first
            # But we need to find the boundary based on train classification
            # For visualization, we show where tail classes typically are (left side)
            ax.axvline(x=tail_count - 0.5, color='green', linestyle=':', 
                      linewidth=1.5, alpha=0.7, 
                      label=f'Head/Tail boundary ({tail_count} tail classes from train)')
        
        # Statistics text box
        stats_text = (
            f"Total: {stats['total']:,} samples\n"
            f"Classes: {num_classes}\n"
            f"Imbalance: {stats['imbalance_factor']:.1f}x\n"
            f"Head: {stats['num_head']} ({stats['head_pct']:.1f}%)\n"
            f"Tail: {stats['num_tail']} ({stats['tail_pct']:.1f}%)\n"
            f"Head samples: {stats['head_sample_pct']:.1f}%\n"
            f"Tail samples: {stats['tail_sample_pct']:.1f}%\n"
            f"Max: {int(stats['max'])}, Min: {int(stats['min'])}\n"
            f"Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}"
        )
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Class Index (sorted by count)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Samples per Class', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} Split Distribution', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        
        # Use log scale for y-axis if needed
        if stats['max'] > stats['min'] * 10:
            ax.set_yscale('log')
    
    # Overall title
    fig.suptitle('ImageNet-LT Dataset Splits Distribution\n'
                 f'All classes sorted by sample count (Head/Tail threshold = {threshold})',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_path = output_dir / "comprehensive_splits_distribution.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comprehensive splits visualization to: {viz_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS FOR ALL SPLITS")
    print(f"{'='*80}")
    print(f"{'Split':<12} {'Total':<12} {'IF':<10} {'Head%':<8} {'Tail%':<8} {'HeadSamp%':<10} {'TailSamp%':<10}")
    print(f"{'-'*80}")
    for name, _, stats, _ in splits_data:
        print(f"{name:<12} {stats['total']:>10,}  {stats['imbalance_factor']:>8.1f}x  "
              f"{stats['head_pct']:>6.1f}%  {stats['tail_pct']:>6.1f}%  "
              f"{stats['head_sample_pct']:>8.1f}%  {stats['tail_sample_pct']:>8.1f}%")


def generate_class_distribution_report(
    train_counts: List[int],
    expert_counts: List[int],
    gating_counts: List[int],
    output_dir: Path,
    threshold: int = 20
):
    """
    Generate detailed class distribution report across all classes.
    Creates both Markdown and CSV reports showing exact sample counts.
    
    Args:
        train_counts: Original train counts per class
        expert_counts: Expert split counts per class
        gating_counts: Gating split counts per class
        output_dir: Directory to save reports
        threshold: Threshold for head/tail classification
    """
    print(f"\n{'='*80}")
    print("GENERATING CLASS DISTRIBUTION REPORT")
    print(f"{'='*80}")
    
    num_classes = len(train_counts)
    
    # Classify head/tail
    head_mask = np.array(train_counts) > threshold
    tail_mask = ~head_mask
    head_classes = np.where(head_mask)[0]
    tail_classes = np.where(tail_mask)[0]
    
    print(f"Total classes: {num_classes:,}")
    print(f"Head classes (> {threshold} samples): {len(head_classes):,}")
    print(f"Tail classes (≤ {threshold} samples): {len(tail_classes):,}")
    
    # Compute statistics
    total_train = sum(train_counts)
    total_expert = sum(expert_counts)
    total_gating = sum(gating_counts)
    
    # Create comprehensive data
    class_data = []
    for class_id in range(num_classes):
        train_count = train_counts[class_id]
        expert_count = expert_counts[class_id]
        gating_count = gating_counts[class_id]
        
        # Calculate ratios
        train_pct = (train_count / total_train * 100) if total_train > 0 else 0
        expert_pct = (expert_count / total_expert * 100) if total_expert > 0 else 0
        gating_pct = (gating_count / total_gating * 100) if total_gating > 0 else 0
        
        # Verify split (should be ~90% expert, ~10% gating)
        split_ratio = (expert_count / train_count) if train_count > 0 else 0
        
        # Category
        category = "Head" if train_count > threshold else "Tail"
        
        class_data.append({
            'class_id': class_id,
            'category': category,
            'train_samples': train_count,
            'expert_samples': expert_count,
            'gating_samples': gating_count,
            'train_pct': train_pct,
            'expert_pct': expert_pct,
            'gating_pct': gating_pct,
            'split_ratio': split_ratio,
        })
    
    # Sort by train count (descending)
    class_data.sort(key=lambda x: x['train_samples'], reverse=True)
    
    # Generate Markdown report
    md_path = output_dir / "class_distribution_report.md"
    print(f"\nGenerating Markdown report: {md_path}")
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# ImageNet-LT Class Distribution Report\n\n")
        f.write("**Generated:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Classes:** {num_classes:,}\n")
        f.write(f"- **Head Classes (> {threshold} samples):** {len(head_classes):,} ({len(head_classes)/num_classes*100:.1f}%)\n")
        f.write(f"- **Tail Classes (≤ {threshold} samples):** {len(tail_classes):,} ({len(tail_classes)/num_classes*100:.1f}%)\n")
        f.write(f"\n- **Total Train Samples:** {total_train:,}\n")
        f.write(f"- **Total Expert Samples:** {total_expert:,} ({total_expert/total_train*100:.2f}%)\n")
        f.write(f"- **Total Gating Samples:** {total_gating:,} ({total_gating/total_train*100:.2f}%)\n\n")
        
        # Head vs Tail stats
        head_train = sum(train_counts[i] for i in head_classes)
        tail_train = sum(train_counts[i] for i in tail_classes)
        f.write("- **Head Samples:** {:,} ({:.2f}%)\n".format(head_train, head_train/total_train*100))
        f.write("- **Tail Samples:** {:,} ({:.2f}%)\n\n".format(tail_train, tail_train/total_train*100))
        f.write("---\n\n")
        
        # Detailed table
        f.write("## Detailed Class Distribution\n\n")
        f.write("| Class ID | Category | Train | Expert | Gating | Train % | Expert % | Gating % | Split Ratio |\n")
        f.write("|----------|----------|-------|--------|--------|---------|----------|----------|-------------|\n")
        
        # Show all classes
        for cls in class_data:
            f.write("| {class_id} | {category} | {train_samples:,} | {expert_samples} | {gating_samples} | "
                   "{train_pct:.3f}% | {expert_pct:.3f}% | {gating_pct:.3f}% | {split_ratio:.2f} |\n".format(**cls))
    
    print(f"✓ Saved Markdown report: {md_path}")
    
    # Generate CSV report for easier analysis
    csv_path = output_dir / "class_distribution_report.csv"
    print(f"Generating CSV report: {csv_path}")
    
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['class_id', 'category', 'train_samples', 'expert_samples', 'gating_samples',
                     'train_pct', 'expert_pct', 'gating_pct', 'split_ratio']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(class_data)
    
    print(f"✓ Saved CSV report: {csv_path}")
    
    # Also create a compact summary by bins
    summary_path = output_dir / "class_distribution_summary.md"
    print(f"Generating summary report: {summary_path}")
    
    # Bin classes by sample count ranges
    bins = [
        (0, 5, "0-5"),
        (6, 10, "6-10"),
        (11, 20, "11-20"),
        (21, 50, "21-50"),
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, 5000, "1001-5000"),
        (5001, float('inf'), "5000+"),
    ]
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# ImageNet-LT Class Distribution Summary\n\n")
        f.write("**Generated:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("---\n\n")
        
        f.write("## Distribution by Sample Count Ranges\n\n")
        f.write("| Range | # Classes | % Classes | Train Samples | % Train | Expert Samples | Gating Samples |\n")
        f.write("|-------|-----------|-----------|---------------|---------|----------------|----------------|\n")
        
        for min_count, max_count, label in bins:
            mask = (np.array(train_counts) >= min_count) & (np.array(train_counts) <= max_count)
            num_in_range = mask.sum()
            if num_in_range == 0:
                continue
            
            train_in_range = sum(train_counts[i] for i in range(num_classes) if mask[i])
            expert_in_range = sum(expert_counts[i] for i in range(num_classes) if mask[i])
            gating_in_range = sum(gating_counts[i] for i in range(num_classes) if mask[i])
            
            f.write(f"| {label} | {num_in_range:,} | {num_in_range/num_classes*100:.2f}% | "
                   f"{train_in_range:,} | {train_in_range/total_train*100:.2f}% | "
                   f"{expert_in_range:,} | {gating_in_range:,} |\n")
    
    print(f"✓ Saved summary report: {summary_path}")


def split_train_for_expert_and_gating(
    train_dataset,
    train_counts: Dict[int, int],
    expert_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Split train set (from ImageNet_LT_train.txt) into Expert (90%) and Gating (10%).
    Maintains same imbalance ratio!
    
    Args:
        train_dataset: ImageNetLTDataset instance (loaded from ImageNet_LT_train.txt)
        train_counts: Dict mapping class_id to train count
        expert_ratio: Ratio for expert training (default: 0.9 = 90%)
        seed: Random seed
        
    Returns:
        (expert_indices, expert_targets, expert_counts_list,
         gating_indices, gating_targets, gating_counts_list)
    """
    print(f"\n{'='*80}")
    print(f"SPLITTING TRAIN SET (from ImageNet_LT_train.txt)")
    print(f"  Expert: {expert_ratio*100:.0f}% | Gating: {(1-expert_ratio)*100:.0f}%")
    print(f"{'='*80}")
    
    np.random.seed(seed)
    
    # Group indices by class
    print("  Grouping samples by class...")
    indices_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset.samples):
        indices_by_class[label].append(idx)
    print(f"  ✓ Grouped {len(indices_by_class)} classes")
    
    expert_indices = []
    gating_indices = []
    expert_counts = defaultdict(int)
    gating_counts = defaultdict(int)
    
    # Split each class
    print("  Splitting each class...")
    classes_to_process = sorted(indices_by_class.keys())
    print(f"  Processing {len(classes_to_process)} classes...")
    
    for i, cls in enumerate(classes_to_process):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(classes_to_process)} classes processed...")
            
        cls_indices = np.array(indices_by_class[cls])
        total_cls = len(cls_indices)
        
        # Calculate split sizes
        expert_size = int(total_cls * expert_ratio)
        gating_size = total_cls - expert_size
        
        # Shuffle and split
        np.random.shuffle(cls_indices)
        expert_cls_indices = cls_indices[:expert_size].tolist()
        gating_cls_indices = cls_indices[expert_size:].tolist()
        
        expert_indices.extend(expert_cls_indices)
        gating_indices.extend(gating_cls_indices)
        expert_counts[cls] = expert_size
        gating_counts[cls] = gating_size
    
    print("  ✓ Completed splitting all classes")
    
    # Get targets
    print("  Extracting labels from samples...")
    expert_targets = [train_dataset.samples[idx][1] for idx in expert_indices]
    gating_targets = [train_dataset.samples[idx][1] for idx in gating_indices]
    print("  ✓ Extracted all labels")
    
    # Convert counts to lists
    num_classes = max(expert_counts.keys()) + 1 if expert_counts else 1000
    expert_counts_list = [expert_counts.get(i, 0) for i in range(num_classes)]
    gating_counts_list = [gating_counts.get(i, 0) for i in range(num_classes)]
    
    # Verify splits
    print(f"\n  SUCCESS: Expert split:")
    print(f"    Total: {len(expert_indices):,} samples")
    if len(expert_counts) > 0:
        non_zero_expert = [c for c in expert_counts_list if c > 0]
        if non_zero_expert:
            print(f"    Mean samples/class: {np.mean(non_zero_expert):.1f}")
            print(f"    Max samples/class: {max(non_zero_expert):,}")
            print(f"    Classes with samples: {len(non_zero_expert):,}")
    
    print(f"\n  SUCCESS: Gating split:")
    print(f"    Total: {len(gating_indices):,} samples")
    if len(gating_counts) > 0:
        non_zero_gating = [c for c in gating_counts_list if c > 0]
        if non_zero_gating:
            print(f"    Mean samples/class: {np.mean(non_zero_gating):.1f}")
            print(f"    Max samples/class: {max(non_zero_gating):,}")
            print(f"    Classes with samples: {len(non_zero_gating):,}")
    
    # Verify no overlap
    expert_set = set(expert_indices)
    gating_set = set(gating_indices)
    assert len(expert_set & gating_set) == 0, "Expert and Gating splits overlap!"
    print("\n  ✓ No overlap between expert and gating splits")
    
    return (expert_indices, expert_targets, expert_counts_list,
            gating_indices, gating_targets, gating_counts_list)


def split_val_into_val_test_tunev(
    val_dataset,
    val_counts: Dict[int, int],
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]:
    """
    Split test set (from ImageNet_LT_test.txt) into Test/Val/TuneV with 8:1:1 ratio.
    
    Split Strategy:
    - Test split: 80% (main test set)
    - Val split: 10% (validation set)
    - TuneV split: 10% (tuning/validation set)
    
    For ImageNet-LT with ~50,000 samples: Test≈40,000, Val≈5,000, TuneV≈5,000
    
    Args:
        val_dataset: ImageNetLTDataset instance (loaded from ImageNet_LT_test.txt)
        val_counts: Dict mapping class_id to test set count
        seed: Random seed
        
    Returns:
        (val_indices, val_targets, test_indices, test_targets, tunev_indices, tunev_targets)
    """
    print(f"\n{'='*80}")
    print("SPLITTING TEST SET (from ImageNet_LT_test.txt) INTO TEST/VAL/TUNEV (80%/10%/10%)")
    print(f"{'='*80}")
    
    np.random.seed(seed)
    
    # Group indices by class
    print("  Grouping samples by class...")
    indices_by_class = defaultdict(list)
    for idx, (_, label) in enumerate(val_dataset.samples):
        indices_by_class[label].append(idx)
    print(f"  ✓ Grouped {len(indices_by_class)} classes")
    
    val_indices = []
    test_indices = []
    tunev_indices = []
    
    # Split each class with 80%:10%:10% ratio (Test:Val:TuneV)
    print("  Splitting each class (80% Test : 10% Val : 10% TuneV)...")
    classes_to_process = sorted(indices_by_class.keys())
    print(f"  Processing {len(classes_to_process)} classes...")
    
    for i, cls in enumerate(classes_to_process):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(classes_to_process)} classes processed...")
            
        cls_indices = np.array(indices_by_class[cls])
        total_cls = len(cls_indices)
        
        # Calculate split sizes: Test (80%) : Val (10%) : TuneV (10%)
        # Test gets 80%, Val gets 10%, TuneV gets 10%
        test_size = int(total_cls * 0.8)
        val_size = int(total_cls * 0.1)
        tunev_size = total_cls - test_size - val_size
        
        # Shuffle and split (order: test, val, tunev)
        np.random.shuffle(cls_indices)
        cls_test_indices = cls_indices[:test_size].tolist()
        cls_val_indices = cls_indices[test_size:test_size+val_size].tolist()
        cls_tunev_indices = cls_indices[test_size+val_size:].tolist()
        
        test_indices.extend(cls_test_indices)
        val_indices.extend(cls_val_indices)
        tunev_indices.extend(cls_tunev_indices)
    
    print("  ✓ Completed splitting all classes")
    
    # Get targets
    print("  Extracting labels from samples...")
    val_targets = [val_dataset.samples[idx][1] for idx in val_indices]
    test_targets = [val_dataset.samples[idx][1] for idx in test_indices]
    tunev_targets = [val_dataset.samples[idx][1] for idx in tunev_indices]
    print("  ✓ Extracted all labels")
    
    print(f"\n  SUCCESS: Test split (80%): {len(test_indices):,} samples")
    print(f"  SUCCESS: Val split (10%): {len(val_indices):,} samples")
    print(f"  SUCCESS: TuneV split (10%): {len(tunev_indices):,} samples")
    
    return (val_indices, val_targets, test_indices, test_targets, tunev_indices, tunev_targets)


def save_splits_to_json(
    splits_dict: Dict,
    output_dir: Path,
    class_counts: Optional[Dict] = None
):
    """Save all splits to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("SAVING SPLITS TO JSON FILES")
    print(f"{'='*80}")
    
    for split_name, split_data in splits_dict.items():
        if isinstance(split_data, dict) and 'indices' in split_data:
            indices_to_save = split_data['indices']
            targets_to_save = split_data.get('targets', [])
        else:
            indices_to_save = split_data
            targets_to_save = []
        
        # Save indices
        indices_path = output_dir / f"{split_name}_indices.json"
        with open(indices_path, 'w') as f:
            json.dump(indices_to_save, f)
        print(f"  ✓ Saved {split_name}_indices.json: {len(indices_to_save):,} samples")
        
        # Save targets if available
        if targets_to_save:
            targets_path = output_dir / f"{split_name}_targets.json"
            with open(targets_path, 'w') as f:
                json.dump(targets_to_save, f)
            print(f"  ✓ Saved {split_name}_targets.json: {len(targets_to_save):,} targets")
    
    # Save class counts and class-to-group mapping
    if class_counts:
        counts_path = output_dir / "train_class_counts.json"
        if isinstance(class_counts, dict):
            # Convert to list if needed
            max_class = max(class_counts.keys()) if class_counts else 0
            counts_list = [class_counts.get(i, 0) for i in range(max_class + 1)]
        else:
            counts_list = class_counts
        
        with open(counts_path, 'w') as f:
            json.dump(counts_list, f)
        print(f"  ✓ Saved train_class_counts.json: {len(counts_list)} classes")
        
        # Save class-to-group mapping (based on train counts, threshold=20)
        # This ensures head/tail classification is consistent across all splits
        threshold = 20
        class_to_group = []
        for i, count in enumerate(counts_list):
            # 0 = head, 1 = tail
            group = 1 if count <= threshold else 0
            class_to_group.append(group)
        
        group_path = output_dir / "class_to_group.json"
        with open(group_path, 'w') as f:
            json.dump(class_to_group, f)
        
        num_head = sum(1 for g in class_to_group if g == 0)
        num_tail = sum(1 for g in class_to_group if g == 1)
        print(f"  ✓ Saved class_to_group.json: {num_head} head classes, {num_tail} tail classes")
        print(f"    (Head/tail classification based on train counts, threshold = {threshold})")


def create_imagenet_lt_splits(
    data_dir: str = "data/imagenet_lt",
    train_label_file: str = "ImageNet_LT_train.txt",
    val_label_file: str = "ImageNet_LT_test.txt",
    output_dir: str = "data/imagenet_lt_splits",
    seed: int = 42,
    expert_ratio: float = 0.9,
    log_file: Optional[str] = None
) -> Tuple[Dict, Dict]:
    """
    Create ImageNet-LT dataset splits.
    
    Split Strategy:
    - Train set (from ImageNet_LT_train.txt) → Expert (90%) + Gating (10%)
    - Test set (from ImageNet_LT_test.txt) → Test (80%) + Val (10%) + TuneV (10%)
    
    Args:
        data_dir: Directory containing images and label files
        train_label_file: Name of train label file (ImageNet_LT_train.txt)
        val_label_file: Name of test label file (ImageNet_LT_test.txt) - will be split into Test/Val/TuneV
        output_dir: Output directory for splits
        seed: Random seed
        expert_ratio: Ratio for expert training (default: 0.9)
        log_file: Optional log file path
        
    Returns:
        Tuple of (splits_dict, class_weights_dict)
    """
    # Setup logging if log_file is provided
    original_stdout = sys.stdout
    log_file_handle = None
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_handle = open(log_path, 'w', encoding='utf-8')
        
        # Create a class that writes to both stdout and log file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = TeeOutput(original_stdout, log_file_handle)
        print(f"\n{'='*80}")
        print(f"LOGGING TO FILE: {log_path}")
        print(f"STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
    
    try:
        print("="*80)
        print("CREATING ImageNet-LT DATASET SPLITS")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Data directory: {data_dir}")
        print(f"  Train label file: {train_label_file} (will be split into Expert 90% + Gating 10%)")
        print(f"  Test label file: {val_label_file} (will be split into Test 80% + Val 10% + TuneV 10%)")
        print(f"  Output directory: {output_dir}")
        print(f"  Random seed: {seed}")
        print(f"  Expert ratio: {expert_ratio} (90% for expert training, 10% for gating)")
        
        # Load datasets
        print(f"\n{'='*80}")
        print("STEP 1: Loading datasets...")
        print(f"{'='*80}")
        
        train_dataset = ImageNetLTDataset(data_dir, train_label_file, transform=None)
        val_dataset = ImageNetLTDataset(data_dir, val_label_file, transform=None)
        
        # Analyze distribution
        output_path = Path(output_dir)
        train_counts, val_counts, total_counts = analyze_imagenet_lt_distribution(
            train_dataset, val_dataset, output_path
        )
        
        # Split train into expert and gating
        print(f"\n{'='*80}")
        print("STEP 2: Splitting train set...")
        print(f"{'='*80}")
        
        (expert_indices, expert_targets, expert_counts_list,
         gating_indices, gating_targets, gating_counts_list) = split_train_for_expert_and_gating(
            train_dataset, train_counts, expert_ratio, seed
        )
        
        # Split test set (from ImageNet_LT_test.txt) into Test/Val/TuneV
        print(f"\n{'='*80}")
        print("STEP 3: Splitting test set (from ImageNet_LT_test.txt) into Test/Val/TuneV...")
        print(f"{'='*80}")
        
        (val_indices, val_targets, test_indices, test_targets,
         tunev_indices, tunev_targets) = split_val_into_val_test_tunev(
            val_dataset, val_counts, seed + 1
        )
        
        # Prepare splits dictionary
        splits = {
            'expert': {
                'indices': expert_indices,
                'targets': expert_targets
            },
            'gating': {
                'indices': gating_indices,
                'targets': gating_targets
            },
            'val': {
                'indices': val_indices,
                'targets': val_targets
            },
            'test': {
                'indices': test_indices,
                'targets': test_targets
            },
            'tunev': {
                'indices': tunev_indices,
                'targets': tunev_targets
            }
        }
        
        # Save splits
        output_path = Path(output_dir)
        save_splits_to_json(splits, output_path, train_counts)
        
        # Compute class weights for reweighting
        print(f"\n{'='*80}")
        print("STEP 4: Computing class weights...")
        print(f"{'='*80}")
        
        total_train_samples = sum(train_counts.values())
        num_classes = len(train_counts)
        
        # Class weights = inverse of class frequency (for importance weighting)
        class_weights = {}
        for cls in range(num_classes):
            count = train_counts.get(cls, 0)
            if count > 0:
                class_weights[cls] = total_train_samples / (num_classes * count)
            else:
                class_weights[cls] = 0.0
        
        # Save class weights
        weights_path = output_path / "class_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(class_weights, f)
        print(f"  ✓ Saved class_weights.json")
        
        # Visualize tail proportion to verify long-tail distribution
        print(f"\n{'='*80}")
        print("STEP 5: Visualizing tail proportion...")
        print(f"{'='*80}")
        
        tail_threshold = 20
        tail_prop = visualize_tail_proportion(train_counts, tail_threshold, output_path)
        
        # Calculate counts for val, test, tunev splits
        val_counts_list = [0] * num_classes
        test_counts_list = [0] * num_classes
        tunev_counts_list = [0] * num_classes
        
        for target in val_targets:
            if 0 <= target < num_classes:
                val_counts_list[target] += 1
        
        for target in test_targets:
            if 0 <= target < num_classes:
                test_counts_list[target] += 1
        
        for target in tunev_targets:
            if 0 <= target < num_classes:
                tunev_counts_list[target] += 1
        
        # Create comprehensive visualization with all 6 splits
        print(f"\n{'='*80}")
        print("STEP 6: Creating comprehensive splits visualization...")
        print(f"{'='*80}")
        
        train_counts_list = [train_counts.get(i, 0) for i in range(num_classes)]
        
        visualize_all_splits_comprehensive(
            train_counts_list,
            expert_counts_list,
            gating_counts_list,
            val_counts_list,
            test_counts_list,
            tunev_counts_list,
            output_path,
            threshold=tail_threshold
        )
        
        # Generate class distribution report (CSV + Markdown)
        print(f"\n{'='*80}")
        print("STEP 7: Generating class distribution report...")
        print(f"{'='*80}")
        
        generate_class_distribution_report(
            [train_counts.get(i, 0) for i in range(num_classes)],
            expert_counts_list,
            gating_counts_list,
            output_path,
            threshold=tail_threshold
        )
        
        print(f"\n{'='*80}")
        print("SUCCESS: All splits created!")
        print(f"{'='*80}")
        print(f"\nGenerated files:")
        print(f"  - All split JSON files (expert, gating, val, test, tunev)")
        print(f"  - train_class_counts.json")
        print(f"  - class_weights.json")
        print(f"  - tail_proportion_analysis.png")
        print(f"  - comprehensive_splits_distribution.png (6 splits: train, expert, gating, val, test, tunev)")
        print(f"  - class_distribution_report.md (Markdown format)")
        print(f"  - class_distribution_report.csv (CSV format)")
        print(f"  - class_distribution_summary.md (Summary by bins)")
        
        return splits, class_weights
        
    finally:
        # Restore stdout and close log file
        if log_file_handle is not None:
            sys.stdout = original_stdout
            log_file_handle.close()
            print(f"\n[Log saved to: {log_file}]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ImageNet-LT dataset splits"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/imagenet_lt',
        help='Directory containing images and label files'
    )
    parser.add_argument(
        '--train-label-file',
        type=str,
        default='ImageNet_LT_train.txt',
        help='Train label file name'
    )
    parser.add_argument(
        '--val-label-file',
        type=str,
        default='ImageNet_LT_test.txt',
        help='Val label file name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/imagenet_lt_splits',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
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
        default='logs/imagenet_lt_splits.log',
        help='Path to log file'
    )
    
    args = parser.parse_args()
    
    create_imagenet_lt_splits(
        data_dir=args.data_dir,
        train_label_file=args.train_label_file,
        val_label_file=args.val_label_file,
        output_dir=args.output_dir,
        seed=args.seed,
        expert_ratio=args.expert_ratio,
        log_file=args.log_file
    )

