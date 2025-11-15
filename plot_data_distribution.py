#!/usr/bin/env python3
"""
Script để vẽ biểu đồ phân phối dữ liệu cho CIFAR-100-LT dataset.
Plots data distribution figures for different splits.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def load_class_counts(splits_dir: str) -> Dict[str, List[int]]:
    """Load class counts for all available splits."""
    splits_dir = Path(splits_dir)
    counts = {}
    
    # List of possible class count files
    count_files = {
        'train': 'train_class_counts.json',
        'expert': 'expert_class_counts.json',
        'gating': 'gating_class_counts.json',
    }
    
    for split_name, filename in count_files.items():
        file_path = splits_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                counts[split_name] = json.load(f)
            print(f"Loaded {split_name}: {sum(counts[split_name]):,} samples")
        else:
            print(f"Warning: {split_name} not found: {file_path}")
    
    return counts


def load_group_config(splits_dir: str) -> Optional[Dict]:
    """Load group configuration."""
    splits_dir = Path(splits_dir)
    config_path = splits_dir / 'group_config.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def plot_distribution(
    class_counts: Dict[str, List[int]],
    group_config: Optional[Dict] = None,
    output_path: str = "data_distribution.png",
    figsize: tuple = (14, 8)
):
    """
    Vẽ biểu đồ phân phối dữ liệu.
    
    Args:
        class_counts: Dictionary mapping split names to class count lists
        group_config: Group configuration (head/tail boundaries)
        output_path: Path to save the figure
        figsize: Figure size
    """
    num_splits = len(class_counts)
    if num_splits == 0:
        print("ERROR: No data to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('CIFAR-100-LT Data Distribution', fontsize=16, fontweight='bold')
    
    # Color palette
    colors = {
        'train': '#1f77b4',    # Blue
        'expert': '#ff7f0e',   # Orange
        'gating': '#2ca02c',   # Green
    }
    
    # Determine group boundaries
    if group_config and 'group_boundaries' in group_config:
        tail_start = group_config['group_boundaries'][0] + 1
    else:
        # Default: classes 0-68 are head, 69-99 are tail
        tail_start = 69
    
    classes = list(range(100))
    
    # 1. All splits comparison (log scale)
    ax1 = axes[0, 0]
    for split_name, counts in class_counts.items():
        if len(counts) != 100:
            print(f"Warning: {split_name} has {len(counts)} classes, expected 100")
            continue
        
        total = sum(counts)
        ax1.semilogy(
            classes, counts,
            'o-', alpha=0.7,
            color=colors.get(split_name, 'gray'),
            label=f'{split_name.upper()} (n={total:,})',
            markersize=3,
            linewidth=2
        )
    
    # Add group boundaries
    ax1.axvline(x=tail_start - 0.5, color='red', linestyle='--', 
                alpha=0.6, linewidth=1.5, label='Head/Tail boundary')
    
    ax1.set_xlabel('Class Index (Head → Tail)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Sample Count (log scale)', fontsize=11, fontweight='bold')
    ax1.set_title('All Splits Comparison (Log Scale)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Add group labels
    ax1.text(34, ax1.get_ylim()[1]*0.3, 'HEAD', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=10, fontweight='bold')
    ax1.text(84, ax1.get_ylim()[1]*0.3, 'TAIL', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
            fontsize=10, fontweight='bold')
    
    # 2. Individual distributions (bar plots)
    for idx, (split_name, counts) in enumerate(class_counts.items()):
        if idx >= 3:  # Only plot first 3 splits
            break
        
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        ax = axes[row, col]
        
        # Create bar plot with different colors for head and tail
        head_counts = counts[:tail_start]
        tail_counts = counts[tail_start:]
        head_classes = classes[:tail_start]
        tail_classes = classes[tail_start:]
        
        ax.bar(head_classes, head_counts, alpha=0.7, color='lightgreen', 
              label='Head', width=0.8)
        ax.bar(tail_classes, tail_counts, alpha=0.7, color='lightcoral', 
              label='Tail', width=0.8)
        
        # Add boundary line
        ax.axvline(x=tail_start - 0.5, color='red', linestyle='--', 
                  alpha=0.6, linewidth=1.5)
        
        # Calculate statistics
        total = sum(counts)
        head_total = sum(head_counts)
        tail_total = sum(tail_counts)
        if_counts = counts[0] / counts[99] if counts[99] > 0 else 0
        
        # Styling
        ax.set_title(f'{split_name.upper()} Distribution\n'
                    f'Total: {total:,} samples | IF: {if_counts:.1f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Class Index', fontsize=10)
        ax.set_ylabel('Sample Count', fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        
        # Add statistics text
        stats_text = f'Head: {head_total:,} ({head_total/total*100:.1f}%)\n'
        stats_text += f'Tail: {tail_total:,} ({tail_total/total*100:.1f}%)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Hide unused subplots
    used_subplots = min(3, len(class_counts))
    if used_subplots < 3:
        for idx in range(used_subplots, 3):
            row = (idx + 1) // 2
            col = (idx + 1) % 2
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved distribution plot to: {output_path}")
    
    plt.close()


def plot_simple_distribution(
    class_counts: List[int],
    split_name: str = "Data",
    output_path: str = "data_distribution_simple.png",
    tail_start: int = 69
):
    """
    Vẽ biểu đồ phân phối đơn giản cho một split.
    
    Args:
        class_counts: List of class counts (length 100)
        split_name: Name of the split
        output_path: Path to save the figure
        tail_start: First class index of tail group
    """
    if len(class_counts) != 100:
        print(f"Warning: Expected 100 classes, got {len(class_counts)}")
    
    classes = list(range(len(class_counts)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot with different colors for head and tail
    head_counts = class_counts[:tail_start]
    tail_counts = class_counts[tail_start:]
    head_classes = classes[:tail_start]
    tail_classes = classes[tail_start:]
    
    ax.bar(head_classes, head_counts, alpha=0.7, color='lightgreen', 
          label='Head', width=0.8)
    ax.bar(tail_classes, tail_counts, alpha=0.7, color='lightcoral', 
          label='Tail', width=0.8)
    
    # Add boundary line
    ax.axvline(x=tail_start - 0.5, color='red', linestyle='--', 
              alpha=0.6, linewidth=1.5, label='Head/Tail boundary')
    
    # Calculate statistics
    total = sum(class_counts)
    head_total = sum(head_counts)
    tail_total = sum(tail_counts)
    if_counts = class_counts[0] / class_counts[-1] if class_counts[-1] > 0 else 0
    
    # Styling
    ax.set_title(f'{split_name.upper()} Distribution\n'
                f'Total: {total:,} samples | IF: {if_counts:.1f}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Class Index (Head → Tail)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Count (log scale)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Add statistics text
    stats_text = f'Head (0-{tail_start-1}): {head_total:,} ({head_total/total*100:.1f}%)\n'
    stats_text += f'Tail ({tail_start}-99): {tail_total:,} ({tail_total/total*100:.1f}%)\n'
    stats_text += f'Imbalance Factor: {if_counts:.1f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved simple distribution plot to: {output_path}")
    
    plt.close()


def main():
    """Main function to plot data distributions."""
    # Configuration
    splits_dir = "./data/cifar100_lt_if100_splits_fixed"
    output_dir = "./results/data_distribution"
    
    print("=" * 80)
    print("PLOTTING DATA DISTRIBUTION")
    print("=" * 80)
    
    # Load data
    print("\nLoading class counts...")
    class_counts = load_class_counts(splits_dir)
    
    if not class_counts:
        print("ERROR: No class counts found!")
        return
    
    # Load group config
    group_config = load_group_config(splits_dir)
    if group_config:
        print(f"Loaded group config: {group_config['group_names']}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot comprehensive distribution
    print("\nCreating comprehensive distribution plot...")
    plot_distribution(
        class_counts,
        group_config,
        output_path=output_dir / "data_distribution_comprehensive.png"
    )
    
    # Plot individual distributions
    print("\nCreating individual distribution plots...")
    for split_name, counts in class_counts.items():
        tail_start = 69
        if group_config and 'group_boundaries' in group_config:
            tail_start = group_config['group_boundaries'][0] + 1
        
        plot_simple_distribution(
            counts,
            split_name=split_name,
            output_path=output_dir / f"data_distribution_{split_name}.png",
            tail_start=tail_start
        )
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

