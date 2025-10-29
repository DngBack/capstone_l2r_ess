import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from torch.utils.data import Subset

def get_cifar100_lt_counts_paper(imb_factor=100):
    """Tính số samples cho mỗi class theo paper (giữ nguyên 50,000 samples)"""
    img_max = 500.0  # CIFAR-100 có 500 samples/class
    
    counts = []
    for cls_idx in range(100):
        # Exponential profile: n_i = n_max * (IF)^(-i/(C-1))
        num = img_max * (imb_factor ** (-cls_idx / 99.0))
        counts.append(max(1, int(num)))
    
    # Normalize để có đúng 50,000 samples
    total_current = sum(counts)
    target_total = 50000
    
    # Scale factor
    scale_factor = target_total / total_current
    
    # Apply scaling
    scaled_counts = []
    for count in counts:
        scaled_count = max(1, int(count * scale_factor))
        scaled_counts.append(scaled_count)
    
    # Adjust để có đúng 50,000 samples
    current_total = sum(scaled_counts)
    diff = target_total - current_total
    
    if diff > 0:
        # Thêm vào class đầu tiên
        scaled_counts[0] += diff
    elif diff < 0:
        # Bớt từ class đầu tiên
        scaled_counts[0] = max(1, scaled_counts[0] + diff)
    
    return scaled_counts

def get_class_to_group_by_threshold_paper(class_counts, threshold=20):
    """Chia classes thành head/tail groups theo paper (threshold=20)"""
    class_to_group = torch.zeros(100, dtype=torch.long)
    
    for class_idx, count in enumerate(class_counts):
        if count > threshold:
            class_to_group[class_idx] = 0  # Head group
        else:
            class_to_group[class_idx] = 1  # Tail group
    
    return class_to_group

def create_longtail_train_set_paper(cifar_train, imb_factor=100, seed=42):
    """Tạo long-tail train set theo paper (giữ nguyên 50,000 samples)"""
    np.random.seed(seed)
    
    target_counts = get_cifar100_lt_counts_paper(imb_factor)
    train_targets = np.array(cifar_train.targets)
    
    lt_train_indices = []
    for cls in range(100):
        cls_indices = np.where(train_targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        lt_train_indices.extend(sampled.tolist())
    
    return Subset(cifar_train, lt_train_indices), target_counts

def create_longtail_test_val_sets_paper(cifar_test, train_class_counts, seed=42):
    """Tạo test/val sets với cùng distribution như train set"""
    np.random.seed(seed)
    
    test_targets = np.array(cifar_test.targets)
    
    # Tính tỷ lệ của mỗi class trong train set
    total_train_samples = sum(train_class_counts)
    class_proportions = [count / total_train_samples for count in train_class_counts]
    
    # Tách 20% làm validation, 80% làm test
    val_size = int(0.2 * len(cifar_test))
    test_size = len(cifar_test) - val_size
    
    # Tính số samples cần cho mỗi class trong validation set
    target_val_counts = [int(prop * val_size) for prop in class_proportions]
    
    # Đảm bảo tổng bằng val_size
    current_total = sum(target_val_counts)
    diff = val_size - current_total
    
    if diff > 0:
        target_val_counts[0] += diff
    elif diff < 0:
        target_val_counts[0] = max(1, target_val_counts[0] + diff)
    
    # Tính số samples cần cho mỗi class trong test set
    target_test_counts = [int(prop * test_size) for prop in class_proportions]
    
    # Đảm bảo tổng bằng test_size
    current_total = sum(target_test_counts)
    diff = test_size - current_total
    
    if diff > 0:
        target_test_counts[0] += diff
    elif diff < 0:
        target_test_counts[0] = max(1, target_test_counts[0] + diff)
    
    # Sample từ test set theo target counts
    lt_val_indices = []
    lt_test_indices = []
    
    for cls in range(100):
        cls_indices = np.where(test_targets == cls)[0]
        
        # Sample cho validation
        val_num_to_sample = min(target_val_counts[cls], len(cls_indices))
        val_sampled = np.random.choice(cls_indices, val_num_to_sample, replace=False)
        lt_val_indices.extend(val_sampled.tolist())
        
        # Sample cho test
        test_num_to_sample = min(target_test_counts[cls], len(cls_indices))
        test_sampled = np.random.choice(cls_indices, test_num_to_sample, replace=False)
        lt_test_indices.extend(test_sampled.tolist())
    
    val_dataset = Subset(cifar_test, lt_val_indices)
    test_dataset = Subset(cifar_test, lt_test_indices)
    
    return val_dataset, test_dataset, target_val_counts, target_test_counts

def analyze_data_distribution_paper():
    """Phân tích phân phối data theo paper"""
    
    print("=" * 80)
    print("BAO CAO PHAN PHOI DATA (PAPER COMPLIANT)")
    print("=" * 80)
    
    # 1. Load original CIFAR-100
    print(f"\n1. ORIGINAL CIFAR-100 DATASET:")
    cifar_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    
    print(f"   Original Train samples: {len(cifar_train):,}")
    print(f"   Original Test samples: {len(cifar_test):,}")
    print(f"   Original samples per class: 500 (train), 100 (test)")
    
    # 2. Generate Long-tail distribution theo paper
    print(f"\n2. LONG-TAIL DISTRIBUTION (IF=100, Paper Compliant):")
    target_counts = get_cifar100_lt_counts_paper(imb_factor=100)
    
    print(f"   Target counts per class:")
    print(f"     Min: {min(target_counts)}")
    print(f"     Max: {max(target_counts)}")
    print(f"     Mean: {np.mean(target_counts):.1f}")
    print(f"     Total: {sum(target_counts):,}")
    
    # 3. Create long-tail train set theo paper
    print(f"\n3. LONG-TAIL TRAIN SET (Paper Compliant):")
    train_lt, train_class_counts = create_longtail_train_set_paper(cifar_train, imb_factor=100, seed=42)
    
    print(f"   Train samples (after down sample): {len(train_lt):,}")
    print(f"   Reduction ratio: {len(train_lt)/len(cifar_train)*100:.1f}%")
    
    # 4. Group analysis với threshold=20 theo paper
    print(f"\n4. GROUP ANALYSIS (Threshold=20, Paper Compliant):")
    class_to_group = get_class_to_group_by_threshold_paper(train_class_counts, threshold=20)
    
    head_classes = torch.sum(class_to_group == 0)
    tail_classes = torch.sum(class_to_group == 1)
    
    head_samples = sum(train_class_counts[i] for i in range(100) if class_to_group[i] == 0)
    tail_samples = sum(train_class_counts[i] for i in range(100) if class_to_group[i] == 1)
    
    print(f"   Head classes: {head_classes} ({head_classes/100*100:.1f}%)")
    print(f"   Tail classes: {tail_classes} ({tail_classes/100*100:.1f}%)")
    print(f"   Head samples: {head_samples:,} ({head_samples/sum(train_class_counts)*100:.1f}%)")
    print(f"   Tail samples: {tail_samples:,} ({tail_samples/sum(train_class_counts)*100:.1f}%)")
    
    # 5. Create long-tail test/val sets
    print(f"\n5. LONG-TAIL TEST/VAL SETS (Paper Compliant):")
    val_lt, test_lt, val_class_counts, test_class_counts = create_longtail_test_val_sets_paper(
        cifar_test, train_class_counts, seed=42
    )
    
    print(f"   Val samples (20% of test): {len(val_lt):,}")
    print(f"   Test samples (80% of test): {len(test_lt):,}")
    
    # 6. Paper comparison
    print(f"\n6. PAPER COMPARISON:")
    print(f"   Paper Table 3:")
    print(f"     ntrain: 50,000")
    print(f"     ntest: 10,000")
    print(f"     Tail prop.: 0.03")
    
    print(f"\n   Our Implementation:")
    print(f"     ntrain: {len(train_lt):,}")
    print(f"     ntest: {len(test_lt):,}")
    print(f"     nval: {len(val_lt):,}")
    print(f"     Tail prop.: {tail_samples/sum(train_class_counts):.3f}")
    
    # 7. Detailed class distribution
    print(f"\n7. DETAILED CLASS DISTRIBUTION:")
    print(f"   First 10 classes:")
    for i in range(10):
        group = "Head" if class_to_group[i] == 0 else "Tail"
        print(f"     Class {i:2d}: {train_class_counts[i]:3d} samples ({group})")
    
    print(f"\n   Last 10 classes:")
    for i in range(90, 100):
        group = "Head" if class_to_group[i] == 0 else "Tail"
        print(f"     Class {i:2d}: {train_class_counts[i]:3d} samples ({group})")
    
    # 8. Create visualization
    print(f"\n8. CREATING VISUALIZATION...")
    
    plt.figure(figsize=(20, 15))
    
    # Class distribution
    plt.subplot(3, 3, 1)
    plt.bar(range(100), train_class_counts, color='skyblue', alpha=0.7)
    plt.axhline(y=20, color='red', linestyle='--', label='Threshold=20')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Long-tail Class Distribution (Train, Paper Compliant)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Group distribution
    plt.subplot(3, 3, 2)
    groups = ['Head', 'Tail']
    group_counts = [head_samples, tail_samples]
    colors = ['lightgreen', 'lightcoral']
    plt.bar(groups, group_counts, color=colors, alpha=0.7)
    plt.ylabel('Number of Samples')
    plt.title('Group Distribution (Train, Paper Compliant)')
    plt.grid(True, alpha=0.3)
    
    # Class distribution (log scale)
    plt.subplot(3, 3, 3)
    plt.semilogy(range(100), train_class_counts, 'b-o', markersize=3)
    plt.axhline(y=20, color='red', linestyle='--', label='Threshold=20')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples (log scale)')
    plt.title('Long-tail Distribution (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(3, 3, 4)
    sorted_counts = sorted(train_class_counts, reverse=True)
    cumulative = np.cumsum(sorted_counts)
    plt.plot(range(100), cumulative, 'g-', linewidth=2)
    plt.xlabel('Class Index (sorted)')
    plt.ylabel('Cumulative Samples')
    plt.title('Cumulative Distribution (Train)')
    plt.grid(True, alpha=0.3)
    
    # Test set distribution
    plt.subplot(3, 3, 5)
    plt.bar(range(100), test_class_counts, color='lightblue', alpha=0.7)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Test Set Distribution (Paper Compliant)')
    plt.grid(True, alpha=0.3)
    
    # Validation set distribution
    plt.subplot(3, 3, 6)
    plt.bar(range(100), val_class_counts, color='lightyellow', alpha=0.7)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Validation Set Distribution (Paper Compliant)')
    plt.grid(True, alpha=0.3)
    
    # Sample size comparison
    plt.subplot(3, 3, 7)
    datasets = ['Original Train', 'Long-tail Train', 'Test', 'Validation']
    sizes = [len(cifar_train), len(train_lt), len(test_lt), len(val_lt)]
    colors = ['red', 'blue', 'green', 'orange']
    plt.bar(datasets, sizes, color=colors, alpha=0.7)
    plt.ylabel('Number of Samples')
    plt.title('Dataset Size Comparison (Paper Compliant)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Tail proportion comparison
    plt.subplot(3, 3, 8)
    datasets = ['Train', 'Test', 'Validation']
    tail_props = [tail_samples/sum(train_class_counts), 
                  sum(test_class_counts[i] for i in range(100) if class_to_group[i] == 1)/sum(test_class_counts),
                  sum(val_class_counts[i] for i in range(100) if class_to_group[i] == 1)/sum(val_class_counts)]
    plt.bar(datasets, tail_props, color=['blue', 'green', 'orange'], alpha=0.7)
    plt.ylabel('Tail Proportion')
    plt.title('Tail Proportion Comparison (Paper Compliant)')
    plt.grid(True, alpha=0.3)
    
    # Class count distribution
    plt.subplot(3, 3, 9)
    plt.hist(train_class_counts, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    plt.axvline(x=20, color='red', linestyle='--', label='Threshold=20')
    plt.xlabel('Samples per Class')
    plt.ylabel('Number of Classes')
    plt.title('Class Count Distribution (Paper Compliant)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./data_distribution_paper_compliant.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved to: ./data_distribution_paper_compliant.png")
    
    # 9. Save detailed report
    report = {
        'paper_specifications': {
            'imb_factor': 100,
            'threshold': 20,
            'target_train_samples': 50000,
            'target_test_samples': 10000,
            'target_tail_proportion': 0.03
        },
        'original': {
            'train_samples': len(cifar_train),
            'test_samples': len(cifar_test),
            'train_samples_per_class': 500,
            'test_samples_per_class': 100
        },
        'long_tail': {
            'train_samples': len(train_lt),
            'test_samples': len(test_lt),
            'val_samples': len(val_lt),
            'head_classes': head_classes.item(),
            'tail_classes': tail_classes.item(),
            'head_samples': head_samples,
            'tail_samples': tail_samples,
            'tail_proportion': tail_samples/sum(train_class_counts)
        },
        'class_counts': {
            'train': train_class_counts,
            'test': test_class_counts,
            'val': val_class_counts
        },
        'class_to_group': class_to_group.tolist()
    }
    
    with open('./data_distribution_paper_compliant.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Detailed report saved to: ./data_distribution_paper_compliant.json")
    
    print(f"\n" + "=" * 80)
    print("DATA DISTRIBUTION REPORT COMPLETED (PAPER COMPLIANT)!")
    print("=" * 80)
    
    return report

if __name__ == '__main__':
    analyze_data_distribution_paper()
