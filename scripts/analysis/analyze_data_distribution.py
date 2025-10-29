import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
from torch.utils.data import Subset

def get_cifar100_lt_counts(imb_factor=100):
    """Tính số samples cho mỗi class theo exponential profile"""
    img_max = 500.0  # CIFAR-100 có 500 samples/class
    
    counts = []
    for cls_idx in range(100):
        # Exponential profile: n_i = n_max * (IF)^(-i/(C-1))
        num = img_max * (imb_factor ** (-cls_idx / 99.0))
        counts.append(max(1, int(num)))
    
    return counts

def find_threshold_for_tail_proportion(class_counts, target_tail_prop=0.03):
    """Tìm threshold để đạt target tail proportion"""
    
    # Sort classes by sample count
    sorted_indices = np.argsort(class_counts)[::-1]  # Descending order
    
    total_samples = sum(class_counts)
    target_tail_samples = int(total_samples * target_tail_prop)
    
    # Tìm threshold
    cumulative_samples = 0
    threshold = None
    
    for i, class_idx in enumerate(sorted_indices):
        cumulative_samples += class_counts[class_idx]
        
        if cumulative_samples >= target_tail_samples:
            threshold = class_counts[class_idx]
            break
    
    return threshold

def get_class_to_group_by_threshold(class_counts, threshold):
    """Chia classes thành head/tail groups theo threshold"""
    class_to_group = torch.zeros(100, dtype=torch.long)
    
    for class_idx, count in enumerate(class_counts):
        if count > threshold:
            class_to_group[class_idx] = 0  # Head group
        else:
            class_to_group[class_idx] = 1  # Tail group
    
    return class_to_group

def create_longtail_train_set(cifar_train, imb_factor=100, seed=42):
    """Tạo long-tail train set (số samples giảm xuống)"""
    np.random.seed(seed)
    
    target_counts = get_cifar100_lt_counts(imb_factor)
    train_targets = np.array(cifar_train.targets)
    
    lt_train_indices = []
    for cls in range(100):
        cls_indices = np.where(train_targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        lt_train_indices.extend(sampled.tolist())
    
    return Subset(cifar_train, lt_train_indices), target_counts

def create_longtail_test_val_sets(cifar_test, train_class_counts, seed=42):
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

def analyze_data_distribution():
    """Phân tích phân phối data và tạo báo cáo"""
    
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
    
    # 2. Generate Long-tail distribution
    print(f"\n2. LONG-TAIL DISTRIBUTION (IF=100):")
    target_counts = get_cifar100_lt_counts(imb_factor=100)
    
    print(f"   Target counts per class:")
    print(f"     Min: {min(target_counts)}")
    print(f"     Max: {max(target_counts)}")
    print(f"     Mean: {np.mean(target_counts):.1f}")
    print(f"     Total: {sum(target_counts):,}")
    
    # 3. Create long-tail train set
    print(f"\n3. LONG-TAIL TRAIN SET:")
    train_lt, train_class_counts = create_longtail_train_set(cifar_train, imb_factor=100, seed=42)
    
    print(f"   Train samples (after down sample): {len(train_lt):,}")
    print(f"   Reduction ratio: {len(train_lt)/len(cifar_train)*100:.1f}%")
    
    # 4. Find threshold để đạt 3% tail proportion
    print(f"\n4. FINDING THRESHOLD FOR 3% TAIL PROPORTION:")
    threshold = find_threshold_for_tail_proportion(train_class_counts, target_tail_prop=0.03)
    print(f"   Threshold: {threshold}")
    
    # 5. Group analysis
    print(f"\n5. GROUP ANALYSIS (Threshold={threshold}):")
    class_to_group = get_class_to_group_by_threshold(train_class_counts, threshold)
    
    head_classes = torch.sum(class_to_group == 0)
    tail_classes = torch.sum(class_to_group == 1)
    
    head_samples = sum(train_class_counts[i] for i in range(100) if class_to_group[i] == 0)
    tail_samples = sum(train_class_counts[i] for i in range(100) if class_to_group[i] == 1)
    
    print(f"   Head classes: {head_classes} ({head_classes/100*100:.1f}%)")
    print(f"   Tail classes: {tail_classes} ({tail_classes/100*100:.1f}%)")
    print(f"   Head samples: {head_samples:,} ({head_samples/sum(train_class_counts)*100:.1f}%)")
    print(f"   Tail samples: {tail_samples:,} ({tail_samples/sum(train_class_counts)*100:.1f}%)")
    
    # 6. Create long-tail test/val sets
    print(f"\n6. LONG-TAIL TEST/VAL SETS:")
    val_lt, test_lt, val_class_counts, test_class_counts = create_longtail_test_val_sets(
        cifar_test, train_class_counts, seed=42
    )
    
    print(f"   Val samples (20% of test): {len(val_lt):,}")
    print(f"   Test samples (80% of test): {len(test_lt):,}")
    
    # 7. Paper comparison
    print(f"\n7. PAPER COMPARISON:")
    print(f"   Paper Table 3:")
    print(f"     ntrain: 50,000")
    print(f"     ntest: 10,000")
    print(f"     Tail prop.: 0.03")
    
    print(f"\n   Our Implementation:")
    print(f"     ntrain: {len(train_lt):,}")
    print(f"     ntest: {len(test_lt):,}")
    print(f"     nval: {len(val_lt):,}")
    print(f"     Tail prop.: {tail_samples/sum(train_class_counts):.3f}")
    
    # 8. Detailed class distribution
    print(f"\n8. DETAILED CLASS DISTRIBUTION:")
    print(f"   First 10 classes:")
    for i in range(10):
        group = "Head" if class_to_group[i] == 0 else "Tail"
        print(f"     Class {i:2d}: {train_class_counts[i]:3d} samples ({group})")
    
    print(f"\n   Last 10 classes:")
    for i in range(90, 100):
        group = "Head" if class_to_group[i] == 0 else "Tail"
        print(f"     Class {i:2d}: {train_class_counts[i]:3d} samples ({group})")
    
    # 9. Create visualization
    print(f"\n9. CREATING VISUALIZATION...")
    
    plt.figure(figsize=(20, 15))
    
    # Class distribution
    plt.subplot(3, 3, 1)
    plt.bar(range(100), train_class_counts, color='skyblue', alpha=0.7)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Long-tail Class Distribution (Train)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Group distribution
    plt.subplot(3, 3, 2)
    groups = ['Head', 'Tail']
    group_counts = [head_samples, tail_samples]
    colors = ['lightgreen', 'lightcoral']
    plt.bar(groups, group_counts, color=colors, alpha=0.7)
    plt.ylabel('Number of Samples')
    plt.title('Group Distribution (Train)')
    plt.grid(True, alpha=0.3)
    
    # Class distribution (log scale)
    plt.subplot(3, 3, 3)
    plt.semilogy(range(100), train_class_counts, 'b-o', markersize=3)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
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
    plt.title('Test Set Distribution')
    plt.grid(True, alpha=0.3)
    
    # Validation set distribution
    plt.subplot(3, 3, 6)
    plt.bar(range(100), val_class_counts, color='lightyellow', alpha=0.7)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Validation Set Distribution')
    plt.grid(True, alpha=0.3)
    
    # Sample size comparison
    plt.subplot(3, 3, 7)
    datasets = ['Original Train', 'Long-tail Train', 'Test', 'Validation']
    sizes = [len(cifar_train), len(train_lt), len(test_lt), len(val_lt)]
    colors = ['red', 'blue', 'green', 'orange']
    plt.bar(datasets, sizes, color=colors, alpha=0.7)
    plt.ylabel('Number of Samples')
    plt.title('Dataset Size Comparison')
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
    plt.title('Tail Proportion Comparison')
    plt.grid(True, alpha=0.3)
    
    # Class count distribution
    plt.subplot(3, 3, 9)
    plt.hist(train_class_counts, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
    plt.xlabel('Samples per Class')
    plt.ylabel('Number of Classes')
    plt.title('Class Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./data_distribution_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Visualization saved to: ./data_distribution_report.png")
    
    # 10. Save detailed report
    report = {
        'original': {
            'train_samples': len(cifar_train),
            'test_samples': len(cifar_test),
            'train_samples_per_class': 500,
            'test_samples_per_class': 100
        },
        'long_tail': {
            'imb_factor': 100,
            'threshold': threshold,
            'train_samples': len(train_lt),
            'test_samples': len(test_lt),
            'val_samples': len(val_lt),
            'reduction_ratio': len(train_lt)/len(cifar_train),
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
    
    with open('./data_distribution_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Detailed report saved to: ./data_distribution_report.json")
    
    print(f"\n" + "=" * 80)
    print("DATA DISTRIBUTION REPORT COMPLETED!")
    print("=" * 80)
    
    return report

if __name__ == '__main__':
    analyze_data_distribution()
