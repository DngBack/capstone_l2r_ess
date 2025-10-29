import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import ResNet backbone
import sys
sys.path.append('./src')
from models.backbones.resnet_cifar import CIFARResNet32

def get_cifar100_lt_counts_paper(imb_factor=100):
    """Tính số samples cho mỗi class theo paper (KHÔNG normalize)"""
    img_max = 500.0  # CIFAR-100 có 500 samples/class
    
    counts = []
    for cls_idx in range(100):
        # Exponential profile: n_i = n_max * (IF)^(-i/(C-1))
        num = img_max * (imb_factor ** (-cls_idx / 99.0))
        counts.append(max(1, int(num)))
    
    return counts

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
    """Tạo long-tail train set theo paper (KHÔNG normalize)"""
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
    """Tạo test/val sets - CÙNG distribution như train set"""
    np.random.seed(seed)
    
    test_targets = np.array(cifar_test.targets)
    
    # Tính target distribution từ train
    total_train = sum(train_class_counts)
    train_proportions = [count / total_train for count in train_class_counts]
    
    # Tạo test/val sets với cùng distribution
    val_indices = []
    test_indices = []
    
    for cls in range(100):
        cls_indices = np.where(test_targets == cls)[0]
        n_available = len(cls_indices)
        
        # Số samples cần theo tỷ lệ train
        n_needed = int(train_proportions[cls] * 10000)  # 10k total
        n_needed = min(n_needed, n_available)
        
        if n_needed > 0:
            sampled = np.random.choice(cls_indices, n_needed, replace=False)
            
            # Tách 20% val, 80% test
            n_val = int(0.2 * n_needed)
            val_indices.extend(sampled[:n_val].tolist())
            test_indices.extend(sampled[n_val:].tolist())
    
    val_dataset = Subset(cifar_test, val_indices)
    test_dataset = Subset(cifar_test, test_indices)
    
    # Tính class counts cho val và test
    val_targets = [cifar_test.targets[i] for i in val_indices]
    test_targets = [cifar_test.targets[i] for i in test_indices]
    
    val_class_counts = [val_targets.count(i) for i in range(100)]
    test_class_counts = [test_targets.count(i) for i in range(100)]
    
    print(f"Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")
    print(f"Val head classes: {sum(1 for c in val_class_counts if c > 20)}")
    print(f"Val tail classes: {sum(1 for c in val_class_counts if c <= 20)}")
    
    return val_dataset, test_dataset, val_class_counts, test_class_counts

def get_cifar100_lt_dataloaders_paper(batch_size=128, num_workers=4, imb_factor=100, seed=42):
    """Tạo dataloaders theo paper specifications"""
    
    # Data transforms theo paper
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load original CIFAR-100
    cifar_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    # Create long-tail train set
    train_lt, train_class_counts = create_longtail_train_set_paper(cifar_train, imb_factor, seed)
    
    # Create test/val sets
    val_lt, test_lt, val_class_counts, test_class_counts = create_longtail_test_val_sets_paper(
        cifar_test, train_class_counts, seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_lt, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_lt, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_lt, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Group information
    class_to_group = get_class_to_group_by_threshold_paper(train_class_counts, threshold=20)
    
    # Group priors
    group_priors = torch.zeros(2)
    for i in range(100):
        group_priors[class_to_group[i]] += train_class_counts[i]
    group_priors = group_priors / group_priors.sum()
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_class_counts': train_class_counts,
        'val_class_counts': val_class_counts,
        'test_class_counts': test_class_counts,
        'class_to_group': class_to_group,
        'group_priors': group_priors
    }

def evaluate_metrics(model, dataloader, class_to_group, device):
    """Tính balanced error và worst-group error theo paper - average of class-wise errors"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Tính class-wise errors (theo paper định nghĩa)
    class_errors = []
    for cls in range(100):
        cls_indices = (all_targets == cls)
        if cls_indices.sum() > 0:  # Nếu có samples của class này
            cls_error = (all_preds[cls_indices] != all_targets[cls_indices]).float().mean()
            class_errors.append(cls_error.item())
        else:
            class_errors.append(0.0)  # Nếu không có samples, error = 0
    
    # Balanced Error = average of class-wise errors (theo paper)
    balanced_error = np.mean(class_errors)
    
    # Worst-group Error = max of class-wise errors (theo paper)
    worst_group_error = np.max(class_errors)
    
    # Head/Tail errors (để so sánh và debug)
    head_indices = (class_to_group[all_targets] == 0)
    tail_indices = (class_to_group[all_targets] == 1)
    
    head_error = (all_preds[head_indices] != all_targets[head_indices]).float().mean()
    tail_error = (all_preds[tail_indices] != all_targets[tail_indices]).float().mean()
    
    # Standard accuracy
    standard_acc = (all_preds == all_targets).float().mean()
    
    return {
        'balanced_error': balanced_error,
        'worst_group_error': worst_group_error,
        'head_error': head_error.item(),
        'tail_error': tail_error.item(),
        'standard_acc': standard_acc.item()
    }

def train_ce_expert_paper():
    """Train CE expert theo paper specifications"""
    
    print("=" * 80)
    print("TRAINING CE EXPERT (PAPER COMPLIANT)")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data loaders
    print("\n1. LOADING DATA...")
    data_info = get_cifar100_lt_dataloaders_paper(
        batch_size=128,
        num_workers=4,
        imb_factor=100,
        seed=42
    )
    
    train_loader = data_info['train_loader']
    val_loader = data_info['val_loader']
    test_loader = data_info['test_loader']
    train_class_counts = data_info['train_class_counts']
    class_to_group = data_info['class_to_group']
    group_priors = data_info['group_priors']
    
    print(f"   Train samples: {len(train_loader.dataset):,}")
    print(f"   Val samples: {len(val_loader.dataset):,}")
    print(f"   Test samples: {len(test_loader.dataset):,}")
    print(f"   Head classes: {torch.sum(class_to_group == 0)}")
    print(f"   Tail classes: {torch.sum(class_to_group == 1)}")
    print(f"   Tail proportion: {torch.sum(class_to_group == 1).float() / 100:.3f}")
    
    # Model
    print("\n2. CREATING MODEL...")
    backbone = CIFARResNet32(dropout_rate=0.0, init_weights=True)
    model = nn.Sequential(
        backbone,
        nn.Linear(backbone.get_feature_dim(), 100)
    )
    model = model.to(device)
    
    # Loss and optimizer theo paper (Table 3 + F.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate schedule theo paper F.1
    # Manual LR scheduling: warmup 15 steps, then epoch-based decays at 96, 192, 224
    def get_lr(epoch, iteration, total_iters_per_epoch):
        """Manual LR scheduling theo paper specifications"""
        global_iter = epoch * total_iters_per_epoch + iteration
        
        # Warmup: 15 steps đầu với linear warmup
        if global_iter < 15:
            return 0.4 * (global_iter + 1) / 15
        
        # Post-warmup: epoch-based decays
        lr = 0.4
        if epoch >= 224:
            lr *= 0.001  # 0.1^3
        elif epoch >= 192:
            lr *= 0.01   # 0.1^2
        elif epoch >= 96:
            lr *= 0.1
        
        return lr

    
    # Training
    print("\n3. TRAINING...")
    epochs = 256
    
    # Manual LR scheduling - không cần scheduler object
    
    print(f"   Model: ResNet-32")
    print(f"   Loss: CrossEntropyLoss")
    print(f"   Optimizer: SGD(lr=0.4, momentum=0.9, weight_decay=1e-4)")
    print(f"   Scheduler: Warm-up + Decay at [96, 192, 224] epochs")
    print(f"   Epochs: {epochs}")
    best_val_acc = 0.0
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            # Update learning rate manually theo paper
            current_lr = get_lr(epoch, batch_idx, len(train_loader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%', 'LR': f'{current_lr:.6f}'})
        
        # Validation - sử dụng evaluate_metrics function
        val_metrics = evaluate_metrics(model, val_loader, class_to_group, device)
        val_acc = val_metrics['standard_acc'] * 100
        val_head_acc = (1 - val_metrics['head_error']) * 100
        val_tail_acc = (1 - val_metrics['tail_error']) * 100
        val_balanced_error = val_metrics['balanced_error']
        val_worst_group_error = val_metrics['worst_group_error']
        
        # Test accuracy - sử dụng evaluate_metrics function
        test_metrics = evaluate_metrics(model, test_loader, class_to_group, device)
        test_acc = test_metrics['standard_acc'] * 100
        test_head_acc = (1 - test_metrics['head_error']) * 100
        test_tail_acc = (1 - test_metrics['tail_error']) * 100
        test_balanced_error = test_metrics['balanced_error']
        test_worst_group_error = test_metrics['worst_group_error']
        
        train_acc = 100. * train_correct / train_total
        
        train_losses.append(train_loss / len(train_loader))
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            os.makedirs('./checkpoints/experts/cifar100_lt_if100', exist_ok=True)
            torch.save(model.state_dict(), './checkpoints/experts/cifar100_lt_if100/ce_expert_best.pth')
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"\n   Epoch {epoch:3d}: Train Loss: {train_loss/len(train_loader):.4f}")
            print(f"     Train Acc: {train_acc:.2f}%")
            print(f"     Val Acc: {val_acc:.2f}% (Head: {val_head_acc:.2f}%, Tail: {val_tail_acc:.2f}%)")
            print(f"     Val Balanced Error: {val_balanced_error:.4f}, Worst-group Error: {val_worst_group_error:.4f}")
            print(f"     Test Acc: {test_acc:.2f}% (Head: {test_head_acc:.2f}%, Tail: {test_tail_acc:.2f}%)")
            print(f"     Test Balanced Error: {test_balanced_error:.4f}, Worst-group Error: {test_worst_group_error:.4f}")
            print(f"     LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"     Best Val Acc: {best_val_acc:.2f}%")
        
        # Manual LR scheduling - không cần gọi scheduler.step()
    
    print(f"\n   Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model for testing
    model.load_state_dict(torch.load('./checkpoints/experts/cifar100_lt_if100/ce_expert_best.pth'))
    
    # Test - Final evaluation với paper metrics
    print("\n4. FINAL TESTING...")
    model.eval()
    test_logits = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_logits.append(output.cpu())
            test_targets.append(target.cpu())
    
    # Final metrics evaluation
    final_test_metrics = evaluate_metrics(model, test_loader, class_to_group, device)
    final_val_metrics = evaluate_metrics(model, val_loader, class_to_group, device)
    
    print(f"   Final Test Acc: {final_test_metrics['standard_acc']*100:.2f}%")
    print(f"   Final Test Balanced Error: {final_test_metrics['balanced_error']:.4f}")
    print(f"   Final Test Worst-group Error: {final_test_metrics['worst_group_error']:.4f}")
    print(f"   Final Test Head Error: {final_test_metrics['head_error']:.4f}")
    print(f"   Final Test Tail Error: {final_test_metrics['tail_error']:.4f}")
    print(f"   Final Val Acc: {final_val_metrics['standard_acc']*100:.2f}%")
    print(f"   Final Val Balanced Error: {final_val_metrics['balanced_error']:.4f}")
    print(f"   Final Val Worst-group Error: {final_val_metrics['worst_group_error']:.4f}")
    
    # Export logits
    print("\n5. EXPORTING LOGITS...")
    test_logits = torch.cat(test_logits, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Save logits
    os.makedirs('./outputs/logits/cifar100_lt_if100/ce_baseline', exist_ok=True)
    torch.save(test_logits, './outputs/logits/cifar100_lt_if100/ce_baseline/test_logits.pt')
    torch.save(test_targets, './outputs/logits/cifar100_lt_if100/ce_baseline/test_targets.pt')
    
    # Also save val logits for plugin training
    model.eval()
    val_logits = []
    val_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_logits.append(output.cpu())
            val_targets.append(target.cpu())
    
    val_logits = torch.cat(val_logits, dim=0)
    val_targets = torch.cat(val_targets, dim=0)
    
    torch.save(val_logits, './outputs/logits/cifar100_lt_if100/ce_baseline/val_logits.pt')
    torch.save(val_targets, './outputs/logits/cifar100_lt_if100/ce_baseline/val_targets.pt')
    
    # Save train logits for plugin training
    model.eval()
    train_logits = []
    train_targets = []
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_logits.append(output.cpu())
            train_targets.append(target.cpu())
    
    train_logits = torch.cat(train_logits, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    
    torch.save(train_logits, './outputs/logits/cifar100_lt_if100/ce_baseline/train_logits.pt')
    torch.save(train_targets, './outputs/logits/cifar100_lt_if100/ce_baseline/train_targets.pt')
    
    print(f"   Logits saved to: ./outputs/logits/cifar100_lt_if100/ce_baseline/")
    
    # Save training info
    training_info = {
        'model': 'ResNet-32',
        'loss': 'CrossEntropyLoss',
        'optimizer': 'SGD(lr=0.4, momentum=0.9, weight_decay=1e-4)',
        'scheduler': 'Manual: Warmup(15 steps) + epoch decays at [96, 192, 224]',
        'epochs': epochs,
        'best_val_acc': best_val_acc,
        'final_test_acc': final_test_metrics['standard_acc'],
        'final_test_balanced_error': final_test_metrics['balanced_error'],
        'final_test_worst_group_error': final_test_metrics['worst_group_error'],
        'final_test_head_error': final_test_metrics['head_error'],
        'final_test_tail_error': final_test_metrics['tail_error'],
        'final_val_acc': final_val_metrics['standard_acc'],
        'final_val_balanced_error': final_val_metrics['balanced_error'],
        'final_val_worst_group_error': final_val_metrics['worst_group_error'],
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'head_classes': torch.sum(class_to_group == 0).item(),
        'tail_classes': torch.sum(class_to_group == 1).item(),
        'tail_proportion': torch.sum(class_to_group == 1).float().item() / 100,
        'train_class_counts': train_class_counts,
        'class_to_group': class_to_group.tolist(),
        'group_priors': group_priors.tolist()
    }
    
    with open('./outputs/logits/cifar100_lt_if100/ce_baseline/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"   Training info saved to: ./outputs/logits/cifar100_lt_if100/ce_baseline/training_info.json")
    
    # Plot training curves
    print("\n6. PLOTTING TRAINING CURVES...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./outputs/logits/cifar100_lt_if100/ce_baseline/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Training curves saved to: ./outputs/logits/cifar100_lt_if100/ce_baseline/training_curves.png")
    
    print("\n" + "=" * 80)
    print("CE EXPERT TRAINING COMPLETED!")
    print("=" * 80)
    
    return training_info

if __name__ == '__main__':
    train_ce_expert_paper()
