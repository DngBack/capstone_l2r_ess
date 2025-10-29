#!/usr/bin/env python3
"""
CE Expert Training Script - Paper Compliant
===========================================

Trains CE baseline expert exactly as described in paper Appendix F.1.
This is the base model before applying the LtR plugin.

Paper Reference: "Learning to Reject Meets Long-Tail Learning" (ICLR 2024)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import time

# ============================================================================
# PAPER CONFIGURATION (Appendix F.1)
# ============================================================================

PAPER_CONFIG = {
    'dataset': {
        'name': 'cifar100_lt',
        'num_classes': 100,
        'imb_factor': 100,  # Imbalance factor
        'train_samples': 50000,  # Total training samples
        'test_samples': 10000,   # Total test samples
    },
    'model': {
        'architecture': 'resnet32',  # ResNet-32 as specified in paper
        'num_classes': 100,
    },
    'training': {
        'epochs': 256,           # Paper: 256 epochs
        'batch_size': 128,      # Paper: batch size 128
        'lr': 0.1,              # Paper: base learning rate 0.1
        'momentum': 0.9,        # Paper: momentum 0.9
        'weight_decay': 1e-4,   # Paper: weight decay 1e-4
        'milestones': [96, 192, 224],  # Paper: LR decay at epochs 96, 192, 224
        'gamma': 0.1,           # Paper: LR decay factor 0.1
        'warmup_epochs': 15,    # Paper: warmup for 15 steps
    },
    'data_augmentation': {
        'train': {
            'random_crop': {'size': 32, 'padding': 4},
            'random_horizontal_flip': {'p': 0.5},
            'normalize': {
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761]
            }
        },
        'test': {
            'normalize': {
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761]
            }
        }
    },
    'output': {
        'checkpoint_dir': './checkpoints/experts_paper',
        'logits_dir': './outputs/logits_paper',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# RESNET-32 MODEL (Paper Compliant)
# ============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet32(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet32, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet32(num_classes=100):
    """ResNet-32 as specified in paper."""
    return ResNet32(BasicBlock, [5, 5, 5], num_classes)

# ============================================================================
# DATA LOADING
# ============================================================================

def get_cifar100_lt_counts(imb_factor=100, num_classes=100):
    """Generate CIFAR-100-LT class counts using exponential profile."""
    img_max = 500.0  # Original CIFAR-100 has 500 samples per class
    
    counts = []
    for cls_idx in range(num_classes):
        # Exponential profile: n_i = n_max * (IF)^(-i/(C-1))
        num = img_max * (imb_factor ** (-cls_idx / (num_classes - 1.0)))
        counts.append(max(1, int(num)))
    
    return counts

def create_longtail_dataset(cifar_dataset, imb_factor=100, seed=42):
    """Create long-tail version of CIFAR dataset."""
    np.random.seed(seed)
    targets = np.array(cifar_dataset.targets)
    num_classes = 100
    
    # Get target counts
    target_counts = get_cifar100_lt_counts(imb_factor, num_classes)
    
    # Sample indices for each class
    lt_indices = []
    for cls in range(num_classes):
        cls_indices = np.where(targets == cls)[0]
        num_to_sample = min(target_counts[cls], len(cls_indices))
        
        # Random sample without replacement
        sampled = np.random.choice(cls_indices, num_to_sample, replace=False)
        lt_indices.extend(sampled.tolist())
    
    lt_indices = np.array(lt_indices)
    lt_targets = targets[lt_indices]
    
    print(f"Created CIFAR-100-LT dataset:")
    print(f"  Total samples: {len(lt_indices):,}")
    print(f"  Head class (0): {target_counts[0]} samples")
    print(f"  Tail class (99): {target_counts[99]} samples")
    print(f"  Imbalance factor: {target_counts[0] / target_counts[99]:.1f}")
    
    return Subset(cifar_dataset, lt_indices)

def get_dataloaders():
    """Get train and test dataloaders with paper-compliant transforms."""
    
    # Data transforms (Paper Appendix F.1)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=PAPER_CONFIG['data_augmentation']['train']['normalize']['mean'],
            std=PAPER_CONFIG['data_augmentation']['train']['normalize']['std']
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=PAPER_CONFIG['data_augmentation']['test']['normalize']['mean'],
            std=PAPER_CONFIG['data_augmentation']['test']['normalize']['std']
        )
    ])
    
    # Load CIFAR-100 datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create long-tail versions
    lt_train_dataset = create_longtail_dataset(train_dataset, PAPER_CONFIG['dataset']['imb_factor'])
    
    # Create test set with same long-tail distribution
    lt_test_dataset = create_longtail_dataset(test_dataset, PAPER_CONFIG['dataset']['imb_factor'])
    
    # Create dataloaders
    train_loader = DataLoader(
        lt_train_dataset, 
        batch_size=PAPER_CONFIG['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        lt_test_dataset, 
        batch_size=PAPER_CONFIG['training']['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def test_epoch(model, test_loader, criterion, device):
    """Test for one epoch."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc

def save_checkpoint(model, epoch, train_acc, test_acc, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
    torch.save(checkpoint, filepath)

def export_logits(model, dataloader, device, output_path):
    """Export logits for all samples."""
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Exporting logits"):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    
    torch.save({
        'logits': all_logits,
        'targets': all_targets
    }, output_path)
    
    print(f"Exported logits: {all_logits.shape} to {output_path}")

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function."""
    print("=" * 80)
    print("CE EXPERT TRAINING - PAPER COMPLIANT")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dataset: CIFAR-100-LT (IF={PAPER_CONFIG['dataset']['imb_factor']})")
    print(f"Model: ResNet-32")
    print(f"Epochs: {PAPER_CONFIG['training']['epochs']}")
    print(f"Batch size: {PAPER_CONFIG['training']['batch_size']}")
    print(f"Learning rate: {PAPER_CONFIG['training']['lr']}")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(PAPER_CONFIG['seed'])
    np.random.seed(PAPER_CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(PAPER_CONFIG['seed'])
    
    # Create output directories
    checkpoint_dir = Path(PAPER_CONFIG['output']['checkpoint_dir'])
    logits_dir = Path(PAPER_CONFIG['output']['logits_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading CIFAR-100-LT datasets...")
    train_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating ResNet-32 model...")
    model = resnet32(num_classes=PAPER_CONFIG['dataset']['num_classes']).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=PAPER_CONFIG['training']['lr'],
        momentum=PAPER_CONFIG['training']['momentum'],
        weight_decay=PAPER_CONFIG['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=PAPER_CONFIG['training']['milestones'],
        gamma=PAPER_CONFIG['training']['gamma']
    )
    
    # Training loop
    print(f"\nStarting training for {PAPER_CONFIG['training']['epochs']} epochs...")
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(PAPER_CONFIG['training']['epochs']):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print progress
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{PAPER_CONFIG['training']['epochs']}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%, "
              f"LR={current_lr:.5f}, Time={epoch_time:.1f}s")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_model_path = checkpoint_dir / "best_ce_expert.pth"
            save_checkpoint(model, epoch, train_acc, test_acc, best_model_path)
            print(f"  -> New best! Test Acc={test_acc:.2f}% (Epoch {epoch+1})")
    
    # Final results
    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best test accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
    
    # Load best model
    best_model_path = checkpoint_dir / "best_ce_expert.pth"
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Export logits
    print(f"\nExporting logits...")
    train_logits_path = logits_dir / "ce_expert_train_logits.pth"
    test_logits_path = logits_dir / "ce_expert_test_logits.pth"
    
    export_logits(model, train_loader, DEVICE, train_logits_path)
    export_logits(model, test_loader, DEVICE, test_logits_path)
    
    # Also export for compatibility with existing plugin scripts
    print(f"\nExporting logits for plugin compatibility...")
    
    # Export in the format expected by existing scripts
    plugin_logits_dir = Path('./outputs/logits/cifar100_lt_if100/ce_baseline')
    plugin_logits_dir.mkdir(parents=True, exist_ok=True)
    
    # Export train logits
    train_data = torch.load(train_logits_path, map_location='cpu')
    torch.save(train_data['logits'].to(torch.float16), plugin_logits_dir / "train_logits.pt")
    torch.save(train_data['logits'].to(torch.float16), plugin_logits_dir / "expert_logits.pt")
    
    # Export test logits
    test_data = torch.load(test_logits_path, map_location='cpu')
    torch.save(test_data['logits'].to(torch.float16), plugin_logits_dir / "test_logits.pt")
    
    # Export validation logits (use test for now)
    torch.save(test_data['logits'].to(torch.float16), plugin_logits_dir / "val_logits.pt")
    torch.save(test_data['logits'].to(torch.float16), plugin_logits_dir / "tunev_logits.pt")
    
    print(f"✅ Plugin-compatible logits exported to: {plugin_logits_dir}")
    
    print(f"\n" + "=" * 80)
    print("CE EXPERT TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Best model saved to: {best_model_path}")
    print(f"Train logits saved to: {train_logits_path}")
    print(f"Test logits saved to: {test_logits_path}")
    print(f"Final test accuracy: {best_test_acc:.2f}%")

if __name__ == '__main__':
    main()
