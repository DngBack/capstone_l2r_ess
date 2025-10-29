import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import collections

# Import our custom modules
from src.models.experts import Expert
from src.models.losses import LogitAdjustLoss, BalancedSoftmaxLoss
from src.metrics.calibration import TemperatureScaler
from src.data.dataloader_utils import get_expert_training_dataloaders

# --- EXPERT CONFIGURATIONS ---
EXPERT_CONFIGS = {
    'ce': {
        'name': 'ce_baseline',
        'loss_type': 'ce',
        'epochs': 200,  # Reduced from 256
        'lr': 0.1,
        'weight_decay': 5e-4,  # Increased from 1e-4
        'dropout_rate': 0.0,  # No dropout for baseline
        'milestones': [60, 120, 160],  # Adjusted for 200 epochs
        'gamma': 0.1
    },
    'logitadjust': {
        'name': 'logitadjust_baseline', 
        'loss_type': 'logitadjust',
        'epochs': 256,
        'lr': 0.1,
        'weight_decay': 5e-4,  # Slightly higher regularization for imbalanced data
        'dropout_rate': 0.1,  # Light dropout for imbalanced data
        'milestones': [160,180],
        'gamma': 0.1
    },
    'balsoftmax': {
        'name': 'balsoftmax_baseline',
        'loss_type': 'balsoftmax', 
        'epochs': 256,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'dropout_rate': 0.1,  # Light dropout for imbalanced data
        'milestones': [96, 192, 224],
        'gamma': 0.1
    }
}

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'data_root': './data',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',
        'num_classes': 100,
        'num_groups': 2,
    },
    'train_params': {
        'batch_size': 128,
        'momentum': 0.9,
        'warmup_steps': 10,
    },
    'output': {
        'checkpoints_dir': './checkpoints/experts',
        'logits_dir': './outputs/logits',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---

def get_dataloaders(use_expert_split=True):
    """
    Get train and validation dataloaders.
    
    Args:
        use_expert_split: If True, use expert split (90% of train), else use full train
    """
    print("Loading CIFAR-100-LT datasets...")
    
    if use_expert_split:
        print("  Using EXPERT split (90% of train) for training")
    else:
        print("  Using FULL train split for training")
    
    train_loader, val_loader = get_expert_training_dataloaders(
        batch_size=CONFIG['train_params']['batch_size'],
        num_workers=4,
        use_expert_split=use_expert_split,
        splits_dir=CONFIG['dataset']['splits_dir']
    )
    
    print(f"  Train loader: {len(train_loader)} batches ({len(train_loader.dataset):,} samples)")
    print(f"  Val loader: {len(val_loader)} batches ({len(val_loader.dataset):,} samples)")
    
    return train_loader, val_loader


def get_loss_function(loss_type, train_loader):
    """Create appropriate loss function based on type."""
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    print("Calculating class counts for loss function...")
    
    # Get class counts from dataset
    if hasattr(train_loader.dataset, 'cifar_dataset'):
        train_targets = np.array(train_loader.dataset.cifar_dataset.targets)[train_loader.dataset.indices]
    elif hasattr(train_loader.dataset, 'dataset'):
        train_targets = np.array(train_loader.dataset.dataset.targets)[train_loader.dataset.indices]
    else:
        train_targets = np.array(train_loader.dataset.targets)
    
    class_counts = [count for _, count in sorted(collections.Counter(train_targets).items())]
    
    if loss_type == 'logitadjust':
        return LogitAdjustLoss(class_counts=class_counts)
    elif loss_type == 'balsoftmax':
        return BalancedSoftmaxLoss(class_counts=class_counts)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported.")


def load_class_weights(splits_dir):
    """Load class weights for reweighted validation metrics."""
    weights_path = Path(splits_dir) / 'class_weights.json'
    
    if not weights_path.exists():
        print(f"Warning: {weights_path} not found, using uniform weights")
        return np.ones(100) / 100
    
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(weights_data, list):
        weights = np.array(weights_data)
    elif isinstance(weights_data, dict):
        weights = np.array([weights_data[str(i)] for i in range(100)])
    else:
        raise ValueError(f"Unexpected format for class weights: {type(weights_data)}")
    
    return weights

def validate_model(model, val_loader, device, class_weights=None):
    """
    Validate model with reweighted metrics.
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader (balanced val split)
        device: Device to use
        class_weights: Class weights for reweighting (from training distribution)
    
    Returns:
        overall_acc: Overall accuracy (unweighted, on balanced val)
        reweighted_acc: Reweighted accuracy (simulates long-tail performance)
        group_accs: Group-wise accuracies
    """
    model.eval()
    correct = 0
    total = 0
    
    # For reweighted accuracy
    class_correct = np.zeros(100)
    class_total = np.zeros(100)
    
    group_correct = {'head': 0, 'tail': 0}
    group_total = {'head': 0, 'tail': 0}
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy for reweighting
            for i, target in enumerate(targets):
                target_class = target.item()
                pred = predicted[i].item()
                
                class_total[target_class] += 1
                if pred == target_class:
                    class_correct[target_class] += 1
                
                # Group-wise accuracy (Head: 0-49, Tail: 50-99)
                if target_class < 50:  # Head classes
                    group_total['head'] += 1
                    if pred == target_class:
                        group_correct['head'] += 1
                else:  # Tail classes
                    group_total['tail'] += 1
                    if pred == target_class:
                        group_correct['tail'] += 1
    
    # Standard accuracy (on balanced val set)
    overall_acc = 100 * correct / total
    
    # Reweighted accuracy (simulates long-tail performance)
    if class_weights is not None:
        # Per-class accuracy
        class_acc = np.zeros(100)
        for i in range(100):
            if class_total[i] > 0:
                class_acc[i] = class_correct[i] / class_total[i]
            else:
                class_acc[i] = 0.0
        
        # Reweighted accuracy using training distribution weights
        reweighted_acc = 100 * np.sum(class_acc * class_weights)
    else:
        reweighted_acc = overall_acc
    
    # Group-wise accuracies
    group_accs = {}
    for group in ['head', 'tail']:
        if group_total[group] > 0:
            group_accs[group] = 100 * group_correct[group] / group_total[group]
        else:
            group_accs[group] = 0.0
    
    return overall_acc, reweighted_acc, group_accs

def export_logits_for_all_splits(model, expert_name):
    """Export logits for all dataset splits."""
    print(f"Exporting logits for expert '{expert_name}'...")
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    output_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name'] / expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define splits to export
    splits_info = [
        # From training set (CIFAR-100 train)
        {'name': 'train', 'dataset_type': 'train', 'file': 'train_indices.json'},
        {'name': 'expert', 'dataset_type': 'train', 'file': 'expert_indices.json'},
        {'name': 'gating', 'dataset_type': 'train', 'file': 'gating_indices.json'},
        # From test set (CIFAR-100 test) - balanced splits
        {'name': 'val', 'dataset_type': 'test', 'file': 'val_indices.json'},
        {'name': 'test', 'dataset_type': 'test', 'file': 'test_indices.json'},
        {'name': 'tunev', 'dataset_type': 'test', 'file': 'tunev_indices.json'},
    ]
    
    for split_info in splits_info:
        split_name = split_info['name']
        dataset_type = split_info['dataset_type']
        indices_file = split_info['file']
        indices_path = splits_dir / indices_file
        
        if not indices_path.exists():
            print(f"  Warning: {indices_file} not found, skipping {split_name}")
            continue
            
        # Load appropriate base dataset
        if dataset_type == 'train':
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=True, transform=transform)
        else:
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=False, transform=transform)
        
        # Load indices and create subset
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        subset = Subset(base_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=4)
        
        # Export logits
        all_logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Exporting {split_name}"):
                logits = model.get_calibrated_logits(inputs.to(DEVICE))
                all_logits.append(logits.cpu())
        
        all_logits = torch.cat(all_logits)
        torch.save(all_logits.to(torch.float16), output_dir / f"{split_name}_logits.pt")
        print(f"  Exported {split_name}: {len(indices):,} samples")
    
    print(f"[SUCCESS] All logits exported to: {output_dir}")

# --- CORE TRAINING FUNCTIONS ---

def train_single_expert(expert_key, use_expert_split=True):
    """
    Train a single expert based on its configuration.
    
    Args:
        expert_key: Key identifying the expert ('ce', 'logitadjust', 'balsoftmax')
        use_expert_split: If True, use expert split (90% of train), else use full train
    """
    if expert_key not in EXPERT_CONFIGS:
        raise ValueError(f"Expert '{expert_key}' not found in EXPERT_CONFIGS")
    
    expert_config = EXPERT_CONFIGS[expert_key]
    expert_name = expert_config['name']
    loss_type = expert_config['loss_type']
    
    print(f"\n{'='*60}")
    print(f"[EXPERT] TRAINING EXPERT: {expert_name.upper()}")
    print(f"[EXPERT] Loss Type: {loss_type.upper()}")
    print(f"[EXPERT] Splits Dir: {CONFIG['dataset']['splits_dir']}")
    print(f"{'='*60}")
    
    # Setup
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    train_loader, val_loader = get_dataloaders(use_expert_split=use_expert_split)
    
    # Model and loss
    model = Expert(
        num_classes=CONFIG['dataset']['num_classes'],
        backbone_name='cifar_resnet32',
        dropout_rate=expert_config['dropout_rate'],
        init_weights=True
    ).to(DEVICE)
    
    criterion = get_loss_function(loss_type, train_loader)
    print(f"[SUCCESS] Loss Function: {type(criterion).__name__}")
    
    # Print model summary
    print("[INFO] Model Architecture:")
    model.summary()
    
    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        lr=expert_config['lr'],
        momentum=CONFIG['train_params']['momentum'],
        weight_decay=expert_config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=expert_config['milestones'], 
        gamma=expert_config['gamma']
    )
    
    # Load class weights for reweighted validation
    class_weights = load_class_weights(CONFIG['dataset']['splits_dir'])
    print(f"[SUCCESS] Loaded class weights for reweighted validation")
    
    # Training setup
    best_acc = 0.0
    best_reweighted_acc = 0.0
    checkpoint_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"best_{expert_name}.pth"
    
    # Training loop
    for epoch in range(expert_config['epochs']):
        # Train
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{expert_config['epochs']}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # Validate with reweighting
        val_acc, reweighted_acc, group_accs = validate_model(
            model, val_loader, DEVICE, class_weights=class_weights
        )
        
        print(f"Epoch {epoch+1:3d}: Loss={running_loss/len(train_loader):.4f}, "
              f"Val Acc={val_acc:.2f}% (Reweighted={reweighted_acc:.2f}%), "
              f"Head={group_accs['head']:.1f}%, Tail={group_accs['tail']:.1f}%, "
              f"LR={scheduler.get_last_lr()[0]:.5f}")
        
        # Save best model based on reweighted accuracy (better for long-tail)
        if reweighted_acc > best_reweighted_acc:
            best_reweighted_acc = reweighted_acc
            best_acc = val_acc
            print(f"💾 New best! Reweighted={reweighted_acc:.2f}% → Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
    
    # Post-training: Calibration
    print(f"\n--- 🔧 POST-PROCESSING: {expert_name} ---")
    model.load_state_dict(torch.load(best_model_path))
    
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(model, val_loader, DEVICE)
    model.set_temperature(optimal_temp)
    print(f"[SUCCESS] Temperature calibration: T = {optimal_temp:.3f}")
    
    final_model_path = checkpoint_dir / f"final_calibrated_{expert_name}.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Final validation with reweighting
    final_acc, final_reweighted_acc, final_group_accs = validate_model(
        model, val_loader, DEVICE, class_weights=class_weights
    )
    print(f"📊 Final Results:")
    print(f"   Overall Acc: {final_acc:.2f}% (on balanced val)")
    print(f"   Reweighted Acc: {final_reweighted_acc:.2f}% (simulates long-tail)")
    print(f"   Head: {final_group_accs['head']:.1f}%, Tail: {final_group_accs['tail']:.1f}%")
    
    # Export logits
    export_logits_for_all_splits(model, expert_name)
    
    print(f"[SUCCESS] COMPLETED: {expert_name}")
    return final_model_path


def main():
    """Main training script - trains all 3 experts."""
    print("[EXPERT] AR-GSE Expert Training Pipeline")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {CONFIG['dataset']['name']}")
    print(f"Experts to train: {list(EXPERT_CONFIGS.keys())}")
    
    results = {}
    
    for expert_key in EXPERT_CONFIGS.keys():
        try:
            model_path = train_single_expert(expert_key)
            results[expert_key] = {'status': 'success', 'path': model_path}
        except Exception as e:
            print(f"[ERROR] Failed to train {expert_key}: {e}")
            results[expert_key] = {'status': 'failed', 'error': str(e)}
            continue
    
    print(f"\n{'='*60}")
    print("[SUMMARY] TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for expert_key, result in results.items():
        status = "[SUCCESS]" if result['status'] == 'success' else "[ERROR]"
        print(f"{status} {expert_key}: {result['status']}")
        if result['status'] == 'failed':
            print(f"    Error: {result['error']}")
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"\nSuccessfully trained {successful}/{len(EXPERT_CONFIGS)} experts")


if __name__ == '__main__':
    main()