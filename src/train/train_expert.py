import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import math
from pathlib import Path
from tqdm import tqdm
import collections

# Import our custom modules
from src.models.experts import Expert
from src.models.losses import LogitAdjustLoss, BalancedSoftmaxLoss
from src.metrics.calibration import TemperatureScaler
from src.data.dataloader_utils import get_expert_training_dataloaders

# Import paper-compliant metrics (optional, for extended evaluation)
# These functions are available from src.train.train_utils:
# - evaluate_metrics: Compute balanced/worst-group error (paper-compliant)
# - compute_ece: Compute Expected Calibration Error
# - get_probs_and_labels: Get probabilities and labels from model
# Import when needed: from src.train.train_utils import evaluate_metrics, compute_ece

# --- EXPERT CONFIGURATIONS ---
EXPERT_CONFIGS = {
    "ce": {
        "name": "ce_baseline",
        "loss_type": "ce",
        "epochs": 256,  # Paper: 256 epochs for CIFAR
        "lr": 0.4,  # Paper: lr=0.4 (Table 3 + F.1)
        "weight_decay": 1e-4,  # Paper: weight_decay=1e-4
        "dropout_rate": 0.0,  # No dropout for baseline
        "milestones": [96, 192, 224],  # Paper: decay at epochs [96, 192, 224]
        "gamma": 0.1,
        "warmup_epochs": 15,  # Paper: 15 epochs warmup (NOT iterations!)
        "use_manual_lr": True,  # Use manual LR scheduling for paper compliance
        "use_cosine": False,  # Use manual step decay
    },
    "logitadjust": {
        "name": "logitadjust_baseline",
        "loss_type": "logitadjust",
        "epochs": 256,
        "lr": 0.1,
        "weight_decay": 5e-4,  # Slightly higher regularization for imbalanced data
        "dropout_rate": 0.1,  # Light dropout for imbalanced data
        "milestones": [160, 180],
        "gamma": 0.1,
        "use_cosine": False,
    },
    "balsoftmax": {
        "name": "balsoftmax_baseline",
        "loss_type": "balsoftmax",
        "epochs": 256,
        "lr": 0.1,
        "weight_decay": 5e-4,
        "dropout_rate": 0.1,  # Light dropout for imbalanced data
        "milestones": [96, 192, 224],
        "gamma": 0.1,
        "use_cosine": False,
    },
}

# iNaturalist and ImageNet-LT specific configs (cosine scheduler with step decay, 200 epochs)
# Paper: For ImageNet and iNaturalist, warmup 5 epochs to reach base LR 0.4,
#        apply learning rate decay 0.1 at epochs 45, 100, 150, and use cosine annealing
EXPERT_CONFIGS_INATURALIST = {
    "ce": {
        "name": "ce_baseline",
        "loss_type": "ce",
        "epochs": 200,  # Paper: 200 epochs for iNaturalist and ImageNet-LT
        "lr": 0.4,  # Paper: base LR 0.4
        "weight_decay": 1e-4,  # Paper: weight decay 10^-4
        "dropout_rate": 0.0,
        "milestones": [45, 100, 150],  # Paper: decay 0.1 at epochs 45, 100, 150
        "gamma": 0.1,  # Paper: decay factor 0.1
        "warmup_epochs": 5,  # Paper: 5 epochs warmup to reach base LR 0.4
        "use_manual_lr": True,
        "use_cosine": True,  # Paper: use cosine annealing scheduler
    },
    "logitadjust": {
        "name": "logitadjust_baseline",
        "loss_type": "logitadjust",
        "epochs": 200,
        "lr": 0.4,
        "weight_decay": 1e-4,
        "dropout_rate": 0.0,
        "milestones": [45, 100, 150],  # Paper: decay 0.1 at epochs 45, 100, 150
        "gamma": 0.1,
        "warmup_epochs": 5,  # Paper: 5 epochs warmup
        "use_manual_lr": True,
        "use_cosine": True,  # Use cosine annealing
    },
    "balsoftmax": {
        "name": "balsoftmax_baseline",
        "loss_type": "balsoftmax",
        "epochs": 200,
        "lr": 0.4,
        "weight_decay": 1e-4,
        "dropout_rate": 0.0,
        "milestones": [45, 100, 150],  # Paper: decay 0.1 at epochs 45, 100, 150
        "gamma": 0.1,
        "warmup_epochs": 5,  # Paper: 5 epochs warmup
        "use_manual_lr": True,
        "use_cosine": True,  # Use cosine annealing
    },
}

# --- DATASET CONFIGURATIONS ---
DATASET_CONFIGS = {
    "cifar100_lt_if100": {
        "name": "cifar100_lt_if100",
        "data_root": "./data",
        "splits_dir": "./data/cifar100_lt_if100_splits_fixed",
        "num_classes": 100,
        "num_groups": 2,
        "backbone": "cifar_resnet32",
        "batch_size": 128,
        "epochs": 256,
    },
    "inaturalist2018": {
        "name": "inaturalist2018",
        "data_root": "./data",
        "splits_dir": "./data/inaturalist2018_splits",
        "train_json": "./data/train2018.json",
        "val_json": "./data/val2018.json",
        "num_classes": 8142,
        "num_groups": 2,
        "backbone": "resnet50",
        "batch_size": 1024,
        "epochs": 200,
    },
    "imagenet_lt": {
        "name": "imagenet_lt",
        "data_root": "./data",
        "splits_dir": "./data/imagenet_lt_splits",
        "data_dir": "./data/imagenet_lt",
        "train_label_file": "ImageNet_LT_train.txt",
        "val_label_file": "ImageNet_LT_test.txt",
        "num_classes": 1000,
        "num_groups": 2,
        "backbone": "resnet50",
        "batch_size": 512,
        "epochs": 200,
    }
}

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    "dataset": {
        "name": "cifar100_lt_if100",
        "data_root": "./data",
        "splits_dir": "./data/cifar100_lt_if100_splits_fixed",
        "num_classes": 100,
        "num_groups": 2,
    },
    "train_params": {
        "batch_size": 128,
        "momentum": 0.9,
        "warmup_steps": 10,
    },
    "output": {
        "checkpoints_dir": "./checkpoints/experts",
        "logits_dir": "./outputs/logits",
    },
    "seed": 42,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- HELPER FUNCTIONS ---


def get_dataloaders(use_expert_split=True, dataset_name=None, override_batch_size=None):
    """
    Get train and validation dataloaders.

    Args:
        use_expert_split: If True, use expert split (90% of train), else use full train
        dataset_name: Name of dataset ('cifar100_lt_if100' or 'inaturalist2018')
        override_batch_size: Override batch size from command line
    """
    # Use CONFIG dataset if not specified
    if dataset_name is None:
        dataset_name = CONFIG["dataset"]["name"]
    
    if dataset_name == "cifar100_lt_if100":
        print("Loading CIFAR-100-LT datasets...")

        if use_expert_split:
            print("  Using EXPERT split (90% of train) for training")
        else:
            print("  Using FULL train split for training")

        batch_size = override_batch_size if override_batch_size is not None else CONFIG["train_params"]["batch_size"]
        train_loader, val_loader = get_expert_training_dataloaders(
            batch_size=batch_size,
            num_workers=4,
            use_expert_split=use_expert_split,
            splits_dir=CONFIG["dataset"]["splits_dir"],
        )

        print(
            f"  Train loader: {len(train_loader)} batches ({len(train_loader.dataset):,} samples)"
        )
        print(
            f"  Val loader: {len(val_loader)} batches ({len(val_loader.dataset):,} samples)"
        )

        return train_loader, val_loader
    
    elif dataset_name == "inaturalist2018":
        print("Loading iNaturalist 2018 datasets...")
        
        # Import iNaturalist utilities
        from src.data.inaturalist2018_splits import (
            INaturalistDataset, 
            get_inaturalist_transforms
        )
        from torch.utils.data import DataLoader
        
        # Get dataset config
        ds_config = DATASET_CONFIGS["inaturalist2018"]
        
        # Load train and val datasets with indices from splits
        train_transform, eval_transform = get_inaturalist_transforms()
        
        # Load the full datasets
        full_train_dataset = INaturalistDataset(
            ds_config["data_root"], 
            ds_config["train_json"],
            transform=None  # Will apply transform in DataLoader
        )
        full_val_dataset = INaturalistDataset(
            ds_config["data_root"],
            ds_config["val_json"],
            transform=None
        )
        
        # Load split indices
        splits_dir = Path(ds_config["splits_dir"])
        if use_expert_split:
            with open(splits_dir / "expert_indices.json", "r") as f:
                train_indices = json.load(f)
            print("  Using EXPERT split (90% of train) for training")
        else:
            with open(splits_dir / "train_class_counts.json", "r") as f:
                train_class_counts = json.load(f)
            # If no train_indices.json, create full indices
            train_indices = list(range(len(full_train_dataset)))
            print("  Using FULL train split for training")
        
        with open(splits_dir / "val_indices.json", "r") as f:
            val_indices = json.load(f)
        
        # For iNaturalist, we need to adapt the approach
        # Create a simple wrapper that applies transforms
        class INaturalistSubset:
            def __init__(self, base_dataset, indices, transform=None):
                self.base_dataset = base_dataset
                self.indices = indices
                self.transform = transform
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                actual_idx = self.indices[idx]
                img_path, label = self.base_dataset.samples[actual_idx]
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
        
        train_dataset = INaturalistSubset(full_train_dataset, train_indices, train_transform)
        val_dataset = INaturalistSubset(full_val_dataset, val_indices, eval_transform)
        
        # Create dataloaders - use override if provided, else use ds_config
        batch_size = override_batch_size if override_batch_size is not None else ds_config["batch_size"]
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        )
        
        print(
            f"  Train loader: {len(train_loader)} batches ({len(train_dataset):,} samples)"
        )
        print(
            f"  Val loader: {len(val_loader)} batches ({len(val_dataset):,} samples)"
        )
        
        return train_loader, val_loader
    
    elif dataset_name == "imagenet_lt":
        print("Loading ImageNet-LT datasets...")
        
        # Import ImageNet-LT utilities
        from src.data.imagenet_lt_splits import (
            ImageNetLTDataset, 
            get_imagenet_lt_transforms
        )
        from torch.utils.data import DataLoader
        
        # Get dataset config
        ds_config = DATASET_CONFIGS["imagenet_lt"]
        
        # Load train and val datasets with indices from splits
        train_transform, eval_transform = get_imagenet_lt_transforms()
        
        # Load the full datasets
        full_train_dataset = ImageNetLTDataset(
            ds_config["data_dir"], 
            ds_config["train_label_file"],
            transform=None  # Will apply transform in DataLoader
        )
        full_val_dataset = ImageNetLTDataset(
            ds_config["data_dir"],
            ds_config["val_label_file"],
            transform=None
        )
        
        print(f"  Loaded train dataset: {len(full_train_dataset)} samples")
        print(f"  Loaded val dataset: {len(full_val_dataset)} samples")
        
        # Load split indices
        splits_dir = Path(ds_config["splits_dir"])
        if use_expert_split:
            # Use expert split (90% of train set)
            with open(splits_dir / "expert_indices.json", "r") as f:
                train_indices = json.load(f)
            print(f"  Using EXPERT split (90% of train): {len(train_indices):,} samples")
        else:
            # Use full train set (100% of train set)
            # Option 1: Load all indices from expert + gating if available
            expert_indices_path = splits_dir / "expert_indices.json"
            gating_indices_path = splits_dir / "gating_indices.json"
            
            if expert_indices_path.exists() and gating_indices_path.exists():
                # Combine expert + gating to get full train
                with open(expert_indices_path, "r") as f:
                    expert_indices = json.load(f)
                with open(gating_indices_path, "r") as f:
                    gating_indices = json.load(f)
                train_indices = expert_indices + gating_indices
                print(f"  Using FULL train split (expert + gating): {len(train_indices):,} samples")
                print(f"    Expert: {len(expert_indices):,} samples")
                print(f"    Gating: {len(gating_indices):,} samples")
            else:
                # Fallback: use all indices from dataset
                train_indices = list(range(len(full_train_dataset)))
                print(f"  Using FULL train split (all samples): {len(train_indices):,} samples")
        
        with open(splits_dir / "val_indices.json", "r") as f:
            val_indices = json.load(f)
        print(f"  Using VAL split: {len(val_indices)} indices")
        
        # Create a simple wrapper that applies transforms
        class ImageNetLTSubset:
            def __init__(self, base_dataset, indices, transform=None):
                self.base_dataset = base_dataset
                # Validate and filter indices
                max_idx = len(base_dataset.samples) - 1
                valid_indices = [i for i in indices if 0 <= i <= max_idx]
                if len(valid_indices) != len(indices):
                    invalid_count = len(indices) - len(valid_indices)
                    print(f"  Warning: {invalid_count} invalid indices filtered out (max valid index: {max_idx})")
                self.indices = valid_indices
                self.transform = transform
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                if idx >= len(self.indices):
                    raise IndexError(f"Index {idx} out of range for dataset of size {len(self.indices)}")
                actual_idx = self.indices[idx]
                if actual_idx >= len(self.base_dataset.samples):
                    raise IndexError(f"Actual index {actual_idx} out of range for base dataset of size {len(self.base_dataset.samples)}")
                img_path, label = self.base_dataset.samples[actual_idx]
                from PIL import Image
                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
                    image = Image.new('RGB', (224, 224), color='black')
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
        
        train_dataset = ImageNetLTSubset(full_train_dataset, train_indices, train_transform)
        val_dataset = ImageNetLTSubset(full_val_dataset, val_indices, eval_transform)
        
        # Create dataloaders - use override if provided, else use ds_config
        batch_size = override_batch_size if override_batch_size is not None else ds_config["batch_size"]
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        )
        
        print(
            f"  Train loader: {len(train_loader)} batches ({len(train_dataset):,} samples)"
        )
        print(
            f"  Val loader: {len(val_loader)} batches ({len(val_dataset):,} samples)"
        )
        
        return train_loader, val_loader
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_loss_function(loss_type, train_loader):
    """Create appropriate loss function based on type."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    print("Calculating class counts for loss function...")

    # Get class counts from dataset
    # Try different dataset structures for compatibility
    if hasattr(train_loader.dataset, "cifar_dataset"):
        train_targets = np.array(train_loader.dataset.cifar_dataset.targets)[
            train_loader.dataset.indices
        ]
    elif hasattr(train_loader.dataset, "dataset"):
        train_targets = np.array(train_loader.dataset.dataset.targets)[
            train_loader.dataset.indices
        ]
    elif hasattr(train_loader.dataset, "base_dataset"):
        # For INaturalistSubset or ImageNetLTSubset
        train_targets = []
        for idx in train_loader.dataset.indices:
            _, label = train_loader.dataset.base_dataset.samples[idx]
            train_targets.append(label)
        train_targets = np.array(train_targets)
    else:
        train_targets = np.array(train_loader.dataset.targets)

    class_counts = [
        count for _, count in sorted(collections.Counter(train_targets).items())
    ]

    if loss_type == "logitadjust":
        return LogitAdjustLoss(class_counts=class_counts)
    elif loss_type == "balsoftmax":
        return BalancedSoftmaxLoss(class_counts=class_counts)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported.")


def load_class_weights(splits_dir):
    """Load class weights for reweighted validation metrics."""
    weights_path = Path(splits_dir) / "class_weights.json"
    
    # Get number of classes from CONFIG
    num_classes = CONFIG["dataset"]["num_classes"]

    if not weights_path.exists():
        print(f"Warning: {weights_path} not found, using uniform weights")
        return np.ones(num_classes) / num_classes

    with open(weights_path, "r") as f:
        weights_data = json.load(f)

    # Handle both list and dict formats
    if isinstance(weights_data, list):
        weights = np.array(weights_data)
    elif isinstance(weights_data, dict):
        weights = np.array([weights_data[str(i)] for i in range(num_classes)])
    else:
        raise ValueError(f"Unexpected format for class weights: {type(weights_data)}")

    return weights


def load_class_to_group(splits_dir, threshold=20):
    """
    Load class-to-group mapping for head/tail classification.
    
    IMPORTANT: Head/tail classification is ALWAYS based on TRAIN counts, not on val/test counts.
    This ensures consistency: a class that is head on train is head on val/test/tunev as well.
    
    Args:
        splits_dir: Directory containing splits and class counts
        threshold: Threshold for head/tail classification (default: 20 samples)
    
    Returns:
        class_to_group: Array mapping class -> group (0=head, 1=tail), based on TRAIN counts
    """
    splits_path = Path(splits_dir)
    
    # Get number of classes from CONFIG
    num_classes = CONFIG["dataset"]["num_classes"]
    
    # Try to load pre-computed class_to_group.json first (if available)
    group_path = splits_path / "class_to_group.json"
    if group_path.exists():
        print(f"Loading class_to_group from: {group_path}")
        with open(group_path, "r") as f:
            class_to_group = json.load(f)
        
        # Convert to numpy array
        if isinstance(class_to_group, list):
            class_to_group = np.array(class_to_group, dtype=np.int64)
        else:
            # If it's a dict, convert to list
            class_to_group = np.array([class_to_group.get(str(i), 0) for i in range(num_classes)], dtype=np.int64)
        
        # Verify it has correct length
        if len(class_to_group) != num_classes:
            print(f"Warning: class_to_group has {len(class_to_group)} classes, expected {num_classes}")
            print("  Recomputing from train_class_counts.json...")
        else:
            num_head = (class_to_group == 0).sum()
            num_tail = (class_to_group == 1).sum()
            print(f"  Loaded: {num_head} head classes, {num_tail} tail classes (based on train counts)")
            return class_to_group
    
    # Fallback: compute from train_class_counts.json
    counts_path = splits_path / "train_class_counts.json"
    
    if not counts_path.exists():
        print(f"Warning: {counts_path} not found, using midpoint split")
        # Default: split at midpoint
        class_to_group = np.zeros(num_classes, dtype=np.int64)
        class_to_group[num_classes // 2:] = 1
        return class_to_group
    
    print(f"Loading train_class_counts from: {counts_path}")
    print(f"  Computing class_to_group based on train counts (threshold = {threshold})")
    with open(counts_path, "r") as f:
        class_counts = json.load(f)
    
    if isinstance(class_counts, dict):
        class_counts = [class_counts.get(str(i), 0) for i in range(num_classes)]
    
    counts = np.array(class_counts)
    tail_mask = counts <= threshold
    class_to_group = np.zeros(num_classes, dtype=np.int64)
    class_to_group[tail_mask] = 1  # 0=head, 1=tail
    
    num_head = (class_to_group == 0).sum()
    num_tail = (class_to_group == 1).sum()
    print(f"  Computed: {num_head} head classes, {num_tail} tail classes (based on train counts)")
    print(f"  IMPORTANT: This classification applies to ALL splits (val/test/tunev)")
    
    return class_to_group


def validate_model(model, val_loader, device, class_weights=None, class_to_group=None):
    """
    Validate model without reweighting.

    Args:
        model: Model to validate
        val_loader: Validation dataloader (balanced val split)
        device: Device to use
        class_weights: Ignored (kept for compatibility, but not used)
        class_to_group: Array mapping class -> group (0=head, 1=tail)

    Returns:
        overall_acc: Overall accuracy (unweighted, on balanced val)
        reweighted_acc: Same as overall_acc (no reweighting)
        group_accs: Group-wise accuracies
    """
    model.eval()
    correct = 0
    total = 0

    group_correct = {"head": 0, "tail": 0}
    group_total = {"head": 0, "tail": 0}
    
    # Load class_to_group if not provided
    if class_to_group is None:
        class_to_group = load_class_to_group(CONFIG["dataset"]["splits_dir"])

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Group-wise accuracy using class_to_group mapping
            for i, target in enumerate(targets):
                target_class = target.item()
                pred = predicted[i].item()

                group_id = class_to_group[target_class]
                group_name = "head" if group_id == 0 else "tail"
                group_total[group_name] += 1
                if pred == target_class:
                    group_correct[group_name] += 1

    # Standard accuracy (on balanced val set)
    overall_acc = 100 * correct / total
    reweighted_acc = overall_acc  # No reweighting, same as overall

    # Group-wise accuracies
    group_accs = {}
    for group in ["head", "tail"]:
        if group_total[group] > 0:
            group_accs[group] = 100 * group_correct[group] / group_total[group]
        else:
            group_accs[group] = 0.0

    return overall_acc, reweighted_acc, group_accs


def export_logits_for_all_splits(model, expert_name):
    """Export logits for all dataset splits."""
    print(f"Exporting logits for expert '{expert_name}'...")
    model.eval()

    dataset_name = CONFIG["dataset"]["name"]
    splits_dir = Path(CONFIG["dataset"]["splits_dir"])
    output_dir = (
        Path(CONFIG["output"]["logits_dir"]) / CONFIG["dataset"]["name"] / expert_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define splits to export
    splits_info = [
        {"name": "train", "file": "train_indices.json"},
        {"name": "expert", "file": "expert_indices.json"},
        {"name": "gating", "file": "gating_indices.json"},
        {"name": "val", "file": "val_indices.json"},
        {"name": "test", "file": "test_indices.json"},
        {"name": "tunev", "file": "tunev_indices.json"},
    ]

    for split_info in splits_info:
        split_name = split_info["name"]
        indices_file = split_info["file"]
        indices_path = splits_dir / indices_file

        if not indices_path.exists():
            print(f"  Warning: {indices_file} not found, skipping {split_name}")
            continue

        # Load indices
        with open(indices_path, "r") as f:
            indices = json.load(f)
        
        # Create dataset based on type
        if dataset_name == "cifar100_lt_if100":
            # Use CIFAR transforms
            from src.data.enhanced_datasets import get_cifar100_transforms
            _, eval_transform = get_cifar100_transforms()

            # Load appropriate base dataset
            if split_name in ["train", "expert", "gating"]:
                base_dataset = torchvision.datasets.CIFAR100(
                    root=CONFIG["dataset"]["data_root"], train=True, transform=None
                )
            else:
                base_dataset = torchvision.datasets.CIFAR100(
                    root=CONFIG["dataset"]["data_root"], train=False, transform=None
                )
            
            from src.data.enhanced_datasets import CIFAR100LTDataset
            dataset = CIFAR100LTDataset(base_dataset, indices, eval_transform)
            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
            
        elif dataset_name == "inaturalist2018":
            # Use iNaturalist transforms
            from src.data.inaturalist2018_splits import (
                INaturalistDataset,
                get_inaturalist_transforms
            )
            _, eval_transform = get_inaturalist_transforms()
            
            # Determine which JSON file to use
            if split_name in ["train", "expert", "gating"]:
                json_file = CONFIG["dataset"]["train_json"]
            else:
                json_file = CONFIG["dataset"]["val_json"]
            
            # Create full dataset first
            full_dataset = INaturalistDataset(
                CONFIG["dataset"]["data_root"],
                json_file,
                transform=None
            )
            
            # Create subset wrapper
            class INaturalistSubset:
                def __init__(self, base_dataset, indices, transform=None):
                    self.base_dataset = base_dataset
                    self.indices = indices
                    self.transform = transform
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    actual_idx = self.indices[idx]
                    img_path, label = self.base_dataset.samples[actual_idx]
                    from PIL import Image
                    image = Image.open(img_path).convert('RGB')
                    
                    if self.transform:
                        image = self.transform(image)
                        
                    return image, label
            
            dataset = INaturalistSubset(full_dataset, indices, eval_transform)
            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)
            
        elif dataset_name == "imagenet_lt":
            # Use ImageNet-LT transforms
            from src.data.imagenet_lt_splits import (
                ImageNetLTDataset,
                get_imagenet_lt_transforms
            )
            _, eval_transform = get_imagenet_lt_transforms()
            
            # Get dataset config
            ds_config = DATASET_CONFIGS["imagenet_lt"]
            
            # Determine which label file to use
            if split_name in ["train", "expert", "gating"]:
                label_file = ds_config["train_label_file"]
            else:
                label_file = ds_config["val_label_file"]
            
            # Create full dataset first
            full_dataset = ImageNetLTDataset(
                ds_config["data_dir"],
                label_file,
                transform=None
            )
            
            # Create subset wrapper (same as in get_dataloaders)
            class ImageNetLTSubset:
                def __init__(self, base_dataset, indices, transform=None):
                    self.base_dataset = base_dataset
                    # Validate and filter indices
                    max_idx = len(base_dataset.samples) - 1
                    valid_indices = [i for i in indices if 0 <= i <= max_idx]
                    if len(valid_indices) != len(indices):
                        invalid_count = len(indices) - len(valid_indices)
                        print(f"    Warning: {invalid_count} invalid indices filtered out (max valid index: {max_idx})")
                    self.indices = valid_indices
                    self.transform = transform
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    if idx >= len(self.indices):
                        raise IndexError(f"Index {idx} out of range for dataset of size {len(self.indices)}")
                    actual_idx = self.indices[idx]
                    if actual_idx >= len(self.base_dataset.samples):
                        raise IndexError(f"Actual index {actual_idx} out of range for base dataset of size {len(self.base_dataset.samples)}")
                    img_path, label = self.base_dataset.samples[actual_idx]
                    from PIL import Image
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        print(f"    Warning: Failed to load {img_path}: {e}")
                        image = Image.new('RGB', (224, 224), color='black')
                    
                    if self.transform:
                        image = self.transform(image)
                        
                    return image, label
            
            dataset = ImageNetLTSubset(full_dataset, indices, eval_transform)
            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)
            
        else:
            raise ValueError(f"Unsupported dataset for export: {dataset_name}")

        # Export logits
        print(f"  Dataset size: {len(dataset):,} samples (after filtering invalid indices)")
        print(f"  Original indices: {len(indices):,} samples")
        if len(dataset) != len(indices):
            print(f"  ⚠️  WARNING: Dataset size ({len(dataset)}) != indices size ({len(indices)})")
        
        all_logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Exporting {split_name}"):
                logits = model.get_calibrated_logits(inputs.to(DEVICE))
                all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits)
        actual_exported = all_logits.shape[0]
        torch.save(all_logits.to(torch.float16), output_dir / f"{split_name}_logits.pt")
        print(f"  Exported {split_name}: {actual_exported:,} logits (dataset had {len(dataset):,} samples)")
        
        if actual_exported != len(dataset):
            print(f"  ⚠️  WARNING: Exported {actual_exported} logits but dataset had {len(dataset)} samples!")

    print(f"[SUCCESS] All logits exported to: {output_dir}")


# --- CORE TRAINING FUNCTIONS ---


def get_manual_lr(
    epoch, iteration, total_iters_per_epoch, base_lr, warmup_epochs=None, warmup_steps=None, milestones=None, gamma=0.1, use_cosine=False, total_epochs=None
):
    """
    Manual LR scheduling theo paper specifications.

    Supports two warmup modes:
    - warmup_epochs: Warmup over N epochs (for CIFAR: 15 epochs, ImageNet/iNaturalist: 5 epochs)
    - warmup_steps: Warmup over N iterations/steps (alternative option, rarely used)

    Args:
        epoch: Current epoch (0-indexed)
        iteration: Current iteration within epoch (0-indexed)
        total_iters_per_epoch: Total iterations per epoch
        base_lr: Base learning rate (0.4 for CE expert)
        warmup_epochs: Number of warmup epochs (for CIFAR: 15, ImageNet/iNaturalist: 5)
        warmup_steps: Number of warmup steps/iterations (alternative, rarely used)
        milestones: List of epochs for decay [96, 192, 224] or [45, 100, 150]
        gamma: Decay factor (0.1)
        use_cosine: If True, use cosine annealing after warmup
        total_epochs: Total number of epochs (required for cosine)

    Returns:
        Current learning rate
    """
    milestones = milestones or []
    current_iteration = epoch * total_iters_per_epoch + iteration
    
    # Warmup: linear warmup to reach base_lr
    # Priority: warmup_epochs > warmup_steps
    if warmup_epochs is not None:
        # Warmup over N epochs - for CIFAR (15 epochs) or ImageNet/iNaturalist (5 epochs)
        if epoch < warmup_epochs:
            current_progress = (epoch * total_iters_per_epoch + iteration) / (
                warmup_epochs * total_iters_per_epoch
            )
            return base_lr * current_progress
    elif warmup_steps is not None:
        # Warmup over N steps (iterations) - alternative option, rarely used
        if current_iteration < warmup_steps:
            current_progress = current_iteration / warmup_steps
            return base_lr * current_progress

    # Post-warmup: apply step decay first (if milestones exist), then cosine annealing
    # Step 1: Apply step decay to get current base LR
    current_base_lr = base_lr
    if len(milestones) > 0:
        # Calculate decay factor based on how many milestones we've passed
        decay_count = sum(1 for m in milestones if epoch >= m)
        current_base_lr = base_lr * (gamma**decay_count)
    
    # Step 2: Apply cosine annealing if enabled
    if use_cosine and total_epochs:
        # Cosine annealing: apply cosine from after warmup to end of training
        # Base LR changes at milestones, but cosine continues with new base LR
        # For ImageNet/iNaturalist: warmup_epochs (5 epochs), then cosine
        warmup_epochs_used = warmup_epochs if warmup_epochs is not None else 0
        if warmup_steps is not None and warmup_epochs is None:
            # Only use warmup_steps if warmup_epochs is not specified
            # If warmup_steps, warmup ends very early (5 iterations), so start cosine from epoch 0
            warmup_epochs_used = 0
        
        # Calculate cosine progress from warmup end to total epochs
        if epoch >= warmup_epochs_used:
            progress = (epoch - warmup_epochs_used) / (total_epochs - warmup_epochs_used)
            # Clamp progress to [0, 1] to avoid issues
            progress = min(max(progress, 0.0), 1.0)
            lr = current_base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Should not reach here if warmup logic is correct, but just in case
            lr = current_base_lr
        return lr
    
    # Default: only step decay (no cosine)
    return current_base_lr


def train_single_expert(expert_key, use_expert_split=True, override_epochs=None, override_batch_size=None):
    """
    Train a single expert based on its configuration.

    Args:
        expert_key: Key identifying the expert ('ce', 'logitadjust', 'balsoftmax')
        use_expert_split: If True, use expert split (90% of train), else use full train
        override_epochs: Override number of epochs (from command line)
        override_batch_size: Override batch size (from command line)
    """
    # Select expert configs based on dataset
    dataset_name = CONFIG["dataset"]["name"]
    if dataset_name in ["inaturalist2018", "imagenet_lt"]:
        if expert_key not in EXPERT_CONFIGS_INATURALIST:
            raise ValueError(f"Expert '{expert_key}' not found in EXPERT_CONFIGS_INATURALIST")
        expert_configs = EXPERT_CONFIGS_INATURALIST
    else:
        if expert_key not in EXPERT_CONFIGS:
            raise ValueError(f"Expert '{expert_key}' not found in EXPERT_CONFIGS")
        expert_configs = EXPERT_CONFIGS
    
    expert_config = expert_configs[expert_key].copy()  # Make a copy to avoid modifying original
    
    # Apply overrides
    if override_epochs is not None:
        expert_config["epochs"] = override_epochs
        print(f"[OVERRIDE] Epochs set to {override_epochs}")
    
    expert_name = expert_config["name"]
    loss_type = expert_config["loss_type"]

    print(f"\n{'=' * 60}")
    print(f"[EXPERT] TRAINING EXPERT: {expert_name.upper()}")
    print(f"[EXPERT] Loss Type: {loss_type.upper()}")
    print(f"[EXPERT] Splits Dir: {CONFIG['dataset']['splits_dir']}")
    print(f"{'=' * 60}")

    # Setup
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    train_loader, val_loader = get_dataloaders(
        use_expert_split=use_expert_split,
        dataset_name=CONFIG["dataset"]["name"],
        override_batch_size=override_batch_size
    )

    # Model and loss - get backbone from config if available
    backbone_name = CONFIG["dataset"].get("backbone", "cifar_resnet32")
    
    model = Expert(
        num_classes=CONFIG["dataset"]["num_classes"],
        backbone_name=backbone_name,
        dropout_rate=expert_config["dropout_rate"],
        init_weights=True,
    ).to(DEVICE)

    criterion = get_loss_function(loss_type, train_loader)
    print(f"[SUCCESS] Loss Function: {type(criterion).__name__}")

    # Print model summary
    print("[INFO] Model Architecture:")
    model.summary()

    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=expert_config["lr"],
        momentum=CONFIG["train_params"]["momentum"],
        weight_decay=expert_config["weight_decay"],
    )

    # Use manual LR scheduling for CE expert (paper compliance), otherwise use MultiStepLR or Cosine
    use_manual_lr = expert_config.get("use_manual_lr", False)
    use_cosine = expert_config.get("use_cosine", False)
    
    if use_manual_lr:
        scheduler = None  # Manual LR scheduling, no scheduler object needed
        warmup_epochs = expert_config.get("warmup_epochs", None)
        warmup_steps = expert_config.get("warmup_steps", None)
        # Print scheduling info
        if warmup_steps is not None:
            warmup_info = f"warmup={warmup_steps} steps to reach base LR"
        else:
            warmup_info = f"warmup={warmup_epochs} epochs"
        
        if use_cosine:
            if len(expert_config.get("milestones", [])) > 0:
                print(
                    f"[INFO] Using manual LR scheduling ({warmup_info}, decay 0.1 at epochs {expert_config['milestones']}, cosine annealing)"
                )
            else:
                print(
                    f"[INFO] Using manual cosine LR scheduling ({warmup_info}, cosine annealing)"
                )
        else:
            if len(expert_config.get("milestones", [])) > 0:
                print(
                    f"[INFO] Using manual LR scheduling ({warmup_info}, decay 0.1 at epochs {expert_config['milestones']})"
                )
            else:
                print(
                    f"[INFO] Using manual LR scheduling ({warmup_info})"
                )
    elif use_cosine:
        warmup_epochs = expert_config.get("warmup_epochs", 0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=expert_config["epochs"] - warmup_epochs,
            eta_min=0.0
        )
        print(
            f"[INFO] Using CosineAnnealingLR scheduler (T_max={expert_config['epochs'] - warmup_epochs}, warmup={warmup_epochs})"
        )
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=expert_config["milestones"],
            gamma=expert_config["gamma"],
        )
        print(
            f"[INFO] Using MultiStepLR scheduler (milestones={expert_config['milestones']}, gamma={expert_config['gamma']})"
        )

    # Note: Class weights removed - no reweighting during training/validation

    # Training setup
    best_val_acc = 0.0
    checkpoint_dir = (
        Path(CONFIG["output"]["checkpoints_dir"]) / CONFIG["dataset"]["name"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"best_{expert_name}.pth"

    # Training loop
    for epoch in range(expert_config["epochs"]):
        # Train
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{expert_config['epochs']}")
        ):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Manual LR scheduling for CE expert (paper compliance)
            if use_manual_lr:
                current_lr = get_manual_lr(
                    epoch=epoch,
                    iteration=batch_idx,
                    total_iters_per_epoch=len(train_loader),
                    base_lr=expert_config["lr"],
                    warmup_epochs=expert_config.get("warmup_epochs", None),
                    warmup_steps=expert_config.get("warmup_steps", None),
                    milestones=expert_config.get("milestones", []),
                    gamma=expert_config.get("gamma", 0.1),
                    use_cosine=expert_config.get("use_cosine", False),
                    total_epochs=expert_config["epochs"],
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Update scheduler (only if not using manual LR)
        if not use_manual_lr and scheduler is not None:
            scheduler.step()

        # Validate model
        val_acc, _, group_accs = validate_model(
            model, val_loader, DEVICE, class_weights=None
        )

        # Get current LR
        if use_manual_lr:
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler
                else optimizer.param_groups[0]["lr"]
            )

        print(
            f"Epoch {epoch + 1:3d}: Loss={running_loss / len(train_loader):.4f}, "
            f"Val Acc={val_acc:.2f}%, "
            f"Head={group_accs['head']:.1f}%, Tail={group_accs['tail']:.1f}%, "
            f"LR={current_lr:.6f}"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(
                f"New best! Val Acc={val_acc:.2f}% → Saving to {best_model_path}"
            )
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

    # Final validation
    final_acc, _, final_group_accs = validate_model(
        model, val_loader, DEVICE, class_weights=None
    )
    print("📊 Final Results:")
    print(f"   Overall Acc: {final_acc:.2f}% (on balanced val)")
    print(f"   Head: {final_group_accs['head']:.1f}%, Tail: {final_group_accs['tail']:.1f}%")

    # Export logits
    export_logits_for_all_splits(model, expert_name)

    print(f"[SUCCESS] COMPLETED: {expert_name}")
    return final_model_path
