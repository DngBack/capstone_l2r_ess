"""
Loading functions for inference.

This module provides functions to load models, data, and parameters
for single image inference and comparison. It imports and reuses functions
from run_balanced_plugin_gating.py to avoid code duplication.
"""

from pathlib import Path
import json
import numpy as np
import torch
import torchvision
from typing import Dict, List, Tuple, Optional

from src.models.experts import Expert
from src.data.datasets import get_eval_augmentations

# Import functions from run_balanced_plugin_gating.py
# We need to set CFG before calling these functions
import run_balanced_plugin_gating as plugin_gating

# Configuration
DATASET = "cifar100_lt_if100"
NUM_CLASSES = 100
NUM_GROUPS = 2
TAIL_THRESHOLD = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
SPLITS_DIR = Path(f"./data/{DATASET}_splits_fixed")
CHECKPOINTS_DIR = Path(f"./checkpoints")
RESULTS_DIR = Path(f"./results/ltr_plugin/{DATASET}")

EXPERT_NAMES = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]
EXPERT_DISPLAY_NAMES = ["CE", "LogitAdjust", "BalancedSoftmax"]

# Export paths for use in other modules
# Export all constants and functions
__all__ = [
    'DATASET', 'NUM_CLASSES', 'NUM_GROUPS', 'TAIL_THRESHOLD', 'DEVICE',
    'SPLITS_DIR', 'CHECKPOINTS_DIR', 'RESULTS_DIR',
    'EXPERT_NAMES', 'EXPERT_DISPLAY_NAMES',
    'load_class_to_group', 'load_test_sample_with_image',
    'load_ce_expert', 'load_all_experts', 'load_gating_network',
    'load_plugin_params', 'load_ce_only_plugin_params',
]


def _setup_cfg():
    """Setup CFG in plugin_gating module from loaders constants."""
    cfg = plugin_gating.setup_config(DATASET)
    plugin_gating.CFG = cfg


def load_class_to_group() -> torch.Tensor:
    """Build class-to-group mapping: tail classes (train count <= 20) → group 1.
    
    Imports from run_balanced_plugin_gating.py to avoid duplication.
    """
    _setup_cfg()
    # Call build_class_to_group from plugin_gating
    result = plugin_gating.build_class_to_group()
    # Ensure tensor is on correct device
    if result.device != DEVICE:
        result = result.to(DEVICE)
    # Update print message to match expected format
    num_head = (result.cpu().numpy() == 0).sum()
    num_tail = (result.cpu().numpy() == 1).sum()
    print(f"📊 Groups: {num_head} head classes, {num_tail} tail classes")
    return result


def load_test_sample_with_image(class_idx: Optional[int] = None) -> Tuple[torch.Tensor, int, np.ndarray, str]:
    """Load a single test sample."""
    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
    
    indices_file = SPLITS_DIR / "test_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        test_indices = json.load(f)
    
    class_to_group = load_class_to_group().cpu().numpy()
    
    if class_idx is None:
        tail_classes = np.where(class_to_group == 1)[0]
        class_idx = np.random.choice(tail_classes)
        print(f"🎲 Randomly selected tail class: {class_idx}")
    else:
        print(f"🎯 Selected class: {class_idx}")
    
    # Find indices of samples from this class in test set
    class_samples = []
    for idx in test_indices:
        if dataset.targets[idx] == class_idx:
            class_samples.append(idx)
    
    if len(class_samples) == 0:
        raise ValueError(f"No test samples found for class {class_idx}")
    
    selected_idx = np.random.choice(class_samples)
    image, label = dataset[selected_idx]
    
    class_name = dataset.classes[class_idx]
    
    # Convert PIL to numpy for display
    image_array = np.array(image)  # [32, 32, 3]
    
    # Apply transforms for model input
    transform = get_eval_augmentations()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 32, 32]
    
    is_tail = class_to_group[class_idx] == 1
    group_str = "tail" if is_tail else "head"
    
    print(f"\n📷 Selected sample:")
    print(f"   Class: {class_idx} ({class_name}) - {group_str}")
    print(f"   True label: {label}")
    print(f"   Image shape: {image_array.shape}")
    
    return image_tensor, label, image_array, class_name


def load_ce_expert() -> Expert:
    """Load CE expert model."""
    expert = Expert(num_classes=NUM_CLASSES, backbone_name="cifar_resnet32")
    checkpoint_path = CHECKPOINTS_DIR / "experts" / DATASET / "final_calibrated_ce_baseline.pth"
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Expert checkpoints are saved directly as state_dict
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    
    # Handle temperature separately to avoid shape mismatch
    temp_value = None
    if "temperature" in state_dict:
        temp_tensor = state_dict["temperature"]
        # Handle both scalar and tensor cases
        if temp_tensor.dim() == 0:  # scalar
            temp_value = temp_tensor.item()
        else:  # tensor with shape [1] or similar
            temp_value = temp_tensor.item() if temp_tensor.numel() == 1 else temp_tensor[0].item()
        # Remove temperature from state_dict to avoid shape mismatch
        state_dict = {k: v for k, v in state_dict.items() if k != "temperature"}
    
    # Load state_dict (without temperature)
    expert.load_state_dict(state_dict, strict=False)
    
    # Set temperature separately if available
    if temp_value is not None:
        expert.set_temperature(temp_value)
    
    expert.eval()
    expert = expert.to(DEVICE)
    
    print(f"✅ Loaded CE expert from {checkpoint_path}")
    return expert


def load_all_experts() -> List[Expert]:
    """Load all 3 expert models."""
    experts = []
    expert_files = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]
    
    for expert_file in expert_files:
        expert = Expert(num_classes=NUM_CLASSES, backbone_name="cifar_resnet32")
        checkpoint_path = CHECKPOINTS_DIR / "experts" / DATASET / f"final_calibrated_{expert_file}.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        
        # Expert checkpoints are saved directly as state_dict
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        
        # Handle temperature separately to avoid shape mismatch
        temp_value = None
        if "temperature" in state_dict:
            temp_tensor = state_dict["temperature"]
            # Handle both scalar and tensor cases
            if temp_tensor.dim() == 0:  # scalar
                temp_value = temp_tensor.item()
            else:  # tensor with shape [1] or similar
                temp_value = temp_tensor.item() if temp_tensor.numel() == 1 else temp_tensor[0].item()
            # Remove temperature from state_dict to avoid shape mismatch
            state_dict = {k: v for k, v in state_dict.items() if k != "temperature"}
        
        # Load state_dict (without temperature)
        expert.load_state_dict(state_dict, strict=False)
        
        # Set temperature separately if available
        if temp_value is not None:
            expert.set_temperature(temp_value)
        
        expert.eval()
        expert = expert.to(DEVICE)
        
        experts.append(expert)
        print(f"✅ Loaded {expert_file} expert")
    
    return experts


def load_gating_network():
    """Load trained gating network.
    
    Imports from run_balanced_plugin_gating.py to avoid duplication.
    """
    _setup_cfg()
    # Call load_gating_network from plugin_gating with correct device
    result = plugin_gating.load_gating_network(device=DEVICE)
    # Ensure gating network is on correct device
    result = result.to(DEVICE)
    return result


def load_plugin_params() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load optimized plugin parameters for MoE from WORST mode results JSON.
    Selects config with rejection rate closest to 0.3.
    """
    results_path = RESULTS_DIR / "ltr_plugin_gating_worst.json"
    
    if not results_path.exists():
        print(f"⚠️  Plugin results not found at {results_path}")
        print("   Using default parameters...")
        return np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.0
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    results_per_point = results.get("results_per_point", [])
    
    if len(results_per_point) == 0:
        print("⚠️  No results found, using defaults")
        return np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.0
    
    # Find config with rejection rate closest to 0.3
    best_result = None
    best_diff = float('inf')
    
    for r in results_per_point:
        rej_rate = 1.0 - r["test_metrics"]["coverage"]
        diff = abs(rej_rate - 0.3)
        if diff < best_diff:
            best_diff = diff
            best_result = r
    
    if best_result is None:
        best_result = results_per_point[0]
    
    alpha = np.array(best_result["alpha"])
    mu = np.array(best_result["mu"])
    # cost_test is the relevant cost on test set; keep same key name
    cost = best_result.get("cost_test", 0.0)
    
    print(f"✅ Loaded plugin params (worst mode fallback) from {results_path}")
    print(f"   α = {alpha}, μ = {mu}, cost = {cost:.4f}")
    
    return alpha, mu, cost


def load_ce_only_plugin_params() -> Tuple[np.ndarray, np.ndarray, float]:
    """Load optimized plugin parameters for CE-only from balanced mode results JSON."""
    results_path = RESULTS_DIR / "ltr_plugin_ce_only_balanced.json"
    
    if not results_path.exists():
        print(f"⚠️  CE-only plugin results not found at {results_path}")
        print("   Using default parameters...")
        return np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.0
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    results_per_cost = results.get("results_per_cost", [])
    
    if len(results_per_cost) == 0:
        print("⚠️  No results found, using defaults")
        return np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.0
    
    # Find config with rejection rate closest to 0.3
    best_result = None
    best_diff = float('inf')
    
    for r in results_per_cost:
        rej_rate = 1.0 - r["val_metrics"]["coverage"]
        diff = abs(rej_rate - 0.3)
        if diff < best_diff:
            best_diff = diff
            best_result = r
    
    if best_result is None:
        best_result = results_per_cost[0]
    
    alpha = np.array(best_result["alpha"])
    mu = np.array(best_result["mu"])
    cost = best_result.get("cost_val", best_result.get("cost_test", 0.0))
    
    print(f"✅ Loaded CE-only plugin params (worst mode) from {results_path}")
    print(f"   α = {alpha}, μ = {mu}, cost = {cost:.4f}")
    
    return alpha, mu, cost
