#!/usr/bin/env python3
"""
Demo functions for comparing MoE + Plugin vs Paper Method (CE + Plugin)

This module provides functions for loading models, running inference pipelines,
and visualizing results. Designed to be imported into Jupyter notebooks.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from src.models.experts import Expert
from src.models.gating_network_map import GatingNetwork, GatingMLP
from src.models.gating import GatingFeatureBuilder
from src.data.datasets import get_eval_augmentations

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = "cifar100_lt_if100"
NUM_CLASSES = 100
NUM_GROUPS = 2
TAIL_THRESHOLD = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
SPLITS_DIR = Path(f"./data/{DATASET}_splits_fixed")
CHECKPOINTS_DIR = Path(f"./checkpoints")
RESULTS_DIR = Path(f"./results/ltr_plugin/{DATASET}")
OUTPUT_DIR = Path("./results/demo_single_image")

EXPERT_NAMES = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]
EXPERT_DISPLAY_NAMES = ["CE", "LogitAdjust", "BalancedSoftmax"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_class_to_group() -> torch.Tensor:
    """Build class-to-group mapping: tail classes (train count <= 20) → group 1."""
    counts_path = SPLITS_DIR / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(NUM_CLASSES)]
    counts = np.array(class_counts)
    tail_mask = counts <= TAIL_THRESHOLD
    class_to_group = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_to_group[tail_mask] = 1  # 0=head, 1=tail
    
    num_head = (class_to_group == 0).sum()
    num_tail = (class_to_group == 1).sum()
    print(f"📊 Groups: {num_head} head classes, {num_tail} tail classes")
    
    return torch.tensor(class_to_group, dtype=torch.long, device=DEVICE)


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


def load_gating_network() -> GatingNetwork:
    """Load trained gating network."""
    num_experts = len(EXPERT_NAMES)
    gating = GatingNetwork(
        num_experts=num_experts, 
        num_classes=NUM_CLASSES, 
        routing="dense"
    ).to(DEVICE)
    
    compact_dim = 7 * num_experts + 3
    gating.mlp = GatingMLP(
        input_dim=compact_dim,
        num_experts=num_experts,
        hidden_dims=[256, 128],
        dropout=0.1,
        activation='relu',
    ).to(DEVICE)
    
    checkpoint_path = CHECKPOINTS_DIR / "gating_map" / DATASET / "final_gating.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Gating checkpoints may have "model_state_dict" key or be direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        gating.load_state_dict(checkpoint["model_state_dict"])
    else:
        gating.load_state_dict(checkpoint)
    
    gating.eval()
    
    print(f"✅ Loaded gating network from {checkpoint_path}")
    return gating


def load_plugin_params() -> Tuple[np.ndarray, np.ndarray, float]:
    """Load optimized plugin parameters for MoE from results JSON."""
    results_path = RESULTS_DIR / "ltr_plugin_gating_balanced.json"
    
    if not results_path.exists():
        print(f"⚠️  Plugin results not found at {results_path}")
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
    
    print(f"✅ Loaded plugin params from {results_path}")
    print(f"   α = {alpha}, μ = {mu}, cost = {cost:.4f}")
    
    return alpha, mu, cost


def load_ce_only_plugin_params() -> Tuple[np.ndarray, np.ndarray, float]:
    """Load optimized plugin parameters for CE-only from results JSON."""
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
    
    print(f"✅ Loaded CE-only plugin params from {results_path}")
    print(f"   α = {alpha}, μ = {mu}, cost = {cost:.4f}")
    
    return alpha, mu, cost


# ============================================================================
# PLUGIN CLASS (Shared by both methods)
# ============================================================================

class BalancedLtRPlugin:
    """Simplified plugin for single image inference."""
    def __init__(self, class_to_group: torch.Tensor, alpha: np.ndarray, mu: np.ndarray, cost: float):
        self.class_to_group = class_to_group
        self.alpha_group = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
        self.mu_group = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
        self.cost = cost
        self.num_groups = len(alpha)
    
    def _alpha_class(self) -> torch.Tensor:
        return self.alpha_group[self.class_to_group]
    
    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]
    
    def _alpha_hat_class(self) -> torch.Tensor:
        K = float(self.num_groups)
        alpha_hat_group = self.alpha_group / max(K, 1.0)
        return alpha_hat_group[self.class_to_group]
    
    def predict(self, posterior: torch.Tensor) -> int:
        """h*(x) = argmax_y (1/α̂[y]) * p_y(x)"""
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        reweighted = posterior / alpha_hat.unsqueeze(0)
        return reweighted.argmax(dim=-1).item()
    
    def reject(self, posterior: torch.Tensor) -> bool:
        """r(x) = 1 if max_y(1/α̂[y]*p_y) < Σ_y'(1/α̂[y'] - μ[y'])*p_y' - c"""
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        mu = self._mu_class()
        
        inv_alpha_hat = 1.0 / alpha_hat
        max_reweighted = (posterior * inv_alpha_hat.unsqueeze(0)).max(dim=-1)[0]
        threshold = ((inv_alpha_hat - mu).unsqueeze(0) * posterior).sum(dim=-1)
        
        return (max_reweighted < (threshold - self.cost)).item()


# ============================================================================
# PAPER METHOD (CE + Plugin)
# ============================================================================

def paper_method_pipeline(
    image_tensor: torch.Tensor,
    ce_expert_model: Expert,
    class_to_group: torch.Tensor,
    plugin_alpha: np.ndarray,
    plugin_mu: np.ndarray,
    plugin_cost: float
) -> Dict:
    """Paper method: CE expert + Plugin (as proposed in paper)"""
    with torch.no_grad():
        logits = ce_expert_model(image_tensor)  # [1, 100]
        probs = F.softmax(logits, dim=-1)  # [1, 100]
        posterior = probs  # [1, 100]
        
        # Apply plugin (same as our method but only with CE expert)
        plugin = BalancedLtRPlugin(class_to_group, plugin_alpha, plugin_mu, plugin_cost)
        plugin_pred = plugin.predict(posterior)
        plugin_reject = plugin.reject(posterior)
        
        # Compute plugin confidence (reweighted value, can be > 1)
        eps = 1e-12
        alpha_hat = plugin._alpha_hat_class().clamp(min=eps)
        reweighted = posterior / alpha_hat.unsqueeze(0)
        plugin_confidence = reweighted.max(dim=-1)[0].item()
        
        # Also compute max probability (actual confidence in [0, 1])
        max_probability = posterior.max(dim=-1)[0].item()
        
        return {
            'method': 'Paper Method (CE + Plugin)',
            'prediction': plugin_pred,
            'confidence': max_probability,  # Actual probability
            'plugin_confidence': plugin_confidence,  # Reweighted value (can be > 1)
            'probabilities': probs[0].cpu().numpy(),
            'logits': logits[0].cpu().numpy(),
            'reject': plugin_reject,
            'plugin_params': {
                'alpha': plugin_alpha.tolist() if isinstance(plugin_alpha, np.ndarray) else plugin_alpha,
                'mu': plugin_mu.tolist() if isinstance(plugin_mu, np.ndarray) else plugin_mu,
                'cost': float(plugin_cost)
            }
        }


# ============================================================================
# OUR METHOD (MoE + Plugin)
# ============================================================================

def our_method_pipeline(
    image_tensor: torch.Tensor,
    experts_list: List[Expert],
    gating_net: GatingNetwork,
    class_to_group: torch.Tensor,
    plugin_alpha: np.ndarray,
    plugin_mu: np.ndarray,
    plugin_cost: float
) -> Dict:
    """Our method: 3 Experts + Gating + Plugin"""
    with torch.no_grad():
        # Step 1: Get expert logits
        expert_logits_list = []
        expert_probs_list = []
        expert_predictions = []
        
        for expert in experts_list:
            logits = expert(image_tensor)  # [1, 100]
            probs = F.softmax(logits, dim=-1)  # [1, 100]
            pred = probs.argmax(dim=-1).item()
            
            expert_logits_list.append(logits)
            expert_probs_list.append(probs)
            expert_predictions.append(pred)
        
        # Stack: [3, 1, 100] -> [1, 3, 100]
        expert_logits = torch.stack(expert_logits_list, dim=0).transpose(0, 1)  # [1, 3, 100]
        expert_posteriors = torch.stack(expert_probs_list, dim=0).transpose(0, 1)  # [1, 3, 100]
        
        # Step 2: Gating network
        feat_builder = GatingFeatureBuilder()
        features = feat_builder(expert_logits)  # [1, 7*3+3] = [1, 24]
        
        gating_logits = gating_net.mlp(features)  # [1, 3]
        gating_weights = gating_net.router(gating_logits)  # [1, 3]
        
        # Step 3: Mixture posterior
        mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)  # [1, 100]
        
        # Step 4: Plugin (Balanced L2R)
        plugin = BalancedLtRPlugin(class_to_group, plugin_alpha, plugin_mu, plugin_cost)
        plugin_pred = plugin.predict(mixture_posterior)
        plugin_reject = plugin.reject(mixture_posterior)
        
        # Compute plugin confidence (reweighted value, can be > 1)
        eps = 1e-12
        alpha_hat = plugin._alpha_hat_class().clamp(min=eps)
        reweighted = mixture_posterior / alpha_hat.unsqueeze(0)
        plugin_confidence = reweighted.max(dim=-1)[0].item()
        
        # Also compute max probability (actual confidence in [0, 1])
        max_probability = mixture_posterior.max(dim=-1)[0].item()
        
        return {
            'method': 'Our Method (MoE + Gating + Plugin)',
            'expert_logits': expert_logits[0].cpu().numpy(),
            'expert_probs': expert_posteriors[0].cpu().numpy(),
            'expert_predictions': expert_predictions,
            'gating_weights': gating_weights[0].cpu().numpy(),
            'mixture_posterior': mixture_posterior[0].cpu().numpy(),
            'prediction': plugin_pred,
            'confidence': max_probability,  # Actual probability
            'plugin_confidence': plugin_confidence,  # Reweighted value (can be > 1)
            'reject': plugin_reject,
            'plugin_params': {
                'alpha': plugin_alpha.tolist() if isinstance(plugin_alpha, np.ndarray) else plugin_alpha,
                'mu': plugin_mu.tolist() if isinstance(plugin_mu, np.ndarray) else plugin_mu,
                'cost': float(plugin_cost)
            }
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_comparison(image_array, true_label, baseline_result, our_result, class_names, class_to_group):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(20, 14))
    
    # Row 1: Image và Top Predictions
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(image_array)
    ax1.set_title(f"Input Image\nClass: {true_label} ({class_names[true_label]})\n{'Tail' if class_to_group[true_label].item() == 1 else 'Head'}", fontsize=11)
    ax1.axis('off')
    
    # Paper Method - Top 5 Predictions
    ax2 = plt.subplot(3, 3, 2)
    probs_baseline = baseline_result['probabilities']
    top5_indices = np.argsort(probs_baseline)[-5:][::-1]
    top5_probs = probs_baseline[top5_indices]
    colors_baseline = ['green' if idx == true_label else ('red' if idx == baseline_result['prediction'] else 'gray') 
                      for idx in top5_indices]
    ax2.barh(range(5), top5_probs, color=colors_baseline)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([f"{class_names[idx]}" for idx in top5_indices], fontsize=9)
    ax2.set_xlabel('Probability')
    conf_display = f"Prob: {baseline_result['confidence']:.3f}"
    if 'plugin_confidence' in baseline_result:
        conf_display += f"\nScore: {baseline_result['plugin_confidence']:.3f}"
    ax2.set_title(f"Paper Method (CE + Plugin)\nPred: {class_names[baseline_result['prediction']]}\n{conf_display}\nReject: {'YES' if baseline_result['reject'] else 'NO'}", fontsize=11)
    ax2.invert_yaxis()
    
    # Our Method - Top 5 Predictions
    ax3 = plt.subplot(3, 3, 3)
    probs_our = our_result['mixture_posterior']
    top5_indices_our = np.argsort(probs_our)[-5:][::-1]
    top5_probs_our = probs_our[top5_indices_our]
    colors_our = ['green' if idx == true_label else ('red' if idx == our_result['prediction'] else 'gray') 
                 for idx in top5_indices_our]
    ax3.barh(range(5), top5_probs_our, color=colors_our)
    ax3.set_yticks(range(5))
    ax3.set_yticklabels([f"{class_names[idx]}" for idx in top5_indices_our], fontsize=9)
    ax3.set_xlabel('Probability')
    conf_display_our = f"Prob: {our_result['confidence']:.3f}"
    if 'plugin_confidence' in our_result:
        conf_display_our += f"\nScore: {our_result['plugin_confidence']:.3f}"
    ax3.set_title(f"Our Method (MoE + Plugin)\nPred: {class_names[our_result['prediction']]}\n{conf_display_our}\nReject: {'YES' if our_result['reject'] else 'NO'}", fontsize=11)
    ax3.invert_yaxis()
    
    # Row 2: Expert Contributions
    ax4 = plt.subplot(3, 3, 4)
    expert_names_short = EXPERT_DISPLAY_NAMES
    expert_preds = our_result['expert_predictions']
    expert_correct = ['[OK]' if p == true_label else '[X]' for p in expert_preds]
    
    y_pos = np.arange(len(expert_names_short))
    bars = ax4.barh(y_pos, [1, 1, 1], color=['skyblue', 'lightcoral', 'lightgreen'])
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{name} {status}" for name, status in zip(expert_names_short, expert_correct)], fontsize=10)
    ax4.set_xlabel('Contribution')
    ax4.set_title('Expert Predictions', fontsize=11)
    ax4.set_xlim([0, 1.2])
    for i, (pred, name) in enumerate(zip(expert_preds, expert_names_short)):
        ax4.text(0.5, i, f"→ {class_names[pred]}", va='center', fontsize=9)
    
    # Gating Weights
    ax5 = plt.subplot(3, 3, 5)
    gating_weights = our_result['gating_weights']
    ax5.barh(expert_names_short, gating_weights, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax5.set_xlabel('Weight')
    ax5.set_title(f'Gating Weights\n(Sum: {gating_weights.sum():.3f})', fontsize=11)
    ax5.set_xlim([0, 1])
    
    # Expert Confidences
    ax6 = plt.subplot(3, 3, 6)
    expert_probs_array = our_result['expert_probs']
    expert_confidences = [np.max(probs) for probs in expert_probs_array]
    ax6.barh(expert_names_short, expert_confidences, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax6.set_xlabel('Max Probability')
    ax6.set_title('Expert Confidences', fontsize=11)
    ax6.set_xlim([0, 1])
    
    # Row 3: Comparison Metrics
    ax7 = plt.subplot(3, 3, 7)
    methods = ['Paper\nMethod', 'Our\nMethod']
    confidences = [baseline_result['confidence'], our_result['confidence']]
    colors_conf = ['orange', 'green']
    bars = ax7.bar(methods, confidences, color=colors_conf, alpha=0.7)
    ax7.set_ylabel('Max Probability')
    ax7.set_title('Probability Comparison', fontsize=11)
    ax7.set_ylim([0, 1])
    for bar, conf in zip(bars, confidences):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{conf:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Prediction Accuracy
    ax8 = plt.subplot(3, 3, 8)
    baseline_correct = baseline_result['prediction'] == true_label
    our_correct = our_result['prediction'] == true_label
    accuracies = [int(baseline_correct), int(our_correct)]
    colors_acc = ['red' if not acc else 'green' for acc in accuracies]
    bars = ax8.bar(methods, accuracies, color=colors_acc, alpha=0.7)
    ax8.set_ylabel('Correct (1) / Wrong (0)')
    ax8.set_title('Prediction Accuracy', fontsize=11)
    ax8.set_ylim([-0.1, 1.1])
    ax8.set_yticks([0, 1])
    for bar, acc in zip(bars, accuracies):
        status = '[CORRECT]' if acc else '[WRONG]'
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                status, ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Rejection Decision
    ax9 = plt.subplot(3, 3, 9)
    rejections = [int(baseline_result['reject']), int(our_result['reject'])]
    colors_rej = ['red' if rej else 'green' for rej in rejections]
    bars = ax9.bar(methods, rejections, color=colors_rej, alpha=0.7)
    ax9.set_ylabel('Reject (1) / Accept (0)')
    ax9.set_title('Rejection Decision', fontsize=11)
    ax9.set_ylim([-0.1, 1.1])
    ax9.set_yticks([0, 1])
    for bar, rej in zip(bars, rejections):
        status = 'REJECT' if rej else 'ACCEPT'
        ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                status, ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    return fig
