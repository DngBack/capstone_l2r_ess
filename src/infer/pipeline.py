#!/usr/bin/env python3
"""
Inference functions for comparing MoE + Plugin vs Paper Method (CE + Plugin)

This module provides functions for running inference pipelines and visualizing results.
Model loading functions are in src.infer.loaders. Designed to be imported into Jupyter notebooks.
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

# Add project root to path (for absolute imports)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from src.models.experts import Expert
from src.models.gating_network_map import GatingNetwork, GatingMLP
from src.models.gating import GatingFeatureBuilder
from src.data.datasets import get_eval_augmentations

# Import loading functions from loaders (relative import)
from .loaders import (
    load_class_to_group,
    load_test_sample_with_image,
    load_ce_expert,
    load_all_experts,
    load_gating_network,
    load_plugin_params,
    DATASET,
    NUM_CLASSES,
    NUM_GROUPS,
    TAIL_THRESHOLD,
    DEVICE,
    EXPERT_NAMES,
    EXPERT_DISPLAY_NAMES,
    SPLITS_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
OUTPUT_DIR = Path("./results/infer_single_image")

# Re-export constants for backward compatibility
__all__ = [
    'load_class_to_group',
    'load_test_sample_with_image',
    'load_ce_expert',
    'load_all_experts',
    'load_gating_network',
    'load_plugin_params',
    'DATASET',
    'NUM_CLASSES',
    'NUM_GROUPS',
    'DEVICE',
    'OUTPUT_DIR',
    'EXPERT_NAMES',
    'EXPERT_DISPLAY_NAMES',
]


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
        # print(self.alpha_group)
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


class GeneralizedLtRPlugin:
    """Worst mode plugin with β parameter (Theorem 12).
    
    Reject rule: max_y p_y / (α̂_[y]) < Σ_y (1/α̂_[y] − μ_[y]) p_y − c,
    with α̂_k = α_k · β_k.
    """
    def __init__(self, class_to_group: torch.Tensor, alpha: np.ndarray, mu: np.ndarray, beta: np.ndarray, cost: float):
        self.class_to_group = class_to_group
        self.alpha_group = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
        self.mu_group = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
        self.beta_group = torch.tensor(beta, dtype=torch.float32, device=DEVICE)
        self.cost = cost
        self.num_groups = len(alpha)
    
    def _u_class(self) -> torch.Tensor:
        """u = β / α per group → expand to class level"""
        eps = 1e-12
        u_group = self.beta_group / self.alpha_group.clamp(min=eps)
        return u_group[self.class_to_group]
    
    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]
    
    def reject(self, posterior: torch.Tensor) -> bool:
        """r(x) = 1 if max_y(p_y * u) < Σ_y((u - μ) * p_y) - c"""
        u = self._u_class().unsqueeze(0)  # [1, C]
        mu = self._mu_class().unsqueeze(0)  # [1, C]
        
        max_reweighted = (posterior * u).max(dim=-1)[0]  # [1]
        threshold = ((u - mu) * posterior).sum(dim=-1)  # [1]
        
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
    """Paper method: CE expert + Plugin (Plugin chỉ dùng để reject, prediction từ CE gốc)"""
    with torch.no_grad():
        logits = ce_expert_model(image_tensor)  # [1, 100]
        probs = F.softmax(logits, dim=-1)  # [1, 100]
        posterior = probs  # [1, 100]
        
        # Prediction từ model gốc (CE expert)
        original_prediction = posterior.argmax(dim=-1).item()
        
        # Apply plugin CHỈ để reject (không dùng để predict)
        plugin = BalancedLtRPlugin(class_to_group, plugin_alpha, plugin_mu, plugin_cost)
        plugin_reject = plugin.reject(posterior)
        
        # Compute plugin confidence (reweighted value, can be > 1) - chỉ để hiển thị
        eps = 1e-12
        alpha_hat = plugin._alpha_hat_class().clamp(min=eps)
        reweighted = posterior / alpha_hat.unsqueeze(0)
        plugin_confidence = reweighted.max(dim=-1)[0].item()
        
        # Also compute max probability (actual confidence in [0, 1])
        max_probability = posterior.max(dim=-1)[0].item()
        
        return {
            'method': 'Paper Method (CE + Plugin)',
            'prediction': original_prediction,  # Prediction từ CE gốc
            'confidence': max_probability,  # Actual probability
            'plugin_confidence': plugin_confidence,  # Reweighted value (can be > 1)
            'probabilities': probs[0].cpu().numpy(),
            'logits': logits[0].cpu().numpy(),
            'reject': plugin_reject,  # Plugin chỉ dùng để reject
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
    plugin_cost: float,
    plugin_beta: Optional[np.ndarray] = None
) -> Dict:
    """
    Our method: 3 Experts + Gating + Plugin
    
    Args:
        plugin_beta: Optional beta parameter for worst mode. If None, uses balanced mode.
    """
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
        
        # Step 2: Gating network (dùng structure gốc của GatingNetwork)
        gating_weights, aux_outputs = gating_net.compute_weights_from_logits(expert_logits)  # [1, 3]
        
        # Step 3: Mixture posterior
        mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)  # [1, 100]
        
        # Prediction từ model gốc (mixture posterior)
        original_prediction = mixture_posterior.argmax(dim=-1).item()
        
        # Step 4: Plugin CHỈ để reject (không dùng để predict)
        # Use worst mode (GeneralizedLtRPlugin) if beta is provided, else balanced mode
        if plugin_beta is not None:
            plugin = GeneralizedLtRPlugin(class_to_group, plugin_alpha, plugin_mu, plugin_beta, plugin_cost)
            plugin_reject = plugin.reject(mixture_posterior)
            # Compute plugin confidence for worst mode
            eps = 1e-12
            u = plugin._u_class().unsqueeze(0)
            reweighted = mixture_posterior * u
            plugin_confidence = reweighted.max(dim=-1)[0].item()
            plugin_mode = "worst"
        else:
            plugin = BalancedLtRPlugin(class_to_group, plugin_alpha, plugin_mu, plugin_cost)
            plugin_reject = plugin.reject(mixture_posterior)
            # Compute plugin confidence for balanced mode
            eps = 1e-12
            alpha_hat = plugin._alpha_hat_class().clamp(min=eps)
            reweighted = mixture_posterior / alpha_hat.unsqueeze(0)
            plugin_confidence = reweighted.max(dim=-1)[0].item()
            plugin_mode = "balanced"
        
        # Also compute max probability (actual confidence in [0, 1])
        max_probability = mixture_posterior.max(dim=-1)[0].item()
        
        return {
            'method': 'Our Method (MoE + Gating + Plugin)',
            'expert_logits': expert_logits[0].cpu().numpy(),
            'expert_probs': expert_posteriors[0].cpu().numpy(),
            'expert_predictions': expert_predictions,
            'gating_weights': gating_weights[0].cpu().numpy(),
            'mixture_posterior': mixture_posterior[0].cpu().numpy(),
            'prediction': original_prediction,  # Prediction từ mixture gốc
            'confidence': max_probability,  # Actual probability
            'plugin_confidence': plugin_confidence,  # Reweighted value (can be > 1)
            'reject': plugin_reject,
            'plugin_params': {
                'alpha': plugin_alpha.tolist() if isinstance(plugin_alpha, np.ndarray) else plugin_alpha,
                'mu': plugin_mu.tolist() if isinstance(plugin_mu, np.ndarray) else plugin_mu,
                'beta': plugin_beta.tolist() if plugin_beta is not None and isinstance(plugin_beta, np.ndarray) else (plugin_beta if plugin_beta is not None else None),
                'cost': float(plugin_cost),
                'mode': plugin_mode
            }
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def compute_rejection_thresholds_from_test_set(
    target_rejection: float = 0.4
) -> Optional[Dict[str, float]]:
    """
    Compute rejection thresholds from test set for all methods.
    
    This loads posteriors from test set logits and computes thresholds
    based on target rejection rate (using quantile).
    
    If test logits are not available, can estimate from fallback posteriors
    (e.g., from current sample or a small subset).
    
    Args:
        target_rejection: Target rejection rate (default: 0.4 = 40% rejection)
        fallback_posteriors: Optional dict with posteriors for fallback estimation
                           Format: {'ce_baseline': [100], 'logitadjust_baseline': [100], ...}
    
    Returns:
        Dict with thresholds for each method, or None if cannot compute:
        {'ce_baseline': float, 'logitadjust_baseline': float, 
         'balsoftmax_baseline': float, 'gating_mixture': float}
    """
    from pathlib import Path
    
    # Try loading from test logits first
    logits_dir = Path(f"./outputs/logits/{DATASET}")
    
    if not logits_dir.exists():
        print(f"⚠️  Test logits directory not found: {logits_dir}")
        print("   Cannot compute rejection thresholds from test set.")
        print("   💡 Thresholds require full test set to compute quantiles correctly.")
        print("   💡 Visualization will proceed without threshold lines (thresholds=None).")
        return None
    
    posteriors = {}
    
    # Load expert posteriors
    for expert_name in EXPERT_NAMES:
        logits_path = logits_dir / expert_name / "test_logits.pt"
        if logits_path.exists():
            logits = torch.load(logits_path, map_location=DEVICE).float()
            posteriors[expert_name] = F.softmax(logits, dim=-1)
        else:
            print(f"⚠️  Missing logits: {logits_path}")
            return None
    
    # Load gating mixture posterior
    expert_logits_list = []
    for expert_name in EXPERT_NAMES:
        logits_path = logits_dir / expert_name / "test_logits.pt"
        logits = torch.load(logits_path, map_location=DEVICE).float()
        expert_logits_list.append(logits)
    
    # Stack expert logits
    expert_logits = torch.stack(expert_logits_list, dim=0).transpose(0, 1)  # [N, 3, 100]
    
    # Compute gating mixture (dùng structure gốc của GatingNetwork)
    gating_net = load_gating_network()
    expert_posteriors = F.softmax(expert_logits, dim=-1)
    
    gating_weights, _ = gating_net.compute_weights_from_logits(expert_logits)
    
    if torch.isnan(gating_weights).any():
        N, E = expert_logits.shape[0], expert_logits.shape[1]
        gating_weights = torch.ones(N, E, device=DEVICE) / E
    
    mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)
    posteriors["gating_mixture"] = mixture_posterior
    
    # Compute thresholds using quantile
    thresholds = {}
    for name, posterior in posteriors.items():
        max_probs = posterior.max(dim=1)[0].detach().cpu().numpy()
        tau = float(np.quantile(max_probs, target_rejection))
        thresholds[name] = tau
    
    print(f"✅ Computed rejection thresholds (target rejection: {target_rejection:.1%})")
    for name, tau in thresholds.items():
        display_name = EXPERT_DISPLAY_NAMES[EXPERT_NAMES.index(name)] if name in EXPERT_NAMES else "Gating Mixture"
        print(f"   {display_name}: τ = {tau:.4f}")
    
    return thresholds


def plot_full_class_distribution(
    true_label: int,
    paper_result: Dict,
    our_result: Dict,
    class_names: List[str],
    thresholds: Optional[Dict[str, float]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot full-class posterior distribution for single sample.
    
    Similar to plot_single_sample_full_class_distribution.py but for inference use.
    
    Args:
        true_label: True class label
        paper_result: Result from paper_method_pipeline() with 'probabilities' key
        our_result: Result from our_method_pipeline() with 'expert_probs' and 'mixture_posterior' keys
        class_names: List of class names
        thresholds: Optional dict with rejection thresholds for each method
                   Format: {'ce_baseline': float, 'logitadjust_baseline': float, 
                           'balsoftmax_baseline': float, 'gating_mixture': float}
        title: Optional custom title for the plot
    
    Returns:
        matplotlib Figure object
    """
    x = np.arange(NUM_CLASSES)
    
    # Extract posteriors
    ce_posterior = paper_result['probabilities']  # [100]
    expert_probs = our_result['expert_probs']  # [3, 100]
    la_posterior = expert_probs[1]  # LogitAdjust
    bs_posterior = expert_probs[2]  # BalSoftmax
    gating_posterior = our_result['mixture_posterior']  # [100]
    
    # Method colors (consistent with plot_single_sample_full_class_distribution.py)
    METHOD_COLORS = {
        "ce_baseline": "#1f77b4",      # Blue
        "logitadjust_baseline": "#ff7f0e",  # Orange
        "balsoftmax_baseline": "#2ca02c",   # Green
        "gating_mixture": "#d62728",        # Red
    }
    
    DISPLAY_NAMES = {
        "ce_baseline": "CE Baseline",
        "logitadjust_baseline": "LogitAdjust",
        "balsoftmax_baseline": "BalSoftmax",
        "gating_mixture": "ARE (Gating)",
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot posteriors
    posteriors_dict = {
        'ce_baseline': ce_posterior,
        'logitadjust_baseline': la_posterior,
        'balsoftmax_baseline': bs_posterior,
        'gating_mixture': gating_posterior,
    }
    
    METHOD_ORDER = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline", "gating_mixture"]
    
    max_values = []
    text_y = 0.92
    
    for method_name in METHOD_ORDER:
        probs = posteriors_dict[method_name]
        color = METHOD_COLORS.get(method_name, "#999999")
        linewidth = 2.4 if method_name == "gating_mixture" else 1.8
        alpha = 0.95 if method_name == "gating_mixture" else 0.6
        
        ax.plot(x, probs, linewidth=linewidth, color=color, alpha=alpha, 
                label=DISPLAY_NAMES[method_name])
        ax.fill_between(x, probs, alpha=0.06 if method_name == "gating_mixture" else 0.04, 
                       color=color)
        
        # Plot threshold if provided
        if thresholds and method_name in thresholds:
            threshold = thresholds[method_name]
            ax.axhline(threshold, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
        
        max_prob = float(probs.max())
        max_values.append(max_prob)
        pred = int(probs.argmax())
        is_correct = pred == true_label
        
        # Decision text
        if thresholds and method_name in thresholds:
            threshold = thresholds[method_name]
            accept = max_prob >= threshold
            if accept and is_correct:
                decision = "True Accept (KEEP)"
                box_color = "#2ca02c"
            elif accept and not is_correct:
                decision = "FALSE ACCEPT"
                box_color = "#d62728"
            elif not accept and is_correct:
                decision = "FALSE REJECT"
                box_color = "#d62728"
            else:
                decision = "True Reject (DROP)"
                box_color = "#2ca02c"
            
            threshold_text = f", τ={threshold:.2f}"
        else:
            decision = "[CORRECT]" if is_correct else "[WRONG]"
            box_color = "#2ca02c" if is_correct else "#d62728"
            threshold_text = ""
        
        ax.text(
            0.015,
            text_y,
            f"{DISPLAY_NAMES[method_name]}: {decision} (pred={pred}{threshold_text})",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.18, edgecolor=box_color),
        )
        text_y -= 0.11
    
    # Mark true class
    peak_prob = max(max_values)
    ax.axvline(true_label, color="#2ca02c", linestyle="-", linewidth=2.5, alpha=0.8)
    ax.annotate(
        f"True class {true_label}",
        xy=(true_label, min(0.95, peak_prob * 1.02 + 1e-3)),
        xytext=(true_label + 4, min(0.98, peak_prob * 1.1 + 1e-3)),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4),
        fontsize=10,
        color="#2ca02c",
        fontweight='bold'
    )
    
    # X-axis ticks
    tick_step = 10
    tick_positions = sorted(set(list(range(0, NUM_CLASSES, tick_step)) + [true_label]))
    tick_labels = [f"{t}\n(label)" if t == true_label else str(t) for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=9)
    
    # Labels and title
    ax.set_xlabel("Class ID", fontsize=14, fontweight='bold')
    ax.set_ylabel("Posterior Probability", fontsize=14, fontweight='bold')
    
    if title is None:
        title = f"Full-Class Posterior Distribution\nTrue Class: {true_label} ({class_names[true_label]})"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Y-axis limits
    y_max = min(1.0, max(max_values) * 1.2 + 1e-3)
    ax.set_ylim(0, y_max)
    ax.set_xlim(-2, NUM_CLASSES - 1)
    ax.grid(True, linestyle="--", alpha=0.35)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.5, label=DISPLAY_NAMES[m]) 
        for m in METHOD_ORDER
    ]
    if thresholds:
        legend_handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, 
                                   label="Rejection threshold"))
    legend_handles.append(Line2D([0], [0], marker="|", color="#2ca02c", linewidth=2.5, 
                               markersize=12, label="True class"))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    return fig


def plot_ce_only_full_class_distribution(
    true_label: int,
    paper_result: Dict,
    class_names: List[str],
    threshold: Optional[float] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot full-class posterior distribution for CE + Plugin (paper method only).

    This is a simplified version of plot_full_class_distribution that focuses on
    the paper method (CE expert + plugin) to show CE behaviour in isolation.

    Args:
        true_label: True class label
        paper_result: Result from paper_method_pipeline() with 'probabilities' key
        class_names: List of class names
        threshold: Optional rejection threshold τ for CE (max-prob rule)
        title: Optional custom title for the plot

    Returns:
        matplotlib Figure object
    """
    x = np.arange(NUM_CLASSES)

    # CE posterior from paper method
    ce_posterior = paper_result["probabilities"]  # [100]

    color = "#1f77b4"
    display_name = "CE Baseline (Paper CE + Plugin)"

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(x, ce_posterior, linewidth=2.0, color=color, alpha=0.9, label=display_name)
    ax.fill_between(x, ce_posterior, alpha=0.06, color=color)

    max_prob = float(np.max(ce_posterior))
    pred = int(np.argmax(ce_posterior))
    is_correct = pred == true_label

    # Optional threshold line and decision text (Chow-style max-prob rule)
    text_y = 0.9
    if threshold is not None:
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1.5, alpha=0.9)
        accept = max_prob >= threshold
        if accept and is_correct:
            decision = "True Accept (KEEP)"
            box_color = "#2ca02c"
        elif accept and not is_correct:
            decision = "FALSE ACCEPT"
            box_color = "#d62728"
        elif not accept and is_correct:
            decision = "FALSE REJECT"
            box_color = "#d62728"
        else:
            decision = "True Reject (DROP)"
            box_color = "#2ca02c"

        threshold_text = f", τ={threshold:.2f}"
    else:
        decision = "[CORRECT]" if is_correct else "[WRONG]"
        box_color = "#2ca02c" if is_correct else "#d62728"
        threshold_text = ""

    ax.text(
        0.02,
        text_y,
        f"{display_name}: {decision} (pred={pred}{threshold_text})",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.18, edgecolor=box_color),
    )

    # Mark true class
    peak_prob = float(np.max(ce_posterior))
    ax.axvline(true_label, color="#2ca02c", linestyle="-", linewidth=2.5, alpha=0.8)
    ax.annotate(
        f"True class {true_label}",
        xy=(true_label, min(0.95, peak_prob * 1.02 + 1e-3)),
        xytext=(true_label + 4, min(0.98, peak_prob * 1.1 + 1e-3)),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4),
        fontsize=10,
        color="#2ca02c",
        fontweight="bold",
    )

    # X-axis ticks
    tick_step = 10
    tick_positions = sorted(set(list(range(0, NUM_CLASSES, tick_step)) + [true_label]))
    tick_labels = [f"{t}\n(label)" if t == true_label else str(t) for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=9)

    # Labels and title
    ax.set_xlabel("Class ID", fontsize=14, fontweight="bold")
    ax.set_ylabel("Posterior Probability", fontsize=14, fontweight="bold")

    if title is None:
        title = f"Paper Method (CE + Plugin) – Full-Class Posterior\nTrue Class: {true_label} ({class_names[true_label]})"
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # Y-axis limits
    y_max = min(1.0, peak_prob * 1.2 + 1e-3)
    ax.set_ylim(0, y_max)
    ax.set_xlim(-2, NUM_CLASSES - 1)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=2.5, label=display_name),
    ]
    if threshold is not None:
        legend_handles.append(
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="Rejection threshold (max-prob)")
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="|",
            color="#2ca02c",
            linewidth=2.5,
            markersize=12,
            label="True class",
        )
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    return fig


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
