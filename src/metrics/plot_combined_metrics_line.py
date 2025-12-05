#!/usr/bin/env python3
"""
Combined Line Plot: All 4 Metrics in One Figure
================================================

Combines NLL, Brier, Max-prob ECE, and Class-wise ECE in a single plot
with normalization and clear legend.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

@dataclass
class Config:
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100"
    gating_checkpoint: str = (
        "./checkpoints/gating_map/cifar100_lt_if100/best_gating.pth"
    )
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"

    expert_names: List[str] = field(
        default_factory=lambda: [
            "ce_baseline",
            "logitadjust_baseline",
            "balsoftmax_baseline",
        ]
    )

    num_classes: int = 100
    tail_threshold: int = 20
    n_bins: int = 15

    seed: int = 42


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# Data loading (reuse functions)
# ================================


def load_expert_logits(expert_name: str, split: str) -> torch.Tensor:
    path = Path(CFG.logits_dir) / expert_name / f"{split}_logits.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return torch.load(path, map_location=DEVICE).float()


def load_all_expert_logits(split: str) -> torch.Tensor:
    logits_list = []
    for expert_name in CFG.expert_names:
        logits = load_expert_logits(expert_name, split)
        logits_list.append(logits)
    return torch.stack(logits_list, dim=0).transpose(0, 1)


def load_labels(split: str) -> torch.Tensor:
    cand = Path(CFG.logits_dir) / CFG.expert_names[0] / f"{split}_targets.pt"
    if cand.exists():
        t = torch.load(cand, map_location=DEVICE)
        if isinstance(t, torch.Tensor):
            return t.to(device=DEVICE, dtype=torch.long)
    import torchvision

    indices_file = Path(CFG.splits_dir) / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    return torch.tensor(
        [ds.targets[i] for i in indices], dtype=torch.long, device=DEVICE
    )


def load_gating_network():
    from src.models.gating_network_map import GatingNetwork, GatingMLP

    num_experts = len(CFG.expert_names)
    gating = GatingNetwork(
        num_experts=num_experts, num_classes=CFG.num_classes, routing="dense"
    ).to(DEVICE)
    compact_dim = 7 * num_experts + 3
    gating.mlp = GatingMLP(
        input_dim=compact_dim,
        num_experts=num_experts,
        hidden_dims=[256, 128],
        dropout=0.1,
        activation="relu",
    ).to(DEVICE)
    checkpoint_path = Path(CFG.gating_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()
    return gating


@torch.no_grad()
def compute_mixture_posterior(expert_logits: torch.Tensor, gating_net) -> torch.Tensor:
    from src.models.gating import GatingFeatureBuilder

    expert_posteriors = F.softmax(expert_logits, dim=-1)
    feat_builder = GatingFeatureBuilder()
    features = feat_builder(expert_logits)
    gating_logits = gating_net.mlp(features)
    gating_weights = gating_net.router(gating_logits)
    
    if torch.isnan(gating_weights).any():
        N, E = expert_logits.shape[0], expert_logits.shape[1]
        gating_weights = torch.ones(N, E, device=DEVICE) / E
    
    mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)
    return mixture_posterior


def load_class_weights(device: str = DEVICE) -> torch.Tensor:
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]
    
    counts = np.array(class_counts, dtype=np.float64)
    total_train = counts.sum()
    train_probs = counts / total_train
    weights = train_probs * CFG.num_classes
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ================================
# Metrics computation
# ================================


@torch.no_grad()
def compute_nll(posterior: torch.Tensor, labels: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> float:
    eps = 1e-12
    p_true = posterior[torch.arange(len(labels)), labels].clamp(min=eps)
    nll_per_sample = -torch.log(p_true)
    
    if class_weights is not None:
        sample_weights = class_weights[labels]
        nll = (sample_weights * nll_per_sample).sum() / sample_weights.sum()
    else:
        nll = nll_per_sample.mean()
    
    return float(nll.item())


@torch.no_grad()
def compute_brier_score(posterior: torch.Tensor, labels: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> float:
    one_hot = F.one_hot(labels, num_classes=posterior.shape[1]).float()
    brier_per_sample = ((posterior - one_hot) ** 2).sum(dim=1)
    
    if class_weights is not None:
        sample_weights = class_weights[labels]
        brier = (sample_weights * brier_per_sample).sum() / sample_weights.sum()
    else:
        brier = brier_per_sample.mean()
    
    return float(brier.item())


@torch.no_grad()
def compute_max_prob_ece(posterior: torch.Tensor, labels: torch.Tensor, n_bins: int = 15, class_weights: Optional[torch.Tensor] = None) -> float:
    max_probs = posterior.max(dim=1)[0].cpu().numpy()
    preds = posterior.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    accuracies = (preds == labels_np).astype(float)
    
    if class_weights is not None:
        sample_weights = class_weights[labels].cpu().numpy()
        total_weight = sample_weights.sum()
    else:
        sample_weights = np.ones(len(max_probs))
        total_weight = len(max_probs)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i + 1])
        
        if bin_mask.sum() == 0:
            continue
        
        bin_weights = sample_weights[bin_mask]
        bin_weight_sum = bin_weights.sum()
        
        if bin_weight_sum == 0:
            continue
        
        avg_conf = (bin_weights * max_probs[bin_mask]).sum() / bin_weight_sum
        avg_acc = (bin_weights * accuracies[bin_mask]).sum() / bin_weight_sum
        
        bin_weight_frac = bin_weight_sum / total_weight
        ece += bin_weight_frac * abs(avg_conf - avg_acc)
    
    return float(ece)


@torch.no_grad()
def compute_classwise_ece(posterior: torch.Tensor, labels: torch.Tensor, n_bins: int = 15, class_weights: Optional[torch.Tensor] = None) -> float:
    num_classes = posterior.shape[1]
    per_class_ece = np.zeros(num_classes)
    
    if class_weights is not None:
        sample_weights = class_weights[labels].cpu().numpy()
        total_weight = sample_weights.sum()
    else:
        sample_weights = np.ones(len(labels))
        total_weight = len(labels)
    
    for k in range(num_classes):
        p_k = posterior[:, k].cpu().numpy()
        y_k = (labels == k).cpu().numpy().astype(float)
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece_k = 0.0
        
        for i in range(n_bins):
            bin_mask = (p_k >= bin_edges[i]) & (p_k < bin_edges[i + 1])
            if i == n_bins - 1:
                bin_mask = (p_k >= bin_edges[i]) & (p_k <= bin_edges[i + 1])
            
            if bin_mask.sum() == 0:
                continue
            
            bin_weights = sample_weights[bin_mask]
            bin_weight_sum = bin_weights.sum()
            
            if bin_weight_sum == 0:
                continue
            
            avg_conf = (bin_weights * p_k[bin_mask]).sum() / bin_weight_sum
            avg_acc = (bin_weights * y_k[bin_mask]).sum() / bin_weight_sum
            
            bin_weight_frac = bin_weight_sum / total_weight
            ece_k += bin_weight_frac * abs(avg_conf - avg_acc)
        
        per_class_ece[k] = ece_k
    
    return float(per_class_ece.mean())


# ================================
# Plotting
# ================================


def normalize_metric(values: List[float], lower_is_better: bool = True) -> List[float]:
    """Normalize metric to [0, 1] where 0 = best, 1 = worst."""
    v = np.array(values)
    v_min, v_max = v.min(), v.max()
    
    if v_max == v_min:
        return [0.0] * len(values)
    
    if lower_is_better:
        # Lower is better: normalize so best (min) = 0, worst (max) = 1
        normalized = (v - v_min) / (v_max - v_min)
    else:
        # Higher is better: normalize so best (max) = 0, worst (min) = 1
        normalized = (v_max - v) / (v_max - v_min)
    
    return normalized.tolist()


def plot_combined_metrics(
    methods: List[str],
    nll_vals: List[float],
    brier_vals: List[float],
    max_ece_vals: List[float],
    cw_ece_vals: List[float],
    out_path: Path,
):
    """Plot all 4 metrics as grouped bar chart with methods as bars."""
    
    # Reduced height to make figure more compact and widen aspect ratio
    fig, ax = plt.subplots(figsize=(9, 5.2))
    
    # Define colors for each method (consistent with other plots)
    method_colors = {
        "CE Baseline": "#1f77b4",      # Blue
        "LogitAdjust": "#ff7f0e",       # Orange
        "BalSoftmax": "#2ca02c",        # Green
        "Gating Mixture": "#d62728",    # Red
    }
    
    # Normalize all metrics to [0, 1] for comparison
    nll_norm = normalize_metric(nll_vals, lower_is_better=True)
    brier_norm = normalize_metric(brier_vals, lower_is_better=True)
    max_ece_norm = normalize_metric(max_ece_vals, lower_is_better=True)
    cw_ece_norm = normalize_metric(cw_ece_vals, lower_is_better=True)
    
    # Metrics on x-axis
    metrics = ["NLL", "Brier Score", "Max-prob ECE", "Class-wise ECE"]
    metric_data = [nll_norm, brier_norm, max_ece_norm, cw_ece_norm]
    
    # Number of methods and metrics
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    # Set up bar positions
    x = np.arange(n_metrics)
    base_width = 0.22  # Increased base width for better visibility
    spacing = 0.02  # Small spacing between bars in a group
    
    # Gating Mixture gets slightly wider bar for better visibility
    gating_width_multiplier = 1.15
    
    # Calculate widths for each method
    widths = [base_width * gating_width_multiplier if method == "Gating Mixture" else base_width 
              for method in methods]
    
    # Calculate total width needed for each group (centered around x positions)
    total_group_width = sum(widths) + (n_methods - 1) * spacing
    
    # Calculate starting position (left edge of first bar)
    start_pos = -total_group_width / 2
    
    # Plot bars for each method
    current_x = start_pos
    for i, method in enumerate(methods):
        method_values = [metric_data[j][i] for j in range(n_metrics)]
        width = widths[i]
        
        # Position is center of bar, so add half width
        positions = x + current_x + width / 2
        color = method_colors.get(method, 'gray')
        
        # Highlight Gating Mixture with thicker border and slightly higher alpha
        if method == "Gating Mixture":
            edgewidth = 2.5
            alpha = 0.95
        else:
            edgewidth = 1.5
            alpha = 0.9
        
        bars = ax.bar(positions, method_values, width, 
                     label=method, color=color, alpha=alpha, 
                     edgecolor='black', linewidth=edgewidth, zorder=5)
        
        # Move to next bar position (right edge of current bar + spacing)
        current_x += width + spacing
    
    ax.set_xlabel("Metric", fontsize=16, fontweight='bold')
    ax.set_ylabel("Normalized Score (0 = Best, 1 = Worst)", fontsize=16, fontweight='bold')
    ax.set_title("Comprehensive Metrics Comparison: Single Models vs Gating Mixture\n" +
                 "(All metrics normalized to [0,1] scale for fair comparison)",
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=0, ha='center', fontsize=15)
    ax.set_ylim([-0.05, 1.1])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5, axis='y', zorder=0)
    
    # Legend for methods (single legend for left column)
    legend = ax.legend(loc='upper left', fontsize=14, framealpha=0.95,
                      fancybox=True, shadow=True, ncol=1, columnspacing=1.0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    
    split = "test"
    
    print("\n" + "="*70)
    print("COMBINED METRICS LINE PLOT")
    print("="*70 + "\n")
    
    # Load data
    labels = load_labels(split)
    class_weights = load_class_weights(DEVICE)
    
    # Load posteriors
    posteriors = {}
    for expert_name in CFG.expert_names:
        logits = load_expert_logits(expert_name, split)
        posteriors[expert_name] = F.softmax(logits, dim=-1)
    
    expert_logits_all = load_all_expert_logits(split)
    gating_net = load_gating_network()
    posteriors["gating_mixture"] = compute_mixture_posterior(expert_logits_all, gating_net)
    
    # Methods
    methods = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline", "gating_mixture"]
    display_names = {
        "ce_baseline": "CE Baseline",
        "logitadjust_baseline": "LogitAdjust",
        "balsoftmax_baseline": "BalSoftmax",
        "gating_mixture": "Gating Mixture",
    }
    method_labels = [display_names[m] for m in methods]
    
    # Compute metrics
    print("Computing metrics...")
    nll_vals, brier_vals, max_ece_vals, cw_ece_vals = [], [], [], []
    
    for method_name in methods:
        posterior = posteriors[method_name]
        
        nll = compute_nll(posterior, labels, class_weights=class_weights)
        brier = compute_brier_score(posterior, labels, class_weights=class_weights)
        max_ece = compute_max_prob_ece(posterior, labels, CFG.n_bins, class_weights=class_weights)
        cw_ece = compute_classwise_ece(posterior, labels, CFG.n_bins, class_weights=class_weights)
        
        nll_vals.append(nll)
        brier_vals.append(brier)
        max_ece_vals.append(max_ece)
        cw_ece_vals.append(cw_ece)
        
        print(f"  {display_names[method_name]:<20} NLL={nll:.4f} Brier={brier:.4f} "
              f"Max-ECE={max_ece:.4f} cw-ECE={cw_ece:.4f}")
    
    # Create combined plot
    Path(CFG.results_dir).mkdir(parents=True, exist_ok=True)
    
    out_path = Path(CFG.results_dir) / "combined_metrics_line_comparison.png"
    plot_combined_metrics(
        methods=method_labels,
        nll_vals=nll_vals,
        brier_vals=brier_vals,
        max_ece_vals=max_ece_vals,
        cw_ece_vals=cw_ece_vals,
        out_path=out_path,
    )
    
    print("\n" + "="*70)
    print("✓ Combined plot created successfully!")
    print(f"  Saved to: {out_path}")
    print("="*70)


if __name__ == "__main__":
    main()

