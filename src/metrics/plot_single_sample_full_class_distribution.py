#!/usr/bin/env python3
"""
Full-class posterior visualization for single tail samples.

Creates a figure with two panels (Case 1: should keep, Case 2: should reject)
and overlays the posterior of CE, LogitAdjust, BalSoftmax, and Gating Mixture
across all 100 classes (x-axis = class id). Horizontal dashed lines indicate
each method's rejection threshold (estimated from test confidence quantile).
Annotations highlight where CE/LA/BS fail (false reject/accept) while ARE succeeds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


@dataclass
class Config:
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100"
    gating_checkpoint: str = "./checkpoints/gating_map/cifar100_lt_if100/best_gating.pth"
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
    seed: int = 42
    target_rejection: float = 0.4


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHOD_ORDER = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline", "gating_mixture"]
METHOD_COLORS = {
    "ce_baseline": "#1f77b4",
    "logitadjust_baseline": "#ff7f0e",
    "balsoftmax_baseline": "#2ca02c",
    "gating_mixture": "#d62728",
}
DISPLAY_NAMES = {
    "ce_baseline": "CE Baseline",
    "logitadjust_baseline": "LogitAdjust",
    "balsoftmax_baseline": "BalSoftmax",
    "gating_mixture": "ARE (Gating)",
}


def load_logits(expert_name: str, split: str) -> torch.Tensor:
    path = Path(CFG.logits_dir) / expert_name / f"{split}_logits.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing logits: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False).float()


def load_labels(split: str) -> torch.Tensor:
    cand = Path(CFG.logits_dir) / CFG.expert_names[0] / f"{split}_targets.pt"
    if cand.exists():
        labels = torch.load(cand, map_location=DEVICE, weights_only=False)
        if isinstance(labels, torch.Tensor):
            return labels.to(device=DEVICE, dtype=torch.long)

    import torchvision

    indices_file = Path(CFG.splits_dir) / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    return torch.tensor([ds.targets[i] for i in indices], dtype=torch.long, device=DEVICE)


def load_gating_network():
    from src.models.gating_network_map import GatingNetwork, GatingMLP

    num_experts = len(CFG.expert_names)
    gating = GatingNetwork(num_experts=num_experts, num_classes=CFG.num_classes, routing="dense").to(DEVICE)

    compact_dim = 7 * num_experts + 3
    gating.mlp = GatingMLP(
        input_dim=compact_dim,
        num_experts=num_experts,
        hidden_dims=[256, 128],
        dropout=0.1,
        activation="relu",
    ).to(DEVICE)

    ckpt = torch.load(Path(CFG.gating_checkpoint), map_location=DEVICE, weights_only=False)
    gating.load_state_dict(ckpt["model_state_dict"])
    gating.eval()
    return gating


def compute_mixture_posterior(expert_logits: torch.Tensor, gating_net) -> torch.Tensor:
    from src.models.gating import GatingFeatureBuilder

    with torch.no_grad():
        expert_posteriors = F.softmax(expert_logits, dim=-1)
        feat_builder = GatingFeatureBuilder()
        features = feat_builder(expert_logits)
        gating_logits = gating_net.mlp(features)
        gating_weights = gating_net.router(gating_logits)

        if torch.isnan(gating_weights).any():
            num_experts = expert_logits.shape[1]
            gating_weights = torch.ones_like(gating_weights)
            gating_weights /= num_experts

        mixture = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)
        return mixture


def build_tail_mask() -> torch.Tensor:
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        counts = np.array([class_counts[str(i)] for i in range(CFG.num_classes)])
    else:
        counts = np.array(class_counts)
    tail_mask = counts <= CFG.tail_threshold
    return torch.tensor(tail_mask, dtype=torch.bool, device=DEVICE)


def prepare_posteriors():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    labels = load_labels("test")
    tail_class_mask = build_tail_mask()
    is_tail = tail_class_mask[labels]

    posteriors: Dict[str, torch.Tensor] = {}
    for name in CFG.expert_names:
        logits = load_logits(name, "test")
        posteriors[name] = F.softmax(logits, dim=-1)

    expert_logits = torch.stack(
        [load_logits(name, "test") for name in CFG.expert_names],
        dim=1,
    )
    gating_net = load_gating_network()
    posteriors["gating_mixture"] = compute_mixture_posterior(expert_logits, gating_net)

    return labels, is_tail, posteriors


def find_case_ce_underconfidence(labels, is_tail, ce_post, gating_post) -> int:
    ce_preds = ce_post.argmax(dim=1)
    gating_preds = gating_post.argmax(dim=1)

    mask = is_tail & (ce_preds == labels) & (gating_preds == labels)
    if mask.sum() == 0:
        raise RuntimeError("No tail samples where CE and gating are correct.")

    ce_true = ce_post[mask, labels[mask]]
    gating_true = gating_post[mask, labels[mask]]
    diff = gating_true - ce_true
    ce_low_conf_mask = ce_true < 0.55
    if ce_low_conf_mask.any():
        diff = diff.clone()
        diff[~ce_low_conf_mask] = -1.0
    idx_within = diff.argmax().item()
    global_idx = torch.nonzero(mask, as_tuple=False)[idx_within].item()
    return global_idx


def find_case_should_keep(
    labels,
    is_tail,
    posteriors: Dict[str, torch.Tensor],
    thresholds: Dict[str, float],
) -> int:
    ce_post = posteriors["ce_baseline"]
    la_post = posteriors["logitadjust_baseline"]
    bs_post = posteriors["balsoftmax_baseline"]
    gating_post = posteriors["gating_mixture"]

    tail_indices = torch.nonzero(is_tail, as_tuple=False).squeeze().tolist()
    if isinstance(tail_indices, int):
        tail_indices = [tail_indices]

    for idx in tail_indices:
        y = labels[idx].item()
        ce_pred = ce_post[idx].argmax().item()
        ce_conf = ce_post[idx, ce_pred].item()
        gating_pred = gating_post[idx].argmax().item()
        gating_conf = gating_post[idx, gating_pred].item()

        la_pred = la_post[idx].argmax().item()
        la_conf = la_post[idx, la_pred].item()
        bs_pred = bs_post[idx].argmax().item()
        bs_conf = bs_post[idx, bs_pred].item()

        ce_false_reject = ce_pred == y and ce_conf < thresholds["ce_baseline"]
        gating_correct_keep = gating_pred == y and gating_conf >= thresholds["gating_mixture"]
        other_correct_keep = (
            (la_pred == y and la_conf >= thresholds["logitadjust_baseline"])
            or (bs_pred == y and bs_conf >= thresholds["balsoftmax_baseline"])
        )

        if ce_false_reject and gating_correct_keep and other_correct_keep:
            return idx

    raise RuntimeError("Could not find tail sample that should be kept (CE fails, others & ARE correct).")


def find_case_should_reject(
    labels,
    is_tail,
    posteriors: Dict[str, torch.Tensor],
    thresholds: Dict[str, float],
) -> int:
    ce_post = posteriors["ce_baseline"]
    la_post = posteriors["logitadjust_baseline"]
    bs_post = posteriors["balsoftmax_baseline"]
    gating_post = posteriors["gating_mixture"]

    tail_indices = torch.nonzero(is_tail, as_tuple=False).squeeze().tolist()
    if isinstance(tail_indices, int):
        tail_indices = [tail_indices]

    for idx in tail_indices:
        y = labels[idx].item()

        ce_pred = ce_post[idx].argmax().item()
        ce_conf = ce_post[idx, ce_pred].item()

        la_pred = la_post[idx].argmax().item()
        la_conf = la_post[idx, la_pred].item()
        bs_pred = bs_post[idx].argmax().item()
        bs_conf = bs_post[idx, bs_pred].item()

        gating_pred = gating_post[idx].argmax().item()
        gating_conf = gating_post[idx, gating_pred].item()

        la_false_accept = la_pred != y and la_conf >= thresholds["logitadjust_baseline"]
        bs_false_accept = bs_pred != y and bs_conf >= thresholds["balsoftmax_baseline"]
        gating_correct_reject = gating_conf < thresholds["gating_mixture"]

        ce_optional = ce_pred == y and ce_conf >= thresholds["ce_baseline"] or True

        if (la_false_accept or bs_false_accept) and gating_correct_reject and ce_optional:
            return idx

    raise RuntimeError("Could not find tail sample that should be rejected (LA/BS fail, ARE correct).")


def compute_thresholds(posteriors: Dict[str, torch.Tensor], target_rejection: float) -> Dict[str, float]:
    thresholds = {}
    for name, posterior in posteriors.items():
        max_probs = posterior.max(dim=1)[0].detach().cpu().numpy()
        tau = float(np.quantile(max_probs, target_rejection))
        thresholds[name] = tau
    return thresholds


def decision_text(max_prob: float, threshold: float, is_correct: bool) -> Tuple[str, str]:
    accept = max_prob >= threshold
    if accept and is_correct:
        return "True Accept (KEEP)", "#2ca02c"
    if accept and not is_correct:
        return "FALSE ACCEPT", "#d62728"
    if not accept and is_correct:
        return "FALSE REJECT", "#d62728"
    return "True Reject (DROP)", "#2ca02c"


def plot_case(ax, sample_idx: int, labels, posteriors, thresholds, title: str):
    x = np.arange(CFG.num_classes)
    label = labels[sample_idx].item()
    max_values = []
    text_y = 0.92

    for method_name in METHOD_ORDER:
        posterior = posteriors[method_name]
        probs = posterior[sample_idx].detach().cpu().numpy()
        color = METHOD_COLORS.get(method_name, "#999999")
        linewidth = 2.4 if method_name == "gating_mixture" else 1.8
        alpha = 0.95 if method_name == "gating_mixture" else 0.6

        ax.plot(x, probs, linewidth=linewidth, color=color, alpha=alpha, label=DISPLAY_NAMES[method_name])
        ax.fill_between(x, probs, alpha=0.06 if method_name == "gating_mixture" else 0.04, color=color)

        threshold = thresholds[method_name]
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1.5, alpha=0.9)

        max_prob = float(probs.max())
        max_values.append(max_prob)
        pred = int(probs.argmax())
        text, box_color = decision_text(max_prob, threshold, pred == label)
        ax.text(
            0.015,
            text_y,
            f"{DISPLAY_NAMES[method_name]}: {text} (pred={pred}, τ={threshold:.2f})",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.18, edgecolor=box_color),
        )
        text_y -= 0.11

    peak_prob = max(max_values)
    ax.axvline(label, color="#2ca02c", linestyle="-", linewidth=2.5, alpha=0.8)
    ax.annotate(
        f"True class {label}",
        xy=(label, min(0.95, peak_prob * 1.02 + 1e-3)),
        xytext=(label + 4, min(0.98, peak_prob * 1.1 + 1e-3)),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4),
        fontsize=9,
        color="#2ca02c",
    )
    tick_step = 5
    tick_positions = sorted(set(list(range(0, CFG.num_classes, tick_step)) + [label]))
    tick_labels = [f"{t} (label)" if t == label else str(t) for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Posterior Probability")
    ax.set_title(title, fontsize=15, fontweight="bold")
    y_max = min(1.0, max(max_values) * 1.2 + 1e-3)
    ax.set_ylim(0, y_max)
    ax.set_xlim(-2, CFG.num_classes - 1)
    ax.grid(True, linestyle="--", alpha=0.35)


def main():
    labels, is_tail, posteriors = prepare_posteriors()
    thresholds = compute_thresholds(posteriors, CFG.target_rejection)

    idx_case1 = find_case_should_keep(labels, is_tail, posteriors, thresholds)
    idx_case2 = find_case_should_reject(labels, is_tail, posteriors, thresholds)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    plot_case(
        axes[0],
        idx_case1,
        labels,
        posteriors,
        thresholds,
        title=f"Case 1 – Tail class {labels[idx_case1].item()} should be KEPT (CE under-confident, ARE fixes)",
    )

    plot_case(
        axes[1],
        idx_case2,
        labels,
        posteriors,
        thresholds,
        title=f"Case 2 – Tail class {labels[idx_case2].item()} should be REJECTED (LA/BS over-confident, ARE fixes)",
    )

    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linewidth=2.5, label=DISPLAY_NAMES[m]) for m in METHOD_ORDER
    ]
    legend_handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Model threshold"))
    legend_handles.append(Line2D([0], [0], marker="|", color="#2ca02c", linewidth=2.5, markersize=12, label="True class (vertical line)"))
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        fontsize=10,
        frameon=True,
        framealpha=0.9,
    )

    out_path = Path(CFG.results_dir) / "single_sample_full_class_distribution.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved full-class distribution figure: {out_path}")


if __name__ == "__main__":
    main()


