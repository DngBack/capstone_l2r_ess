#!/usr/bin/env python3
"""
Standalone Balanced Plug-in with Gating (3 Experts) per "Learning to Reject Meets Long-Tail Learning"
=======================================================================================================

- Loads CIFAR-100-LT splits and 3 expert logits (CE, LogitAdjust, BalancedSoftmax)
- Loads trained gating network to combine experts
- Builds head/tail groups from train class counts (tail <= 20 samples)
- Implements Theorem 1 decision rules (classifier and rejector)
- Implements Algorithm 1 (power-iteration) over α and 1D λ grid for μ
- Runs theory-compliant cost sweep: optimize (α, μ) per cost, one RC point per cost
- Evaluates on test; computes balanced error RC and AURC; saves JSON and plots

Inputs (expected existing):
- Splits dir: ./data/cifar100_lt_if100_splits_fixed/
- Logits dir: ./outputs/logits/cifar100_lt_if100/{expert_name}/{split}_logits.pt
- Gating checkpoint: ./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth

Outputs:
- results/ltr_plugin/cifar100_lt_if100/ltr_plugin_gating_balanced.json
- results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_balanced_gating_test.png
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ================================
# Config
# ================================
@dataclass
class Config:
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100"
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"
    gating_checkpoint: str = (
        "./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth"
    )

    expert_names: List[str] = (
        None  # Will be set to ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]
    )

    num_classes: int = 100
    num_groups: int = 2

    # Tail definition per paper: tail if train count <= 20
    tail_threshold: int = 20

    # Optimizer settings - extended range including paper values {1, 6, 11}
    mu_lambda_grid: List[float] = (
        -5.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        5.0,
        6.0,
        8.0,
        11.0,
        15.0,
        20.0,
    )
    power_iter_iters: int = 20  # More iterations for convergence
    power_iter_damping: float = 0.5  # Higher damping for stability

    # Cost sweep (theory-compliant): one RC point per cost
    cost_sweep: List[float] = ()  # unused when target_rejections set
    # Target rejection grid to match paper plots exactly
    target_rejections: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    seed: int = 42


# Initialize expert names
def init_config():
    cfg = Config()
    if cfg.expert_names is None:
        cfg.expert_names = [
            "ce_baseline",
            "logitadjust_baseline",
            "balsoftmax_baseline",
        ]
    return cfg


CFG = init_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# Load gating network
# ================================


def load_gating_network():
    """Load the pre-trained gating network."""
    from src.models.gating_network_map import GatingNetwork

    print(f"Loading gating network from: {CFG.gating_checkpoint}")
    checkpoint = torch.load(
        CFG.gating_checkpoint, map_location=DEVICE, weights_only=False
    )

    gating = GatingNetwork(
        num_experts=len(CFG.expert_names), num_classes=CFG.num_classes, routing="dense"
    ).to(DEVICE)

    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()
    print("✓ Gating network loaded successfully")

    return gating


# ================================
# IO helpers
# ================================


def load_expert_logits(split: str, device: str = DEVICE) -> torch.Tensor:
    """Load and stack expert logits for a split."""
    logits_list = []

    for expert_name in CFG.expert_names:
        path = Path(CFG.logits_dir) / expert_name / f"{split}_logits.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing logits: {path}")

        logits_e = torch.load(path, map_location=device).float()
        logits_list.append(logits_e)
        print(f"  Loaded {expert_name}: {logits_e.shape}")

    # Stack: [E, N, C] → transpose → [N, E, C]
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    print(f"✓ Stacked expert logits: {logits.shape}")

    return logits


def compute_mixture_posterior(
    logits: torch.Tensor, gating: nn.Module, device: str = DEVICE
) -> torch.Tensor:
    """
    Compute mixture posterior using gating network.

    Args:
        logits: [N, E, C] tensor of expert logits
        gating: GatingNetwork module

    Returns:
        mixture_posterior: [N, C] tensor
    """
    print(f"Computing mixture posterior from logits shape: {logits.shape}")

    # Convert logits to posteriors
    expert_posteriors = F.softmax(logits, dim=-1)  # [N, E, C]

    # Get gating weights
    with torch.no_grad():
        weights, _ = gating(expert_posteriors)  # [N, E]

    # Compute mixture: η̃(x) = Σ_e w_e · p^(e)(y|x)
    mixture_posterior = torch.sum(
        weights.unsqueeze(-1) * expert_posteriors,
        dim=1,  # Sum over experts
    )  # [N, C]

    print(f"✓ Mixture posterior shape: {mixture_posterior.shape}")

    return mixture_posterior


def load_labels(split: str, device: str = DEVICE) -> torch.Tensor:
    # Prefer saved targets alongside logits
    for expert_name in CFG.expert_names:
        cand = Path(CFG.logits_dir) / expert_name / f"{split}_targets.pt"
        if cand.exists():
            t = torch.load(cand, map_location=device)
            if isinstance(t, torch.Tensor):
                return t.to(device=device, dtype=torch.long)

    # Fallback: reconstruct from CIFAR100 and indices
    import torchvision

    indices_file = Path(CFG.splits_dir) / f"{split}_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        indices = json.load(f)
    is_train = split in ("expert", "gating", "train")
    ds = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    labels = torch.tensor(
        [ds.targets[i] for i in indices], dtype=torch.long, device=device
    )
    return labels


def load_class_weights(device: str = DEVICE) -> torch.Tensor:
    """Load inverse class weights for importance weighting to re-weight test set to training distribution."""
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]

    counts = np.array(class_counts, dtype=np.float64)
    total_train = counts.sum()

    # Calculate inverse weights to re-weight test set to training distribution
    train_probs = counts / total_train
    test_probs = np.ones(CFG.num_classes) / CFG.num_classes  # Balanced test set

    # Importance weights = train_probs / test_probs = train_probs * num_classes
    weights = train_probs * CFG.num_classes

    print(
        f"Training distribution: head={train_probs[0]:.6f}, tail={train_probs[-1]:.6f}"
    )
    print(f"Test distribution (balanced): {1.0 / CFG.num_classes:.6f}")
    print(f"Importance weights: head={weights[0]:.6f}, tail={weights[-1]:.6f}")
    print(f"Weight ratio (head/tail): {weights[0] / weights[-1]:.1f}x")

    return torch.tensor(weights, dtype=torch.float32, device=device)


def ensure_dirs():
    Path(CFG.results_dir).mkdir(parents=True, exist_ok=True)


# ================================
# Group construction per paper
# ================================


def build_class_to_group() -> torch.Tensor:
    counts_path = Path(CFG.splits_dir) / "train_class_counts.json"
    with open(counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)
    if isinstance(class_counts, dict):
        class_counts = [class_counts[str(i)] for i in range(CFG.num_classes)]
    counts = np.array(class_counts)
    tail_mask = counts <= CFG.tail_threshold
    class_to_group = np.zeros(CFG.num_classes, dtype=np.int64)
    class_to_group[tail_mask] = 1  # 0=head, 1=tail
    print(
        f"Groups: head={(class_to_group == 0).sum()}, tail={(class_to_group == 1).sum()}"
    )
    return torch.tensor(class_to_group, dtype=torch.long, device=DEVICE)


# ================================
# Plug-in model (Theorem 1)
# ================================
class BalancedLtRPlugin(nn.Module):
    def __init__(self, class_to_group: torch.Tensor):
        super().__init__()
        self.class_to_group = class_to_group  # [C]
        num_groups = int(class_to_group.max().item() + 1)
        self.register_buffer("alpha_group", torch.ones(num_groups))  # α[g]
        self.register_buffer("mu_group", torch.zeros(num_groups))  # μ[g]
        self.register_buffer("cost", torch.tensor(0.0))  # c

    def set_params(self, alpha_g: torch.Tensor, mu_g: torch.Tensor, cost: float):
        self.alpha_group = alpha_g.to(self.alpha_group.device)
        self.mu_group = mu_g.to(self.mu_group.device)
        self.cost = torch.tensor(float(cost), device=self.cost.device)

    def _alpha_class(self) -> torch.Tensor:
        return self.alpha_group[self.class_to_group]

    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]

    @torch.no_grad()
    def predict(self, posterior: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        alpha = self._alpha_class().clamp(min=eps)
        reweighted = posterior / alpha.unsqueeze(0)
        return reweighted.argmax(dim=-1)

    @torch.no_grad()
    def reject(
        self, posterior: torch.Tensor, cost: Optional[float] = None
    ) -> torch.Tensor:
        eps = 1e-12
        alpha = self._alpha_class().clamp(min=eps)
        mu = self._mu_class()
        inv_alpha = 1.0 / alpha
        max_reweighted = (posterior * inv_alpha.unsqueeze(0)).max(dim=-1)[0]
        threshold = ((inv_alpha - mu).unsqueeze(0) * posterior).sum(dim=-1)
        c = self.cost.item() if cost is None else float(cost)
        return max_reweighted < (threshold - c)


# ================================
# Metrics
# ================================
@torch.no_grad()
def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    reject: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    accept = ~reject
    if accept.sum() == 0:
        num_groups = int(class_to_group.max().item() + 1)
        return {
            "selective_error": 1.0,
            "coverage": 0.0,
            "group_errors": [1.0] * num_groups,
            "balanced_error": 1.0,
            "worst_group_error": 1.0,
        }
    preds_a = preds[accept]
    labels_a = labels[accept]
    errors = (preds_a != labels_a).float()
    selective_error = errors.mean().item()
    coverage = accept.float().mean().item()
    groups = class_to_group[labels_a]
    num_groups = int(class_to_group.max().item() + 1)

    group_errors = []

    if class_weights is not None:
        device = labels.device
        class_weights = class_weights.to(device)
        print(
            f"DEBUG: Using importance weighting with class_weights shape: {class_weights.shape}"
        )
        print(
            f"DEBUG: Class weights range: {class_weights.min().item():.6f} to {class_weights.max().item():.6f}"
        )

        for g in range(num_groups):
            mask = groups == g

            if mask.sum() == 0:
                group_errors.append(1.0)
            else:
                y_g = labels_a[mask]
                preds_g = preds_a[mask]

                sample_weights = class_weights[y_g]
                errors_in_group = (preds_g != y_g).float()

                weighted_errors = (sample_weights * errors_in_group).sum().item()
                total_weight = sample_weights.sum().item()

                if total_weight > 0:
                    group_errors.append(weighted_errors / total_weight)
                else:
                    group_errors.append(1.0)
    else:
        print("DEBUG: NOT using importance weighting")
        for g in range(num_groups):
            mask = groups == g

            if mask.sum() == 0:
                group_errors.append(1.0)
            else:
                num_errors_in_group = errors[mask].sum().item()
                num_accepted_in_group = mask.sum().item()
                conditional_error = num_errors_in_group / num_accepted_in_group
                group_errors.append(conditional_error)

    balanced_error = float(np.mean(group_errors))
    worst_group_error = float(np.max(group_errors))
    return {
        "selective_error": selective_error,
        "coverage": coverage,
        "group_errors": group_errors,
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
    }


# ================================
# Algorithm 1 (Power-iteration)
# ================================
@torch.no_grad()
def initialize_alpha(labels: torch.Tensor, class_to_group: torch.Tensor) -> np.ndarray:
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    N = len(labels)
    for g in range(K):
        group_mask = class_to_group[labels] == g
        alpha[g] = group_mask.sum().float().item() / N
    return alpha


@torch.no_grad()
def update_alpha_from_coverage(
    reject: torch.Tensor, labels: torch.Tensor, class_to_group: torch.Tensor
) -> np.ndarray:
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    accept = ~reject
    N = len(labels)
    if accept.sum() == 0:
        return np.ones(K, dtype=np.float64) * 0.5

    for g in range(K):
        in_group = class_to_group[labels] == g
        accepted_in_group = accept & in_group
        empirical_cov = accepted_in_group.sum().float().item() / max(N, 1)
        alpha[g] = float(np.clip(empirical_cov, 1e-6, 1.0))
    return alpha


@torch.no_grad()
def power_iter_search(
    plugin: BalancedLtRPlugin,
    posterior: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    mu: np.ndarray,
    cost: float,
    num_iters: int,
    damping: float,
    class_weights: Optional[torch.Tensor] = None,
    verbose: bool = False,
    target_rejection: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    alpha = initialize_alpha(labels, class_to_group)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    K = int(class_to_group.max().item() + 1)
    beta_scale = 1.0 / float(K)

    for it in range(num_iters):
        alpha_hat = alpha.astype(np.float32)
        alpha_t = torch.tensor(alpha_hat, dtype=torch.float32, device=DEVICE)
        c_it = cost
        if target_rejection is not None:
            c_it = compute_cost_for_target_rejection(
                posterior, class_to_group, alpha, mu, target_rejection
            )
        plugin.set_params(alpha_t, mu_t, c_it)
        preds = plugin.predict(posterior)
        rej = plugin.reject(posterior)
        alpha_new = update_alpha_from_coverage(rej, labels, class_to_group)
        if damping > 0.0:
            alpha = (1.0 - damping) * alpha + damping * alpha_new
        else:
            alpha = alpha_new
        if verbose and (it % 10 == 0 or it == num_iters - 1):
            m = compute_metrics(preds, labels, rej, class_to_group, class_weights)
            print(
                f"   [PI] iter={it + 1} cov={m['coverage']:.3f} bal={m['balanced_error']:.4f}"
            )
        if np.max(np.abs(alpha_new - alpha)) < 1e-4:
            break

    alpha_hat = alpha.astype(np.float32)
    alpha_t = torch.tensor(alpha_hat, dtype=torch.float32, device=DEVICE)
    c_fin = cost
    if target_rejection is not None:
        c_fin = compute_cost_for_target_rejection(
            posterior, class_to_group, alpha, mu, target_rejection
        )
    plugin.set_params(alpha_t, mu_t, c_fin)
    preds = plugin.predict(posterior)
    rej = plugin.reject(posterior)
    metrics = compute_metrics(preds, labels, rej, class_to_group, class_weights)
    return alpha, metrics


@torch.no_grad()
def compute_cost_for_target_rejection(
    posterior: torch.Tensor,
    class_to_group: torch.Tensor,
    alpha: np.ndarray,
    mu: np.ndarray,
    target_rejection: float,
) -> float:
    """Compute cost c to achieve a target rejection rate r using posterior."""
    eps = 1e-12
    K = int(class_to_group.max().item() + 1)
    C = int(class_to_group.numel())
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    if alpha_t.numel() == K:
        alpha_t = alpha_t[class_to_group]
        mu_t = mu_t[class_to_group]

    inv_alpha = 1.0 / alpha_t.clamp(min=eps)
    max_rew = (posterior * inv_alpha.unsqueeze(0)).max(dim=-1)[0]
    thresh_base = ((inv_alpha - mu_t).unsqueeze(0) * posterior).sum(dim=-1)
    t = thresh_base - max_rew
    t_sorted = torch.sort(t)[0]
    q = max(0.0, min(1.0, 1.0 - float(target_rejection)))
    idx = int(round(q * (len(t_sorted) - 1)))
    return float(t_sorted[idx].item())


# ================================
# RC curve (balanced error) and AURC
# ================================
@torch.no_grad()
def compute_rc_curve(
    plugin: BalancedLtRPlugin,
    posterior: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    alpha: np.ndarray,
    mu: np.ndarray,
    cost_grid: List[float],
) -> Dict[str, np.ndarray]:
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    rejection_rates, balanced_errors = [], []
    for c in cost_grid:
        plugin.set_params(alpha_t, mu_t, c)
        preds = plugin.predict(posterior)
        rej = plugin.reject(posterior)
        m = compute_metrics(preds, labels, rej, class_to_group)
        rejection_rates.append(1.0 - m["coverage"])
        balanced_errors.append(m["balanced_error"])
    r = np.array(rejection_rates)
    e = np.array(balanced_errors)
    idx = np.argsort(r)
    r, e = r[idx], e[idx]
    aurc = float(np.trapz(e, r))
    return {"rejection_rates": r, "balanced_errors": e, "aurc": aurc}


# ================================
# Plotting
# ================================


def plot_rc(rc: Dict[str, np.ndarray], out_path: Path):
    r = rc["rejection_rates"]
    e_balanced = rc["balanced_errors"]
    aurc_balanced = rc["aurc"]

    # Extract worst errors and group errors if available
    e_worst = rc.get("worst_errors", None)
    aurc_worst = rc.get("aurc_worst", None)
    head_errors = rc.get("head_errors", None)
    tail_errors = rc.get("tail_errors", None)

    # Calculate tail - head error gap
    error_gap = None
    aurc_gap = None
    if head_errors is not None and tail_errors is not None:
        error_gap = tail_errors - head_errors
        if error_gap.size > 1:
            aurc_gap = float(np.trapz(error_gap, r))

    # Create figure with subplots
    if head_errors is not None or tail_errors is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        ax1, ax2, ax3 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        ax2 = None
        ax3 = None

    # Main RC curve plot (left subplot)
    ax1.plot(
        r,
        e_balanced,
        "o-",
        color="green",
        linewidth=2,
        markersize=6,
        label=f"Balanced Error (AURC={aurc_balanced:.4f})",
    )

    if e_worst is not None:
        ax1.plot(
            r,
            e_worst,
            "s--",
            color="red",
            linewidth=2,
            markersize=5,
            label=f"Worst Group Error (AURC={aurc_worst:.4f})"
            if aurc_worst
            else "Worst Group Error",
        )

    ax1.set_xlabel("Proportion of Rejections", fontsize=11)
    ax1.set_ylabel("Error Rate", fontsize=11)
    ax1.set_title(
        "Balanced Error vs Rejection Rate (Gating)", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=10)
    ax1.set_xlim([0, 1])
    y_max = max(
        e_balanced.max() if e_balanced.size > 0 else 0,
        e_worst.max() if e_worst is not None and e_worst.size > 0 else 0,
    )
    ax1.set_ylim([0, min(1.05, y_max * 1.1)])

    # Group errors plot (middle subplot)
    if ax2 is not None and head_errors is not None and tail_errors is not None:
        ax2.plot(
            r,
            head_errors,
            "o-",
            color="blue",
            linewidth=2,
            markersize=6,
            label="Head Group Error",
        )
        ax2.plot(
            r,
            tail_errors,
            "s-",
            color="orange",
            linewidth=2,
            markersize=6,
            label="Tail Group Error",
        )

        ax2.set_xlabel("Proportion of Rejections", fontsize=11)
        ax2.set_ylabel("Group Error Rate", fontsize=11)
        ax2.set_title("Head vs Tail Group Errors", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=10)
        ax2.set_xlim([0, 1])
        y_max_groups = max(
            head_errors.max() if head_errors.size > 0 else 0,
            tail_errors.max() if tail_errors.size > 0 else 0,
        )
        ax2.set_ylim([0, min(1.05, y_max_groups * 1.1)])

    # Error gap plot (right subplot)
    if ax3 is not None and error_gap is not None:
        # Positive values mean tail > head (tail worse)
        # Negative values mean tail < head (head worse)
        color = "purple"
        ax3.plot(
            r,
            error_gap,
            "^-",
            color=color,
            linewidth=2,
            markersize=6,
            label=f"Tail - Head Gap (AUC={aurc_gap:.4f})"
            if aurc_gap
            else "Tail - Head Gap",
        )
        # Add horizontal line at y=0 for reference
        ax3.axhline(
            y=0, color="black", linestyle=":", linewidth=1, alpha=0.5, label="Zero Gap"
        )

        ax3.set_xlabel("Proportion of Rejections", fontsize=11)
        ax3.set_ylabel("Error Gap (Tail - Head)", fontsize=11)
        ax3.set_title("Error Gap: Tail vs Head", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best", fontsize=10)
        ax3.set_xlim([0, 1])

        # Auto-scale y-axis based on data
        gap_max = abs(error_gap.max()) if error_gap.size > 0 else 0
        gap_min = abs(error_gap.min()) if error_gap.size > 0 else 0
        y_limit = max(gap_max, gap_min) * 1.2
        ax3.set_ylim(
            [-y_limit if y_limit > 0 else -0.1, y_limit if y_limit > 0 else 0.1]
        )

        # Add text annotation for interpretation
        if error_gap.max() > 0:
            ax3.text(
                0.02,
                0.98,
                "Positive = Tail worse",
                transform=ax3.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ================================
# Main
# ================================


def main():
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    ensure_dirs()

    print("\n" + "=" * 70)
    print("LOADING DATA AND GATING")
    print("=" * 70)

    # Load gating network
    gating = load_gating_network()

    # Load logits and labels for all splits
    print("\nLoading expert logits...")
    logits_tunev = load_expert_logits("tunev", DEVICE)
    logits_val = load_expert_logits("val", DEVICE)
    logits_test = load_expert_logits("test", DEVICE)

    print("\nLoading labels...")
    labels_tunev = load_labels("tunev", DEVICE)
    labels_val = load_labels("val", DEVICE)
    labels_test = load_labels("test", DEVICE)

    print("\nComputing mixture posteriors using gating network...")
    posterior_tunev = compute_mixture_posterior(logits_tunev, gating, DEVICE)
    posterior_val = compute_mixture_posterior(logits_val, gating, DEVICE)
    posterior_test = compute_mixture_posterior(logits_test, gating, DEVICE)

    print("Building class-to-group mapping (tail <= 20)...")
    class_to_group = build_class_to_group()

    print("Loading class weights for importance weighting...")
    class_weights = load_class_weights(DEVICE)

    # Check baseline balanced error on test set
    ce_pred_test = posterior_test.argmax(dim=-1)
    groups_test = class_to_group[labels_test]
    num_groups = int(class_to_group.max().item() + 1)

    dummy_reject = torch.zeros(len(labels_test), dtype=torch.bool, device=DEVICE)
    baseline_metrics = compute_metrics(
        ce_pred_test, labels_test, dummy_reject, class_to_group, class_weights
    )

    baseline_balanced_error = baseline_metrics["balanced_error"]
    print(f"\nBaseline Gating balanced error (TEST) = {baseline_balanced_error:.4f}")
    print(f"Baseline Gating group errors = {baseline_metrics['group_errors']}")
    print(
        f"Baseline Gating overall accuracy (TEST) = {(ce_pred_test == labels_test).float().mean().item():.4f}"
    )

    print("\nCreating plug-in model...")
    plugin = BalancedLtRPlugin(class_to_group).to(DEVICE)

    # Target rejection grid (paper-style points)
    results_per_cost: List[Dict] = []
    targets = list(CFG.target_rejections)
    for i, target_rej in enumerate(targets):
        print(f"\n=== Target {i + 1}/{len(targets)}: rejection={target_rej:.1f} ===")

        print("   Step 1: Optimizing (alpha, mu) on tunev for each mu...")
        candidates = []
        for lam in CFG.mu_lambda_grid:
            mu = np.array([0.0, float(lam)], dtype=np.float64)
            alpha_found, _ = power_iter_search(
                plugin,
                posterior_tunev,
                labels_tunev,
                class_to_group,
                mu=mu,
                cost=0.0,
                num_iters=CFG.power_iter_iters,
                damping=CFG.power_iter_damping,
                class_weights=class_weights,
                verbose=False,
                target_rejection=target_rej,
            )
            candidates.append((alpha_found, mu))
            print(f"     mu={lam:5.1f}: alpha={alpha_found}")

        print("   Step 2: Selecting best mu based on val performance...")
        best = {
            "objective": float("inf"),
            "alpha": None,
            "mu": None,
            "val_metrics": None,
        }

        for alpha, mu in candidates:
            K = int(class_to_group.max().item() + 1)
            alpha_eval = alpha.astype(np.float32)

            cost_val = compute_cost_for_target_rejection(
                posterior_val, class_to_group, alpha, mu, target_rej
            )

            plugin.set_params(
                torch.tensor(alpha_eval, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                float(cost_val),
            )
            preds_val = plugin.predict(posterior_val)
            rej_val = plugin.reject(posterior_val)
            m_val = compute_metrics(
                preds_val, labels_val, rej_val, class_to_group, class_weights
            )

            print(
                f"     mu={mu[1]:5.1f}: val_bal={m_val['balanced_error']:.4f} val_cov={m_val['coverage']:.3f}"
            )

            if m_val["balanced_error"] < best["objective"]:
                best = {
                    "objective": m_val["balanced_error"],
                    "alpha": alpha,
                    "mu": mu,
                    "val_metrics": m_val,
                }

        # Local refinement around best mu
        print("   Step 3: Local refinement around best mu...")
        best_lam = float(best["mu"][1])
        refine_step = 2.0
        for refine_iter in range(4):
            tried = []
            for lam in (best_lam - refine_step, best_lam + refine_step):
                mu = np.array([0.0, float(lam)], dtype=np.float64)
                alpha_found, _ = power_iter_search(
                    plugin,
                    posterior_tunev,
                    labels_tunev,
                    class_to_group,
                    mu=mu,
                    cost=0.0,
                    num_iters=CFG.power_iter_iters,
                    damping=CFG.power_iter_damping,
                    class_weights=class_weights,
                    verbose=False,
                    target_rejection=target_rej,
                )

                alpha_eval = alpha_found.astype(np.float32)
                cost_val = compute_cost_for_target_rejection(
                    posterior_val, class_to_group, alpha_found, mu, target_rej
                )
                plugin.set_params(
                    torch.tensor(alpha_eval, dtype=torch.float32, device=DEVICE),
                    torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                    float(cost_val),
                )
                preds_val = plugin.predict(posterior_val)
                rej_val = plugin.reject(posterior_val)
                m_val = compute_metrics(
                    preds_val, labels_val, rej_val, class_to_group, class_weights
                )

                tried.append((lam, m_val["balanced_error"], alpha_found, mu, m_val))

            lam_better, obj_better, alpha_better, mu_better, metr_better = min(
                tried, key=lambda x: x[1]
            )
            if obj_better < best["objective"]:
                best = {
                    "objective": obj_better,
                    "alpha": alpha_better,
                    "mu": mu_better,
                    "val_metrics": metr_better,
                }
                best_lam = lam_better
                print(
                    f"     Refine {refine_iter + 1}: mu={lam_better:.2f} val_bal={obj_better:.4f}"
                )
            refine_step *= 0.5

        print(
            f"   Final selection: mu={best['mu'][1]:.2f} val_bal={best['val_metrics']['balanced_error']:.4f} val_cov={best['val_metrics']['coverage']:.3f}"
        )
        print(f"   Best alpha: {best['alpha']}")

        # Evaluate best (alpha, mu, c) on test
        alpha_best = np.array(best["alpha"], dtype=np.float64)
        mu_best = np.array(best["mu"], dtype=np.float64)

        m_val = best["val_metrics"]

        cost_test = compute_cost_for_target_rejection(
            posterior_test, class_to_group, alpha_best, mu_best, target_rej
        )

        K = int(class_to_group.max().item() + 1)
        alpha_eval = alpha_best.astype(np.float32)
        plugin.set_params(
            torch.tensor(alpha_eval, dtype=torch.float32, device=DEVICE),
            torch.tensor(mu_best, dtype=torch.float32, device=DEVICE),
            float(cost_test),
        )
        preds_test = plugin.predict(posterior_test)
        rej_test = plugin.reject(posterior_test)
        m_test = compute_metrics(
            preds_test, labels_test, rej_test, class_to_group, class_weights
        )

        print(
            f"   Target={target_rej:.1f}  VAL: bal={m_val['balanced_error']:.4f} cov={m_val['coverage']:.3f}"
        )
        print(
            f"   Target={target_rej:.1f}  TEST: bal={m_test['balanced_error']:.4f} cov={m_test['coverage']:.3f}"
        )

        results_per_cost.append(
            {
                "target_rejection": float(target_rej),
                "cost_val": float(
                    compute_cost_for_target_rejection(
                        posterior_val, class_to_group, alpha_best, mu_best, target_rej
                    )
                ),
                "cost_test": float(cost_test),
                "alpha": alpha_best.tolist(),
                "mu": mu_best.tolist(),
                "selection_method": "val_based",
                "val_metrics": {
                    "balanced_error": float(m_val["balanced_error"]),
                    "coverage": float(m_val["coverage"]),
                    "rejection_rate": float(1.0 - m_val["coverage"]),
                    "group_errors": [float(x) for x in m_val["group_errors"]],
                    "worst_group_error": float(m_val["worst_group_error"]),
                },
                "test_metrics": {
                    "balanced_error": float(m_test["balanced_error"]),
                    "coverage": float(m_test["coverage"]),
                    "rejection_rate": float(1.0 - m_test["coverage"]),
                    "group_errors": [float(x) for x in m_test["group_errors"]],
                    "worst_group_error": float(m_test["worst_group_error"]),
                },
            }
        )

    # Build unified RC curve (balanced) from target points
    r_val = np.array([1.0 - r["val_metrics"]["coverage"] for r in results_per_cost])
    e_val = np.array([r["val_metrics"]["balanced_error"] for r in results_per_cost])
    e_worst_val = np.array(
        [r["val_metrics"]["worst_group_error"] for r in results_per_cost]
    )
    head_errors_val = np.array(
        [r["val_metrics"]["group_errors"][0] for r in results_per_cost]
    )
    tail_errors_val = np.array(
        [r["val_metrics"]["group_errors"][1] for r in results_per_cost]
    )

    r_test = np.array([1.0 - r["test_metrics"]["coverage"] for r in results_per_cost])
    e_test = np.array([r["test_metrics"]["balanced_error"] for r in results_per_cost])
    e_worst_test = np.array(
        [r["test_metrics"]["worst_group_error"] for r in results_per_cost]
    )
    head_errors_test = np.array(
        [r["test_metrics"]["group_errors"][0] for r in results_per_cost]
    )
    tail_errors_test = np.array(
        [r["test_metrics"]["group_errors"][1] for r in results_per_cost]
    )

    idx_v = np.argsort(r_val)
    r_val, e_val = r_val[idx_v], e_val[idx_v]
    e_worst_val = e_worst_val[idx_v]
    head_errors_val = head_errors_val[idx_v]
    tail_errors_val = tail_errors_val[idx_v]

    idx_t = np.argsort(r_test)
    r_test, e_test = r_test[idx_t], e_test[idx_t]
    e_worst_test = e_worst_test[idx_t]
    head_errors_test = head_errors_test[idx_t]
    tail_errors_test = tail_errors_test[idx_t]

    aurc_val = (
        float(np.trapz(e_val, r_val))
        if r_val.size > 1
        else float(e_val.mean() if e_val.size else 0.0)
    )
    aurc_worst_val = (
        float(np.trapz(e_worst_val, r_val))
        if r_val.size > 1
        else float(e_worst_val.mean() if e_worst_val.size else 0.0)
    )
    aurc_test = (
        float(np.trapz(e_test, r_test))
        if r_test.size > 1
        else float(e_test.mean() if e_test.size else 0.0)
    )
    aurc_worst_test = (
        float(np.trapz(e_worst_test, r_test))
        if r_test.size > 1
        else float(e_worst_test.mean() if e_worst_test.size else 0.0)
    )

    # Calculate error gap (tail - head) for both val and test
    error_gap_test = tail_errors_test - head_errors_test
    error_gap_val = tail_errors_val - head_errors_val
    aurc_gap_test = (
        float(np.trapz(error_gap_test, r_test))
        if error_gap_test.size > 1
        else float(error_gap_test.mean() if error_gap_test.size else 0.0)
    )
    aurc_gap_val = (
        float(np.trapz(error_gap_val, r_val))
        if error_gap_val.size > 1
        else float(error_gap_val.mean() if error_gap_val.size else 0.0)
    )

    save_dict = {
        "objective": "balanced",
        "description": "Gating with 3 experts (CE, LogitAdjust, BalancedSoftmax) - Targeted rejection grid with val-based hyperparameter selection per paper Algorithm 1",
        "method": "plug-in_balanced_val_selection_with_gating",
        "experts": CFG.expert_names,
        "hyperparameter_selection": "val_based",
        "algorithm": "Algorithm 1 from paper - optimize (α,μ) on tunev, select μ on val",
        "results_per_cost": results_per_cost,
        "rc_curve": {
            "val": {
                "rejection_rates": r_val.tolist(),
                "balanced_errors": e_val.tolist(),
                "worst_errors": e_worst_val.tolist(),
                "head_errors": head_errors_val.tolist(),
                "tail_errors": tail_errors_val.tolist(),
                "error_gap": error_gap_val.tolist(),
                "aurc": aurc_val,
                "aurc_worst": aurc_worst_val,
                "aurc_gap": aurc_gap_val,
            },
            "test": {
                "rejection_rates": r_test.tolist(),
                "balanced_errors": e_test.tolist(),
                "worst_errors": e_worst_test.tolist(),
                "head_errors": head_errors_test.tolist(),
                "tail_errors": tail_errors_test.tolist(),
                "error_gap": error_gap_test.tolist(),
                "aurc": aurc_test,
                "aurc_worst": aurc_worst_test,
                "aurc_gap": aurc_gap_test,
            },
        },
    }

    out_json = Path(CFG.results_dir) / "ltr_plugin_gating_balanced.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)
    print(f"\n✓ Saved results to: {out_json}")

    # Plot test RC curve with worst error and head/tail errors
    plot_path = (
        Path(CFG.results_dir) / "ltr_rc_curves_balanced_gating_test_old_iter.png"
    )
    plot_rc(
        {
            "rejection_rates": r_test,
            "balanced_errors": e_test,
            "worst_errors": e_worst_test,
            "head_errors": head_errors_test,
            "tail_errors": tail_errors_test,
            "error_gap": error_gap_test,
            "aurc": aurc_test,
            "aurc_worst": aurc_worst_test,
            "aurc_gap": aurc_gap_test,
        },
        plot_path,
    )
    print(f"✓ Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
