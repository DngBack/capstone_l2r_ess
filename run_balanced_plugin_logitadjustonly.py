#!/usr/bin/env python3
"""
Balanced Plug-in with ONLY LogitAdjust expert (no gating)
=========================================================

- Loads CIFAR-100-LT splits and uses ONLY logitadjust expert logits
- Builds head/tail groups from train class counts (tail <= 20 samples)
- Implements Theorem 1 decision rules (classifier and rejector)
- Implements Algorithm 1 (power-iteration) over α and 1D λ grid for μ
- Runs theory-compliant cost sweep: optimize (α, μ) per cost, one RC point per cost
- Evaluates on test; computes balanced error RC and AURC; saves JSON and plots

Inputs (expected existing):
- Splits dir: ./data/cifar100_lt_if100_splits_fixed/
- Logits dir: ./outputs/logits/cifar100_lt_if100/logitadjust_baseline/{split}_logits.pt

Outputs:
- results/ltr_plugin/cifar100_lt_if100/ltr_plugin_logitadjustonly_balanced.json
- results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_balanced_logitadjustonly_test.png
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
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

    # Use only logitadjust expert
    expert_name: str = "logitadjust_baseline"

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

    # Target rejection grid to match paper plots exactly
    target_rejections: List[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

    seed: int = 42


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# IO helpers
# ================================


def load_expert_logits(split: str, device: str = DEVICE) -> torch.Tensor:
    """Load logits from single expert."""
    path = Path(CFG.logits_dir) / CFG.expert_name / f"{split}_logits.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing logits: {path}")
    logits = torch.load(path, map_location=device).float()
    print(f"✓ Loaded {CFG.expert_name} logits: {logits.shape}")
    return logits


def load_labels(split: str, device: str = DEVICE) -> torch.Tensor:
    # Prefer saved targets alongside logits
    cand = Path(CFG.logits_dir) / CFG.expert_name / f"{split}_targets.pt"
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

    # CORRECTED: Calculate inverse weights to re-weight test set to training distribution
    # Test set is balanced (1/num_classes per class), training set is long-tail
    train_probs = counts / total_train
    test_probs = np.ones(CFG.num_classes) / CFG.num_classes  # Balanced test set

    # Importance weights = train_probs / test_probs = train_probs * num_classes
    # This up-weights head classes and down-weights tail classes
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

    def _alpha_hat_class(self) -> torch.Tensor:
        # α̂_k = α_k · β_k ; for balanced β_k = 1/K ⇒ α̂ = α / K
        K = float(self.alpha_group.numel())
        alpha_hat_group = self.alpha_group / max(K, 1.0)
        return alpha_hat_group[self.class_to_group]

    @torch.no_grad()
    def predict(self, posterior: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        reweighted = posterior / alpha_hat.unsqueeze(0)
        return reweighted.argmax(dim=-1)

    @torch.no_grad()
    def reject(
        self, posterior: torch.Tensor, cost: Optional[float] = None
    ) -> torch.Tensor:
        eps = 1e-12
        alpha_hat = self._alpha_hat_class().clamp(min=eps)
        mu = self._mu_class()
        inv_alpha_hat = 1.0 / alpha_hat
        max_reweighted = (posterior * inv_alpha_hat.unsqueeze(0)).max(dim=-1)[0]
        threshold = ((inv_alpha_hat - mu).unsqueeze(0) * posterior).sum(dim=-1)
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
        prop = group_mask.sum().float().item() / max(N, 1)
        alpha[g] = float(K * prop)
    return alpha


@torch.no_grad()
def update_alpha_from_coverage(
    reject: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> np.ndarray:
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    accept = ~reject
    N = len(labels)
    if accept.sum() == 0:
        return np.ones(K, dtype=np.float64) * 0.5

    if class_weights is not None:
        cw = class_weights.to(labels.device)
        sample_w = cw[labels]
        total_weight = sample_w.sum().item()
        total_weight = max(total_weight, 1e-12)
        for g in range(K):
            in_group = class_to_group[labels] == g
            accepted_in_group = accept & in_group
            w_acc_g = sample_w[accepted_in_group].sum().item()
            cov_g = float(np.clip(w_acc_g / total_weight, 1e-12, 1.0))
            alpha[g] = float(K * cov_g)
        return alpha

    for g in range(K):
        in_group = class_to_group[labels] == g
        accepted_in_group = accept & in_group
        empirical_cov = accepted_in_group.sum().float().item() / max(N, 1)
        alpha[g] = float(K * np.clip(empirical_cov, 1e-6, 1.0))
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
        alpha_new = update_alpha_from_coverage(
            rej, labels, class_to_group, class_weights=class_weights
        )
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
    """Compute cost c to achieve a target rejection rate r."""
    eps = 1e-12
    K = int(class_to_group.max().item() + 1)
    C = int(class_to_group.numel())
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    if alpha_t.numel() == K:
        alpha_group = alpha_t
        mu_group = mu_t
        alpha_hat_group = alpha_group / max(float(K), 1.0)
        alpha_t = alpha_hat_group[class_to_group]
        mu_t = mu_group[class_to_group]
    inv_alpha_hat = 1.0 / alpha_t.clamp(min=eps)
    max_rew = (posterior * inv_alpha_hat.unsqueeze(0)).max(dim=-1)[0]
    thresh_base = ((inv_alpha_hat - mu_t).unsqueeze(0) * posterior).sum(dim=-1)
    t = thresh_base - max_rew
    t_sorted = torch.sort(t)[0]
    q = max(0.0, min(1.0, 1.0 - float(target_rejection)))
    idx = int(round(q * (len(t_sorted) - 1)))
    return float(t_sorted[idx].item())


# ================================
# Plotting
# ================================


def plot_rc_dual(
    r: np.ndarray,
    e_bal: np.ndarray,
    e_wst: np.ndarray,
    aurc_bal: float,
    aurc_wst: float,
    out_path: Path,
):
    # Prepare data for plotting: filter to <= 0.8 and ensure point at 0.8
    max_rejection_plot = 0.8
    mask_plot = r <= max_rejection_plot
    r_plot = r[mask_plot].copy()
    e_bal_plot = e_bal[mask_plot].copy()
    e_wst_plot = e_wst[mask_plot].copy()
    
    # If the last point is less than 0.8, interpolate to add a point at exactly 0.8
    if r_plot.size > 0 and r_plot[-1] < max_rejection_plot:
        e_bal_at_08 = float(np.interp(max_rejection_plot, r, e_bal))
        e_wst_at_08 = float(np.interp(max_rejection_plot, r, e_wst))
        r_plot = np.append(r_plot, max_rejection_plot)
        e_bal_plot = np.append(e_bal_plot, e_bal_at_08)
        e_wst_plot = np.append(e_wst_plot, e_wst_at_08)
    
    plt.figure(figsize=(7, 5))
    plt.plot(r_plot, e_bal_plot, "o-", color="green", label=f"Balanced (AURC={aurc_bal:.4f})")
    plt.plot(
        r_plot, e_wst_plot, "s-", color="royalblue", label=f"Worst-group (AURC={aurc_wst:.4f})"
    )
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Error")
    plt.title("Balanced and Worst-group Error vs Rejection Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 0.8])
    ymax = 0.0
    if e_bal.size:
        ymax = max(ymax, float(e_bal.max()))
    if e_wst.size:
        ymax = max(ymax, float(e_wst.max()))
    plt.ylim([0, min(1.05, ymax * 1.1 if ymax > 0 else 1.0)])
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

    print(f"Loading {CFG.expert_name} logits...")
    expert_logits_tunev = load_expert_logits("tunev", DEVICE)
    labels_tunev = load_labels("tunev", DEVICE)

    expert_logits_val = load_expert_logits("val", DEVICE)
    labels_val = load_labels("val", DEVICE)

    print("Computing posteriors using softmax...")
    posterior_tunev = F.softmax(expert_logits_tunev, dim=-1)
    posterior_val = F.softmax(expert_logits_val, dim=-1)

    print("Building class-to-group mapping (tail <= 20)...")
    class_to_group = build_class_to_group()

    print("Loading class weights for importance weighting...")
    class_weights = load_class_weights(DEVICE)

    # Check baseline balanced error on test set
    expert_logits_test = load_expert_logits("test", DEVICE)
    labels_test = load_labels("test", DEVICE)
    posterior_test = F.softmax(expert_logits_test, dim=-1)

    # Compute baseline balanced error (no rejection) with importance weighting
    mix_pred_test = posterior_test.argmax(dim=-1)
    groups_test = class_to_group[labels_test]
    num_groups = int(class_to_group.max().item() + 1)

    dummy_reject = torch.zeros(len(labels_test), dtype=torch.bool, device=DEVICE)
    baseline_metrics = compute_metrics(
        mix_pred_test, labels_test, dummy_reject, class_to_group, class_weights
    )

    baseline_balanced_error = baseline_metrics["balanced_error"]
    print(f"Baseline {CFG.expert_name} balanced error (TEST) = {baseline_balanced_error:.4f}")
    print(f"Baseline {CFG.expert_name} group errors = {baseline_metrics['group_errors']}")
    print(
        f"Baseline {CFG.expert_name} overall accuracy (TEST) = {(mix_pred_test == labels_test).float().mean().item():.4f}"
    )

    print("Creating plug-in model...")
    plugin = BalancedLtRPlugin(class_to_group).to(DEVICE)

    # Target rejection grid (paper-style points)
    results_per_cost: List[Dict] = []
    targets = list(CFG.target_rejections)
    for i, target_rej in enumerate(targets):
        print(f"\n=== Target {i + 1}/{len(targets)}: rejection={target_rej:.1f} ===")

        # Follow paper Algorithm 1 - optimize on tunev, select on val
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

        # Select best mu based on val performance
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
        K_groups = int(class_to_group.max().item() + 1)
        print(
            f"   Alpha sum: {np.sum(best['alpha']):.4f} (should be ≈ {K_groups * best['val_metrics']['coverage']:.4f})"
        )
        print(f"   Alpha_hat (=alpha/K): {best['alpha'] / float(K_groups)}")

        # Evaluate best (alpha, mu, c) on val and test
        alpha_best = np.array(best["alpha"], dtype=np.float64)
        mu_best = np.array(best["mu"], dtype=np.float64)

        m_val = best["val_metrics"]
        if len(m_val["group_errors"]) >= 2:
            head_err_v = float(m_val["group_errors"][0])
            tail_err_v = float(m_val["group_errors"][1])
            gap_v = tail_err_v - head_err_v
        else:
            head_err_v = tail_err_v = gap_v = float("nan")

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

        wge_v = (
            float(m_val["worst_group_error"])
            if "worst_group_error" in m_val
            else float("nan")
        )
        if len(m_test["group_errors"]) >= 2:
            head_err_t = float(m_test["group_errors"][0])
            tail_err_t = float(m_test["group_errors"][1])
            gap_t = tail_err_t - head_err_t
        else:
            head_err_t = tail_err_t = gap_t = float("nan")
        wge_t = (
            float(m_test["worst_group_error"])
            if "worst_group_error" in m_test
            else float("nan")
        )
        print(
            f"      VAL: worst={wge_v:.4f} | head={head_err_v:.4f} | tail={tail_err_v:.4f} | tail-head={gap_v:.4f}"
        )
        print(
            f"      TEST: worst={wge_t:.4f} | head={head_err_t:.4f} | tail={tail_err_t:.4f} | tail-head={gap_t:.4f}"
        )

        if abs(target_rej - 0.0) < 1e-6:
            print(f"   alpha (group) learned = {alpha_best}")
            mix_pred_test = posterior_test.argmax(dim=-1)
            mix_acc_test = (mix_pred_test == labels_test).float().mean().item()
            plugin_acc_test = 1.0 - m_test["balanced_error"]
            print(f"   Baseline {CFG.expert_name} acc (TEST) = {mix_acc_test:.4f}")
            print(f"   Plugin balanced complement (1-bErr) ~ {plugin_acc_test:.4f}")

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
                    "worst_group_error": float(m_val["worst_group_error"]),
                    "coverage": float(m_val["coverage"]),
                    "rejection_rate": float(1.0 - m_val["coverage"]),
                    "group_errors": [float(x) for x in m_val["group_errors"]],
                },
                "test_metrics": {
                    "balanced_error": float(m_test["balanced_error"]),
                    "worst_group_error": float(m_test["worst_group_error"]),
                    "coverage": float(m_test["coverage"]),
                    "rejection_rate": float(1.0 - m_test["coverage"]),
                    "group_errors": [float(x) for x in m_test["group_errors"]],
                },
            }
        )

    # Build unified RC curve (balanced) from target points
    r_val = np.array([1.0 - r["val_metrics"]["coverage"] for r in results_per_cost])
    e_val = np.array([r["val_metrics"]["balanced_error"] for r in results_per_cost])
    w_val = np.array([r["val_metrics"]["worst_group_error"] for r in results_per_cost])
    gap_val = np.array(
        [
            r["val_metrics"]["group_errors"][1] - r["val_metrics"]["group_errors"][0]
            for r in results_per_cost
        ]
    )
    r_test = np.array([1.0 - r["test_metrics"]["coverage"] for r in results_per_cost])
    e_test = np.array([r["test_metrics"]["balanced_error"] for r in results_per_cost])
    w_test = np.array(
        [r["test_metrics"]["worst_group_error"] for r in results_per_cost]
    )
    gap_test = np.array(
        [
            r["test_metrics"]["group_errors"][1] - r["test_metrics"]["group_errors"][0]
            for r in results_per_cost
        ]
    )

    idx_v = np.argsort(r_val)
    r_val, e_val, w_val, gap_val = (
        r_val[idx_v],
        e_val[idx_v],
        w_val[idx_v],
        gap_val[idx_v],
    )
    idx_t = np.argsort(r_test)
    r_test, e_test, w_test, gap_test = (
        r_test[idx_t],
        e_test[idx_t],
        w_test[idx_t],
        gap_test[idx_t],
    )

    # Filter to only include rejection rates <= 0.8 for AURC calculation
    # and ensure we have a point at exactly 0.8 by interpolation if needed
    max_rejection = 0.8
    
    # For val set
    mask_val_aurc = r_val <= max_rejection
    r_val_aurc = r_val[mask_val_aurc].copy()
    e_val_aurc = e_val[mask_val_aurc].copy()
    w_val_aurc = w_val[mask_val_aurc].copy()
    
    if r_val_aurc.size > 0 and r_val_aurc[-1] < max_rejection:
        e_val_at_08 = float(np.interp(max_rejection, r_val, e_val))
        w_val_at_08 = float(np.interp(max_rejection, r_val, w_val))
        r_val_aurc = np.append(r_val_aurc, max_rejection)
        e_val_aurc = np.append(e_val_aurc, e_val_at_08)
        w_val_aurc = np.append(w_val_aurc, w_val_at_08)
    
    # For test set
    mask_test_aurc = r_test <= max_rejection
    r_test_aurc = r_test[mask_test_aurc].copy()
    e_test_aurc = e_test[mask_test_aurc].copy()
    w_test_aurc = w_test[mask_test_aurc].copy()
    
    if r_test_aurc.size > 0 and r_test_aurc[-1] < max_rejection:
        e_test_at_08 = float(np.interp(max_rejection, r_test, e_test))
        w_test_at_08 = float(np.interp(max_rejection, r_test, w_test))
        r_test_aurc = np.append(r_test_aurc, max_rejection)
        e_test_aurc = np.append(e_test_aurc, e_test_at_08)
        w_test_aurc = np.append(w_test_aurc, w_test_at_08)
    
    # Calculate AURC from 0 to 0.8
    aurc_val_bal = (
        float(np.trapz(e_val_aurc, r_val_aurc))
        if r_val_aurc.size > 1
        else float(e_val_aurc.mean() if e_val_aurc.size else 0.0)
    )
    aurc_test_bal = (
        float(np.trapz(e_test_aurc, r_test_aurc))
        if r_test_aurc.size > 1
        else float(e_test_aurc.mean() if e_test_aurc.size else 0.0)
    )
    aurc_val_wst = (
        float(np.trapz(w_val_aurc, r_val_aurc))
        if r_val_aurc.size > 1
        else float(w_val_aurc.mean() if w_val_aurc.size else 0.0)
    )
    aurc_test_wst = (
        float(np.trapz(w_test_aurc, r_test_aurc))
        if r_test_aurc.size > 1
        else float(w_test_aurc.mean() if w_test_aurc.size else 0.0)
    )
    
    # Keep old variable names for backward compatibility
    aurc_val_bal_08 = aurc_val_bal
    aurc_val_wst_08 = aurc_val_wst
    aurc_test_bal_08 = aurc_test_bal
    aurc_test_wst_08 = aurc_test_wst

    save_dict = {
        "objectives": ["balanced", "worst_group"],
        "description": f"Balanced plug-in with ONLY {CFG.expert_name}. Targeted rejection grid (0.0..0.8) with val-based hyperparameter selection per paper Algorithm 1.",
        "method": "plug-in_balanced_val_selection_single_expert",
        "expert_name": CFG.expert_name,
        "hyperparameter_selection": "val_based",
        "algorithm": "Algorithm 1 from paper - optimize (α,μ) on tunev, select μ on val",
        "results_per_cost": results_per_cost,
        "rc_curve": {
            "val": {
                "rejection_rates": r_val.tolist(),
                "balanced_errors": e_val.tolist(),
                "worst_group_errors": w_val.tolist(),
                "tail_minus_head": gap_val.tolist(),
                "aurc_balanced": aurc_val_bal,
                "aurc_worst_group": aurc_val_wst,
                "aurc_balanced_coverage_ge_0_2": aurc_val_bal_08,
                "aurc_worst_group_coverage_ge_0_2": aurc_val_wst_08,
            },
            "test": {
                "rejection_rates": r_test.tolist(),
                "balanced_errors": e_test.tolist(),
                "worst_group_errors": w_test.tolist(),
                "tail_minus_head": gap_test.tolist(),
                "aurc_balanced": aurc_test_bal,
                "aurc_worst_group": aurc_test_wst,
                "aurc_balanced_coverage_ge_0_2": aurc_test_bal_08,
                "aurc_worst_group_coverage_ge_0_2": aurc_test_wst_08,
            },
        },
    }

    out_json = Path(CFG.results_dir) / "ltr_plugin_logitadjustonly_balanced.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)
    print(f"Saved results to: {out_json}")

    # Print AURCs
    print(f"Val AURC - Balanced: {aurc_val_bal:.4f} | Worst-group: {aurc_val_wst:.4f}")
    print(
        f"Test AURC - Balanced: {aurc_test_bal:.4f} | Worst-group: {aurc_test_wst:.4f}"
    )
    print(
        f"Val AURC (coverage>=0.2) - Balanced: {aurc_val_bal_08:.4f} | Worst-group: {aurc_val_wst_08:.4f}"
    )
    print(
        f"Test AURC (coverage>=0.2) - Balanced: {aurc_test_bal_08:.4f} | Worst-group: {aurc_test_wst_08:.4f}"
    )

    # Plot test RC curves (both metrics)
    plot_path = Path(CFG.results_dir) / "ltr_rc_curves_balanced_logitadjustonly_test.png"
    plot_rc_dual(r_test, e_test, w_test, aurc_test_bal, aurc_test_wst, plot_path)
    print(f"Saved combined plot to: {plot_path}")

    # Plot Tail - Head gap curve
    gap_plot_path = Path(CFG.results_dir) / "ltr_tail_minus_head_logitadjustonly_test.png"
    # Prepare data for plotting: filter to <= 0.8 and ensure point at 0.8
    max_rejection_plot = 0.8
    mask_gap_plot = r_test <= max_rejection_plot
    r_gap_plot = r_test[mask_gap_plot].copy()
    gap_plot = gap_test[mask_gap_plot].copy()
    
    if r_gap_plot.size > 0 and r_gap_plot[-1] < max_rejection_plot:
        gap_at_08 = float(np.interp(max_rejection_plot, r_test, gap_test))
        r_gap_plot = np.append(r_gap_plot, max_rejection_plot)
        gap_plot = np.append(gap_plot, gap_at_08)
    
    plt.figure(figsize=(7, 5))
    plt.plot(r_gap_plot, gap_plot, "d-", color="crimson", label="Tail - Head error")
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Tail Error - Head Error")
    plt.title("Tail-Head Error Gap vs Rejection Rate")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 0.8])
    ymin = float(min(0.0, gap_plot.min() if gap_plot.size else 0.0))
    ymax = float(max(0.0, gap_plot.max() if gap_plot.size else 0.0))
    pad = 0.05 * (ymax - ymin + 1e-8)
    plt.ylim([ymin - pad, ymax + pad])
    plt.tight_layout()
    plt.savefig(gap_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gap plot to: {gap_plot_path}")


if __name__ == "__main__":
    main()

