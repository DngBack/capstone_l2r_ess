#!/usr/bin/env python3
"""
Standalone Balanced Plug-in with Uniform Weighting (3 Experts) per "Learning to Reject Meets Long-Tail Learning"
================================================================================================================

- Loads CIFAR-100-LT splits and 3 expert logits (CE, LogitAdjust, BalSoftmax)
- Uses uniform weighting (1/3, 1/3, 1/3) to combine experts instead of trained gating network
- Builds head/tail groups from train class counts (tail <= 20 samples)
- Implements Theorem 1 decision rules (classifier and rejector)
- Implements Algorithm 1 (power-iteration) over α and 1D λ grid for μ
- Runs theory-compliant cost sweep: optimize (α, μ) per cost, one RC point per cost
- Evaluates on test; computes balanced error RC and AURC; saves JSON and plots

This serves as a baseline to compare against gating network performance.

Inputs (expected existing):
- Splits dir: ./data/cifar100_lt_if100_splits_fixed/
- Logits dir: ./outputs/logits/cifar100_lt_if100/{expert_name}/{split}_logits.pt
- Targets (if available): ./outputs/logits/cifar100_lt_if100/{expert_name}/{split}_targets.pt

Outputs:
- results/ltr_plugin/cifar100_lt_if100/ltr_plugin_uniform_balanced_3experts.json
- results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_balanced_uniform_3experts_test.png
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ================================
# Config
# ================================
@dataclass
class Config:
    dataset_name: str = "cifar100_lt_if100"
    splits_dir: str = "./data/cifar100_lt_if100_splits_fixed"
    logits_dir: str = "./outputs/logits/cifar100_lt_if100"
    results_dir: str = "./results/ltr_plugin/cifar100_lt_if100"

    expert_names: List[str] = field(
        default_factory=lambda: [
            "ce_baseline",
            "logitadjust_baseline",
            "balsoftmax_baseline",
        ]
    )

    num_classes: int = 100
    num_groups: int = 2

    # Tail definition per paper: tail if train count <= 20
    tail_threshold: int = 20

    # Optimizer settings - extended range including paper values {1, 6, 11}
    mu_lambda_grid: List[float] = field(
        default_factory=lambda: [
            -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 11.0, 15.0, 20.0,
        ]
    )
    power_iter_iters: int = 20  # More iterations for convergence
    power_iter_damping: float = 0.5  # Higher damping for stability

    # Cost sweep (theory-compliant): one RC point per cost
    cost_sweep: List[float] = field(default_factory=list)  # unused when target_rejections set
    # Target rejection grid to match paper plots exactly
    target_rejections: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )

    seed: int = 42


CFG = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# IO helpers
# ================================


def load_expert_logits(
    expert_names: List[str], split: str, device: str = DEVICE
) -> torch.Tensor:
    """Load logits from all experts and stack them."""
    logits_list = []

    print(f"Loading logits for {len(expert_names)} experts: {expert_names}")
    for expert_name in expert_names:
        path = Path(CFG.logits_dir) / expert_name / f"{split}_logits.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing logits: {path}")
        logits = torch.load(path, map_location=device).float()
        logits_list.append(logits)
        print(f"  ✓ Loaded {expert_name}: {logits.shape}")

    # Stack: [E, N, C] -> transpose to [N, E, C]
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    print(
        f"✓ Stacked expert logits: {logits.shape} (should be [N, {len(expert_names)}, {CFG.num_classes}])"
    )
    return logits


def load_labels(split: str, device: str = DEVICE) -> torch.Tensor:
    # Prefer saved targets alongside logits
    cand = Path(CFG.logits_dir) / CFG.expert_names[0] / f"{split}_targets.pt"
    if cand.exists():
        t = torch.load(cand, map_location=device)
        if isinstance(t, torch.Tensor):
            return t.to(device=device, dtype=torch.long)

    # Fallback: reconstruct from CIFAR100 and indices
    print(f"Reconstructing labels for {split} from CIFAR100 dataset...")

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


def compute_uniform_mixture_posterior(
    expert_logits: torch.Tensor, device: str = DEVICE
) -> torch.Tensor:
    """Compute mixture posterior using uniform weighting (equal weights for all experts)."""
    # expert_logits: [N, E, C]
    with torch.no_grad():
        # Convert logits to posteriors (for mixture)
        expert_posteriors = F.softmax(expert_logits, dim=-1)  # [N, E, C]
        
        num_experts = expert_logits.shape[1]
        
        print(f"Using uniform weighting for {num_experts} experts")
        
        # Uniform weights: equal weight for all experts
        uniform_weights = torch.ones(expert_logits.shape[0], num_experts, device=device) / num_experts  # [N, E]
        
        # Mixture posterior: η̃(x) = (1/E) * Σ_e p^(e)(y|x)
        mixture_posterior = (uniform_weights.unsqueeze(-1) * expert_posteriors).sum(
            dim=1
        )  # [N, C]

        # Verify mixture is valid probability distribution
        mixture_sum = mixture_posterior.sum(dim=1)
        if not torch.allclose(mixture_sum, torch.ones_like(mixture_sum), atol=1e-5):
            print(f"Warning: Uniform mixture posterior sums: min={mixture_sum.min():.6f}, max={mixture_sum.max():.6f}")

        return mixture_posterior


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
        self.class_to_group = class_to_group
        num_groups = int(class_to_group.max().item() + 1)
        self.register_buffer("alpha_group", torch.ones(num_groups))
        self.register_buffer("mu_group", torch.zeros(num_groups))
        self.register_buffer("cost", torch.tensor(0.0))

    def set_params(self, alpha_g: torch.Tensor, mu_g: torch.Tensor, cost: float):
        self.alpha_group = alpha_g.to(self.alpha_group.device)
        self.mu_group = mu_g.to(self.mu_group.device)
        self.cost = torch.tensor(float(cost), device=self.cost.device)

    def _alpha_class(self) -> torch.Tensor:
        return self.alpha_group[self.class_to_group]

    def _mu_class(self) -> torch.Tensor:
        return self.mu_group[self.class_to_group]

    def _alpha_hat_class(self) -> torch.Tensor:
        # Per Theorem 1: α̂_k = α_[k] (no group weighting for balanced)
        return self._alpha_class()

    @torch.no_grad()
    def predict(self, posterior: torch.Tensor) -> torch.Tensor:
        alpha_hat = self._alpha_hat_class().unsqueeze(0)  # [1, C]
        return (posterior / alpha_hat).argmax(dim=-1)

    @torch.no_grad()
    def reject(
        self, posterior: torch.Tensor, cost: Optional[float] = None
    ) -> torch.Tensor:
        alpha_hat = self._alpha_hat_class().unsqueeze(0)  # [1, C]
        mu = self._mu_class().unsqueeze(0)  # [1, C]
        
        max_reweighted = (posterior / alpha_hat).max(dim=-1)[0]  # [N]
        threshold = ((1.0 / alpha_hat - mu) * posterior).sum(dim=-1)  # [N]
        
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
        K = int(class_to_group.max().item() + 1)
        return {
            "selective_error": 1.0,
            "coverage": 0.0,
            "group_errors": [1.0] * K,
            "balanced_error": 1.0,
            "worst_group_error": 1.0,
        }

    preds_a = preds[accept]
    labels_a = labels[accept]
    groups_a = class_to_group[labels_a]
    errors = (preds_a != labels_a).float()

    K = int(class_to_group.max().item() + 1)
    group_errors = []

    # Coverage calculation
    cov_unw = float(accept.float().mean().item())
    cov_w = None
    if class_weights is not None:
        cw_all = class_weights[labels]
        total_w = float(cw_all.sum().item())
        cov_w = float(cw_all[accept].sum().item() / max(total_w, 1e-12))

    # Group-wise error calculation
    for g in range(K):
        mask = groups_a == g
        if mask.sum() == 0:
            group_errors.append(1.0)
            continue
        
        if class_weights is None:
            group_errors.append(float(errors[mask].mean().item()))
        else:
            # Weighted error within group
            cw = class_weights[labels_a]
            cw_group = cw[mask]
            if cw_group.sum() > 0:
                weighted_error = float((errors[mask] * cw_group).sum() / cw_group.sum())
            else:
                weighted_error = 1.0
            group_errors.append(weighted_error)

    balanced_error = float(np.mean(group_errors))
    worst_group_error = float(np.max(group_errors))

    return {
        "selective_error": float(errors.mean().item()),
        "coverage": cov_w if cov_w is not None else cov_unw,
        "group_errors": group_errors,
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
    }


# ================================
# Algorithm 1 (Power-iteration)
# ================================
@torch.no_grad()
def initialize_alpha(labels: torch.Tensor, class_to_group: torch.Tensor) -> np.ndarray:
    """Initialize α based on group proportions in labels."""
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    for g in range(K):
        prop = (class_to_group[labels] == g).float().mean().item()
        alpha[g] = float(max(prop, 1e-12))
    return alpha


@torch.no_grad()
def update_alpha_from_coverage(
    reject: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Update α based on actual coverage per group."""
    K = int(class_to_group.max().item() + 1)
    alpha = np.zeros(K, dtype=np.float64)
    accept = ~reject
    N = max(1, len(labels))

    if class_weights is None:
        for g in range(K):
            in_group = class_to_group[labels] == g
            cov = float(accept[in_group].float().mean().item() if in_group.sum() > 0 else 0.0)
            alpha[g] = float(max(cov, 1e-12))
        return alpha

    # Weighted coverage calculation
    cw_all = class_weights[labels]
    total_w = float(cw_all.sum().item())
    for g in range(K):
        in_group = class_to_group[labels] == g
        if in_group.sum() == 0:
            alpha[g] = 1e-12
            continue
        cov_w = float(cw_all[accept & in_group].sum().item() / max(total_w, 1e-12))
        alpha[g] = float(max(cov_w, 1e-12))
    
    return alpha


@torch.no_grad()
def compute_cost_for_target_rejection(
    posterior: torch.Tensor,
    class_to_group: torch.Tensor,
    alpha: np.ndarray,
    mu: np.ndarray,
    target_rejection: float,
) -> float:
    """Compute cost threshold to achieve target rejection rate."""
    K = int(class_to_group.max().item() + 1)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    
    alpha_class = alpha_t[class_to_group]  # [C]
    mu_class = mu_t[class_to_group]  # [C]
    
    max_reweighted = (posterior / alpha_class.unsqueeze(0)).max(dim=-1)[0]  # [N]
    threshold_base = ((1.0 / alpha_class - mu_class).unsqueeze(0) * posterior).sum(dim=-1)  # [N]
    
    # Rejection score = threshold_base - max_reweighted - cost
    # Want: rejection_rate = P(rejection_score > 0) = target_rejection
    t = threshold_base - max_reweighted  # [N]
    t_sorted = torch.sort(t)[0]
    
    # Find quantile
    q = max(0.0, min(1.0, 1.0 - float(target_rejection)))
    idx = int(round(q * (len(t_sorted) - 1)))
    
    return float(t_sorted[idx].item())


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
    """Power iteration search for optimal α given μ and cost."""
    
    # Initialize α
    alpha = initialize_alpha(labels, class_to_group)
    
    if verbose:
        print(f"  Power iteration: μ={mu}, cost={cost:.4f}, target_rej={target_rejection}")
        print(f"  Initial α: {alpha}")
    
    for iter_idx in range(num_iters):
        # Set parameters and compute rejection decisions
        plugin.set_params(
            torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
            torch.tensor(mu, dtype=torch.float32, device=DEVICE),
            float(cost)
        )
        
        preds = plugin.predict(posterior)
        reject = plugin.reject(posterior)
        
        # Update α based on actual coverage
        alpha_new = update_alpha_from_coverage(reject, labels, class_to_group, class_weights)
        
        # Damped update
        alpha = damping * alpha + (1.0 - damping) * alpha_new
        
        if verbose and iter_idx % 5 == 0:
            rej_rate = float(reject.float().mean().item())
            print(f"    Iter {iter_idx}: α={alpha}, rej_rate={rej_rate:.4f}")
    
    # Final evaluation
    plugin.set_params(
        torch.tensor(alpha, dtype=torch.float32, device=DEVICE),
        torch.tensor(mu, dtype=torch.float32, device=DEVICE),
        float(cost)
    )
    
    preds = plugin.predict(posterior)
    reject = plugin.reject(posterior)
    metrics = compute_metrics(preds, labels, reject, class_to_group, class_weights)
    
    if verbose:
        print(f"  Final: α={alpha}, metrics={metrics}")
    
    return alpha, metrics


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
    plt.figure(figsize=(7, 5))
    plt.plot(r, e_bal, "o-", color="green", label=f"Balanced (AURC={aurc_bal:.4f})")
    plt.plot(
        r, e_wst, "s-", color="royalblue", label=f"Worst-group (AURC={aurc_wst:.4f})"
    )
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Error")
    plt.title("Balanced and Worst-group Error vs Rejection Rate (Uniform Weighting)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
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

    print("=== Uniform Weighting Baseline ===")
    print(f"Using uniform weights for {len(CFG.expert_names)} experts: {CFG.expert_names}")

    print("\nLoading S1 (tunev) and S2 (val) for selection/evaluation...")
    expert_logits_tunev = load_expert_logits(CFG.expert_names, "tunev", DEVICE)
    labels_tunev = load_labels("tunev", DEVICE)

    expert_logits_val = load_expert_logits(CFG.expert_names, "val", DEVICE)
    labels_val = load_labels("val", DEVICE)

    print("\nComputing mixture posteriors using uniform weighting...")
    posterior_tunev = compute_uniform_mixture_posterior(expert_logits_tunev, DEVICE)
    posterior_val = compute_uniform_mixture_posterior(expert_logits_val, DEVICE)

    print("\nBuilding class-to-group mapping (tail <= 20)...")
    class_to_group = build_class_to_group()

    print("\nLoading class weights for importance weighting...")
    class_weights = load_class_weights(DEVICE)

    # Check baseline balanced error on test set
    expert_logits_test = load_expert_logits(CFG.expert_names, "test", DEVICE)
    labels_test = load_labels("test", DEVICE)
    posterior_test = compute_uniform_mixture_posterior(expert_logits_test, DEVICE)

    # Compute baseline balanced error (no rejection) with importance weighting
    mix_pred_test = posterior_test.argmax(dim=-1)
    groups_test = class_to_group[labels_test]
    num_groups = int(class_to_group.max().item() + 1)

    dummy_reject = torch.zeros(len(labels_test), dtype=torch.bool, device=DEVICE)
    baseline_metrics = compute_metrics(
        mix_pred_test, labels_test, dummy_reject, class_to_group, class_weights
    )

    baseline_balanced_error = baseline_metrics["balanced_error"]
    print(f"\nBaseline Uniform balanced error (TEST) = {baseline_balanced_error:.4f}")
    print(f"Baseline Uniform group errors = {baseline_metrics['group_errors']}")
    print(
        f"Baseline Uniform overall accuracy (TEST) = {(mix_pred_test == labels_test).float().mean().item():.4f}"
    )

    print("\nCreating plug-in model...")
    plugin = BalancedLtRPlugin(class_to_group).to(DEVICE)

    # Target rejection grid (paper-style points)
    results_per_cost: List[Dict] = []
    targets = list(CFG.target_rejections)
    
    for i, target_rej in enumerate(targets):
        print(f"\n--- Target rejection {target_rej:.1f} ({i+1}/{len(targets)}) ---")
        
        best_val_result = {
            "balanced_error": float("inf"),
            "alpha": None,
            "mu": None,
            "cost_val": None,
            "cost_test": None,
            "val_metrics": None,
            "test_metrics": None,
        }
        
        # Grid search over μ (lambda values)
        for mu_lambda in CFG.mu_lambda_grid:
            # For 2-group case: μ = [0, λ] where λ = μ_tail - μ_head
            mu = np.array([0.0, float(mu_lambda)], dtype=np.float64)
            
            # Optimize (α, μ) on S1 (tunev)
            cost_s1 = compute_cost_for_target_rejection(
                posterior_tunev, class_to_group, 
                initialize_alpha(labels_tunev, class_to_group), 
                mu, target_rej
            )
            
            alpha_opt, _ = power_iter_search(
                plugin, posterior_tunev, labels_tunev, class_to_group,
                mu, cost_s1, CFG.power_iter_iters, CFG.power_iter_damping,
                class_weights, verbose=False, target_rejection=target_rej
            )
            
            # Evaluate on S2 (val) for model selection
            cost_s2 = compute_cost_for_target_rejection(
                posterior_val, class_to_group, alpha_opt, mu, target_rej
            )
            
            plugin.set_params(
                torch.tensor(alpha_opt, dtype=torch.float32, device=DEVICE),
                torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                float(cost_s2)
            )
            
            preds_val = plugin.predict(posterior_val)
            reject_val = plugin.reject(posterior_val)
            val_metrics = compute_metrics(
                preds_val, labels_val, reject_val, class_to_group, class_weights
            )
            
            print(f"  λ={mu_lambda:5.1f}: val_bal_err={val_metrics['balanced_error']:.4f}, "
                  f"coverage={val_metrics['coverage']:.3f}")
            
            # Select best based on balanced error on validation
            if val_metrics["balanced_error"] < best_val_result["balanced_error"]:
                # Evaluate on test set with selected hyperparameters
                cost_test = compute_cost_for_target_rejection(
                    posterior_test, class_to_group, alpha_opt, mu, target_rej
                )
                
                plugin.set_params(
                    torch.tensor(alpha_opt, dtype=torch.float32, device=DEVICE),
                    torch.tensor(mu, dtype=torch.float32, device=DEVICE),
                    float(cost_test)
                )
                
                preds_test = plugin.predict(posterior_test)
                reject_test = plugin.reject(posterior_test)
                test_metrics = compute_metrics(
                    preds_test, labels_test, reject_test, class_to_group, class_weights
                )
                
                best_val_result.update({
                    "balanced_error": val_metrics["balanced_error"],
                    "alpha": alpha_opt.tolist(),
                    "mu": mu.tolist(),
                    "mu_lambda": float(mu_lambda),
                    "cost_val": float(cost_s2),
                    "cost_test": float(cost_test),
                    "val_metrics": {
                        "coverage": float(val_metrics["coverage"]),
                        "balanced_error": float(val_metrics["balanced_error"]),
                        "worst_group_error": float(val_metrics["worst_group_error"]),
                        "group_errors": [float(x) for x in val_metrics["group_errors"]],
                    },
                    "test_metrics": {
                        "coverage": float(test_metrics["coverage"]),
                        "balanced_error": float(test_metrics["balanced_error"]),
                        "worst_group_error": float(test_metrics["worst_group_error"]),
                        "group_errors": [float(x) for x in test_metrics["group_errors"]],
                    },
                })
        
        print(f"  ✓ Best λ={best_val_result['mu_lambda']:.1f}: "
              f"val_bal_err={best_val_result['balanced_error']:.4f}, "
              f"test_bal_err={best_val_result['test_metrics']['balanced_error']:.4f}")
        
        results_per_cost.append({
            "target_rejection": float(target_rej),
            **best_val_result
        })

    # Build unified RC curve (balanced) from target points
    r_val = np.array([1.0 - r["val_metrics"]["coverage"] for r in results_per_cost])
    e_val = np.array([r["val_metrics"]["balanced_error"] for r in results_per_cost])
    w_val = np.array([r["val_metrics"]["worst_group_error"] for r in results_per_cost])
    gap_val = np.array([
        r["val_metrics"]["group_errors"][1] - r["val_metrics"]["group_errors"][0] 
        for r in results_per_cost
    ])
    
    r_test = np.array([1.0 - r["test_metrics"]["coverage"] for r in results_per_cost])
    e_test = np.array([r["test_metrics"]["balanced_error"] for r in results_per_cost])
    w_test = np.array([r["test_metrics"]["worst_group_error"] for r in results_per_cost])
    gap_test = np.array([
        r["test_metrics"]["group_errors"][1] - r["test_metrics"]["group_errors"][0] 
        for r in results_per_cost
    ])

    # Sort by rejection rate
    idx_v = np.argsort(r_val)
    r_val, e_val, w_val, gap_val = r_val[idx_v], e_val[idx_v], w_val[idx_v], gap_val[idx_v]
    
    idx_t = np.argsort(r_test)
    r_test, e_test, w_test, gap_test = r_test[idx_t], e_test[idx_t], w_test[idx_t], gap_test[idx_t]

    # Compute AURCs
    aurc_val_bal = (
        float(np.trapezoid(e_val, r_val))
        if r_val.size > 1
        else float(e_val.mean() if e_val.size else 0.0)
    )
    aurc_test_bal = (
        float(np.trapezoid(e_test, r_test))
        if r_test.size > 1
        else float(e_test.mean() if e_test.size else 0.0)
    )
    aurc_val_wst = (
        float(np.trapezoid(w_val, r_val))
        if r_val.size > 1
        else float(w_val.mean() if w_val.size else 0.0)
    )
    aurc_test_wst = (
        float(np.trapezoid(w_test, r_test))
        if r_test.size > 1
        else float(w_test.mean() if w_test.size else 0.0)
    )

    # Practical AURC over coverage >= 0.2 → rejection <= 0.8
    mask_val_08 = r_val <= 0.8 if r_val.size > 1 else np.array([True])
    mask_test_08 = r_test <= 0.8 if r_test.size > 1 else np.array([True])
    
    if r_val.size > 1 and mask_val_08.sum() > 1:
        aurc_val_bal_08 = float(np.trapezoid(e_val[mask_val_08], r_val[mask_val_08]))
        aurc_val_wst_08 = float(np.trapezoid(w_val[mask_val_08], r_val[mask_val_08]))
    else:
        aurc_val_bal_08 = float(e_val.mean() if e_val.size else 0.0)
        aurc_val_wst_08 = float(w_val.mean() if w_val.size else 0.0)

    if r_test.size > 1 and mask_test_08.sum() > 1:
        aurc_test_bal_08 = float(np.trapezoid(e_test[mask_test_08], r_test[mask_test_08]))
        aurc_test_wst_08 = float(np.trapezoid(w_test[mask_test_08], r_test[mask_test_08]))
    else:
        aurc_test_bal_08 = float(e_test.mean() if e_test.size else 0.0)
        aurc_test_wst_08 = float(w_test.mean() if w_test.size else 0.0)

    # Save results
    save_dict = {
        "objectives": ["balanced", "worst_group"],
        "description": "Targeted rejection grid (0.0..0.8) with val-based hyperparameter selection per paper Algorithm 1. Uses 3 experts with UNIFORM weighting (baseline).",
        "method": "plug-in_balanced_val_selection_uniform",
        "weighting": "uniform",
        "hyperparameter_selection": "val_based",
        "algorithm": "Algorithm 1 from paper - optimize (α,μ) on tunev, select μ on val",
        "experts": CFG.expert_names,
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

    out_json = Path(CFG.results_dir) / "ltr_plugin_uniform_balanced_3experts.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2)
    print(f"\nSaved results to: {out_json}")

    # Print AURCs
    print(f"\n=== AURC Results (Uniform Weighting) ===")
    print(f"Val AURC - Balanced: {aurc_val_bal:.4f} | Worst-group: {aurc_val_wst:.4f}")
    print(f"Test AURC - Balanced: {aurc_test_bal:.4f} | Worst-group: {aurc_test_wst:.4f}")
    print(f"Val AURC (coverage>=0.2) - Balanced: {aurc_val_bal_08:.4f} | Worst-group: {aurc_val_wst_08:.4f}")
    print(f"Test AURC (coverage>=0.2) - Balanced: {aurc_test_bal_08:.4f} | Worst-group: {aurc_test_wst_08:.4f}")

    # Plot test RC curves (both metrics)
    plot_path = Path(CFG.results_dir) / "ltr_rc_curves_balanced_uniform_3experts_test.png"
    plot_rc_dual(r_test, e_test, w_test, aurc_test_bal, aurc_test_wst, plot_path)
    print(f"\nSaved combined plot to: {plot_path}")

    # Plot Tail - Head gap curve
    gap_plot_path = Path(CFG.results_dir) / "ltr_tail_minus_head_uniform_3experts_test.png"
    plt.figure(figsize=(7, 5))
    plt.plot(r_test, gap_test, "d-", color="crimson", label="Tail - Head error")
    plt.xlabel("Proportion of Rejections")
    plt.ylabel("Tail Error - Head Error")
    plt.title("Tail-Head Error Gap vs Rejection Rate (Uniform 3-Experts)")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    ymin = float(min(0.0, gap_test.min() if gap_test.size else 0.0))
    ymax = float(max(0.0, gap_test.max() if gap_test.size else 0.0))
    pad = 0.05 * (ymax - ymin + 1e-8)
    plt.ylim([ymin - pad, ymax + pad])
    plt.tight_layout()
    plt.savefig(gap_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gap plot to: {gap_plot_path}")


if __name__ == "__main__":
    main()