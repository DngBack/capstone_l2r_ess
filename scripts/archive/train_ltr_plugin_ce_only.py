"""
Learning to Reject (LtR) Plugin Training - CE Only Mode
=======================================================

Modified version that uses only CE expert logits instead of mixture posterior.
This simplifies the pipeline by bypassing the gating network.

Theory (per paper, Theorem 1):
    Classifier: h_α(x) = argmax_y (η_y(x) / α[y])  ≡ argmax_y (1/α[y])·η_y(x)
    Rejector:   r(x) = 1{ max_y(η_y/α[y]) < Σ_y' (1/α[y'] − μ[y'])·η_{y'}(x) − c }

where:
    - η(x): posterior from CE model (instead of mixture)
    - α: class reweighting coefficients
    - μ: normalization vector for threshold
    - c: rejection cost

Usage:
    # CE only mode with balanced objective
    python train_ltr_plugin_ce_only.py --objective balanced --optimizer power_iter --cost_sweep
    
    # CE only mode with worst-group objective
    python train_ltr_plugin_ce_only.py --objective worst --optimizer power_iter --cost_sweep
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict
import matplotlib.pyplot as plt

from src.models.ltr_plugin import (
    LtRPlugin,
    LtRPluginConfig,
    LtRGridSearchOptimizer,
    LtRPowerIterOptimizer,
    LtRWorstGroupOptimizer,
    RCCurveComputer,
    compute_selective_metrics,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "dataset": {
        "name": "cifar100_lt_if100",
        "splits_dir": "./data/cifar100_lt_if100_splits_fixed",
        "num_classes": 100,
        "num_groups": 2,
        # group_boundaries will be constructed dynamically from train_class_counts.json
        # to follow the paper's rule: tail = classes with ≤ 20 training samples
        "group_boundaries": None,
    },
    "expert": {
        "name": "ce_baseline",  # Only use CE expert
        "logits_dir": "./outputs/logits/cifar100_lt_if100/",
    },
    "ltr": {
        # Parameter mode
        "param_mode": "group",  # 'group' or 'class'
        # Grid ranges (group mode: 2 groups = head/tail)
        "alpha_grid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        "mu_grid": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        "cost_grid": [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.91, 0.95, 0.97, 0.99],
        # Cost sweep for RC curve - fixed grid approach (Paper Method)
        "cost_sweep": [
            0.0,    # Should give ~0.0 rejection rate
            0.001,  # Very low cost
            0.005,  # Very low cost
            0.01,   # Very low cost
            0.02,   # Low cost
            0.03,   # Low cost
            0.04,   # Low cost
            0.05,   # Low cost
            0.06,   # Low cost
            0.07,   # Low cost
            0.08,   # Low cost
            0.09,   # Low cost
            0.1,    # Low cost
            0.12,   # Medium-low cost
            0.14,   # Medium-low cost
            0.16,   # Medium-low cost
            0.18,   # Medium-low cost
            0.2,    # Medium-low cost
            0.25,   # Medium cost
            0.3,    # Medium cost
            0.35,   # Medium cost
            0.4,    # Medium cost
            0.45,   # Medium cost
            0.5,    # Medium-high cost
            0.55,   # Medium-high cost
            0.6,    # High cost
            0.65,   # High cost
            0.7,    # High cost
            0.75,   # High cost
            0.8,    # Very high cost
            0.85,   # Very high cost
            0.9,    # Very high cost
            0.95,   # Very high cost
            0.97,   # Very high cost
            0.99,   # Very high cost
        ],  # Dense grid covering full rejection range
        "target_rejection_rates": [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.98,
            0.99,
        ],
        # Optional λ grid for K=2 (Appendix E.1 style: μ = [0, λ])
        "lambda_grid": [1.0, 6.0, 11.0],
    },
    "evaluation": {
        "use_reweighting": True,
    },
    "output": {
        "results_dir": "./results/ltr_plugin/",
    },
    "seed": 42,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# DATA LOADING
# ============================================================================


def load_single_expert_logits(expert_name, logits_dir, split_name, device="cpu"):
    """Load logits from a single expert."""
    logits_path = Path(logits_dir) / expert_name / f"{split_name}_logits.pt"

    if not logits_path.exists():
        raise FileNotFoundError(f"Logits not found: {logits_path}")

    logits = torch.load(logits_path, map_location=device).float()
    return logits


def load_labels(splits_dir, split_name, device="cpu", prefer_logits_targets=True):
    """Load labels for a split.

    Priority:
    1) If targets tensor saved alongside logits (e.g., outputs/.../{split}_targets.pt), use it.
       This guarantees alignment with the stored logits ordering.
    2) Fallback: reconstruct using CIFAR100 + saved indices (legacy path).
    """
    # Try to locate saved targets next to logits (CE-only path)
    if prefer_logits_targets:
        cand = Path(CONFIG["expert"]["logits_dir"]) / CONFIG["expert"]["name"] / f"{split_name}_targets.pt"
        if cand.exists():
            targets = torch.load(cand, map_location=device)
            if isinstance(targets, torch.Tensor):
                return targets.to(device=device, dtype=torch.long)

    # Fallback to legacy reconstruction via dataset + indices
    import torchvision
    indices_file = f"{split_name}_indices.json"
    with open(Path(splits_dir) / indices_file, "r") as f:
        indices = json.load(f)

    is_train = split_name in ["expert", "gating", "train"]
    dataset = torchvision.datasets.CIFAR100(root="./data", train=is_train, download=False)
    labels = torch.tensor([dataset.targets[i] for i in indices], dtype=torch.long, device=device)
    return labels


def load_sample_weights(splits_dir, split_name, device="cpu"):
    """
    Load sample weights for reweighting evaluation metrics.

    Purpose: REBALANCE test/val distribution to match train distribution.
    """
    weights_path = Path(splits_dir) / "class_weights.json"

    if not weights_path.exists():
        print(f"WARNING: Class weights not found: {weights_path}")
        return None

    with open(weights_path, "r") as f:
        class_weights_list = json.load(f)

    # Convert to tensor
    class_weights = torch.tensor(class_weights_list, dtype=torch.float32, device=device)

    # Get labels to create per-sample weights
    labels = load_labels(splits_dir, split_name, device)
    sample_weights = class_weights[labels]

    return sample_weights


# ============================================================================
# TRAINING
# ============================================================================


def train_ltr_plugin_ce_only(
    objective: str = "balanced",
    cost_sweep: bool = False,
    optimizer_type: str = "power_iter",
    verbose: bool = True,
):
    """
    Train LtR Plugin with CE expert only (no gating network).

    Args:
        objective: 'balanced' or 'worst'
        cost_sweep: if True, train with multiple costs for RC curve
        optimizer_type: 'power_iter' (Algorithm 1), 'grid' (baseline), or 'worst_group' (Algorithm 2)
        verbose: print progress
    """
    print("=" * 70)
    print("LtR PLUGIN TRAINING - CE ONLY MODE")
    print("=" * 70)
    print(f"Expert: {CONFIG['expert']['name']}")
    print(f"Objective: {objective}")
    print(f"Cost sweep: {cost_sweep}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Param mode: {CONFIG['ltr']['param_mode']}")
    print(f"Device: {DEVICE}")

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ========================================================================
    # 1. LOAD DATA (tunev for S1, val for S2)
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. LOADING DATA & BUILDING GROUPS (paper-compliant)")
    print("=" * 70)

    # Build head/tail groups from train_class_counts.json (tail = <=20 samples)
    class_to_group = None
    try:
        counts_path = Path(CONFIG["dataset"]["splits_dir"]) / "train_class_counts.json"
        with open(counts_path, "r", encoding="utf-8") as f:
            class_counts = json.load(f)
        # Accept both list or dict
        if isinstance(class_counts, dict):
            class_counts = [class_counts[str(i)] for i in range(CONFIG["dataset"]["num_classes"])]
        class_counts = np.array(class_counts)

        # Paper rule: tail if count <= 20
        tail_mask = class_counts <= 20
        # Build class_to_group mapping: 0=head, 1=tail
        class_to_group = np.zeros(CONFIG["dataset"]["num_classes"], dtype=np.int64)
        class_to_group[tail_mask] = 1

        derived_boundary_info = {
            "num_head": int((class_to_group == 0).sum()),
            "num_tail": int((class_to_group == 1).sum()),
        }
        # Use ASCII-only print to avoid Windows console encoding issues
        print(
            f"Group construction (paper rule <=20): head={derived_boundary_info['num_head']}, tail={derived_boundary_info['num_tail']}"
        )
    except (OSError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARNING: Failed to build groups from train_class_counts.json: {e}. Falling back to previous boundary.")
        class_to_group = None

    # Load S1 (tunev) and S2 (val) for Algorithm 2 compliance
    if optimizer_type == "worst_group" or objective == "balanced":
        print("Loading S1 (tunev) and S2 (val) for selection/evaluation")

        # S1: tunev split
        expert_logits_s1 = load_single_expert_logits(
            CONFIG["expert"]["name"], CONFIG["expert"]["logits_dir"], "tunev", DEVICE
        )
        labels_s1 = load_labels(CONFIG["dataset"]["splits_dir"], "tunev", DEVICE)
        sample_weights_s1 = None
        if CONFIG["evaluation"]["use_reweighting"]:
            sample_weights_s1 = load_sample_weights(
                CONFIG["dataset"]["splits_dir"], "tunev", DEVICE
            )

        # S2: val split
        expert_logits_s2 = load_single_expert_logits(
            CONFIG["expert"]["name"], CONFIG["expert"]["logits_dir"], "val", DEVICE
        )
        labels_s2 = load_labels(CONFIG["dataset"]["splits_dir"], "val", DEVICE)
        sample_weights_s2 = None
        if CONFIG["evaluation"]["use_reweighting"]:
            sample_weights_s2 = load_sample_weights(
                CONFIG["dataset"]["splits_dir"], "val", DEVICE
            )

        print(f"SUCCESS: S1 (tunev): {expert_logits_s1.shape[0]} samples")
        print(f"SUCCESS: S2 (val): {expert_logits_s2.shape[0]} samples")

        # For backward compatibility, use S1 as main data when present
        expert_logits_val = expert_logits_s1
        labels_val = labels_s1
        sample_weights_val = sample_weights_s1

    else:
        # Use tunev for other optimizers (backward compatibility)
        optimization_split = "tunev"
        expert_logits_val = load_single_expert_logits(
            CONFIG["expert"]["name"],
            CONFIG["expert"]["logits_dir"],
            optimization_split,
            DEVICE,
        )

        labels_val = load_labels(
            CONFIG["dataset"]["splits_dir"], optimization_split, DEVICE
        )

        sample_weights_val = None
        if CONFIG["evaluation"]["use_reweighting"]:
            sample_weights_val = load_sample_weights(
                CONFIG["dataset"]["splits_dir"], optimization_split, DEVICE
            )
            print("SUCCESS: Reweighting enabled")

        print(
            f"SUCCESS: {optimization_split.capitalize()}: {expert_logits_val.shape[0]} samples"
        )

    # ========================================================================
    # 2. COMPUTE POSTERIORS FROM CE LOGITS
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. COMPUTING POSTERIORS FROM CE LOGITS")
    print("=" * 70)

    def compute_ce_posterior(logits, device):
        """Helper function to compute posteriors from CE logits."""
        with torch.no_grad():
            # Direct softmax on CE logits
            posterior = F.softmax(logits, dim=-1)
            return posterior

    # Compute posteriors for main data (S1 or tunev)
    posterior_val = compute_ce_posterior(expert_logits_val, DEVICE)

    # For Algorithm 2, also compute posteriors for S2
    posterior_s1 = None
    posterior_s2 = None
    if optimizer_type == "worst_group":
        posterior_s1 = compute_ce_posterior(expert_logits_s1, DEVICE)
        posterior_s2 = compute_ce_posterior(expert_logits_s2, DEVICE)
        posterior_val = posterior_s1  # Use S1 as main for backward compatibility

    print("SUCCESS: CE posteriors computed")
    print(f"   Shape: {posterior_val.shape}")
    print(
        f"   Range: [{posterior_val.min():.3f}, {posterior_val.max():.3f}]"
    )
    print(
        f"   Sum per sample: {posterior_val.sum(dim=-1).mean():.3f} (should be ~1.0)"
    )

    # ========================================================================
    # 3. CREATE LTR PLUGIN
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. CREATING LtR PLUGIN")
    print("=" * 70)

    ltr_config = LtRPluginConfig(
        num_classes=CONFIG["dataset"]["num_classes"],
        num_groups=CONFIG["dataset"]["num_groups"],
        # If we have an explicit class_to_group, keep default boundaries (won't be used)
        group_boundaries=[0] if CONFIG["dataset"].get("group_boundaries") is None else CONFIG["dataset"]["group_boundaries"],
        param_mode=CONFIG["ltr"]["param_mode"],
        alpha_grid=CONFIG["ltr"]["alpha_grid"],
        mu_grid=CONFIG["ltr"]["mu_grid"],
        cost_grid=CONFIG["ltr"]["cost_grid"],
        objective=objective,
    )

    # Add target_rejection_rates for percentile-based approach
    if cost_sweep and "target_rejection_rates" in CONFIG["ltr"]:
        ltr_config.target_rejection_rates = CONFIG["ltr"]["target_rejection_rates"]

    plugin = LtRPlugin(ltr_config).to(DEVICE)

    # If we successfully derived head/tail mapping, override plugin mapping here
    if class_to_group is not None:
        plugin.class_to_group = torch.tensor(class_to_group, dtype=torch.long, device=DEVICE)

    # Create optimizer based on type
    if optimizer_type == "power_iter":
        print("SUCCESS: Using Power-Iteration Optimizer (Algorithm 1 - Paper)")
        optimizer_obj = LtRPowerIterOptimizer(
            ltr_config, num_iters=50, alpha_init_mode="prior", damping=0.5
        )
    elif optimizer_type == "worst_group":
        print("SUCCESS: Using Worst-Group Optimizer (Algorithm 2 - Paper)")
        optimizer_obj = LtRWorstGroupOptimizer(
            ltr_config,
            num_outer_iters=25,  # Paper F.3: T=25
            learning_rate=1.0,  # Paper F.3: xi=1
            use_power_iter=True,
        )
    else:  # 'grid'
        print("SUCCESS: Using Grid Search Optimizer (Baseline)")
        optimizer_obj = LtRGridSearchOptimizer(ltr_config)

    print("SUCCESS: Plugin created")
    print(f"   Param mode: {ltr_config.param_mode}")
    print(f"   Num groups: {ltr_config.num_groups}")
    print(f"   Alpha grid size: {len(ltr_config.alpha_grid)}")
    print(f"   Mu grid size: {len(ltr_config.mu_grid)}")
    print(f"   Cost grid size: {len(ltr_config.cost_grid)}")

    # ========================================================================
    # 4. OPTIMIZATION
    # ========================================================================

    if cost_sweep:
        # Train with multiple costs - Theory-compliant version
        print("\nMode: Cost Sweep (Theory-Compliant)")
        print("   Each cost c optimizes (alpha, mu) and gives ONE point on RC curve")

        # For worst_group optimizer, pass S1/S2 data
        if optimizer_type == "worst_group" or objective == "balanced":
            results_per_cost, unified_rc = train_with_cost_sweep_ce_only(
                plugin,
                optimizer_obj,
                ltr_config,
                posterior_val,
                labels_val,
                sample_weights_val,
                objective,
                verbose,
                posterior_s1 if optimizer_type == "worst_group" or objective == "balanced" else None,
                labels_s1 if optimizer_type == "worst_group" or objective == "balanced" else None,
                sample_weights_s1 if optimizer_type == "worst_group" or objective == "balanced" else None,
                posterior_s2 if optimizer_type == "worst_group" or objective == "balanced" else None,
                labels_s2 if optimizer_type == "worst_group" or objective == "balanced" else None,
                sample_weights_s2 if optimizer_type == "worst_group" or objective == "balanced" else None,
            )
        else:
            pass
        return results_per_cost, unified_rc
    else:
        # Single training
        print("\n" + "=" * 70)
        print("4. GRID SEARCH OPTIMIZATION")
        print("=" * 70)

        # Call optimizer with appropriate data
        if optimizer_type == "worst_group":
            # Algorithm 2: Use S1 and S2 separately
            best_result = optimizer_obj.search(
                plugin,
                posterior_s1,
                labels_s1,
                posterior_s2,
                labels_s2,
                sample_weights_s1=sample_weights_s1,
                sample_weights_s2=sample_weights_s2,
                verbose=verbose,
            )
        else:
            # Algorithm 1 and Grid Search: Use single dataset
            best_result = optimizer_obj.search(
                plugin,
                posterior_val,
                labels_val,
                sample_weights=sample_weights_val,
                verbose=verbose,
            )

        # Set best parameters
        alpha_tensor = torch.tensor(
            best_result.alpha, dtype=torch.float32, device=DEVICE
        )
        mu_tensor = torch.tensor(best_result.mu, dtype=torch.float32, device=DEVICE)
        plugin.set_parameters(alpha=alpha_tensor, mu=mu_tensor, cost=best_result.cost)

        # ====================================================================
        # 5. EVALUATE ON TEST SET
        # ====================================================================
        print("\n" + "=" * 70)
        print("5. EVALUATION ON TEST SET")
        print("=" * 70)

        test_results = evaluate_on_test_ce_only(
            plugin,
            CONFIG["expert"]["name"],
            CONFIG["expert"]["logits_dir"],
            CONFIG["dataset"]["splits_dir"],
            CONFIG["evaluation"]["use_reweighting"],
        )

        # ====================================================================
        # 6. COMPUTE RC CURVE
        # ====================================================================
        print("\n" + "=" * 70)
        print("6. COMPUTING RC CURVE")
        print("=" * 70)

        rc_computer = RCCurveComputer(ltr_config)

        # VAL RC curve
        rc_data_val = rc_computer.compute_rc_curve(
            plugin,
            posterior_val,
            labels_val,
            alpha=best_result.alpha,
            mu=best_result.mu,
            cost_grid=np.linspace(0.0, 1.0, 200),
            sample_weights=None,
        )

        print(f"SUCCESS: VAL AURC: {rc_data_val['aurc']:.4f}")

        # TEST RC curve
        expert_logits_test = load_single_expert_logits(
            CONFIG["expert"]["name"], CONFIG["expert"]["logits_dir"], "test", DEVICE
        )
        labels_test = load_labels(CONFIG["dataset"]["splits_dir"], "test", DEVICE)

        with torch.no_grad():
            posterior_test = F.softmax(expert_logits_test, dim=-1)

        # Paper-compliant balanced/worst RC curves should not use class reweighting
        sample_weights_test = None

        rc_data_test = rc_computer.compute_rc_curve(
            plugin,
            posterior_test,
            labels_test,
            alpha=best_result.alpha,
            mu=best_result.mu,
            cost_grid=np.linspace(0.0, 1.0, 200),
            sample_weights=None,
        )

        print(f"SUCCESS: TEST AURC: {rc_data_test['aurc']:.4f}")

        # ====================================================================
        # 7. SAVE RESULTS
        # ====================================================================
        print("\n" + "=" * 70)
        print("7. SAVING RESULTS")
        print("=" * 70)

        results_dir = Path(CONFIG["output"]["results_dir"]) / CONFIG["dataset"]["name"]
        results_dir.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "expert": CONFIG["expert"]["name"],
            "objective": objective,
            "param_mode": CONFIG["ltr"]["param_mode"],
            "best_params": {
                "alpha": best_result.alpha.tolist(),
                "mu": best_result.mu.tolist(),
                "cost": best_result.cost,
            },
            "val_metrics": {
                "selective_error": best_result.selective_error,
                "coverage": best_result.coverage,
                "group_errors": best_result.group_errors,
                "worst_group_error": best_result.worst_group_error,
                "aurc": rc_data_val["aurc"],
            },
            "test_metrics": {
                "selective_error": test_results["selective_error"],
                "coverage": test_results["coverage"],
                "group_errors": test_results["group_errors"],
                "worst_group_error": test_results["worst_group_error"],
                "aurc": rc_data_test["aurc"],
            },
            "rc_curves": {
                "val": {
                    "rejection_rates": rc_data_val["rejection_rates"].tolist(),
                    "selective_errors": rc_data_val["selective_errors"].tolist(),
                },
                "test": {
                    "rejection_rates": rc_data_test["rejection_rates"].tolist(),
                    "selective_errors": rc_data_test["selective_errors"].tolist(),
                },
            },
        }

        output_path = results_dir / f"ltr_plugin_ce_only_{objective}.json"
        with open(output_path, "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"SUCCESS: Saved results to: {output_path}")

        # ====================================================================
        # 8. PLOT RC CURVES
        # ====================================================================
        print("\n" + "=" * 70)
        print("8. PLOTTING RC CURVES")
        print("=" * 70)

        plot_rc_curves_ltr(rc_data_val, rc_data_test, results_dir, f"{objective}_ce_only")

        return save_dict


def train_with_cost_sweep_ce_only(
    plugin,
    optimizer_obj,
    ltr_config,
    posterior_val,
    labels_val,
    sample_weights_val,
    objective,
    verbose,
    posterior_s1=None,
    labels_s1=None,
    sample_weights_s1=None,
    posterior_s2=None,
    labels_s2=None,
    sample_weights_s2=None,
):
    """
    Train with multiple rejection costs for RC curve analysis - CE Only Mode.

    THEORY-COMPLIANT VERSION:
    - Each c: optimize (alpha, mu) -> get ONE point (rejection_rate, error)
    - All points form ONE RC curve
    - Compute AURC from this single curve
    """

    print("\n" + "=" * 70)
    print("4. COST SWEEP OPTIMIZATION (Theory-Compliant) - CE Only")
    print("=" * 70)
    print("Each cost c gives ONE optimal point on RC curve")

    # Use fixed cost grid approach (like paper)
    print("Using fixed cost grid approach (Paper Method)")
    cost_sweep = CONFIG["ltr"]["cost_sweep"]
    print(f"   Cost grid: {cost_sweep}")

    # Load test data once (for efficiency)
    expert_logits_test = load_single_expert_logits(
        CONFIG["expert"]["name"], CONFIG["expert"]["logits_dir"], "test", DEVICE
    )
    labels_test = load_labels(CONFIG["dataset"]["splits_dir"], "test", DEVICE)

    with torch.no_grad():
        posterior_test = F.softmax(expert_logits_test, dim=-1)

    sample_weights_test = None
    if CONFIG["evaluation"]["use_reweighting"]:
        sample_weights_test = load_sample_weights(
            CONFIG["dataset"]["splits_dir"], "test", DEVICE
        )

    # Collect results: each cost gives ONE point
    results_per_cost = []

        # For building the unified RC curve - track BOTH balanced and worst errors
    val_points_balanced = {"rejection_rates": [], "errors": [], "costs": []}
    val_points_worst = {"rejection_rates": [], "errors": [], "costs": []}
    test_points_balanced = {"rejection_rates": [], "errors": [], "costs": []}
    test_points_worst = {"rejection_rates": [], "errors": [], "costs": []}

    for i, cost in enumerate(cost_sweep):
        # Only try to access target_rejection_rates if it exists and matches cost_sweep length
        target_rejection_rate = None
        if (
            hasattr(ltr_config, "target_rejection_rates")
            and ltr_config.target_rejection_rates is not None
        ):
            if i < len(ltr_config.target_rejection_rates):
                target_rejection_rate = ltr_config.target_rejection_rates[i]

        print(f"\n{'=' * 70}")
        if target_rejection_rate is not None:
            print(
                f"Training with target_rejection_rate = {target_rejection_rate:.2f}, cost = {cost:.4f}"
            )
        else:
            print(f"Training with rejection_cost = {cost}")
        print(f"{'=' * 70}")

        # For Algorithm 2 (Worst-Group), run full optimization for each cost
        if hasattr(optimizer_obj, "num_outer_iters"):  # Worst-group optimizer
            print(f"   Running Algorithm 2 for cost {cost:.4f}...")

            # Set the cost in the plugin configuration and optimizer config
            plugin.set_parameters(cost=cost)
            ltr_config.cost_grid = [cost]  # Set cost in config for Algorithm 2
            optimizer_obj.config = ltr_config

            # Run Algorithm 2 with this specific cost
            best_result = optimizer_obj.search(
                plugin,
                posterior_s1,
                labels_s1,
                posterior_s2,
                labels_s2,
                sample_weights_s1=sample_weights_s1,
                sample_weights_s2=sample_weights_s2,
                verbose=True,  # Enable verbose to see Algorithm 2 progress
            )

            print(f"   Algorithm 2 completed for cost {cost:.4f}")
            print(f"   alpha = {best_result.alpha}")
            print(f"   mu = {best_result.mu}")
            print(f"   Worst-group error: {best_result.worst_group_error:.4f}")

        else:
            # For Algorithm 1/Grid Search: optimize μ, then lock μ and sweep costs at fixed targets
            ltr_config.cost_grid = [cost]
            optimizer_obj.config = ltr_config

            # If we have a target rejection rate, first find μ at this cost
            if target_rejection_rate is not None:
                best_result_mu = optimizer_obj.search(
                    plugin,
                    posterior_val,
                    labels_val,
                    sample_weights=sample_weights_val,
                    verbose=verbose,
                )
                # Compute cost that achieves the exact target rejection with found (α, μ)
                plugin.set_parameters(
                    alpha=torch.tensor(best_result_mu.alpha, dtype=torch.float32, device=DEVICE),
                    mu=torch.tensor(best_result_mu.mu, dtype=torch.float32, device=DEVICE),
                )
                cost = plugin.compute_cost_for_target_rejection_rate(
                    posterior_val, target_rejection_rate
                )
                # Proper 1D λ reparameterization for K=2: set μ = [0, λ]
                from copy import deepcopy
                lambda_grid = CONFIG["ltr"].get("lambda_grid", None)
                if lambda_grid is None:
                    # finer λ grid per your request
                    lambda_grid = [x/10.0 for x in range(-15, 16)]  # [-1.5, ..., 1.5]
                def eval_with_lambda(lmb):
                    mu_try = np.array([0.0, float(lmb)], dtype=float)
                    return optimizer_obj.search(
                        plugin,
                        posterior_val,
                        labels_val,
                        sample_weights=sample_weights_val,
                        verbose=False,
                        fixed_mu=mu_try,
                        fixed_cost=cost,
                    )
                best_result = None
                best_obj = float('inf')
                for lmb in lambda_grid:
                    res = eval_with_lambda(lmb)
                    if res.objective_value < best_obj:
                        best_obj = res.objective_value
                        best_result = res
                # Final re-evaluate with fixed μ and computed cost
                best_result = optimizer_obj.search(
                    plugin,
                    posterior_val,
                    labels_val,
                    sample_weights=sample_weights_val,
                    verbose=False,
                    fixed_mu=best_result.mu,
                    fixed_cost=cost,
                )
            else:
                # No target provided, just optimize normally for this cost
                best_result = optimizer_obj.search(
                    plugin,
                    posterior_val,
                    labels_val,
                    sample_weights=sample_weights_val,
                    verbose=verbose,
                )

            # Set optimal parameters for this cost
            alpha_tensor = torch.tensor(
                best_result.alpha, dtype=torch.float32, device=DEVICE
            )
            mu_tensor = torch.tensor(best_result.mu, dtype=torch.float32, device=DEVICE)
            plugin.set_parameters(
                alpha=alpha_tensor, mu=mu_tensor, cost=best_result.cost
            )

        # Paper-compliant: single point per cost (no further re-thresholding)
        with torch.no_grad():
            preds_val = plugin.predict_class(posterior_val)
            reject_val = plugin.predict_reject(posterior_val)
            m_val = compute_selective_metrics(preds_val, labels_val, reject_val, plugin.class_to_group, None)
            preds_test = plugin.predict_class(posterior_test)
            reject_test = plugin.predict_reject(posterior_test)
            m_test = compute_selective_metrics(preds_test, labels_test, reject_test, plugin.class_to_group, None)

        val_error = float(np.mean(m_val["group_errors"])) if objective == "balanced" else m_val["worst_group_error"]
        test_error = float(np.mean(m_test["group_errors"])) if objective == "balanced" else m_test["worst_group_error"]

        # Store detailed results
        result_dict = {
            "cost": float(cost),
            "alpha": best_result.alpha.tolist(),
            "mu": best_result.mu.tolist(),
            "val_metrics": {
                "selective_error": float(m_val["selective_error"]),
                "coverage": float(m_val["coverage"]),
                "rejection_rate": float(1.0 - m_val["coverage"]),
                "group_errors": [float(g) for g in m_val["group_errors"]],
                "worst_group_error": float(m_val["worst_group_error"]),
                "balanced_error": val_error,
                "objective_value": best_result.objective_value,
            },
            "test_metrics": {
                "selective_error": float(m_test["selective_error"]),
                "coverage": float(m_test["coverage"]),
                "rejection_rate": float(1.0 - m_test["coverage"]),
                "group_errors": [float(g) for g in m_test["group_errors"]],
                "worst_group_error": float(m_test["worst_group_error"]),
                "balanced_error": test_error,
            },
        }

        results_per_cost.append(result_dict)

        print(f"\nSUCCESS: Cost={cost}:")
        print(f"   alpha = {best_result.alpha}")
        print(f"   mu = {best_result.mu}")
        print(
            f"   VAL: error={val_error:.4f}, coverage={m_val['coverage']:.3f}, rejection={1.0 - m_val['coverage']:.3f}"
        )
        print(
            f"   TEST: error={test_error:.4f}, coverage={m_test['coverage']:.3f}, rejection={1.0 - m_test['coverage']:.3f}"
        )

    # Build unified RC curves from one-point-per-cost (sorted by rejection)
    print(f"\n{'=' * 70}")
    print("5. BUILDING UNIFIED RC CURVES (Balanced & Worst)")
    print("=" * 70)

    def build_rc_curve_from_results(results, metric_key):
        """Helper function to build RC curve from data dictionaries."""
        val_rejection_rates = np.array([1.0 - r["val_metrics"]["coverage"] for r in results])
        test_rejection_rates = np.array([1.0 - r["test_metrics"]["coverage"] for r in results])
        val_errors = np.array([r["val_metrics"][metric_key] for r in results])
        test_errors = np.array([r["test_metrics"][metric_key] for r in results])

        # Sort by rejection rate (CRITICAL for AURC calculation)
        val_sort_idx = np.argsort(val_rejection_rates)
        val_rejection_rates = val_rejection_rates[val_sort_idx]
        val_errors = val_errors[val_sort_idx]

        test_sort_idx = np.argsort(test_rejection_rates)
        test_rejection_rates = test_rejection_rates[test_sort_idx]
        test_errors = test_errors[test_sort_idx]

        # Compute AURC using trapezoid rule
        val_aurc = np.trapz(val_errors, val_rejection_rates)
        test_aurc = np.trapz(test_errors, test_rejection_rates)

        return {
            "val": {
                "rejection_rates": val_rejection_rates,
                "selective_errors": val_errors,
                "aurc": val_aurc,
            },
            "test": {
                "rejection_rates": test_rejection_rates,
                "selective_errors": test_errors,
                "aurc": test_aurc,
            },
        }

    # Build both curves (now based on fixed-grid target points)
    rc_curve_balanced = build_rc_curve_from_results(results_per_cost, "balanced_error")
    rc_curve_worst = build_rc_curve_from_results(results_per_cost, "worst_group_error")

    # Store unified RC curves data
    unified_rc_curves = {
        "balanced": {
            "val": {
                "rejection_rates": rc_curve_balanced["val"]["rejection_rates"].tolist(),
                "selective_errors": rc_curve_balanced["val"][
                    "selective_errors"
                ].tolist(),
                "aurc": float(rc_curve_balanced["val"]["aurc"]),
            },
            "test": {
                "rejection_rates": rc_curve_balanced["test"][
                    "rejection_rates"
                ].tolist(),
                "selective_errors": rc_curve_balanced["test"][
                    "selective_errors"
                ].tolist(),
                "aurc": float(rc_curve_balanced["test"]["aurc"]),
            },
        },
        "worst": {
            "val": {
                "rejection_rates": rc_curve_worst["val"]["rejection_rates"].tolist(),
                "selective_errors": rc_curve_worst["val"]["selective_errors"].tolist(),
                "aurc": float(rc_curve_worst["val"]["aurc"]),
            },
            "test": {
                "rejection_rates": rc_curve_worst["test"]["rejection_rates"].tolist(),
                "selective_errors": rc_curve_worst["test"]["selective_errors"].tolist(),
                "aurc": float(rc_curve_worst["test"]["aurc"]),
            },
        },
    }

    # Save all results
    results_dir = Path(CONFIG["output"]["results_dir"]) / CONFIG["dataset"]["name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "expert": CONFIG["expert"]["name"],
        "objective": objective,
        "param_mode": CONFIG["ltr"]["param_mode"],
        "cost_sweep": cost_sweep,
        "results_per_cost": results_per_cost,
        "unified_rc_curves": unified_rc_curves,  # Both balanced and worst RC curves
        "theory_compliant": True,  # Flag to indicate this follows paper methodology
        "description": "Each cost optimizes (alpha, mu) and contributes ONE point to the unified RC curve",
    }

    output_path = results_dir / f"ltr_plugin_ce_only_cost_sweep_{objective}.json"
    with open(output_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"\nSUCCESS: Saved cost sweep results to: {output_path}")

    # Plot the unified RC curves
    print("\n" + "=" * 70)
    print("6. PLOTTING UNIFIED RC CURVES (Balanced & Worst)")
    print("=" * 70)

    # Plot balanced curve
    plot_rc_curves_ltr(
        rc_curve_balanced["val"],
        rc_curve_balanced["test"],
        results_dir,
        f"balanced_ce_only",
    )

    # Plot worst curve
    plot_rc_curves_ltr(
        rc_curve_worst["val"],
        rc_curve_worst["test"],
        results_dir,
        f"worst_ce_only",
    )

    # Plot both curves together
    plot_rc_curves_ltr_dual(
        rc_curve_balanced["test"],
        rc_curve_worst["test"],
        results_dir,
        "ce_only",
    )

    return results_per_cost, unified_rc_curves


def evaluate_on_test_ce_only(
    plugin, expert_name, logits_dir, splits_dir, use_reweighting
):
    """Evaluate plugin on test set - CE Only Mode."""
    # Load test data
    expert_logits_test = load_single_expert_logits(expert_name, logits_dir, "test", DEVICE)
    labels_test = load_labels(splits_dir, "test", DEVICE)

    # Compute posteriors
    with torch.no_grad():
        posterior_test = F.softmax(expert_logits_test, dim=-1)

        # Predictions
        predictions = plugin.predict_class(posterior_test)
        reject = plugin.predict_reject(posterior_test)

    # Sample weights
    sample_weights_test = None
    if use_reweighting:
        sample_weights_test = load_sample_weights(splits_dir, "test", DEVICE)

    # Metrics
    metrics = compute_selective_metrics(
        predictions, labels_test, reject, plugin.class_to_group, sample_weights_test
    )

    print(f"\nTEST RESULTS:")
    print(f"   Selective error: {metrics['selective_error']:.4f}")
    print(f"   Coverage: {metrics['coverage']:.3f}")
    print(f"   Group errors: {[f'{e:.4f}' for e in metrics['group_errors']]}")
    print(f"   Worst group error: {metrics['worst_group_error']:.4f}")

    return metrics


# ============================================================================
# PLOTTING & AURC
# ============================================================================


def mean_risk(r: np.ndarray, e: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Compute mean risk over rejection range [lo, hi].

    Args:
        r: rejection rates
        e: errors
        lo: lower bound
        hi: upper bound

    Returns:
        mean risk (integral / width)
    """
    mask = (r >= lo) & (r <= hi)
    if mask.sum() < 2:
        return 0.0
    r_m, e_m = r[mask], e[mask]
    integral = np.trapz(e_m, r_m)
    width = max(hi - lo, 1e-6)
    return integral / width


def plot_rc_curves_ltr(
    rc_data_val: Dict, rc_data_test: Dict, output_dir: Path, objective: str
):
    """
    Plot RC curves with 3 panels (similar to train_map_cost_sweep.py):
    1. Error vs Rejection Rate (full range 0-1)
    2. Error vs Rejection Rate (practical range 0-0.8)
    3. AURC Comparison (Full vs Practical 0.2-1.0)

    Uses TEST data for all plots.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract TEST data (paper-compliant: use balanced error for AURC and curve)
    rejection_rates = rc_data_test["rejection_rates"]
    errors = rc_data_test.get("balanced_errors", rc_data_test["selective_errors"])  # fallback
    aurc_test = rc_data_test["aurc"]

    # ========================================================================
    # Plot 1: Error vs Rejection Rate (Full range 0-1)
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(
        rejection_rates,
        errors,
        "o-",
        linewidth=3,
        markersize=6,
        label=f"{objective.capitalize()} (AURC={aurc_test:.4f})",
        color="green",
        markerfacecolor="lightgreen",
        markeredgecolor="darkgreen",
        markeredgewidth=1,
    )

    ax1.set_xlabel("Proportion of Rejections", fontsize=12)
    ax1.set_ylabel("Error", fontsize=12)
    ax1.set_title(f"Error vs Rejection Rate (0-1)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, min(1.05, errors.max() * 1.1)])

    # ========================================================================
    # Plot 2: Error vs Rejection Rate (Practical range 0-0.8)
    # ========================================================================
    ax2 = axes[1]

    # Filter data for rejection rate <= 0.8
    mask = rejection_rates <= 0.8
    rejection_practical = rejection_rates[mask]
    errors_practical = errors[mask]

    ax2.plot(
        rejection_practical,
        errors_practical,
        "o-",
        linewidth=3,
        markersize=6,
        label=f"{objective.capitalize()}",
        color="green",
        markerfacecolor="lightgreen",
        markeredgecolor="darkgreen",
        markeredgewidth=1,
    )

    ax2.set_xlabel("Proportion of Rejections", fontsize=12)
    ax2.set_ylabel("Error", fontsize=12)
    ax2.set_title(f"Error vs Rejection Rate (0-0.8)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.8])
    ax2.set_ylim([0, max(errors_practical) * 1.1 if len(errors_practical) > 0 else 1.0])

    # ========================================================================
    # Plot 3: AURC Comparison (Full 0-1 vs Practical 0.2-1.0)
    # ========================================================================
    ax3 = axes[2]

    # Compute mean risks on balanced error
    mean_risk_full = mean_risk(rejection_rates, errors, 0.0, 1.0)
    mean_risk_practical = mean_risk(rejection_rates, errors, 0.2, 1.0)

    # Bar chart
    x_pos = np.arange(1)
    width = 0.35

    bar1 = ax3.bar(
        x_pos - width / 2,
        [mean_risk_full],
        width,
        label="Full (0-1)",
        color="green",
        alpha=0.7,
    )
    bar2 = ax3.bar(
        x_pos + width / 2,
        [mean_risk_practical],
        width,
        label="Practical (0.2-1.0)",
        color="green",
        alpha=0.4,
        hatch="///",
    )

    ax3.set_ylabel("AURC", fontsize=12)
    ax3.set_title("AURC Comparison (Full vs 0.2-1.0)", fontsize=14, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([objective.capitalize()])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bar1:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar in bar2:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add percentage difference annotation
    if mean_risk_full > 0:
        pct_diff = ((mean_risk_practical - mean_risk_full) / mean_risk_full) * 100
        ax3.text(
            0,
            max(mean_risk_full, mean_risk_practical) * 0.95,
            f"{pct_diff:+.1f}% (Practical vs Full)",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
        )

    plt.tight_layout()

    plot_path = output_dir / f"ltr_rc_curves_{objective}_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"SUCCESS: Saved plot to: {plot_path}")

    plt.close()


def plot_rc_curves_ltr_dual(rc_data_balanced, rc_data_worst, output_dir, suffix=""):
    """
    Plot both balanced and worst error RC curves together for comparison.

    Args:
        rc_data_balanced: dict with 'rejection_rates', 'selective_errors', 'aurc'
        rc_data_worst: dict with 'rejection_rates', 'selective_errors', 'aurc'
        output_dir: Path to save plot
        suffix: suffix for filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Extract data
    rejection_rates_balanced = rc_data_balanced["rejection_rates"]
    errors_balanced = rc_data_balanced["selective_errors"]
    aurc_balanced = rc_data_balanced["aurc"]

    rejection_rates_worst = rc_data_worst["rejection_rates"]
    errors_worst = rc_data_worst["selective_errors"]
    aurc_worst = rc_data_worst["aurc"]

    # Compute AURC for 0-0.8 range
    mask_balanced_08 = rejection_rates_balanced <= 0.8
    if mask_balanced_08.sum() > 1:
        aurc_balanced_08 = np.trapz(
            errors_balanced[mask_balanced_08],
            rejection_rates_balanced[mask_balanced_08],
        )
    else:
        aurc_balanced_08 = 0.0

    mask_worst_08 = rejection_rates_worst <= 0.8
    if mask_worst_08.sum() > 1:
        aurc_worst_08 = np.trapz(
            errors_worst[mask_worst_08], rejection_rates_worst[mask_worst_08]
        )
    else:
        aurc_worst_08 = 0.0

    # ========================================================================
    # Plot 1: Full range (0-1)
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(
        rejection_rates_balanced,
        errors_balanced,
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Balanced Error (AURC={aurc_balanced:.4f})",
        color="blue",
    )
    ax1.plot(
        rejection_rates_worst,
        errors_worst,
        "s-",
        linewidth=2,
        markersize=4,
        label=f"Worst Error (AURC={aurc_worst:.4f})",
        color="red",
    )

    ax1.set_xlabel("Proportion of Rejections", fontsize=12)
    ax1.set_ylabel("Error", fontsize=12)
    ax1.set_title("Error vs Rejection Rate (0-1)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])

    # Auto-adjust ylim to accommodate both curves
    y_max = max(
        errors_balanced.max() if len(errors_balanced) > 0 else 0,
        errors_worst.max() if len(errors_worst) > 0 else 0,
    )
    ax1.set_ylim([0, min(1.05, y_max * 1.1)])

    # ========================================================================
    # Plot 2: Practical range (0-0.8)
    # ========================================================================
    ax2 = axes[1]

    # Filter data for rejection rate <= 0.8
    rejection_balanced_08 = rejection_rates_balanced[mask_balanced_08]
    errors_balanced_08 = errors_balanced[mask_balanced_08]
    rejection_worst_08 = rejection_rates_worst[mask_worst_08]
    errors_worst_08 = errors_worst[mask_worst_08]

    ax2.plot(
        rejection_balanced_08,
        errors_balanced_08,
        "o-",
        linewidth=2,
        markersize=4,
        label=f"Balanced Error (AURC={aurc_balanced_08:.4f})",
        color="blue",
    )
    ax2.plot(
        rejection_worst_08,
        errors_worst_08,
        "s-",
        linewidth=2,
        markersize=4,
        label=f"Worst Error (AURC={aurc_worst_08:.4f})",
        color="red",
    )

    ax2.set_xlabel("Proportion of Rejections", fontsize=12)
    ax2.set_ylabel("Error", fontsize=12)
    ax2.set_title("Error vs Rejection Rate (0-0.8)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.8])
    y_max_08 = max(
        errors_balanced_08.max() if len(errors_balanced_08) > 0 else 0,
        errors_worst_08.max() if len(errors_worst_08) > 0 else 0,
    )
    ax2.set_ylim([0, max(y_max_08 * 1.1, 0.01) if y_max_08 > 0 else 1.0])

    # ========================================================================
    # Plot 3: AURC Comparison (Full 0-1 vs Practical 0-0.8)
    # ========================================================================
    ax3 = axes[2]

    # Bar chart with grouped bars
    x_pos = np.arange(2)
    width = 0.35

    # Full range bars (0-1)
    bars1 = ax3.bar(
        x_pos - width / 2,
        [aurc_balanced, aurc_worst],
        width,
        label="Full (0-1)",
        color=["blue", "red"],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Practical range bars (0-0.8)
    bars2 = ax3.bar(
        x_pos + width / 2,
        [aurc_balanced_08, aurc_worst_08],
        width,
        label="Practical (0-0.8)",
        color=["lightblue", "lightcoral"],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        hatch="///",
    )

    ax3.set_ylabel("AURC", fontsize=12)
    ax3.set_title("AURC Comparison (Full vs Practical)", fontsize=14, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(["Balanced Error", "Worst Error"], fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars, aurc_vals in [
        (bars1, [aurc_balanced, aurc_worst]),
        (bars2, [aurc_balanced_08, aurc_worst_08]),
    ]:
        for bar, aurc_val in zip(bars, aurc_vals):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{aurc_val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Add percentage difference annotations
    if aurc_balanced > 0:
        pct_diff_full = ((aurc_worst - aurc_balanced) / aurc_balanced) * 100
        pct_diff_08 = (
            ((aurc_worst_08 - aurc_balanced_08) / aurc_balanced_08) * 100
            if aurc_balanced_08 > 0
            else 0
        )
        ax3.text(
            0.5,
            max(aurc_balanced, aurc_worst, aurc_balanced_08, aurc_worst_08) * 0.95,
            f"Full: {pct_diff_full:+.1f}%\n0-0.8: {pct_diff_08:+.1f}%",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.6),
        )

    plt.tight_layout()

    plot_path = output_dir / f"ltr_rc_curves_dual_{suffix}_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"SUCCESS: Saved dual plot to: {plot_path}")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train LtR Plugin - CE Only Mode")
    parser.add_argument(
        "--objective",
        type=str,
        default="balanced",
        choices=["balanced", "worst"],
        help="Optimization objective",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="power_iter",
        choices=["power_iter", "grid", "worst_group"],
        help="Optimizer type: power_iter (Algorithm 1), grid (baseline), worst_group (Algorithm 2)",
    )
    parser.add_argument(
        "--cost_sweep",
        action="store_true",
        help="Train with multiple costs for RC curve",
    )
    parser.add_argument(
        "--no_reweight", action="store_true", help="Disable reweighting on test set"
    )

    args = parser.parse_args()

    if args.no_reweight:
        CONFIG["evaluation"]["use_reweighting"] = False

    train_ltr_plugin_ce_only(
        objective=args.objective,
        cost_sweep=args.cost_sweep,
        optimizer_type=args.optimizer,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

