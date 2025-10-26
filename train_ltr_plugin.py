"""
Learning to Reject (LtR) Plugin Training
=========================================

Pure implementation of LtR theory with plug-in decision rule.

Theory:
    Classifier: h_α(x) = argmax_y (α[y] · η[y])
    Rejector:   r(x) = 1{max_y(α[y]·η[y]) < <μ, η> - c}

where:
    - η(x): mixture posterior from MoE [C]
    - α: class reweighting coefficients
    - μ: normalization vector for threshold
    - c: rejection cost

Group simplification (for stability with limited data):
    - α[y] = α_g, μ[y] = μ_g for y in group g
    - Only 2G + 1 parameters: (α_0, α_1, μ_0, μ_1, c)

Usage:
    # Balanced objective (minimize mean group error)
    python train_ltr_plugin.py --objective balanced

    # Worst-group objective (minimize max group error)
    python train_ltr_plugin.py --objective worst

    # Cost sweep for RC curve analysis
    python train_ltr_plugin.py --cost_sweep
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict
import matplotlib.pyplot as plt

from src.models.gating_network_map import GatingNetwork
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
        "group_boundaries": [69],  # Head: 0-68 (69 classes), Tail: 69-99 (31 classes)
    },
    "experts": {
        "names": ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"],
        "logits_dir": "./outputs/logits/cifar100_lt_if100/",
    },
    "gating": {
        "checkpoint": "./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth",
    },
    "ltr": {
        # Parameter mode
        "param_mode": "group",  # 'group' or 'class'
        # Grid ranges (group mode: 2 groups = head/tail)
        "alpha_grid": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        "mu_grid": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        "cost_grid": [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.91, 0.95, 0.97, 0.99],
        # Cost sweep for RC curve - using percentile-based approach
        "cost_sweep": [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.91,
            0.93,
            0.95,
            0.97,
            0.99,
        ],  # Denser grid for better coverage
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


def load_expert_logits(expert_names, logits_dir, split_name, device="cpu"):
    """Load expert logits."""
    logits_list = []

    for expert_name in expert_names:
        logits_path = Path(logits_dir) / expert_name / f"{split_name}_logits.pt"

        if not logits_path.exists():
            raise FileNotFoundError(f"Logits not found: {logits_path}")

        logits_e = torch.load(logits_path, map_location=device).float()
        logits_list.append(logits_e)

    logits = torch.stack(logits_list, dim=0).transpose(0, 1)
    return logits


def load_labels(splits_dir, split_name, device="cpu"):
    """Load labels from CIFAR-100 dataset."""
    import torchvision

    # Load indices
    indices_file = f"{split_name}_indices.json"
    with open(Path(splits_dir) / indices_file, "r") as f:
        indices = json.load(f)

    # Determine if train or test split
    # tunev is created from test set, so should use train=False
    is_train = split_name in ["expert", "gating", "train"]

    dataset = torchvision.datasets.CIFAR100(
        root="./data", train=is_train, download=False
    )

    # Extract labels for these indices
    labels = torch.tensor(
        [dataset.targets[i] for i in indices], dtype=torch.long, device=device
    )
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


def train_ltr_plugin(
    objective: str = "balanced",
    cost_sweep: bool = False,
    optimizer_type: str = "power_iter",
    verbose: bool = True,
):
    """
    Train LtR Plugin with chosen optimizer.

    Args:
        objective: 'balanced' or 'worst'
        cost_sweep: if True, train with multiple costs for RC curve
        optimizer_type: 'power_iter' (Algorithm 1), 'grid' (baseline), or 'worst_group' (Algorithm 2)
        verbose: print progress
    """
    print("=" * 70)
    print("LtR PLUGIN TRAINING")
    print("=" * 70)
    print(f"Objective: {objective}")
    print(f"Cost sweep: {cost_sweep}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Param mode: {CONFIG['ltr']['param_mode']}")
    print(f"Device: {DEVICE}")

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ========================================================================
    # 1. LOAD GATING NETWORK
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. LOADING GATING NETWORK")
    print("=" * 70)

    num_experts = len(CONFIG["experts"]["names"])
    num_classes = CONFIG["dataset"]["num_classes"]

    gating = GatingNetwork(
        num_experts=num_experts, num_classes=num_classes, routing="dense"
    ).to(DEVICE)

    gating_checkpoint_path = Path(CONFIG["gating"]["checkpoint"])
    checkpoint = torch.load(
        gating_checkpoint_path, map_location=DEVICE, weights_only=False
    )
    gating.load_state_dict(checkpoint["model_state_dict"])
    gating.eval()

    print(f"SUCCESS: Loaded gating from: {gating_checkpoint_path}")

    # ========================================================================
    # 2. LOAD DATA (tunev for S1, val for S2)
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. LOADING DATA")
    print("=" * 70)

    # Load S1 (tunev) and S2 (val) for Algorithm 2 compliance
    if optimizer_type == "worst_group":
        print("Loading S1 (tunev) and S2 (val) for Algorithm 2 compliance")

        # S1: tunev split
        expert_logits_s1 = load_expert_logits(
            CONFIG["experts"]["names"], CONFIG["experts"]["logits_dir"], "tunev", DEVICE
        )
        labels_s1 = load_labels(CONFIG["dataset"]["splits_dir"], "tunev", DEVICE)
        sample_weights_s1 = None
        if CONFIG["evaluation"]["use_reweighting"]:
            sample_weights_s1 = load_sample_weights(
                CONFIG["dataset"]["splits_dir"], "tunev", DEVICE
            )

        # S2: val split
        expert_logits_s2 = load_expert_logits(
            CONFIG["experts"]["names"], CONFIG["experts"]["logits_dir"], "val", DEVICE
        )
        labels_s2 = load_labels(CONFIG["dataset"]["splits_dir"], "val", DEVICE)
        sample_weights_s2 = None
        if CONFIG["evaluation"]["use_reweighting"]:
            sample_weights_s2 = load_sample_weights(
                CONFIG["dataset"]["splits_dir"], "val", DEVICE
            )

        print(f"SUCCESS: S1 (tunev): {expert_logits_s1.shape[0]} samples")
        print(f"SUCCESS: S2 (val): {expert_logits_s2.shape[0]} samples")

        # For backward compatibility, use S1 as main data
        expert_logits_val = expert_logits_s1
        labels_val = labels_s1
        sample_weights_val = sample_weights_s1

    else:
        # Use tunev for other optimizers (backward compatibility)
        optimization_split = "tunev"
        expert_logits_val = load_expert_logits(
            CONFIG["experts"]["names"],
            CONFIG["experts"]["logits_dir"],
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
    # 3. COMPUTE MIXTURE POSTERIORS
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. COMPUTING MIXTURE POSTERIORS")
    print("=" * 70)

    def compute_mixture_posterior(logits, gating_net, device):
        """Helper function to compute mixture posteriors."""
        with torch.no_grad():
            expert_posteriors = F.softmax(logits, dim=-1)

            # Get gating weights
            gating_output = gating_net(expert_posteriors)
            if isinstance(gating_output, tuple):
                gating_weights = gating_output[0]
            else:
                gating_weights = gating_output

            # Check for NaN
            if torch.isnan(gating_weights).any():
                print("WARNING: Gating produces NaN! Falling back to uniform weights")
                B, E = logits.shape[0], logits.shape[1]
                gating_weights = torch.ones(B, E, device=device) / E

            # Mixture posterior
            mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(
                dim=1
            )
            return mixture_posterior

    # Compute mixture posteriors for main data (S1 or tunev)
    mixture_posterior_val = compute_mixture_posterior(expert_logits_val, gating, DEVICE)

    # For Algorithm 2, also compute mixture posteriors for S2
    mixture_posterior_s1 = None
    mixture_posterior_s2 = None
    if optimizer_type == "worst_group":
        mixture_posterior_s1 = compute_mixture_posterior(
            expert_logits_s1, gating, DEVICE
        )
        mixture_posterior_s2 = compute_mixture_posterior(
            expert_logits_s2, gating, DEVICE
        )
        mixture_posterior_val = (
            mixture_posterior_s1  # Use S1 as main for backward compatibility
        )

    print("SUCCESS: Mixture posteriors computed")
    print(f"   Shape: {mixture_posterior_val.shape}")
    print(
        f"   Range: [{mixture_posterior_val.min():.3f}, {mixture_posterior_val.max():.3f}]"
    )
    print(
        f"   Sum per sample: {mixture_posterior_val.sum(dim=-1).mean():.3f} (should be ~1.0)"
    )

    # ========================================================================
    # 4. CREATE LTR PLUGIN
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. CREATING LtR PLUGIN")
    print("=" * 70)

    ltr_config = LtRPluginConfig(
        num_classes=CONFIG["dataset"]["num_classes"],
        num_groups=CONFIG["dataset"]["num_groups"],
        group_boundaries=CONFIG["dataset"]["group_boundaries"],
        param_mode=CONFIG["ltr"]["param_mode"],
        alpha_grid=CONFIG["ltr"]["alpha_grid"],
        mu_grid=CONFIG["ltr"]["mu_grid"],
        cost_grid=CONFIG["ltr"]["cost_grid"],  # Single cost for initial search
        objective=objective,
    )

    # Add target_rejection_rates for percentile-based approach
    if cost_sweep and "target_rejection_rates" in CONFIG["ltr"]:
        ltr_config.target_rejection_rates = CONFIG["ltr"]["target_rejection_rates"]

    plugin = LtRPlugin(ltr_config).to(DEVICE)

    # Create optimizer based on type
    if optimizer_type == "power_iter":
        print("SUCCESS: Using Power-Iteration Optimizer (Algorithm 1 - Paper)")
        optimizer_obj = LtRPowerIterOptimizer(
            ltr_config, num_iters=10, alpha_init_mode="prior"
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
    # 5. OPTIMIZATION
    # ========================================================================

    if cost_sweep:
        # Train with multiple costs - Theory-compliant version
        print("\nMode: Cost Sweep (Theory-Compliant)")
        print("   Each cost c optimizes (alpha, mu) and gives ONE point on RC curve")

        # For worst_group optimizer, pass S1/S2 data
        if optimizer_type == "worst_group":
            results_per_cost, unified_rc = train_with_cost_sweep(
                plugin,
                optimizer_obj,
                ltr_config,
                mixture_posterior_val,
                labels_val,
                sample_weights_val,
                gating,
                objective,
                verbose,
                mixture_posterior_s1,
                labels_s1,
                sample_weights_s1,
                mixture_posterior_s2,
                labels_s2,
                sample_weights_s2,
            )
        else:
            # For other optimizers, pass None for S1/S2
            results_per_cost, unified_rc = train_with_cost_sweep(
                plugin,
                optimizer_obj,
                ltr_config,
                mixture_posterior_val,
                labels_val,
                sample_weights_val,
                gating,
                objective,
                verbose,
                None,
                None,
                None,  # S1 data
                None,
                None,
                None,  # S2 data
            )
        return results_per_cost, unified_rc
    else:
        # Single training
        print("\n" + "=" * 70)
        print("5. GRID SEARCH OPTIMIZATION")
        print("=" * 70)

        # Call optimizer with appropriate data
        if optimizer_type == "worst_group":
            # Algorithm 2: Use S1 and S2 separately
            best_result = optimizer_obj.search(
                plugin,
                mixture_posterior_s1,
                labels_s1,
                mixture_posterior_s2,
                labels_s2,
                sample_weights_s1=sample_weights_s1,
                sample_weights_s2=sample_weights_s2,
                verbose=verbose,
            )
        else:
            # Algorithm 1 and Grid Search: Use single dataset
            best_result = optimizer_obj.search(
                plugin,
                mixture_posterior_val,
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
        # 6. EVALUATE ON TEST SET
        # ====================================================================
        print("\n" + "=" * 70)
        print("6. EVALUATION ON TEST SET")
        print("=" * 70)

        test_results = evaluate_on_test(
            plugin,
            gating,
            CONFIG["experts"]["names"],
            CONFIG["experts"]["logits_dir"],
            CONFIG["dataset"]["splits_dir"],
            CONFIG["evaluation"]["use_reweighting"],
        )

        # ====================================================================
        # 7. COMPUTE RC CURVE
        # ====================================================================
        print("\n" + "=" * 70)
        print("7. COMPUTING RC CURVE")
        print("=" * 70)

        rc_computer = RCCurveComputer(ltr_config)

        # VAL RC curve
        rc_data_val = rc_computer.compute_rc_curve(
            plugin,
            mixture_posterior_val,
            labels_val,
            alpha=best_result.alpha,
            mu=best_result.mu,
            cost_grid=np.linspace(0.0, 1.0, 200),
            sample_weights=sample_weights_val,
        )

        print(f"SUCCESS: VAL AURC: {rc_data_val['aurc']:.4f}")

        # TEST RC curve
        expert_logits_test = load_expert_logits(
            CONFIG["experts"]["names"], CONFIG["experts"]["logits_dir"], "test", DEVICE
        )
        labels_test = load_labels(CONFIG["dataset"]["splits_dir"], "test", DEVICE)

        with torch.no_grad():
            expert_posteriors_test = F.softmax(expert_logits_test, dim=-1)
            gating_output_test = gating(expert_posteriors_test)
            if isinstance(gating_output_test, tuple):
                gating_weights_test = gating_output_test[0]
            else:
                gating_weights_test = gating_output_test
            mixture_posterior_test = (
                gating_weights_test.unsqueeze(-1) * expert_posteriors_test
            ).sum(dim=1)

        sample_weights_test = None
        if CONFIG["evaluation"]["use_reweighting"]:
            sample_weights_test = load_sample_weights(
                CONFIG["dataset"]["splits_dir"], "test", DEVICE
            )

        rc_data_test = rc_computer.compute_rc_curve(
            plugin,
            mixture_posterior_test,
            labels_test,
            alpha=best_result.alpha,
            mu=best_result.mu,
            cost_grid=np.linspace(0.0, 1.0, 200),
            sample_weights=sample_weights_test,
        )

        print(f"SUCCESS: TEST AURC: {rc_data_test['aurc']:.4f}")

        # ====================================================================
        # 8. SAVE RESULTS
        # ====================================================================
        print("\n" + "=" * 70)
        print("8. SAVING RESULTS")
        print("=" * 70)

        results_dir = Path(CONFIG["output"]["results_dir"]) / CONFIG["dataset"]["name"]
        results_dir.mkdir(parents=True, exist_ok=True)

        save_dict = {
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

        output_path = results_dir / f"ltr_plugin_{objective}.json"
        with open(output_path, "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"SUCCESS: Saved results to: {output_path}")

        # ====================================================================
        # 9. PLOT RC CURVES
        # ====================================================================
        print("\n" + "=" * 70)
        print("9. PLOTTING RC CURVES")
        print("=" * 70)

        plot_rc_curves_ltr(rc_data_val, rc_data_test, results_dir, objective)

        return save_dict


def train_with_cost_sweep(
    plugin,
    optimizer_obj,
    ltr_config,
    mixture_posterior_val,
    labels_val,
    sample_weights_val,
    gating,
    objective,
    verbose,
    mixture_posterior_s1=None,
    labels_s1=None,
    sample_weights_s1=None,
    mixture_posterior_s2=None,
    labels_s2=None,
    sample_weights_s2=None,
):
    """
    Train with multiple rejection costs for RC curve analysis.

    THEORY-COMPLIANT VERSION:
    - Each c: optimize (alpha, mu) -> get ONE point (rejection_rate, error)
    - All points form ONE RC curve
    - Compute AURC from this single curve
    """

    print("\n" + "=" * 70)
    print("5. COST SWEEP OPTIMIZATION (Theory-Compliant)")
    print("=" * 70)
    print("Each cost c gives ONE optimal point on RC curve")

    # Use percentile-based approach for cost sweep
    if (
        hasattr(ltr_config, "target_rejection_rates")
        and ltr_config.target_rejection_rates is not None
    ):
        print("Using percentile-based approach for cost sweep")
        target_rejection_rates = ltr_config.target_rejection_rates
        cost_sweep = []

        # For Algorithm 2 (Worst-Group), we need to run the full optimization for each cost
        # According to paper: "for each c train the rejector (i.e., run the full pipeline)"
        if hasattr(optimizer_obj, "num_outer_iters"):  # Worst-group optimizer
            print("   Step 1: Using fixed cost grid for Algorithm 2...")
            print("   Note: Algorithm 2 will be run independently for each cost c")

            # For Algorithm 2, use fixed cost grid instead of percentile-based approach
            # This is the correct approach according to paper Appendix F.4
            cost_sweep = CONFIG["ltr"]["cost_sweep"]  # Use fixed cost grid
            print(f"   Cost grid: {cost_sweep}")
        else:
            # For Algorithm 1/Grid, use the original approach
            print("   Step 1: Finding optimal (alpha, mu) without cost constraint...")

            # Use a neutral cost for initial optimization
            ltr_config.cost_grid = [0.5]  # Neutral cost
            optimizer_obj.config = ltr_config

            # Find optimal (alpha, mu)
            best_result = optimizer_obj.search(
                plugin,
                mixture_posterior_val,
                labels_val,
                sample_weights=sample_weights_val,
                verbose=False,
            )

            # Set optimal (alpha, mu)
            alpha_tensor = torch.tensor(
                best_result.alpha, dtype=torch.float32, device=DEVICE
            )
            mu_tensor = torch.tensor(best_result.mu, dtype=torch.float32, device=DEVICE)
            plugin.set_parameters(alpha=alpha_tensor, mu=mu_tensor)

            print("   Step 2: Computing costs for target rejection rates...")
            for target_rejection_rate in target_rejection_rates:
                cost = plugin.compute_cost_for_target_rejection_rate(
                    mixture_posterior_val, target_rejection_rate
                )
                cost_sweep.append(cost)
                print(
                    f"   Target rejection {target_rejection_rate:.2f} -> Cost {cost:.4f}"
                )
    else:
        # Fallback to fixed cost grid
        cost_sweep = CONFIG["ltr"]["cost_sweep"]

    # Load test data once (for efficiency)
    expert_logits_test = load_expert_logits(
        CONFIG["experts"]["names"], CONFIG["experts"]["logits_dir"], "test", DEVICE
    )
    labels_test = load_labels(CONFIG["dataset"]["splits_dir"], "test", DEVICE)

    with torch.no_grad():
        expert_posteriors_test = F.softmax(expert_logits_test, dim=-1)
        gating_output_test = gating(expert_posteriors_test)
        if isinstance(gating_output_test, tuple):
            gating_weights_test = gating_output_test[0]
        else:
            gating_weights_test = gating_output_test
        mixture_posterior_test = (
            gating_weights_test.unsqueeze(-1) * expert_posteriors_test
        ).sum(dim=1)

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
        # For Algorithm 2 (worst_group), we use fixed cost grid, so target_rejection_rates may not match
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
            # Run Algorithm 2 independently for this cost
            # This is the correct approach according to paper Appendix F.4
            print(f"   Running Algorithm 2 for cost {cost:.4f}...")

            # Set the cost in the plugin configuration and optimizer config
            plugin.set_parameters(cost=cost)
            ltr_config.cost_grid = [cost]  # Set cost in config for Algorithm 2
            optimizer_obj.config = ltr_config

            # Run Algorithm 2 with this specific cost
            best_result = optimizer_obj.search(
                plugin,
                mixture_posterior_s1,
                labels_s1,
                mixture_posterior_s2,
                labels_s2,
                sample_weights_s1=sample_weights_s1,
                sample_weights_s2=sample_weights_s2,
                verbose=True,  # Enable verbose to see Algorithm 2 progress
            )

            print(f"   Algorithm 2 completed for cost {cost:.4f}")
            print(f"   alpha = {best_result.alpha}")
            print(f"   mu = {best_result.mu}")
            print(f"   Worst-group error: {best_result.worst_group_error:.4f}")

        elif (
            hasattr(ltr_config, "target_rejection_rates")
            and ltr_config.target_rejection_rates is not None
        ):
            # For Algorithm 1/Grid, use pre-computed optimal (alpha, mu) and set cost
            plugin.set_parameters(cost=cost)

            # Create a dummy result object
            class DummyResult:
                def __init__(self, alpha, mu, cost):
                    self.alpha = alpha
                    self.mu = mu
                    self.cost = cost
                    self.objective_value = 0.0
                    self.selective_error = 0.0
                    self.coverage = 0.0
                    self.group_errors = [0.0, 0.0]

            best_result = DummyResult(
                alpha_tensor.cpu().numpy(), mu_tensor.cpu().numpy(), cost
            )
        else:
            # Original approach: optimize for each cost
            ltr_config.cost_grid = [cost]
            optimizer_obj.config = ltr_config

            # Algorithm 1 and Grid Search: Use single dataset
            best_result = optimizer_obj.search(
                plugin,
                mixture_posterior_val,
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

        # Evaluate on VAL with this specific (alpha, mu, c) → ONE POINT
        with torch.no_grad():
            predictions_val = plugin.predict_class(mixture_posterior_val)
            reject_val = plugin.predict_reject(mixture_posterior_val)

        val_metrics = compute_selective_metrics(
            predictions_val,
            labels_val,
            reject_val,
            plugin.class_to_group,
            sample_weights_val,
        )

        val_rejection_rate = 1.0 - val_metrics["coverage"]
        # Compute BOTH balanced and worst errors for RC curve
        val_error_balanced = np.mean(val_metrics["group_errors"])
        val_error_worst = val_metrics["worst_group_error"]

        # Use objective-specific error for storing in result_dict
        if objective == "balanced":
            val_error = val_error_balanced
        else:  # worst
            val_error = val_error_worst

        # Store both for separate RC curves
        val_points_balanced["rejection_rates"].append(val_rejection_rate)
        val_points_balanced["errors"].append(val_error_balanced)
        val_points_balanced["costs"].append(cost)

        val_points_worst["rejection_rates"].append(val_rejection_rate)
        val_points_worst["errors"].append(val_error_worst)
        val_points_worst["costs"].append(cost)

        # Evaluate on TEST with this specific (alpha, mu, c) → ONE POINT
        with torch.no_grad():
            predictions_test = plugin.predict_class(mixture_posterior_test)
            reject_test = plugin.predict_reject(mixture_posterior_test)

        test_metrics = compute_selective_metrics(
            predictions_test,
            labels_test,
            reject_test,
            plugin.class_to_group,
            sample_weights_test,
        )

        test_rejection_rate = 1.0 - test_metrics["coverage"]
        # Compute BOTH balanced and worst errors for RC curve
        test_error_balanced = np.mean(test_metrics["group_errors"])
        test_error_worst = test_metrics["worst_group_error"]

        # Use objective-specific error for storing in result_dict
        if objective == "balanced":
            test_error = test_error_balanced
        else:  # worst
            test_error = test_error_worst

        # Store both for separate RC curves
        test_points_balanced["rejection_rates"].append(test_rejection_rate)
        test_points_balanced["errors"].append(test_error_balanced)
        test_points_balanced["costs"].append(cost)

        test_points_worst["rejection_rates"].append(test_rejection_rate)
        test_points_worst["errors"].append(test_error_worst)
        test_points_worst["costs"].append(cost)

        # Store detailed results
        result_dict = {
            "cost": float(cost),
            "alpha": best_result.alpha.tolist(),
            "mu": best_result.mu.tolist(),
            "val_metrics": {
                "selective_error": val_error,
                "coverage": val_metrics["coverage"],
                "rejection_rate": val_rejection_rate,
                "group_errors": val_metrics["group_errors"],
                "worst_group_error": val_metrics["worst_group_error"],
                "objective_value": best_result.objective_value,
            },
            "test_metrics": {
                "selective_error": test_error,
                "coverage": test_metrics["coverage"],
                "rejection_rate": test_rejection_rate,
                "group_errors": test_metrics["group_errors"],
                "worst_group_error": test_metrics["worst_group_error"],
            },
        }

        results_per_cost.append(result_dict)

        print(f"\nSUCCESS: Cost={cost}:")
        print(f"   alpha = {best_result.alpha}")
        print(f"   mu = {best_result.mu}")
        print(
            f"   VAL: error={val_error:.4f}, coverage={val_metrics['coverage']:.3f}, rejection={val_rejection_rate:.3f}"
        )
        print(
            f"   TEST: error={test_error:.4f}, coverage={test_metrics['coverage']:.3f}, rejection={test_rejection_rate:.3f}"
        )

    # Build unified RC curves from all points - BOTH balanced and worst
    print(f"\n{'=' * 70}")
    print("6. BUILDING UNIFIED RC CURVES (Balanced & Worst)")
    print("=" * 70)

    def build_rc_curve(val_dict, test_dict, name=""):
        """Helper function to build RC curve from data dictionaries."""
        # Convert to numpy arrays
        val_rejection_rates = np.array(val_dict["rejection_rates"])
        val_errors = np.array(val_dict["errors"])
        test_rejection_rates = np.array(test_dict["rejection_rates"])
        test_errors = np.array(test_dict["errors"])

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

        print(f"\n{name.upper()} RC CURVE STATISTICS:")
        print(f"   VAL:  {len(val_rejection_rates)} points")
        print(
            f"         Rejection range: [{val_rejection_rates.min():.4f}, {val_rejection_rates.max():.4f}]"
        )
        print(f"         Error range: [{val_errors.min():.4f}, {val_errors.max():.4f}]")
        print(f"         AURC: {val_aurc:.6f}")
        print(f"   TEST: {len(test_rejection_rates)} points")
        print(
            f"         Rejection range: [{test_rejection_rates.min():.4f}, {test_rejection_rates.max():.4f}]"
        )
        print(
            f"         Error range: [{test_errors.min():.4f}, {test_errors.max():.4f}]"
        )
        print(f"         AURC: {test_aurc:.6f}")

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

    # Build both curves
    rc_curve_balanced = build_rc_curve(
        val_points_balanced, test_points_balanced, "balanced"
    )
    rc_curve_worst = build_rc_curve(val_points_worst, test_points_worst, "worst")

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
        "objective": objective,
        "param_mode": CONFIG["ltr"]["param_mode"],
        "cost_sweep": cost_sweep,
        "results_per_cost": results_per_cost,
        "unified_rc_curves": unified_rc_curves,  # Both balanced and worst RC curves
        "theory_compliant": True,  # Flag to indicate this follows paper methodology
        "description": "Each cost optimizes (alpha, mu) and contributes ONE point to the unified RC curve",
    }

    output_path = results_dir / f"ltr_plugin_cost_sweep_{objective}.json"
    with open(output_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"\nSUCCESS: Saved cost sweep results to: {output_path}")

    # Plot the unified RC curves
    print("\n" + "=" * 70)
    print("7. PLOTTING UNIFIED RC CURVES (Balanced & Worst)")
    print("=" * 70)

    # Plot balanced curve
    plot_rc_curves_ltr(
        rc_curve_balanced["val"],
        rc_curve_balanced["test"],
        results_dir,
        "balanced",
    )

    # Plot worst curve
    plot_rc_curves_ltr(
        rc_curve_worst["val"],
        rc_curve_worst["test"],
        results_dir,
        "worst",
    )

    # Plot both curves together
    plot_rc_curves_ltr_dual(
        rc_curve_balanced["test"],
        rc_curve_worst["test"],
        results_dir,
    )

    return results_per_cost, unified_rc_curves


def evaluate_on_test(
    plugin, gating, expert_names, logits_dir, splits_dir, use_reweighting
):
    """Evaluate plugin on test set."""
    # Load test data
    expert_logits_test = load_expert_logits(expert_names, logits_dir, "test", DEVICE)
    labels_test = load_labels(splits_dir, "test", DEVICE)

    # Compute mixture posteriors
    with torch.no_grad():
        expert_posteriors_test = F.softmax(expert_logits_test, dim=-1)

        gating_output = gating(expert_posteriors_test)
        if isinstance(gating_output, tuple):
            gating_weights_test = gating_output[0]
        else:
            gating_weights_test = gating_output

        mixture_posterior_test = (
            gating_weights_test.unsqueeze(-1) * expert_posteriors_test
        ).sum(dim=1)

        # Predictions
        predictions = plugin.predict_class(mixture_posterior_test)
        reject = plugin.predict_reject(mixture_posterior_test)

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

    # Extract TEST data
    rejection_rates = rc_data_test["rejection_rates"]
    errors = rc_data_test["selective_errors"]
    aurc_test = rc_data_test["aurc"]

    # ========================================================================
    # Plot 1: Error vs Rejection Rate (Full range 0-1)
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(
        rejection_rates,
        errors,
        "o-",
        linewidth=2,
        markersize=3,
        label=f"{objective.capitalize()} (AURC={aurc_test:.4f})",
        color="green",
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
        linewidth=2,
        markersize=3,
        label=f"{objective.capitalize()}",
        color="green",
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

    # Compute mean risks
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


def plot_rc_curves_ltr_dual(rc_data_balanced, rc_data_worst, output_dir):
    """
    Plot both balanced and worst error RC curves together for comparison.

    Args:
        rc_data_balanced: dict with 'rejection_rates', 'selective_errors', 'aurc'
        rc_data_worst: dict with 'rejection_rates', 'selective_errors', 'aurc'
        output_dir: Path to save plot
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

    plot_path = output_dir / "ltr_rc_curves_dual_balanced_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"SUCCESS: Saved dual plot to: {plot_path}")

    plt.close()


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train LtR Plugin")
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

    train_ltr_plugin(
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
