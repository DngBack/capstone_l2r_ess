#!/usr/bin/env python3
"""
Plot Comparison Figures for LtR Plugin Methods
===============================================

Creates 4 line plots comparing different methods:
1. Balanced mode - Head Error vs Rejection Rate
2. Balanced mode - Tail Error vs Rejection Rate
3. Worst-group mode - Head Error vs Rejection Rate
4. Worst-group mode - Tail Error vs Rejection Rate
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        plt.style.use("seaborn")
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("./results/ltr_plugin/cifar100_lt_if100")
OUTPUT_DIR = Path("./results/comparison_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Method configuration
METHODS_CONFIG = {
    "ce_only_balanced": {
        "name": "CE-only",
        "style": {"linestyle": "-", "marker": "o", "linewidth": 2, "markersize": 6},
    },
    "logitadjustonly_balanced": {
        "name": "LogitAdjust-only",
        "style": {"linestyle": "-", "marker": "s", "linewidth": 2, "markersize": 6},
    },
    "balonly_balanced": {
        "name": "BalSoftmax-only",
        "style": {"linestyle": "-", "marker": "^", "linewidth": 2, "markersize": 6},
    },
    "uniform_balanced_3experts": {
        "name": "Uniform 3-Experts",
        "style": {"linestyle": "--", "marker": "s", "linewidth": 2, "markersize": 6},
    },
    "gating_balanced": {
        "name": "MoE (Gating)",
        "style": {"linestyle": "-.", "marker": "D", "linewidth": 2, "markersize": 6},
    },
    "ce_only_worst": {
        "name": "CE-only",
        "style": {"linestyle": "-", "marker": "o", "linewidth": 2, "markersize": 6},
    },
    "logitadjustonly_worst": {
        "name": "LogitAdjust-only",
        "style": {"linestyle": "-", "marker": "s", "linewidth": 2, "markersize": 6},
    },
    "balonly_worst": {
        "name": "BalSoftmax-only",
        "style": {"linestyle": "-", "marker": "^", "linewidth": 2, "markersize": 6},
    },
    "uniform_worst_3experts": {
        "name": "Uniform 3-Experts",
        "style": {"linestyle": "--", "marker": "s", "linewidth": 2, "markersize": 6},
    },
    "gating_worst": {
        "name": "MoE (Gating)",
        "style": {"linestyle": "-.", "marker": "D", "linewidth": 2, "markersize": 6},
    },
}

# ============================================================================
# DATA LOADING
# ============================================================================


def load_json_results(file_path: Path) -> Dict:
    """Load JSON results file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_rc_curve(
    data: Dict, split: str = "test"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract RC curve data: rejection_rates, balanced_errors, worst_group_errors, group_errors.
    Returns: (rejection_rates, head_errors, tail_errors, balanced_errors)
    """
    rc_curve = data.get("rc_curve", {})
    
    # Try nested structure first (balanced methods)
    if isinstance(rc_curve, dict) and split in rc_curve:
        curve_data = rc_curve[split]
        rejection_rates = np.array(curve_data.get("rejection_rates", []))
        balanced_errors = np.array(curve_data.get("balanced_errors", []))
        group_errors = curve_data.get("group_errors", [])
        
        if group_errors and isinstance(group_errors[0], list):
            head_errors = np.array([ge[0] for ge in group_errors if len(ge) > 0])
            tail_errors = np.array([ge[1] for ge in group_errors if len(ge) > 1])
        else:
            # Fallback: use worst_group_error as proxy
            worst_errors = np.array(curve_data.get("worst_group_errors", []))
            head_errors = worst_errors
            tail_errors = worst_errors
        
        # Sort by rejection rate
        if len(rejection_rates) > 0:
            sorted_idx = np.argsort(rejection_rates)
            rejection_rates = rejection_rates[sorted_idx]
            balanced_errors = balanced_errors[sorted_idx]
            if len(head_errors) == len(rejection_rates):
                head_errors = head_errors[sorted_idx]
            if len(tail_errors) == len(rejection_rates):
                tail_errors = tail_errors[sorted_idx]
        
        return rejection_rates, head_errors, tail_errors, balanced_errors
    
    # Try flat structure (worst-group methods)
    if isinstance(rc_curve, dict):
        # For worst-group methods, extract from results_per_point
        results = data.get("results_per_point", [])
        if results:
            rejection_rates_list = []
            head_errors_list = []
            tail_errors_list = []
            balanced_errors_list = []
            
            for result in results:
                metrics = result.get("test_metrics", {})
                if metrics:
                    rejection_rate = 1.0 - metrics.get("coverage", 1.0)
                    group_errors = metrics.get("group_errors", [0.0, 0.0])
                    balanced_error = metrics.get("balanced_error", 0.0)
                    
                    rejection_rates_list.append(rejection_rate)
                    head_errors_list.append(group_errors[0] if len(group_errors) > 0 else 0.0)
                    tail_errors_list.append(group_errors[1] if len(group_errors) > 1 else 0.0)
                    balanced_errors_list.append(balanced_error)
            
            if len(rejection_rates_list) > 0:
                rejection_rates = np.array(rejection_rates_list)
                head_errors = np.array(head_errors_list)
                tail_errors = np.array(tail_errors_list)
                balanced_errors = np.array(balanced_errors_list)
                
                # Sort by rejection rate
                sorted_idx = np.argsort(rejection_rates)
                rejection_rates = rejection_rates[sorted_idx]
                head_errors = head_errors[sorted_idx]
                tail_errors = tail_errors[sorted_idx]
                balanced_errors = balanced_errors[sorted_idx]
                
                return rejection_rates, head_errors, tail_errors, balanced_errors
        
        # Fallback: use rc_curve data if results_per_point not available
        rejection_rates = np.array(rc_curve.get("rejection_rates", []))
        balanced_errors = np.array(rc_curve.get("balanced_errors", []))
        worst_errors = np.array(rc_curve.get("worst_group_errors", []))
        
        # Use worst_errors as proxy for both head and tail (not ideal but better than nothing)
        if len(rejection_rates) > 0:
            sorted_idx = np.argsort(rejection_rates)
            rejection_rates = rejection_rates[sorted_idx]
            balanced_errors = balanced_errors[sorted_idx]
            head_errors = worst_errors[sorted_idx] if len(worst_errors) == len(rejection_rates) else worst_errors
            tail_errors = worst_errors[sorted_idx] if len(worst_errors) == len(rejection_rates) else worst_errors
            
            return rejection_rates, head_errors, tail_errors, balanced_errors
    
    return np.array([]), np.array([]), np.array([]), np.array([])


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_balanced_head_error():
    """Plot Figure 1: Balanced mode - Head Error vs Rejection Rate."""
    print("Plotting Figure 1: Balanced mode - Head Error...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" in method_key:
            continue  # Skip worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, head_errors, _, _ = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0 and len(head_errors) > 0:
            ax.plot(
                rejection_rates,
                head_errors,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Head Error", fontsize=12)
    ax.set_title("Balanced Mode: Head Error vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_balanced_head_error.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_balanced_tail_error():
    """Plot Figure 2: Balanced mode - Tail Error vs Rejection Rate."""
    print("Plotting Figure 2: Balanced mode - Tail Error...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" in method_key:
            continue  # Skip worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, _, tail_errors, _ = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0 and len(tail_errors) > 0:
            ax.plot(
                rejection_rates,
                tail_errors,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Tail Error", fontsize=12)
    ax.set_title("Balanced Mode: Tail Error vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_balanced_tail_error.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_worst_head_error():
    """Plot Figure 3: Worst-group mode - Head Error vs Rejection Rate."""
    print("Plotting Figure 3: Worst-group mode - Head Error...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" not in method_key:
            continue  # Only worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, head_errors, _, _ = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0 and len(head_errors) > 0:
            ax.plot(
                rejection_rates,
                head_errors,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Head Error", fontsize=12)
    ax.set_title("Worst-group Mode: Head Error vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_worst_head_error.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_worst_tail_error():
    """Plot Figure 4: Worst-group mode - Tail Error vs Rejection Rate."""
    print("Plotting Figure 4: Worst-group mode - Tail Error...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" not in method_key:
            continue  # Only worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, _, tail_errors, _ = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0 and len(tail_errors) > 0:
            ax.plot(
                rejection_rates,
                tail_errors,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Tail Error", fontsize=12)
    ax.set_title("Worst-group Mode: Tail Error vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_worst_tail_error.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_balanced_head_tail_ratio():
    """Plot Figure 5: Balanced mode - Head/Tail Ratio vs Rejection Rate."""
    print("Plotting Figure 5: Balanced mode - Head/Tail Ratio...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" in method_key:
            continue  # Skip worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, head_errors, tail_errors, _ = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0 and len(head_errors) > 0 and len(tail_errors) > 0:
            # Compute ratio: head/tail
            ratio = head_errors / (tail_errors + 1e-10)  # Avoid division by zero
            ax.plot(
                rejection_rates,
                ratio,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Head/Tail Error Ratio", fontsize=12)
    ax.set_title("Balanced Mode: Head/Tail Error Ratio vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Equal (1.0)")
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_balanced_head_tail_ratio.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_worst_head_tail_ratio():
    """Plot Figure 6: Worst-group mode - Head/Tail Ratio vs Rejection Rate."""
    print("Plotting Figure 6: Worst-group mode - Head/Tail Ratio...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" not in method_key:
            continue  # Only worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, head_errors, tail_errors, _ = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0 and len(head_errors) > 0 and len(tail_errors) > 0:
            # Compute ratio: head/tail
            ratio = head_errors / (tail_errors + 1e-10)  # Avoid division by zero
            ax.plot(
                rejection_rates,
                ratio,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Head/Tail Error Ratio", fontsize=12)
    ax.set_title("Worst-group Mode: Head/Tail Error Ratio vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Equal (1.0)")
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_worst_head_tail_ratio.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_balanced_balanced_risk():
    """Plot Figure 7a: Balanced mode - Balanced Risk (Balanced Error only)."""
    print("Plotting Figure 7a: Balanced mode - Balanced Risk...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" in method_key:
            continue  # Skip worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, _, _, balanced_errors = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0 and len(balanced_errors) > 0:
            ax.plot(
                rejection_rates,
                balanced_errors,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Risk (Balanced Error)", fontsize=12)
    ax.set_title("Balanced Mode: Balanced Risk vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_balanced_balanced_risk.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_balanced_worst_risk():
    """Plot Figure 7b: Balanced mode - Worst Risk (Worst-group Error only)."""
    print("Plotting Figure 7b: Balanced mode - Worst Risk...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" in method_key:
            continue  # Skip worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, _, _, _ = extract_rc_curve(data, "test")
        
        # Get worst-group errors
        rc_curve = data.get("rc_curve", {})
        worst_errors = None
        if isinstance(rc_curve, dict) and "test" in rc_curve:
            worst_errors = np.array(rc_curve["test"].get("worst_group_errors", []))
            if len(worst_errors) == len(rejection_rates):
                sorted_idx = np.argsort(rejection_rates)
                worst_errors = worst_errors[sorted_idx]
        
        if len(rejection_rates) > 0 and worst_errors is not None and len(worst_errors) == len(rejection_rates):
            ax.plot(
                rejection_rates,
                worst_errors,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Risk (Worst-group Error)", fontsize=12)
    ax.set_title("Balanced Mode: Worst Risk vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_balanced_worst_risk.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_worst_balanced_risk():
    """Plot Figure 8a: Worst-group mode - Balanced Risk (Balanced Error only)."""
    print("Plotting Figure 8a: Worst-group mode - Balanced Risk...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" not in method_key:
            continue  # Only worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, _, _, balanced_errors = extract_rc_curve(data, "test")
        
        # Ensure balanced_errors is available
        if len(balanced_errors) != len(rejection_rates):
            # Extract from results_per_point if needed
            results = data.get("results_per_point", [])
            if results:
                balanced_errors_list = []
                rejection_rates_list = []
                for result in results:
                    metrics = result.get("test_metrics", {})
                    if metrics:
                        rejection_rate = 1.0 - metrics.get("coverage", 1.0)
                        balanced_error = metrics.get("balanced_error", 0.0)
                        rejection_rates_list.append(rejection_rate)
                        balanced_errors_list.append(balanced_error)
                if len(rejection_rates_list) > 0:
                    rejection_rates = np.array(rejection_rates_list)
                    balanced_errors = np.array(balanced_errors_list)
                    sorted_idx = np.argsort(rejection_rates)
                    rejection_rates = rejection_rates[sorted_idx]
                    balanced_errors = balanced_errors[sorted_idx]
        
        if len(rejection_rates) > 0 and len(balanced_errors) == len(rejection_rates):
            ax.plot(
                rejection_rates,
                balanced_errors,
                label=config["name"],
                **config["style"],
                alpha=0.8,
            )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Risk (Balanced Error)", fontsize=12)
    ax.set_title("Worst-group Mode: Balanced Risk vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_worst_balanced_risk.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


def plot_worst_worst_risk():
    """Plot Figure 8b: Worst-group mode - Worst Risk (Worst-group Error only)."""
    print("Plotting Figure 8b: Worst-group mode - Worst Risk...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_key, config in METHODS_CONFIG.items():
        if "worst" not in method_key:
            continue  # Only worst-group methods
        
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        rejection_rates, head_errors, tail_errors, _ = extract_rc_curve(data, "test")
        
        if len(rejection_rates) > 0:
            # Get worst_group_errors
            rc_curve = data.get("rc_curve", {})
            worst_errors = None
            
            # Try from rc_curve first
            if isinstance(rc_curve, dict):
                worst_errors = np.array(rc_curve.get("worst_group_errors", []))
                if len(worst_errors) == len(rejection_rates):
                    sorted_idx = np.argsort(rejection_rates)
                    worst_errors = worst_errors[sorted_idx]
            
            # Fallback: extract from results_per_point
            if worst_errors is None or len(worst_errors) != len(rejection_rates):
                results = data.get("results_per_point", [])
                if results:
                    worst_errors_list = []
                    rejection_rates_list = []
                    for result in results:
                        metrics = result.get("test_metrics", {})
                        if metrics:
                            rejection_rate = 1.0 - metrics.get("coverage", 1.0)
                            worst_error = metrics.get("worst_group_error", 0.0)
                            rejection_rates_list.append(rejection_rate)
                            worst_errors_list.append(worst_error)
                    
                    if len(rejection_rates_list) > 0:
                        rejection_rates = np.array(rejection_rates_list)
                        worst_errors = np.array(worst_errors_list)
                        sorted_idx = np.argsort(rejection_rates)
                        rejection_rates = rejection_rates[sorted_idx]
                        worst_errors = worst_errors[sorted_idx]
            
            # Last fallback: max(head, tail)
            if worst_errors is None or len(worst_errors) != len(rejection_rates):
                if len(head_errors) > 0 and len(tail_errors) > 0:
                    worst_errors = np.maximum(head_errors, tail_errors)
                else:
                    worst_errors = np.array([0.0] * len(rejection_rates))
            
            if len(worst_errors) == len(rejection_rates):
                ax.plot(
                    rejection_rates,
                    worst_errors,
                    label=config["name"],
                    **config["style"],
                    alpha=0.8,
                )
    
    ax.set_xlabel("Rejection Rate", fontsize=12)
    ax.set_ylabel("Risk (Worst-group Error)", fontsize=12)
    ax.set_title("Worst-group Mode: Worst Risk vs Rejection Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig_worst_worst_risk.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {save_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Generate all comparison figures."""
    print("=" * 70)
    print("GENERATING COMPARISON FIGURES")
    print("=" * 70)
    
    # Generate all 10 figures
    plot_balanced_head_error()
    plot_balanced_tail_error()
    plot_worst_head_error()
    plot_worst_tail_error()
    plot_balanced_head_tail_ratio()
    plot_worst_head_tail_ratio()
    plot_balanced_balanced_risk()
    plot_balanced_worst_risk()
    plot_worst_balanced_risk()
    plot_worst_worst_risk()
    
    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

