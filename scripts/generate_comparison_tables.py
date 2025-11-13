#!/usr/bin/env python3
"""
Generate Comparison Tables for LtR Plugin Methods
==================================================

Creates 4 comparison tables:
1. AURC Summary (Balanced & Worst-group AURC)
2. Metrics at Specific Rejection Rates
3. Group Errors Comparison (Head vs Tail)
4. Hyperparameters Summary (alpha, mu, beta)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = Path("./results/ltr_plugin/cifar100_lt_if100")
OUTPUT_DIR = Path("./results/comparison_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Method mapping
METHOD_NAMES = {
    "ce_only_balanced": "CE-only (Balanced)",
    "ce_only_worst": "CE-only (Worst-group)",
    "logitadjustonly_balanced": "LogitAdjust-only (Balanced)",
    "balonly_balanced": "BalSoftmax-only (Balanced)",
    "balonly_worst": "BalSoftmax-only (Worst-group)",
    "uniform_balanced_2experts": "Uniform 2-Experts (Balanced)",
    "uniform_balanced_3experts": "Uniform 3-Experts (Balanced)",
    "uniform_worst_2experts": "Uniform 2-Experts (Worst-group)",
    "uniform_worst_3experts": "Uniform 3-Experts (Worst-group)",
    "gating_balanced": "MoE (Gating) (Balanced)",
    "gating_worst": "MoE (Gating) (Worst-group)",
}


# ============================================================================
# DATA LOADING
# ============================================================================


def load_json_results(file_path: Path) -> Dict:
    """Load JSON results file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_aurc(data: Dict, split: str = "test") -> Tuple[float, float]:
    """Extract AURC balanced and worst-group from data."""
    rc_curve = data.get("rc_curve", {})
    
    # Try nested structure first (balanced methods)
    if isinstance(rc_curve, dict) and split in rc_curve:
        aurc_bal = rc_curve[split].get("aurc_balanced", 0.0)
        aurc_worst = rc_curve[split].get("aurc_worst_group", 0.0)
        if aurc_bal > 0 or aurc_worst > 0:
            return float(aurc_bal), float(aurc_worst)
    
    # Try flat structure (worst-group methods)
    if isinstance(rc_curve, dict):
        aurc_bal = rc_curve.get("aurc_balanced", 0.0)
        aurc_worst = rc_curve.get("aurc_worst_group", 0.0)
        if aurc_bal > 0 or aurc_worst > 0:
            return float(aurc_bal), float(aurc_worst)
    
    return 0.0, 0.0


def extract_metrics_at_rejection(
    data: Dict, target_rejection: float, split: str = "test"
) -> Optional[Dict]:
    """Extract metrics at specific rejection rate."""
    results = data.get("results_per_cost", data.get("results_per_point", []))
    for result in results:
        if abs(result.get("target_rejection", -1) - target_rejection) < 1e-6:
            metrics = result.get(f"{split}_metrics", result.get("test_metrics", {}))
            if metrics:
                return {
                    "rejection_rate": 1.0 - metrics.get("coverage", 1.0),
                    "balanced_error": metrics.get("balanced_error", 0.0),
                    "worst_group_error": metrics.get("worst_group_error", 0.0),
                    "group_errors": metrics.get("group_errors", [0.0, 0.0]),
                }
    return None


def extract_hyperparameters(data: Dict, target_rejection: float = 0.2) -> Dict:
    """Extract hyperparameters at specific rejection rate."""
    results = data.get("results_per_cost", data.get("results_per_point", []))
    for result in results:
        if abs(result.get("target_rejection", -1) - target_rejection) < 1e-6:
            return {
                "alpha": result.get("alpha", [1.0, 1.0]),
                "mu": result.get("mu", [0.0, 0.0]),
                "beta": result.get("beta", None),
                "mu_lambda": result.get("mu_lambda", None),
            }
    return {"alpha": [1.0, 1.0], "mu": [0.0, 0.0], "beta": None, "mu_lambda": None}


# ============================================================================
# TABLE 1: AURC Summary
# ============================================================================


def generate_table1_aurc_summary() -> pd.DataFrame:
    """Generate Table 1: AURC Summary."""
    rows = []
    
    for method_key, method_name in METHOD_NAMES.items():
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        aurc_bal, aurc_worst = extract_aurc(data, "test")
        
        # Determine method type
        if "worst" in method_key:
            method_type = "Worst-group"
        else:
            method_type = "Balanced"
        
        # Determine expert type
        if "ce_only" in method_key:
            expert_type = "Single (CE)"
        elif "logitadjustonly" in method_key:
            expert_type = "Single (LogitAdjust)"
        elif "balonly" in method_key:
            expert_type = "Single (BalSoftmax)"
        elif "uniform" in method_key:
            if "2experts" in method_key:
                expert_type = "Ensemble 2-Experts"
            else:
                expert_type = "Ensemble 3-Experts"
        elif "gating" in method_key:
            expert_type = "MoE (Gating)"
        else:
            expert_type = "Unknown"
        
        rows.append({
            "Method": method_name,
            "Expert Type": expert_type,
            "Objective": method_type,
            "AURC (Balanced)": f"{aurc_bal:.4f}",
            "AURC (Worst-group)": f"{aurc_worst:.4f}",
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(["Expert Type", "Objective"])
    return df


# ============================================================================
# TABLE 2: Metrics at Specific Rejection Rates
# ============================================================================


def generate_table2_metrics_at_rejection() -> pd.DataFrame:
    """Generate Table 2: Metrics at specific rejection rates."""
    rejection_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    rows = []
    
    for method_key, method_name in METHOD_NAMES.items():
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        
        for target_rej in rejection_rates:
            metrics = extract_metrics_at_rejection(data, target_rej, "test")
            if metrics is None:
                continue
            
            rows.append({
                "Method": method_name,
                "Target Rejection": f"{target_rej:.1f}",
                "Actual Rejection": f"{metrics['rejection_rate']:.3f}",
                "Balanced Error": f"{metrics['balanced_error']:.4f}",
                "Worst-group Error": f"{metrics['worst_group_error']:.4f}",
                "Head Error": f"{metrics['group_errors'][0]:.4f}",
                "Tail Error": f"{metrics['group_errors'][1]:.4f}",
            })
    
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# TABLE 3: Group Errors Comparison
# ============================================================================


def generate_table3_group_errors() -> pd.DataFrame:
    """Generate Table 3: Group Errors Comparison."""
    rows = []
    
    for method_key, method_name in METHOD_NAMES.items():
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        
        # Get metrics at rejection rate 0.0 (baseline)
        metrics = extract_metrics_at_rejection(data, 0.0, "test")
        if metrics is None:
            continue
        
        head_err = metrics["group_errors"][0]
        tail_err = metrics["group_errors"][1]
        gap = tail_err - head_err
        
        # Get metrics at rejection rate 0.4 (moderate rejection)
        metrics_04 = extract_metrics_at_rejection(data, 0.4, "test")
        if metrics_04:
            head_err_04 = metrics_04["group_errors"][0]
            tail_err_04 = metrics_04["group_errors"][1]
            gap_04 = tail_err_04 - head_err_04
        else:
            head_err_04 = head_err
            tail_err_04 = tail_err
            gap_04 = gap
        
        rows.append({
            "Method": method_name,
            "Head Error (r=0.0)": f"{head_err:.4f}",
            "Tail Error (r=0.0)": f"{tail_err:.4f}",
            "Gap (r=0.0)": f"{gap:.4f}",
            "Head Error (r=0.4)": f"{head_err_04:.4f}",
            "Tail Error (r=0.4)": f"{tail_err_04:.4f}",
            "Gap (r=0.4)": f"{gap_04:.4f}",
        })
    
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# TABLE 4: Hyperparameters Summary
# ============================================================================


def generate_table4_hyperparameters() -> pd.DataFrame:
    """Generate Table 4: Hyperparameters Summary."""
    rows = []
    
    for method_key, method_name in METHOD_NAMES.items():
        json_file = RESULTS_DIR / f"ltr_plugin_{method_key}.json"
        if not json_file.exists():
            continue
        
        data = load_json_results(json_file)
        
        # Get hyperparameters at rejection rate 0.2
        hparams = extract_hyperparameters(data, 0.2)
        
        alpha = hparams.get("alpha", [1.0, 1.0])
        mu = hparams.get("mu", [0.0, 0.0])
        beta = hparams.get("beta", None)
        mu_lambda = hparams.get("mu_lambda", None)
        
        if mu_lambda is None and len(mu) >= 2:
            mu_lambda = mu[1] - mu[0]
        
        rows.append({
            "Method": method_name,
            "α_head": f"{alpha[0]:.4f}" if len(alpha) > 0 else "N/A",
            "α_tail": f"{alpha[1]:.4f}" if len(alpha) > 1 else "N/A",
            "μ_head": f"{mu[0]:.2f}" if len(mu) > 0 else "N/A",
            "μ_tail": f"{mu[1]:.2f}" if len(mu) > 1 else "N/A",
            "λ (μ_tail - μ_head)": f"{mu_lambda:.2f}" if mu_lambda is not None else "N/A",
            "β_head": f"{beta[0]:.4f}" if beta is not None and len(beta) > 0 else "N/A",
            "β_tail": f"{beta[1]:.4f}" if beta is not None and len(beta) > 1 else "N/A",
        })
    
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# OUTPUT FORMATS
# ============================================================================


def save_table_latex(df: pd.DataFrame, file_path: Path, caption: str, label: str):
    """Save table as LaTeX format."""
    latex_str = df.to_latex(
        index=False,
        float_format="%.4f",
        escape=False,
        caption=caption,
        label=label,
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  Saved LaTeX table to: {file_path}")


def save_table_markdown(df: pd.DataFrame, file_path: Path, title: str):
    """Save table as Markdown format."""
    markdown_str = f"# {title}\n\n"
    
    # Create markdown table manually
    # Header
    headers = list(df.columns)
    markdown_str += "| " + " | ".join(headers) + " |\n"
    markdown_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Rows
    for _, row in df.iterrows():
        values = [str(val) for val in row.values]
        markdown_str += "| " + " | ".join(values) + " |\n"
    
    markdown_str += "\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_str)
    print(f"  Saved Markdown table to: {file_path}")


def save_table_csv(df: pd.DataFrame, file_path: Path):
    """Save table as CSV format."""
    df.to_csv(file_path, index=False)
    print(f"  Saved CSV table to: {file_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Generate all comparison tables."""
    print("=" * 70)
    print("GENERATING COMPARISON TABLES")
    print("=" * 70)
    
    # Table 1: AURC Summary
    print("\n1. Generating Table 1: AURC Summary...")
    df1 = generate_table1_aurc_summary()
    save_table_latex(
        df1,
        OUTPUT_DIR / "table1_aurc_summary.tex",
        "AURC Summary: Balanced and Worst-group AURC for all methods",
        "tab:aurc_summary",
    )
    save_table_markdown(df1, OUTPUT_DIR / "table1_aurc_summary.md", "Table 1: AURC Summary")
    save_table_csv(df1, OUTPUT_DIR / "table1_aurc_summary.csv")
    
    # Table 2: Metrics at Specific Rejection Rates
    print("\n2. Generating Table 2: Metrics at Specific Rejection Rates...")
    df2 = generate_table2_metrics_at_rejection()
    save_table_latex(
        df2,
        OUTPUT_DIR / "table2_metrics_at_rejection.tex",
        "Metrics at Specific Rejection Rates",
        "tab:metrics_rejection",
    )
    save_table_markdown(df2, OUTPUT_DIR / "table2_metrics_at_rejection.md", "Table 2: Metrics at Specific Rejection Rates")
    save_table_csv(df2, OUTPUT_DIR / "table2_metrics_at_rejection.csv")
    
    # Table 3: Group Errors Comparison
    print("\n3. Generating Table 3: Group Errors Comparison...")
    df3 = generate_table3_group_errors()
    save_table_latex(
        df3,
        OUTPUT_DIR / "table3_group_errors.tex",
        "Group Errors Comparison: Head vs Tail at different rejection rates",
        "tab:group_errors",
    )
    save_table_markdown(df3, OUTPUT_DIR / "table3_group_errors.md", "Table 3: Group Errors Comparison")
    save_table_csv(df3, OUTPUT_DIR / "table3_group_errors.csv")
    
    # Table 4: Hyperparameters Summary
    print("\n4. Generating Table 4: Hyperparameters Summary...")
    df4 = generate_table4_hyperparameters()
    save_table_latex(
        df4,
        OUTPUT_DIR / "table4_hyperparameters.tex",
        "Hyperparameters Summary at rejection rate 0.2",
        "tab:hyperparameters",
    )
    save_table_markdown(df4, OUTPUT_DIR / "table4_hyperparameters.md", "Table 4: Hyperparameters Summary")
    save_table_csv(df4, OUTPUT_DIR / "table4_hyperparameters.csv")
    
    print("\n" + "=" * 70)
    print("ALL TABLES GENERATED SUCCESSFULLY!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Print summary
    print("\nSummary:")
    print(f"  Table 1: {len(df1)} methods")
    print(f"  Table 2: {len(df2)} method-rejection combinations")
    print(f"  Table 3: {len(df3)} methods")
    print(f"  Table 4: {len(df4)} methods")


if __name__ == "__main__":
    main()

