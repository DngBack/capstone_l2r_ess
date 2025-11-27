"""
Complete Feature Ablation Study Script
======================================

Tích hợp training và plugin evaluation (balanced + worst) để tạo một file CSV ablation study đầy đủ.

Workflow:
1. Train gating network với từng feature preset
2. Run balanced plugin evaluation với từng trained gating
3. Run worst plugin evaluation với từng trained gating
4. Tổng hợp tất cả kết quả vào một CSV file

Usage:
    python scripts/ablation_feature_study_complete.py --dataset cifar100_lt_if100
    python scripts/ablation_feature_study_complete.py --dataset cifar100_lt_if100 --presets all minimal uncertainty_only
"""

import sys
import subprocess
import json
import os
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gating import FEATURE_PRESETS, FeatureConfig


def run_training_experiment(
    preset_name: str,
    dataset: str,
    routing: str = "dense",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    log_dir: Path = None,
) -> Dict:
    """Run a single training experiment with given feature preset."""
    print(f"\n{'=' * 80}")
    print(f"[TRAINING] Running experiment: {preset_name}")
    print(f"{'=' * 80}")

    if log_dir is None:
        log_dir = Path("./logs/ablation_feature_study")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{dataset}_{preset_name}_{timestamp}.log"

    # Get absolute path to script
    script_path = project_root / "src" / "train" / "train_gating_map.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--dataset",
        dataset,
        "--routing",
        routing,
        "--feature-preset",
        preset_name,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--log-file",
        str(log_file),
    ]

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{str(project_root)}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(project_root),  # Set working directory to project root
        )

        results = parse_training_results(log_file, preset_name)
        results["status"] = "success"
        results["train_log_file"] = str(log_file)

        return results

    except subprocess.CalledProcessError as e:
        print(f"❌ Error training {preset_name}:")
        print(e.stdout)
        print(e.stderr)
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
            "train_log_file": str(log_file),
        }
    except Exception as e:
        print(f"❌ Unexpected error in {preset_name}: {e}")
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
        }


def parse_training_results(log_file: Path, preset_name: str) -> Dict:
    """Parse training results from log file."""
    results = {
        "preset": preset_name,
        "balanced_acc": None,
        "val_loss": None,
        "feature_dim": None,
        "enabled_features": None,
    }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract best balanced accuracy
        if "Best balanced acc:" in content:
            for line in content.split("\n"):
                if "Best balanced acc:" in line:
                    try:
                        acc_str = line.split("Best balanced acc:")[1].strip().split()[0]
                        results["balanced_acc"] = float(acc_str)
                    except:
                        pass

        # Extract best validation loss
        if "Best val loss:" in content:
            for line in content.split("\n"):
                if "Best val loss:" in line:
                    try:
                        loss_str = line.split("Best val loss:")[1].strip().split()[0]
                        results["val_loss"] = float(loss_str)
                    except:
                        pass

        # Extract feature dimension
        if "Feature dim:" in content:
            for line in content.split("\n"):
                if "Feature dim:" in line:
                    try:
                        dim_str = line.split("Feature dim:")[1].strip().split()[0]
                        results["feature_dim"] = int(dim_str)
                    except:
                        pass

        # Extract enabled features
        if "Feature config:" in content:
            for line in content.split("\n"):
                if "Feature config:" in line:
                    results["enabled_features"] = line.split("Feature config:")[
                        1
                    ].strip()

    except Exception as e:
        print(f"⚠️  Warning: Could not parse training results from {log_file}: {e}")

    return results


def run_balanced_plugin_evaluation(
    preset_name: str,
    dataset: str,
    log_dir: Path = None,
) -> Dict:
    """Run balanced plugin evaluation for a trained gating network."""
    print(f"\n{'=' * 80}")
    print(f"[PLUGIN BALANCED] Running evaluation: {preset_name}")
    print(f"{'=' * 80}")

    if log_dir is None:
        log_dir = Path("./logs/ablation_feature_study")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"plugin_balanced_{dataset}_{preset_name}_{timestamp}.log"

    # Construct checkpoint path (use absolute path)
    checkpoint_path = (
        project_root
        / "checkpoints"
        / "gating_map"
        / dataset
        / f"final_gating_{preset_name}.pth"
    )
    if not checkpoint_path.exists():
        # Fallback to default checkpoint name
        checkpoint_path = (
            project_root / "checkpoints" / "gating_map" / dataset / "final_gating.pth"
        )

    # Get absolute path to script
    plugin_script_path = project_root / "run_balanced_plugin_gating.py"

    cmd = [
        sys.executable,
        str(plugin_script_path),
        "--dataset",
        dataset,
        "--checkpoint",
        str(checkpoint_path),
        "--log-file",
        str(log_file),
    ]

    print(f"Using checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"⚠️  WARNING: Checkpoint not found: {checkpoint_path}")

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{str(project_root)}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(project_root),  # Set working directory to project root
        )

        # Parse plugin results
        plugin_results = parse_balanced_plugin_results(log_file, preset_name, dataset)
        plugin_results["status"] = "success"
        plugin_results["plugin_balanced_log_file"] = str(log_file)

        return plugin_results

    except subprocess.CalledProcessError as e:
        print(f"❌ Error in balanced plugin evaluation {preset_name}:")
        print(e.stdout)
        print(e.stderr)
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
            "plugin_balanced_log_file": str(log_file),
        }
    except Exception as e:
        print(f"❌ Unexpected error in balanced plugin evaluation {preset_name}: {e}")
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
        }


def run_worst_plugin_evaluation(
    preset_name: str,
    dataset: str,
    log_dir: Path = None,
) -> Dict:
    """Run worst plugin evaluation for a trained gating network."""
    print(f"\n{'=' * 80}")
    print(f"[PLUGIN WORST] Running evaluation: {preset_name}")
    print(f"{'=' * 80}")

    if log_dir is None:
        log_dir = Path("./logs/ablation_feature_study")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"plugin_worst_{dataset}_{preset_name}_{timestamp}.log"

    # Construct checkpoint path (use absolute path)
    checkpoint_path = (
        project_root
        / "checkpoints"
        / "gating_map"
        / dataset
        / f"final_gating_{preset_name}.pth"
    )
    if not checkpoint_path.exists():
        # Fallback to default checkpoint name
        checkpoint_path = (
            project_root / "checkpoints" / "gating_map" / dataset / "final_gating.pth"
        )

    # Get absolute path to script
    plugin_script_path = project_root / "run_worst_plugin_gating.py"

    cmd = [
        sys.executable,
        str(plugin_script_path),
        "--dataset",
        dataset,
        "--checkpoint",
        str(checkpoint_path),
        "--log-file",
        str(log_file),
    ]

    print(f"Using checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        print(f"⚠️  WARNING: Checkpoint not found: {checkpoint_path}")

    # Set PYTHONPATH to include project root
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{str(project_root)}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(project_root),  # Set working directory to project root
        )

        # Parse plugin results
        plugin_results = parse_worst_plugin_results(log_file, preset_name, dataset)
        plugin_results["status"] = "success"
        plugin_results["plugin_worst_log_file"] = str(log_file)

        return plugin_results

    except subprocess.CalledProcessError as e:
        print(f"❌ Error in worst plugin evaluation {preset_name}:")
        print(e.stdout)
        print(e.stderr)
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
            "plugin_worst_log_file": str(log_file),
        }
    except Exception as e:
        print(f"❌ Unexpected error in worst plugin evaluation {preset_name}: {e}")
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
        }


def parse_balanced_plugin_results(
    log_file: Path, preset_name: str, dataset: str
) -> Dict:
    """Parse balanced plugin evaluation results from log file."""
    results = {
        "preset": preset_name,
        "baseline_balanced_error": None,
        "baseline_head_error": None,
        "baseline_tail_error": None,
        "plugin_balanced_aurc_balanced": None,
        "plugin_balanced_aurc_worst_group": None,
        "plugin_balanced_test_balanced_error_r0": None,
        "plugin_balanced_test_head_error_r0": None,
        "plugin_balanced_test_tail_error_r0": None,
    }

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract baseline balanced error
        if "Baseline Gating balanced error (TEST)" in content:
            for line in content.split("\n"):
                if "Baseline Gating balanced error (TEST)" in line:
                    try:
                        err_str = line.split("=")[1].strip()
                        results["baseline_balanced_error"] = float(err_str)
                    except:
                        pass

        # Extract baseline group errors
        if "Baseline Gating group errors" in content:
            for line in content.split("\n"):
                if "Baseline Gating group errors" in line and "[" in line:
                    try:
                        # Parse list format: [0.xxxx, 0.yyyy]
                        parts = line.split("=")[1].strip()
                        import re

                        nums = re.findall(r"[\d.]+", parts)
                        if len(nums) >= 2:
                            results["baseline_head_error"] = float(nums[0])
                            results["baseline_tail_error"] = float(nums[1])
                    except:
                        pass

        # Extract AURC from log
        if "Test AURC - Balanced:" in content:
            for line in content.split("\n"):
                if "Test AURC - Balanced:" in line:
                    try:
                        parts = line.split("Balanced:")[1].split("|")
                        if parts:
                            aurc_str = parts[0].strip()
                            results["plugin_balanced_aurc_balanced"] = float(aurc_str)
                    except:
                        pass

        if "Test AURC - Worst-group:" in content or "Worst-group:" in content:
            for line in content.split("\n"):
                if "Worst-group:" in line:
                    try:
                        parts = line.split("Worst-group:")[1].strip()
                        results["plugin_balanced_aurc_worst_group"] = float(parts)
                    except:
                        pass

        # Try to parse from JSON output file (use absolute path)
        results_json_path = (
            project_root
            / "results"
            / "ltr_plugin"
            / dataset
            / "ltr_plugin_gating_balanced.json"
        )
        if results_json_path.exists():
            try:
                with open(results_json_path, "r") as f:
                    json_data = json.load(f)

                # Get baseline (r=0.0)
                results_per_cost = json_data.get("results_per_cost", [])
                for r in results_per_cost:
                    if abs(r.get("target_rejection", 1.0) - 0.0) < 1e-6:
                        test_metrics = r.get("test_metrics", {})
                        if test_metrics:
                            results["plugin_balanced_test_balanced_error_r0"] = (
                                test_metrics.get("balanced_error")
                            )
                            group_errors = test_metrics.get("group_errors", [])
                            if len(group_errors) >= 2:
                                results["plugin_balanced_test_head_error_r0"] = (
                                    group_errors[0]
                                )
                                results["plugin_balanced_test_tail_error_r0"] = (
                                    group_errors[1]
                                )
                        break

                # Get AURC from rc_curve
                rc_curve = json_data.get("rc_curve", {}).get("test", {})
                if rc_curve:
                    results["plugin_balanced_aurc_balanced"] = rc_curve.get(
                        "aurc_balanced"
                    )
                    results["plugin_balanced_aurc_worst_group"] = rc_curve.get(
                        "aurc_worst_group"
                    )

            except Exception as e:
                print(f"⚠️  Warning: Could not parse JSON results: {e}")

    except Exception as e:
        print(
            f"⚠️  Warning: Could not parse balanced plugin results from {log_file}: {e}"
        )

    return results


def parse_worst_plugin_results(log_file: Path, preset_name: str, dataset: str) -> Dict:
    """Parse worst plugin evaluation results from log file."""
    results = {
        "preset": preset_name,
        "plugin_worst_aurc_balanced": None,
        "plugin_worst_aurc_worst_group": None,
        "plugin_worst_test_balanced_error_r0": None,
        "plugin_worst_test_head_error_r0": None,
        "plugin_worst_test_tail_error_r0": None,
    }

    try:
        # Try to parse from JSON output file (use absolute path)
        results_json_path = (
            project_root
            / "results"
            / "ltr_plugin"
            / dataset
            / "ltr_plugin_gating_worst.json"
        )
        if results_json_path.exists():
            try:
                with open(results_json_path, "r") as f:
                    json_data = json.load(f)

                # Get baseline (r=0.0)
                results_per_point = json_data.get("results_per_point", [])
                for r in results_per_point:
                    if abs(r.get("target_rejection", 1.0) - 0.0) < 1e-6:
                        test_metrics = r.get("test_metrics", {})
                        if test_metrics:
                            results["plugin_worst_test_balanced_error_r0"] = (
                                test_metrics.get("balanced_error")
                            )
                            group_errors = test_metrics.get("group_errors", [])
                            if len(group_errors) >= 2:
                                results["plugin_worst_test_head_error_r0"] = (
                                    group_errors[0]
                                )
                                results["plugin_worst_test_tail_error_r0"] = (
                                    group_errors[1]
                                )
                        break

                # Get AURC from rc_curve
                rc_curve = json_data.get("rc_curve", {})
                if rc_curve:
                    results["plugin_worst_aurc_balanced"] = rc_curve.get(
                        "aurc_balanced"
                    )
                    results["plugin_worst_aurc_worst_group"] = rc_curve.get(
                        "aurc_worst_group"
                    )

            except Exception as e:
                print(f"⚠️  Warning: Could not parse worst plugin JSON results: {e}")

        # Also try to parse from log file
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract AURC from log if available
        if "AURC" in content:
            for line in content.split("\n"):
                if "AURC" in line and "Balanced" in line:
                    try:
                        # Try to extract AURC value
                        import re

                        nums = re.findall(r"[\d.]+", line)
                        if nums:
                            results["plugin_worst_aurc_balanced"] = float(nums[-1])
                    except:
                        pass
                if "AURC" in line and "Worst" in line:
                    try:
                        import re

                        nums = re.findall(r"[\d.]+", line)
                        if nums:
                            results["plugin_worst_aurc_worst_group"] = float(nums[-1])
                    except:
                        pass

    except Exception as e:
        print(f"⚠️  Warning: Could not parse worst plugin results from {log_file}: {e}")

    return results


def run_complete_ablation_study(
    presets: List[str],
    dataset: str,
    routing: str = "dense",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    output_dir: Path = None,
    skip_training: bool = False,
    skip_plugin: bool = False,
    skip_worst_plugin: bool = False,
) -> pd.DataFrame:
    """
    Run complete ablation study: training + balanced plugin + worst plugin evaluation.

    Args:
        skip_training: If True, skip training (assume checkpoints exist)
        skip_plugin: If True, skip plugin evaluation (only train)
        skip_worst_plugin: If True, skip worst plugin evaluation
    """
    if output_dir is None:
        output_dir = Path("./results/ablation_feature_study_complete")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"ablation_complete_{dataset}_{timestamp}.csv"
    json_file = output_dir / f"ablation_complete_{dataset}_{timestamp}.json"

    print(f"\n{'=' * 80}")
    print(f"COMPLETE FEATURE ABLATION STUDY")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset}")
    print(f"Routing: {routing}")
    print(f"Presets to test: {presets}")
    print(f"Output directory: {output_dir}")
    print(f"Skip training: {skip_training}")
    print(f"Skip balanced plugin: {skip_plugin}")
    print(f"Skip worst plugin: {skip_worst_plugin}")
    print(f"{'=' * 80}\n")

    all_results = []

    # Run each preset
    for preset_name in presets:
        if preset_name not in FEATURE_PRESETS:
            print(f"⚠️  Warning: Unknown preset '{preset_name}', skipping...")
            continue

        result = {"preset": preset_name}

        # Step 1: Training
        if not skip_training:
            train_result = run_training_experiment(
                preset_name=preset_name,
                dataset=dataset,
                routing=routing,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                log_dir=output_dir / "logs",
            )
            result.update(train_result)
        else:
            print(
                f"⏭️  Skipping training for {preset_name} (assuming checkpoint exists)"
            )
            result["status"] = "training_skipped"

        # Step 2: Balanced Plugin Evaluation
        if not skip_plugin and result.get("status") != "error":
            balanced_result = run_balanced_plugin_evaluation(
                preset_name=preset_name,
                dataset=dataset,
                log_dir=output_dir / "logs",
            )
            result.update(balanced_result)
        else:
            if skip_plugin:
                print(f"⏭️  Skipping balanced plugin evaluation for {preset_name}")
            else:
                print(
                    f"⏭️  Skipping balanced plugin evaluation for {preset_name} (training failed)"
                )

        # Step 3: Worst Plugin Evaluation
        if not skip_worst_plugin and result.get("status") != "error":
            worst_result = run_worst_plugin_evaluation(
                preset_name=preset_name,
                dataset=dataset,
                log_dir=output_dir / "logs",
            )
            result.update(worst_result)
        else:
            if skip_worst_plugin:
                print(f"⏭️  Skipping worst plugin evaluation for {preset_name}")
            else:
                print(
                    f"⏭️  Skipping worst plugin evaluation for {preset_name} (training failed)"
                )

        all_results.append(result)

    # Add feature breakdown
    feature_breakdown = []
    for preset_name in presets:
        if preset_name in FEATURE_PRESETS:
            config = FEATURE_PRESETS[preset_name]
            feature_breakdown.append(
                {
                    "preset": preset_name,
                    "per_expert_count": len(config.enabled_per_expert_features),
                    "global_count": len(config.enabled_global_features),
                    "total_features": len(config.enabled_per_expert_features)
                    + len(config.enabled_global_features),
                    "per_expert_features": ", ".join(
                        config.enabled_per_expert_features
                    ),
                    "global_features": ", ".join(config.enabled_global_features),
                }
            )

    # Create DataFrame
    df = pd.DataFrame(all_results)

    if feature_breakdown:
        df_features = pd.DataFrame(feature_breakdown)
        df = df.merge(df_features, on="preset", how="left")

    # Reorder columns for better readability
    preferred_order = [
        "preset",
        "status",
        "feature_dim",
        "total_features",
        "per_expert_count",
        "global_count",
        "per_expert_features",
        "global_features",
        "balanced_acc",
        "val_loss",
        "baseline_balanced_error",
        "baseline_head_error",
        "baseline_tail_error",
        # Balanced plugin metrics
        "plugin_balanced_aurc_balanced",
        "plugin_balanced_aurc_worst_group",
        "plugin_balanced_test_balanced_error_r0",
        "plugin_balanced_test_head_error_r0",
        "plugin_balanced_test_tail_error_r0",
        # Worst plugin metrics
        "plugin_worst_aurc_balanced",
        "plugin_worst_aurc_worst_group",
        "plugin_worst_test_balanced_error_r0",
        "plugin_worst_test_head_error_r0",
        "plugin_worst_test_tail_error_r0",
    ]

    # Add remaining columns
    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in preferred_order]
    column_order = existing_cols + remaining_cols

    df = df[column_order]

    # Save results
    df.to_csv(csv_file, index=False)

    # Save JSON with full details
    results_dict = {
        "timestamp": timestamp,
        "dataset": dataset,
        "routing": routing,
        "epochs": epochs,
        "experiments": all_results,
    }
    with open(json_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Print summary
    print(f"\n{'=' * 80}")
    print("COMPLETE ABLATION STUDY SUMMARY")
    print(f"{'=' * 80}")
    print(df.to_string(index=False))
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Complete feature ablation study: training + balanced + worst plugin evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100_lt_if100",
        choices=["cifar100_lt_if100", "inaturalist2018", "imagenet_lt"],
        help="Dataset name",
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="+",
        default=list(FEATURE_PRESETS.keys()),
        help="Feature presets to test (default: all presets)",
    )
    parser.add_argument(
        "--routing",
        type=str,
        default="dense",
        choices=["dense", "top_k"],
        help="Routing strategy",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./results/ablation_feature_study_complete)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step (assume checkpoints already exist)",
    )
    parser.add_argument(
        "--skip-plugin",
        action="store_true",
        help="Skip balanced plugin evaluation step (only train)",
    )
    parser.add_argument(
        "--skip-worst-plugin",
        action="store_true",
        help="Skip worst plugin evaluation step",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    df_results = run_complete_ablation_study(
        presets=args.presets,
        dataset=args.dataset,
        routing=args.routing,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=output_dir,
        skip_training=args.skip_training,
        skip_plugin=args.skip_plugin,
        skip_worst_plugin=args.skip_worst_plugin,
    )

    print("\n✅ Complete ablation study finished!")

    # Print top configurations by different metrics
    if "plugin_balanced_aurc_balanced" in df_results.columns:
        df_sorted = df_results.sort_values(
            "plugin_balanced_aurc_balanced", ascending=True
        )  # Lower is better
        print(
            "\n📊 Top configurations by Balanced Plugin AURC (Balanced) - Lower is better:"
        )
        print(
            df_sorted[
                [
                    "preset",
                    "plugin_balanced_aurc_balanced",
                    "plugin_balanced_aurc_worst_group",
                    "baseline_balanced_error",
                    "feature_dim",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    if "plugin_worst_aurc_worst_group" in df_results.columns:
        df_sorted = df_results.sort_values(
            "plugin_worst_aurc_worst_group", ascending=True
        )  # Lower is better
        print(
            "\n📊 Top configurations by Worst Plugin AURC (Worst-group) - Lower is better:"
        )
        print(
            df_sorted[
                [
                    "preset",
                    "plugin_worst_aurc_worst_group",
                    "plugin_worst_aurc_balanced",
                    "baseline_balanced_error",
                    "feature_dim",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    if "balanced_acc" in df_results.columns:
        df_sorted = df_results.sort_values("balanced_acc", ascending=False)
        print("\n📊 Top configurations by Gating Balanced Accuracy - Higher is better:")
        print(
            df_sorted[["preset", "balanced_acc", "val_loss", "feature_dim"]]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
