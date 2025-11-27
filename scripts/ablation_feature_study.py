"""
Feature Ablation Study Script
=============================

Chạy training với các feature combinations khác nhau để đánh giá ảnh hưởng
của từng feature đến kết quả gating network.

Usage:
    python scripts/ablation_feature_study.py --dataset cifar100_lt_if100
    python scripts/ablation_feature_study.py --dataset cifar100_lt_if100 --presets all minimal uncertainty_only
"""

import sys
import subprocess
import json
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gating import FEATURE_PRESETS, FeatureConfig


def run_training_experiment(
    preset_name: str,
    dataset: str,
    routing: str = "dense",
    epochs: int = 50,  # Shorter for ablation
    batch_size: int = 128,
    lr: float = 1e-3,
    log_dir: Path = None,
) -> Dict:
    """
    Run a single training experiment with given feature preset.

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'=' * 80}")
    print(f"Running experiment: {preset_name}")
    print(f"{'=' * 80}")

    # Create log file path
    if log_dir is None:
        log_dir = Path("./logs/ablation_feature_study")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{dataset}_{preset_name}_{timestamp}.log"

    # Build command
    cmd = [
        sys.executable,
        "src/train/train_gating_map.py",
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

    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse results from log file
        results = parse_training_results(log_file, preset_name)
        results["status"] = "success"
        results["log_file"] = str(log_file)

        return results

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running experiment {preset_name}:")
        print(e.stdout)
        print(e.stderr)
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
            "log_file": str(log_file),
        }
    except Exception as e:
        print(f"❌ Unexpected error in {preset_name}: {e}")
        return {
            "preset": preset_name,
            "status": "error",
            "error": str(e),
        }


def parse_training_results(log_file: Path, preset_name: str) -> Dict:
    """
    Parse training results from log file.

    Looks for:
    - Best balanced accuracy
    - Best validation loss
    - Feature dimension
    """
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

        # Extract enabled features from feature config print
        if "Feature config:" in content:
            for line in content.split("\n"):
                if "Feature config:" in line:
                    results["enabled_features"] = line.split("Feature config:")[
                        1
                    ].strip()

    except Exception as e:
        print(f"⚠️  Warning: Could not parse results from {log_file}: {e}")

    return results


def run_ablation_study(
    presets: List[str],
    dataset: str,
    routing: str = "dense",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    output_dir: Path = None,
) -> pd.DataFrame:
    """
    Run ablation study across multiple feature presets.

    Returns:
        DataFrame with results for all experiments
    """
    if output_dir is None:
        output_dir = Path("./results/ablation_feature_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"ablation_results_{dataset}_{timestamp}.json"
    csv_file = output_dir / f"ablation_results_{dataset}_{timestamp}.csv"

    print(f"\n{'=' * 80}")
    print(f"FEATURE ABLATION STUDY")
    print(f"{'=' * 80}")
    print(f"Dataset: {dataset}")
    print(f"Routing: {routing}")
    print(f"Presets to test: {presets}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}\n")

    all_results = []

    # Run each preset
    for preset_name in presets:
        if preset_name not in FEATURE_PRESETS:
            print(f"⚠️  Warning: Unknown preset '{preset_name}', skipping...")
            continue

        result = run_training_experiment(
            preset_name=preset_name,
            dataset=dataset,
            routing=routing,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            log_dir=output_dir / "logs",
        )
        all_results.append(result)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Add feature breakdown
    feature_breakdown = []
    for preset_name in presets:
        if preset_name in FEATURE_PRESETS:
            config = FEATURE_PRESETS[preset_name]
            feature_breakdown.append(
                {
                    "preset": preset_name,
                    "per_expert": len(config.enabled_per_expert_features),
                    "global": len(config.enabled_global_features),
                    "total_features": len(config.enabled_per_expert_features)
                    + len(config.enabled_global_features),
                }
            )

    if feature_breakdown:
        df_features = pd.DataFrame(feature_breakdown)
        df = df.merge(df_features, on="preset", how="left")

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
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Print summary
    print(f"\n{'=' * 80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'=' * 80}")
    print(df.to_string(index=False))
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {results_file}")

    return df


def create_custom_feature_combinations() -> Dict[str, FeatureConfig]:
    """
    Create additional custom feature combinations for detailed ablation.

    Returns:
        Dictionary of custom FeatureConfig objects
    """
    custom = {}

    # Individual feature removals (remove one at a time from "all")
    all_config = FEATURE_PRESETS["all"]

    # Remove entropy
    custom["no_entropy"] = FeatureConfig(
        use_entropy=False,
        use_topk_mass=all_config.use_topk_mass,
        use_residual_mass=all_config.use_residual_mass,
        use_max_probs=all_config.use_max_probs,
        use_top_gap=all_config.use_top_gap,
        use_cosine_sim=all_config.use_cosine_sim,
        use_kl_to_mean=all_config.use_kl_to_mean,
        use_mean_entropy=all_config.use_mean_entropy,
        use_mean_class_var=all_config.use_mean_class_var,
        use_std_max_conf=all_config.use_std_max_conf,
        top_k=all_config.top_k,
    )

    # Remove disagreement features
    custom["no_disagreement"] = FeatureConfig(
        use_entropy=all_config.use_entropy,
        use_topk_mass=all_config.use_topk_mass,
        use_residual_mass=all_config.use_residual_mass,
        use_max_probs=all_config.use_max_probs,
        use_top_gap=all_config.use_top_gap,
        use_cosine_sim=False,
        use_kl_to_mean=False,
        use_mean_entropy=all_config.use_mean_entropy,
        use_mean_class_var=False,
        use_std_max_conf=all_config.use_std_max_conf,
        top_k=all_config.top_k,
    )

    # Only uncertainty (entropy + confidence)
    custom["uncertainty_confidence"] = FeatureConfig(
        use_entropy=True,
        use_topk_mass=False,
        use_residual_mass=False,
        use_max_probs=True,
        use_top_gap=True,
        use_cosine_sim=False,
        use_kl_to_mean=False,
        use_mean_entropy=True,
        use_mean_class_var=False,
        use_std_max_conf=True,
        top_k=all_config.top_k,
    )

    return custom


def main():
    parser = argparse.ArgumentParser(
        description="Run feature ablation study for gating network"
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
        help="Number of training epochs (reduced for faster ablation)",
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
        help="Output directory for results (default: ./results/ablation_feature_study)",
    )
    parser.add_argument(
        "--include-custom",
        action="store_true",
        help="Include custom feature combinations in addition to presets",
    )

    args = parser.parse_args()

    # Get presets to test
    presets_to_test = args.presets.copy()

    # Add custom combinations if requested
    if args.include_custom:
        custom = create_custom_feature_combinations()
        presets_to_test.extend(custom.keys())
        # Add to FEATURE_PRESETS temporarily
        FEATURE_PRESETS.update(custom)

    # Run ablation study
    output_dir = Path(args.output_dir) if args.output_dir else None
    df_results = run_ablation_study(
        presets=presets_to_test,
        dataset=args.dataset,
        routing=args.routing,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=output_dir,
    )

    print("\n✅ Ablation study completed!")

    # Print top configurations
    if "balanced_acc" in df_results.columns:
        df_sorted = df_results.sort_values("balanced_acc", ascending=False)
        print("\n📊 Top configurations by balanced accuracy:")
        print(
            df_sorted[["preset", "balanced_acc", "val_loss", "feature_dim"]]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
