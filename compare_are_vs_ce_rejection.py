#!/usr/bin/env python3
"""
Compare ARE+Plugin vs CE+Plugin rejection decisions.

This script:
1. Loads full test set
2. Loads CE expert and Gating network models
3. Loads plugin parameters for both methods across different rejection rates
4. Runs inference on all samples
5. Compares rejection decisions between CE+Plugin and ARE+Plugin
6. Finds cases where ARE makes better decisions:
   - CE accepts wrong predictions but ARE rejects (True Reject for ARE, False Accept for CE)
   - CE rejects correct predictions but ARE accepts (True Accept for ARE, False Reject for CE)
7. Saves results to text files
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torchvision

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from src.models.experts import Expert
from src.models.gating_network_map import GatingNetwork
from src.data.datasets import get_eval_augmentations
from src.infer.loaders import (
    load_class_to_group,
    load_ce_expert,
    load_gating_network,
    load_plugin_params,
    SPLITS_DIR,
    RESULTS_DIR,
    DEVICE,
    NUM_CLASSES,
)
from src.infer.pipeline import (
    _compute_plugin_threshold_and_max_reweighted,
    BalancedLtRPlugin,
    GeneralizedLtRPlugin,
    _compute_reweighted_scores,
    paper_method_pipeline,
    our_method_pipeline,
)

# Configuration
DATASET = "cifar100_lt_if100"
OUTPUT_DIR = Path("./results/are_vs_ce_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RejectionCase:
    """Represents a case where ARE makes a better decision than CE."""
    sample_idx: int
    true_label: int
    predicted_class_ce: int
    predicted_class_are: int
    is_correct_ce: bool
    is_correct_are: bool
    ce_decision: str  # "accept" or "reject"
    are_decision: str  # "accept" or "reject"
    ce_max_reweighted: float
    are_max_reweighted: float
    ce_threshold: float
    are_threshold: float
    rejection_rate: float
    mode: str  # "balanced" or "worst"
    group: str  # "head" or "tail"
    case_type: str  # "are_true_reject_ce_false_accept" or "are_true_accept_ce_false_reject"


def load_all_plugin_configs(method: str, mode: str) -> List[Dict]:
    """
    Load all plugin configurations from JSON file.
    
    Args:
        method: "moe" or "ce_only"
        mode: "balanced" or "worst"
    
    Returns:
        List of configurations, each with alpha, mu, cost, and rejection_rate
    """
    if method == "moe":
        if mode == "worst":
            results_path = RESULTS_DIR / "ltr_plugin_gating_worst.json"
            results_key = "results_per_point"
            metrics_key = "test_metrics"
        elif mode == "balanced":
            results_path = RESULTS_DIR / "ltr_plugin_gating_balanced.json"
            results_key = "results_per_cost"
            metrics_key = "val_metrics"
        else:
            raise ValueError(f"Invalid mode: {mode}")
    elif method == "ce_only":
        if mode == "worst":
            results_path = RESULTS_DIR / "ltr_plugin_ce_only_worst.json"
            results_key = "results_per_point"
            metrics_key = "test_metrics"
        elif mode == "balanced":
            results_path = RESULTS_DIR / "ltr_plugin_ce_only_balanced.json"
            results_key = "results_per_cost"
            metrics_key = "val_metrics"
        else:
            raise ValueError(f"Invalid mode: {mode}")
    else:
        raise ValueError(f"Invalid method: {method}")
    
    if not results_path.exists():
        print(f"⚠️  Warning: {results_path} not found. Skipping {method} {mode}.")
        return []
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    results_list = results.get(results_key, [])
    
    configs = []
    for r in results_list:
        rejection_rate = 1.0 - r[metrics_key]["coverage"]
        alpha = np.array(r["alpha"])
        mu = np.array(r["mu"])
        
        if method == "moe" and mode == "worst":
            cost = r.get("cost_test", 0.0)
            beta = np.array(r.get("beta", [1.0, 1.0])) if "beta" in r else None
        elif method == "ce_only" and mode == "balanced":
            cost = r.get("cost_val", r.get("cost_test", 0.0))
            beta = None
        else:
            cost = r.get("cost_test", r.get("cost_val", 0.0))
            beta = np.array(r.get("beta", [1.0, 1.0])) if "beta" in r else None
        
        configs.append({
            "rejection_rate": rejection_rate,
            "alpha": alpha,
            "mu": mu,
            "cost": cost,
            "beta": beta,
        })
    
    # Sort by rejection rate
    configs.sort(key=lambda x: x["rejection_rate"])
    
    return configs


def load_test_set() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load full test set.
    
    Returns:
        Tuple of (images, labels, indices) where:
        - images: [N, 3, 32, 32] tensor
        - labels: [N] tensor
        - indices: [N] list of original dataset indices
    """
    import torchvision
    
    print("\n1. Loading test set...")
    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
    
    indices_file = SPLITS_DIR / "test_indices.json"
    with open(indices_file, "r", encoding="utf-8") as f:
        test_indices = json.load(f)
    
    transform = get_eval_augmentations()
    
    images = []
    labels = []
    
    for idx in tqdm(test_indices, desc="Loading images"):
        image, label = dataset[idx]
        image_tensor = transform(image).to(DEVICE)
        images.append(image_tensor)
        labels.append(label)
    
    images = torch.stack(images)  # [N, 3, 32, 32]
    labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    
    print(f"   Loaded {len(images)} test samples")
    
    return images, labels, test_indices


def run_inference_batch(
    images: torch.Tensor,
    ce_expert: Expert,
    gating_network: GatingNetwork,
    all_experts: List[Expert],
    batch_size: int = 128  # kept for signature, ignored in sequential mode
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference on all images sequentially (per-sample) for highest fidelity.
    
    Returns:
        Tuple of (ce_posteriors, are_posteriors) where each is [N, 100]
    """
    print("\n2. Running inference on test set (sequential)...")
    print("   ⚠️  Running per-sample for exact parity with notebook logic (slower).")

    ce_expert.eval()
    gating_network.eval()
    for expert in all_experts:
        expert.eval()
    
    ce_posteriors = []
    are_posteriors = []
    
    with torch.no_grad():
        for i in tqdm(range(len(images)), desc="Inference samples"):
            img = images[i:i+1]  # [1, 3, 32, 32]
            
            # CE expert
            ce_logits = ce_expert(img)
            ce_probs = F.softmax(ce_logits, dim=-1)  # [1, 100]
            ce_posteriors.append(ce_probs.cpu())
            
            # ARE (Gating + Experts)
            expert_logits_list = []
            for expert in all_experts:
                expert_logits = expert(img)  # [1, 100]
                expert_logits_list.append(expert_logits)
            
            expert_logits = torch.stack(expert_logits_list, dim=1)  # [1, 3, 100]
            expert_probs = F.softmax(expert_logits, dim=-1)  # [1, 3, 100]
            
            # Gating weights - use compute_weights_from_logits like in pipeline
            gating_weights, _ = gating_network.compute_weights_from_logits(expert_logits)  # [1, 3]
            
            # Mixture
            mixture_probs = (expert_probs * gating_weights.unsqueeze(-1)).sum(dim=1)  # [1, 100]
            are_posteriors.append(mixture_probs.cpu())
    
    ce_posteriors = torch.cat(ce_posteriors, dim=0)  # [N, 100]
    are_posteriors = torch.cat(are_posteriors, dim=0)  # [N, 100]
    
    print(f"   CE posteriors shape: {ce_posteriors.shape}")
    print(f"   ARE posteriors shape: {are_posteriors.shape}")
    
    return ce_posteriors, are_posteriors


def evaluate_with_pipelines(
    images: torch.Tensor,
    labels: torch.Tensor,
    test_indices: List[int],
    class_to_group: torch.Tensor,
    ce_expert: Expert,
    all_experts: List[Expert],
    gating_network: GatingNetwork,
    mode: str,
    rejection_rate: float = 0.4
) -> List[RejectionCase]:
    """
    Evaluate using the exact notebook pipelines (paper_method_pipeline & our_method_pipeline).
    This runs per-sample for full fidelity.
    
    Args:
        rejection_rate: Target rejection rate for plugin parameters
    """
    print(f"\n   Evaluating mode={mode}, rejection_rate={rejection_rate} with notebook pipelines (sequential)...")
    # Load plugin params as notebook does, with specified rejection_rate
    ce_alpha, ce_mu, ce_cost = load_plugin_params(method="ce_only", mode=mode, rejection_rate=rejection_rate)
    moe_alpha, moe_mu, moe_cost = load_plugin_params(method="moe", mode=mode, rejection_rate=rejection_rate)
    
    better_cases = []
    class_to_group_np = class_to_group.cpu().numpy()
    
    for idx in tqdm(range(len(images)), desc=f"Pipeline eval ({mode})"):
        img = images[idx:idx+1]  # [1, 3, 32, 32]
        true_label = labels[idx].item()
        
        # Paper method (CE + plugin)
        paper_result = paper_method_pipeline(
            img, ce_expert, class_to_group, ce_alpha, ce_mu, ce_cost
        )
        ce_pred = paper_result['prediction']
        ce_reject = bool(paper_result['reject'])
        ce_accept = not ce_reject
        is_correct_ce = ce_pred == true_label
        
        # Our method (MoE + plugin)
        our_result = our_method_pipeline(
            img,
            all_experts,
            gating_network,
            class_to_group,
            moe_alpha,
            moe_mu,
            moe_cost,
            plugin_beta=None  # notebook passes None even for worst
        )
        are_pred = our_result['prediction']
        are_reject = bool(our_result['reject'])
        are_accept = not are_reject
        is_correct_are = are_pred == true_label
        
        # Case logic: only keep if ARE correct decision and CE wrong decision
        case_type = None
        # Case 1: CE False Accept (accept wrong), ARE True Reject (reject wrong)
        if ce_accept and not is_correct_ce and (not are_accept) and (not is_correct_are):
            case_type = "are_true_reject_ce_false_accept"
        # Case 2: CE False Reject (reject correct), ARE True Accept (accept correct)
        elif (not ce_accept) and is_correct_ce and are_accept and is_correct_are:
            case_type = "are_true_accept_ce_false_reject"
        
        if case_type is not None:
            # Need thresholds and max_rew for logging
            ce_max_rew, ce_threshold = _compute_plugin_threshold_and_max_reweighted(
                paper_result['probabilities'],
                {"alpha": ce_alpha, "mu": ce_mu, "cost": ce_cost},
                class_to_group,
                plugin_beta=None
            )
            are_max_rew, are_threshold = _compute_plugin_threshold_and_max_reweighted(
                our_result['mixture_posterior'],
                {"alpha": moe_alpha, "mu": moe_mu, "cost": moe_cost},
                class_to_group,
                plugin_beta=None
            )
            
            group_idx = class_to_group_np[true_label]
            group_str = "tail" if group_idx == 1 else "head"
            
            case = RejectionCase(
                sample_idx=test_indices[idx],
                true_label=true_label,
                predicted_class_ce=ce_pred,
                predicted_class_are=are_pred,
                is_correct_ce=is_correct_ce,
                is_correct_are=is_correct_are,
                ce_decision="accept" if ce_accept else "reject",
                are_decision="accept" if are_accept else "reject",
                ce_max_reweighted=ce_max_rew,
                are_max_reweighted=are_max_rew,
                ce_threshold=ce_threshold,
                are_threshold=are_threshold,
                rejection_rate=rejection_rate,  # Use the rejection_rate parameter
                mode=mode,
                group=group_str,
                case_type=case_type,
            )
            better_cases.append(case)
    
    return better_cases


def compare_decisions(
    ce_posteriors: torch.Tensor,
    are_posteriors: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    ce_config: Dict,
    are_config: Dict,
    test_indices: List[int],
    mode: str
) -> List[RejectionCase]:
    """
    Compare rejection decisions between CE+Plugin and ARE+Plugin.
    
    Returns:
        List of RejectionCase where ARE makes better decisions
    """
    ce_alpha = ce_config["alpha"]
    ce_mu = ce_config["mu"]
    ce_cost = ce_config["cost"]
    ce_beta = ce_config.get("beta")
    
    are_alpha = are_config["alpha"]
    are_mu = are_config["mu"]
    are_cost = are_config["cost"]
    are_beta = are_config.get("beta")
    
    rejection_rate = ce_config["rejection_rate"]
    
    class_to_group_np = class_to_group.cpu().numpy()
    
    better_cases = []
    
    for idx in tqdm(range(len(ce_posteriors)), desc=f"Comparing (RR={rejection_rate:.2f})"):
        ce_posterior = ce_posteriors[idx].numpy()  # [100]
        are_posterior = are_posteriors[idx].numpy()  # [100]
        true_label = labels[idx].item()
        
        # Predictions
        ce_pred = ce_posterior.argmax()
        are_pred = are_posterior.argmax()
        
        is_correct_ce = (ce_pred == true_label)
        is_correct_are = (are_pred == true_label)
        
        # Compute plugin thresholds and max reweighted
        ce_max_rew, ce_threshold = _compute_plugin_threshold_and_max_reweighted(
            ce_posterior,
            {"alpha": ce_alpha, "mu": ce_mu, "cost": ce_cost},
            class_to_group,
            plugin_beta=ce_beta
        )
        
        are_max_rew, are_threshold = _compute_plugin_threshold_and_max_reweighted(
            are_posterior,
            {"alpha": are_alpha, "mu": are_mu, "cost": are_cost},
            class_to_group,
            plugin_beta=are_beta
        )
        
        # Decisions: reject if max_reweighted < threshold
        ce_accept = ce_max_rew >= ce_threshold
        are_accept = are_max_rew >= are_threshold
        
        ce_decision = "accept" if ce_accept else "reject"
        are_decision = "accept" if are_accept else "reject"
        
        # Find cases where ARE makes better decisions
        case_type = None
        
        # Case 1: CE False Accept (accept wrong), ARE True Reject (reject wrong)
        if ce_accept and not is_correct_ce and (not are_accept) and (not is_correct_are):
            case_type = "are_true_reject_ce_false_accept"
        
        # Case 2: CE False Reject (reject correct), ARE True Accept (accept correct)
        elif (not ce_accept) and is_correct_ce and are_accept and is_correct_are:
            case_type = "are_true_accept_ce_false_reject"
        
        if case_type is not None:
            group_idx = class_to_group_np[true_label]
            group_str = "tail" if group_idx == 1 else "head"
            
            case = RejectionCase(
                sample_idx=test_indices[idx],
                true_label=true_label,
                predicted_class_ce=ce_pred,
                predicted_class_are=are_pred,
                is_correct_ce=is_correct_ce,
                is_correct_are=is_correct_are,
                ce_decision=ce_decision,
                are_decision=are_decision,
                ce_max_reweighted=ce_max_rew,
                are_max_reweighted=are_max_rew,
                ce_threshold=ce_threshold,
                are_threshold=are_threshold,
                rejection_rate=rejection_rate,
                mode=mode,
                group=group_str,
                case_type=case_type,
            )
            better_cases.append(case)
    
    return better_cases


def save_results(
    all_cases: List[RejectionCase],
    mode: str,
    output_dir: Path
):
    """Save comparison results to text files."""
    
    print(f"\n3. Saving results for {mode} mode...")
    
    # Filter by mode
    mode_cases = [c for c in all_cases if c.mode == mode]
    
    if len(mode_cases) == 0:
        print(f"   No cases found for {mode} mode")
        return
    
    # Group by case type and rejection rate
    case_type_1 = [c for c in mode_cases if c.case_type == "are_true_reject_ce_false_accept"]
    case_type_2 = [c for c in mode_cases if c.case_type == "are_true_accept_ce_false_reject"]
    
    # Group by head/tail
    head_cases_1 = [c for c in case_type_1 if c.group == "head"]
    tail_cases_1 = [c for c in case_type_1 if c.group == "tail"]
    head_cases_2 = [c for c in case_type_2 if c.group == "head"]
    tail_cases_2 = [c for c in case_type_2 if c.group == "tail"]
    
    # Summary file
    summary_path = output_dir / f"summary_{mode}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ARE vs CE Rejection Decision Comparison - {mode.upper()} Mode\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Summary Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total cases where ARE is better: {len(mode_cases)}\n\n")
        
        f.write("Case Type 1: ARE True Reject, CE False Accept (wrong predictions)\n")
        f.write(f"  Total: {len(case_type_1)}\n")
        f.write(f"    Head: {len(head_cases_1)}\n")
        f.write(f"    Tail: {len(tail_cases_1)}\n")
        # List sample indices for Case Type 1
        if len(case_type_1) > 0:
            sample_indices_1 = sorted([c.sample_idx for c in case_type_1])
            f.write(f"  Sample indices: {sample_indices_1[:50]}")  # Show first 50
            if len(sample_indices_1) > 50:
                f.write(f" ... (and {len(sample_indices_1) - 50} more)")
            f.write("\n")
        f.write("\n")
        
        f.write("Case Type 2: ARE True Accept, CE False Reject (correct predictions)\n")
        f.write(f"  Total: {len(case_type_2)}\n")
        f.write(f"    Head: {len(head_cases_2)}\n")
        f.write(f"    Tail: {len(tail_cases_2)}\n")
        # List sample indices for Case Type 2
        if len(case_type_2) > 0:
            sample_indices_2 = sorted([c.sample_idx for c in case_type_2])
            f.write(f"  Sample indices: {sample_indices_2[:50]}")  # Show first 50
            if len(sample_indices_2) > 50:
                f.write(f" ... (and {len(sample_indices_2) - 50} more)")
            f.write("\n")
        f.write("\n")
        
        # Group by rejection rate
        f.write("Breakdown by Rejection Rate:\n")
        f.write("-" * 80 + "\n")
        rejection_rates = sorted(set(c.rejection_rate for c in mode_cases))
        for rr in rejection_rates:
            rr_cases = [c for c in mode_cases if abs(c.rejection_rate - rr) < 1e-4]
            rr_type1 = [c for c in rr_cases if c.case_type == "are_true_reject_ce_false_accept"]
            rr_type2 = [c for c in rr_cases if c.case_type == "are_true_accept_ce_false_reject"]
            f.write(f"Rejection Rate {rr:.2f}:\n")
            f.write(f"  Type 1 (ARE True Reject): {len(rr_type1)}\n")
            if len(rr_type1) > 0:
                rr_type1_indices = sorted([c.sample_idx for c in rr_type1])
                f.write(f"    Sample indices: {rr_type1_indices[:30]}")  # Show first 30
                if len(rr_type1_indices) > 30:
                    f.write(f" ... (and {len(rr_type1_indices) - 30} more)")
                f.write("\n")
            f.write(f"  Type 2 (ARE True Accept): {len(rr_type2)}\n")
            if len(rr_type2) > 0:
                rr_type2_indices = sorted([c.sample_idx for c in rr_type2])
                f.write(f"    Sample indices: {rr_type2_indices[:30]}")  # Show first 30
                if len(rr_type2_indices) > 30:
                    f.write(f" ... (and {len(rr_type2_indices) - 30} more)")
                f.write("\n")
            f.write(f"  Total: {len(rr_cases)}\n\n")
    
    print(f"   Saved summary to {summary_path}")
    
    # Save all sample indices for each case type
    indices_file = output_dir / f"sample_indices_{mode}.txt"
    with open(indices_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"All Sample Indices - {mode.upper()} Mode\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Case Type 1: ARE True Reject, CE False Accept (wrong predictions)\n")
        f.write("-" * 80 + "\n")
        if len(case_type_1) > 0:
            sample_indices_1 = sorted([c.sample_idx for c in case_type_1])
            f.write(f"Total: {len(sample_indices_1)} samples\n")
            f.write(f"All indices: {sample_indices_1}\n")
        else:
            f.write("No cases found.\n")
        f.write("\n")
        
        f.write("Case Type 2: ARE True Accept, CE False Reject (correct predictions)\n")
        f.write("-" * 80 + "\n")
        if len(case_type_2) > 0:
            sample_indices_2 = sorted([c.sample_idx for c in case_type_2])
            f.write(f"Total: {len(sample_indices_2)} samples\n")
            f.write(f"All indices: {sample_indices_2}\n")
        else:
            f.write("No cases found.\n")
        f.write("\n")
        
        # Also group by rejection rate
        f.write("Breakdown by Rejection Rate:\n")
        f.write("-" * 80 + "\n")
        rejection_rates = sorted(set(c.rejection_rate for c in mode_cases))
        for rr in rejection_rates:
            rr_cases = [c for c in mode_cases if abs(c.rejection_rate - rr) < 1e-4]
            rr_type1 = [c for c in rr_cases if c.case_type == "are_true_reject_ce_false_accept"]
            rr_type2 = [c for c in rr_cases if c.case_type == "are_true_accept_ce_false_reject"]
            
            f.write(f"\nRejection Rate {rr:.2f}:\n")
            f.write(f"  Type 1 (ARE True Reject): {len(rr_type1)} samples\n")
            if len(rr_type1) > 0:
                rr_type1_indices = sorted([c.sample_idx for c in rr_type1])
                f.write(f"    Indices: {rr_type1_indices}\n")
            
            f.write(f"  Type 2 (ARE True Accept): {len(rr_type2)} samples\n")
            if len(rr_type2) > 0:
                rr_type2_indices = sorted([c.sample_idx for c in rr_type2])
                f.write(f"    Indices: {rr_type2_indices}\n")
    
    print(f"   Saved all sample indices to {indices_file.name}")
    
    # Detailed files
    for cases, case_type_str, type_name in [
        (case_type_1, "are_true_reject_ce_false_accept", "ARE_TrueReject_CE_FalseAccept"),
        (case_type_2, "are_true_accept_ce_false_reject", "ARE_TrueAccept_CE_FalseReject"),
    ]:
        if len(cases) == 0:
            continue
        
        # Group by rejection rate
        for rr in sorted(set(c.rejection_rate for c in cases)):
            rr_cases = [c for c in cases if abs(c.rejection_rate - rr) < 1e-4]
            
            # Separate by head/tail
            head_cases = [c for c in rr_cases if c.group == "head"]
            tail_cases = [c for c in rr_cases if c.group == "tail"]
            
            for group_name, group_cases in [("head", head_cases), ("tail", tail_cases)]:
                if len(group_cases) == 0:
                    continue
                
                detail_path = output_dir / f"{type_name}_{mode}_rr{rr:.2f}_{group_name}.txt"
                with open(detail_path, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"{type_name} - {mode.upper()} Mode - Rejection Rate {rr:.2f} - {group_name.upper()}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Total cases: {len(group_cases)}\n\n")
                    
                    f.write(f"{'Sample':<10} {'True':<6} {'CE Pred':<10} {'ARE Pred':<10} "
                           f"{'CE Decision':<12} {'ARE Decision':<13} "
                           f"{'CE MaxRew':<12} {'ARE MaxRew':<12} "
                           f"{'CE Thresh':<12} {'ARE Thresh':<12}\n")
                    f.write("-" * 80 + "\n")
                    
                    for case in sorted(group_cases, key=lambda x: x.sample_idx):
                        f.write(f"{case.sample_idx:<10} {case.true_label:<6} "
                               f"{case.predicted_class_ce:<10} {case.predicted_class_are:<10} "
                               f"{case.ce_decision:<12} {case.are_decision:<13} "
                               f"{case.ce_max_reweighted:<12.4f} {case.are_max_reweighted:<12.4f} "
                               f"{case.ce_threshold:<12.4f} {case.are_threshold:<12.4f}\n")
                
                print(f"   Saved {len(group_cases)} {group_name} cases to {detail_path.name}")


def main():
    """Main function."""
    print("=" * 80)
    print("ARE vs CE Rejection Decision Comparison")
    print("=" * 80)
    RR_TARGET_SAVE = 0.4  # rejection rate to save images for
    USE_PIPELINE_EVAL = True  # run per-sample with notebook pipelines
    
    # Load models
    print("\n0. Loading models...")
    from src.infer.loaders import load_all_experts
    
    ce_expert = load_ce_expert()
    all_experts = load_all_experts()
    gating_network = load_gating_network()
    class_to_group = load_class_to_group()
    
    # Load test set
    images, labels, test_indices = load_test_set()
    
    modes = ["balanced", "worst"]
    all_cases = []
    
    if USE_PIPELINE_EVAL:
        for mode in modes:
            # Per-sample, per-mode evaluation with notebook pipelines
            cases = evaluate_with_pipelines(
                images, labels, test_indices, class_to_group,
                ce_expert, all_experts, gating_network, mode
            )
            all_cases.extend(cases)
    else:
        # Legacy path (batch posteriors + compare_decisions)
        ce_posteriors, are_posteriors = run_inference_batch(
            images, ce_expert, gating_network, all_experts
        )
        
        # Load plugin configurations
        print("\n3. Loading plugin configurations...")
        
        for mode in modes:
            print(f"\n   Processing {mode} mode...")
            
            ce_configs = load_all_plugin_configs("ce_only", mode)
            are_configs = load_all_plugin_configs("moe", mode)
            
            if len(ce_configs) == 0 or len(are_configs) == 0:
                print(f"   ⚠️  Skipping {mode} mode: missing configurations")
                continue
            
            # Match configurations by rejection rate (with tolerance for floating point errors)
            # Group rejection rates into bins (e.g., 0.0-0.05, 0.05-0.15, 0.15-0.25, etc.)
            def round_to_nearest_rr(rr):
                """Round rejection rate to nearest 0.1 (0.0, 0.1, 0.2, etc.)"""
                return round(rr * 10) / 10
            
            ce_by_rr_rounded = {}
            for c in ce_configs:
                rr_rounded = round_to_nearest_rr(c["rejection_rate"])
                if rr_rounded not in ce_by_rr_rounded:
                    ce_by_rr_rounded[rr_rounded] = []
                ce_by_rr_rounded[rr_rounded].append(c)
            
            are_by_rr_rounded = {}
            for c in are_configs:
                rr_rounded = round_to_nearest_rr(c["rejection_rate"])
                if rr_rounded not in are_by_rr_rounded:
                    are_by_rr_rounded[rr_rounded] = []
                are_by_rr_rounded[rr_rounded].append(c)
            
            # Find common rounded rejection rates
            common_rrs = sorted(set(ce_by_rr_rounded.keys()) & set(are_by_rr_rounded.keys()))
            
            print(f"   Found {len(common_rrs)} common rejection rates: {common_rrs}")
            
            for rr_rounded in common_rrs:
                # Use the first config for each rejection rate (or could average, but first is simpler)
                ce_config = ce_by_rr_rounded[rr_rounded][0]
                are_config = are_by_rr_rounded[rr_rounded][0]
                
                print(f"     Processing rejection rate {rr_rounded:.2f}...")
                print(f"       CE actual RR: {ce_config['rejection_rate']:.4f}")
                print(f"       ARE actual RR: {are_config['rejection_rate']:.4f}")
                
                cases = compare_decisions(
                    ce_posteriors,
                    are_posteriors,
                    labels,
                    class_to_group,
                    ce_config,
                    are_config,
                    test_indices,
                    mode
                )
                
                # Use the rounded rejection rate for consistency
                for case in cases:
                    case.rejection_rate = rr_rounded
                
                all_cases.extend(cases)
                print(f"       Found {len(cases)} cases where ARE is better")
    
    # Save results
    print("\n4. Saving results...")
    for mode in modes:
        save_results(all_cases, mode, OUTPUT_DIR)
    
    # Save sample images for rejection rate target (RR_TARGET_SAVE)
    print(f"\n5. Saving sample images for rejection rate {RR_TARGET_SAVE:.1f}...")
    dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
    class_to_group_np = class_to_group.cpu().numpy()
    base_dir = Path("./infer_samples")
    
    for mode in modes:
        mode_cases = [c for c in all_cases if c.mode == mode and abs(c.rejection_rate - RR_TARGET_SAVE) < 1e-6]
        if len(mode_cases) == 0:
            print(f"   No cases to save for mode={mode}, rr={RR_TARGET_SAVE:.1f}")
            continue
        
        # Only keep beneficial cases: ARE true (accept/reject) while CE wrong
        type1 = [c for c in mode_cases if c.case_type == "are_true_reject_ce_false_accept"]
        type2 = [c for c in mode_cases if c.case_type == "are_true_accept_ce_false_reject"]
        cases_to_save = type1 + type2
        
        saved_count = 0
        for case in cases_to_save:
            class_idx = case.true_label
            group_idx = class_to_group_np[class_idx]
            group_name = "head" if group_idx == 0 else "tail"
            class_name = dataset.classes[class_idx].replace(" ", "_")
            
            out_dir = base_dir / mode / group_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            img, _ = dataset[case.sample_idx]
            filename = f"{group_name}_{class_idx}_{class_name}_idx{case.sample_idx}.png"
            img.save(out_dir / filename)
            saved_count += 1
        
        print(f"   Saved {saved_count} images to {base_dir / mode}")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

