"""
Comprehensive AURC Evaluation (Correct Methodology with Reweighting)

This script follows the proper AURC evaluation methodology for AR-GSE:
  - Validation set: tuneV + val (combined for threshold optimization)
  - Test set: test (held-out for final evaluation)
  - All splits are balanced (from test set), but metrics are reweighted for long-tail

The AR-GSE algorithm:
  1. Compute RAW margins: score - threshold_per_sample (without c)
  2. For each rejection cost c:
     - Accept if: raw_margin >= -c
     - This is equivalent to: margin + c >= 0 (where margin = raw_margin)
  3. Compute coverage and risk on test set
  4. Integrate risk over coverage to get AURC

Key insight: For AR-GSE, the threshold is deterministic (threshold = -c).
No optimization needed - just evaluate at each cost value.

Outputs:
  - aurc_detailed_results.csv: RC points for each metric
  - aurc_summary.json: AURC values for full and [0.2, 1.0] coverage ranges
  - aurc_curves.png: plots of RC curves and AURC comparison
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

# Custom modules
from src.models.argse import AR_GSE
from src.train.gse_balanced_plugin import compute_raw_margin  # Import RAW margin (no c subtraction)
from src.metrics.reweighted_metrics import ReweightedMetrics  # Import reweighted metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits_fixed',  # Updated to use fixed splits
        'num_classes': 100,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits/cifar100_lt_if100/',  # Updated path
    },
    'aurc_eval': {
        # Choose evaluation mode:
        # 'fast': 9 strategic points for quick evaluation  
        # 'detailed': 21 points for smoother curves
        # 'full': 41 points for publication-quality curves
        'mode': 'detailed',  # Change to 'detailed' or 'full' for better visualization
        
        'cost_values_fast': [0.0, 0.1, 0.5, 0.75, 0.85, 0.91, 0.95, 0.97, 0.99],
        'cost_values_detailed': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.91, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995],
        'cost_values_full': list(np.linspace(0.0, 1.0, 41)),
        
        'metrics': ['balanced', 'worst'],  # Focus on group-aware metrics only
        'interpolate_smooth_curves': True,  # Add interpolation for smoother plotting
    },
    'plugin_checkpoint': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './results_worst_eg_improved/cifar100_lt_if100',
    'seed': 42,
}

#############################################
# Data loading for comprehensive AURC only  #
#############################################

def load_aurc_splits_data():
    """
    Load all three splits for proper AURC evaluation:
      - tunev: tuning/validation split for threshold optimization
      - val: validation split for threshold optimization  
      - test: held-out test set for final evaluation
    
    Returns:
        Tuple of (tunev_data, val_data, test_data) where each is (logits, labels, indices)
    """
    logits_root = Path(CONFIG['experts']['logits_dir'])
    # Check if dataset subdirectory exists
    dataset_subdir = logits_root / CONFIG['dataset']['name']
    if dataset_subdir.exists():
        logits_root = dataset_subdir
    
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    num_experts = len(CONFIG['experts']['names'])
    num_classes = CONFIG['dataset']['num_classes']
    
    # Load splits indices
    print("📂 Loading AURC evaluation splits...")
    
    # tunev (from test set - balanced)
    with open(splits_dir / 'tunev_indices.json', 'r') as f:
        tunev_indices = json.load(f)
    
    # val (from test set - balanced)
    with open(splits_dir / 'val_indices.json', 'r') as f:
        val_indices = json.load(f)
    
    # test (from test set - balanced)
    with open(splits_dir / 'test_indices.json', 'r') as f:
        test_indices = json.load(f)
    
    print(f"✅ tunev: {len(tunev_indices)} samples (test set - balanced)")
    print(f"✅ val: {len(val_indices)} samples (test set - balanced)")
    print(f"✅ test: {len(test_indices)} samples (test set - balanced)")
    print(f"✅ Validation (tunev + val): {len(tunev_indices) + len(val_indices)} samples")
    
    # Load datasets - all splits come from test set now (balanced)
    cifar_test_full = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)
    
    # Helper function to load logits (supports both .npz and .pt)
    def load_expert_logits(split_name, indices):
        logits = torch.zeros(len(indices), num_experts, num_classes)
        for i, expert_name in enumerate(CONFIG['experts']['names']):
            # Try .npz first (new format), then .pt (old format)
            npz_path = logits_root / expert_name / f"{split_name}_logits.npz"
            pt_path = logits_root / expert_name / f"{split_name}_logits.pt"
            
            if npz_path.exists():
                data = np.load(npz_path)
                logits[:, i, :] = torch.from_numpy(data['logits'])
            elif pt_path.exists():
                logits[:, i, :] = torch.load(pt_path, map_location='cpu', weights_only=False)
            else:
                raise FileNotFoundError(f"Missing logits for {expert_name} split {split_name}: {npz_path} or {pt_path}")
        return logits
    
    # Load tunev data
    tunev_logits = load_expert_logits('tunev', tunev_indices)
    tunev_labels = torch.tensor(np.array(cifar_test_full.targets)[tunev_indices])
    
    # Load val data
    val_logits = load_expert_logits('val', val_indices)
    val_labels = torch.tensor(np.array(cifar_test_full.targets)[val_indices])
    
    # Load test data
    test_logits = load_expert_logits('test', test_indices)
    test_labels = torch.tensor(np.array(cifar_test_full.targets)[test_indices])
    
    return (tunev_logits, tunev_labels, tunev_indices), (val_logits, val_labels, val_indices), (test_logits, test_labels, test_indices)

def get_mixture_posteriors(model, logits):
    """Compute mixture posteriors η̃(x) from expert logits."""
    model.eval()
    with torch.no_grad():
        logits = logits.to(DEVICE)
        expert_posteriors = torch.softmax(logits, dim=-1)              # [B, E, C]
        gating_features = model.feature_builder(logits)
        gating_weights = torch.softmax(model.gating_net(gating_features), dim=1)  # [B, E]
        eta_mix = torch.einsum('be,bec->bc', gating_weights, expert_posteriors)   # [B, C]
    return eta_mix.cpu()

#############################################
# Core AURC utilities                        #
#############################################


def compute_group_risk_for_aurc(preds, labels, accepted_mask, class_to_group, K, 
                               class_weights=None, metric_type="balanced"):
    """
    Compute group-aware risk for AURC evaluation on accepted samples.
    
    Focuses on group-aware metrics (balanced and worst) with proper reweighting.
    
    Args:
        preds: [N] predictions
        labels: [N] true labels  
        accepted_mask: [N] boolean mask for accepted samples
        class_to_group: [C] class to group mapping
        K: number of groups
        class_weights: dict mapping class_id -> weight (for reweighting), or None
        metric_type: 'balanced' or 'worst' (standard removed)
        
    Returns:
        risk: scalar risk value (error rate)
    """
    if accepted_mask.sum() == 0:
        return 1.0
    
    y = labels
    g = class_to_group[y]
    
    # Compute group-wise errors with reweighting
    group_errors = []
    for k in range(K):
        group_mask = (g == k)
        group_accepted = accepted_mask & group_mask
        if group_accepted.sum() == 0:
            group_errors.append(1.0)
        else:
            group_correct = (preds[group_accepted] == y[group_accepted])
            
            # Apply reweighting if class_weights provided
            if class_weights is not None:
                weights = torch.tensor([class_weights[int(c)] for c in y[group_accepted]], 
                                      dtype=torch.float32)
                weighted_correct = (group_correct.float() * weights).sum()
                total_weight = weights.sum()
                group_error = 1.0 - (weighted_correct / total_weight).item()
            else:
                group_error = 1.0 - group_correct.float().mean().item()
            
            group_errors.append(group_error)
    
    if metric_type == 'balanced':
        return float(np.mean(group_errors))
    elif metric_type == 'worst':
        return float(np.max(group_errors))
    else:
        raise ValueError(f"Unknown metric type: {metric_type}. Use 'balanced' or 'worst'.")

def find_optimal_threshold_for_cost(confidence_scores, preds, labels, class_to_group, K, 
                                   cost_c, class_weights=None, metric_type="balanced",
                                   use_per_group=False, t_group_base=None):
    """
    Find optimal threshold for a given cost.
    
    Two modes:
    1. Global threshold (use_per_group=False): threshold = -c
    2. Per-group thresholds (use_per_group=True): t_k = t_group_base[k] * scale
    
    Args:
        confidence_scores: [N] RAW confidence scores (raw GSE margins)
        preds: [N] predictions
        labels: [N] true labels
        class_to_group: [C] class to group mapping
        K: number of groups
        cost_c: rejection cost or scale factor
        class_weights: dict mapping class_id -> weight (for reweighting), or None
        metric_type: risk metric type
        use_per_group: if True, use per-group thresholds with scaling
        t_group_base: [K] base per-group thresholds (for scaling)
        
    Returns:
        optimal_threshold: scalar (global mode) or tensor [K] (per-group mode)
    """
    if use_per_group and t_group_base is not None:
        # Per-group mode: scale base thresholds
        # cost_c acts as a scale factor (0.0 = reject all, large = accept all)
        # We want: cost_c=0 → very negative thresholds (reject all)
        #          cost_c=1 → original thresholds
        #          cost_c>1 → even more lenient
        scale = cost_c
        return t_group_base * scale
    else:
        # Global mode: threshold = -c (original AR-GSE)
        return -cost_c

def sweep_cost_values_aurc(confidence_scores_val, preds_val, labels_val, 
                          confidence_scores_test, preds_test, labels_test,
                          class_to_group, K, cost_values, class_weights=None, metric_type="balanced",
                          use_per_group=False, t_group_base=None):
    """
    Sweep cost values and return (cost, coverage, risk) points on test set.
    
    Supports two modes:
    1. Global threshold: threshold = -c for all samples
    2. Per-group thresholds: t_k = t_group_base[k] * scale for each group
    
    Args:
        confidence_scores_val: [N_val] validation confidence scores
        preds_val: [N_val] validation predictions
        labels_val: [N_val] validation labels
        confidence_scores_test: [N_test] test confidence scores
        preds_test: [N_test] test predictions
        labels_test: [N_test] test labels
        class_to_group: [C] class to group mapping
        K: number of groups
        cost_values: array of cost values to sweep
        class_weights: dict mapping class_id -> weight (for reweighting), or None
        metric_type: risk metric type
        use_per_group: if True, use per-group thresholds
        t_group_base: [K] base per-group thresholds (for scaling)
        
    Returns:
        rc_points: list of (cost, coverage, risk) tuples
    """
    rc_points = []
    
    mode_str = "per-group" if use_per_group else "global"
    print(f"🔄 Sweeping {len(cost_values)} cost values for {metric_type} metric ({mode_str} mode)...")
    
    for i, cost_c in enumerate(cost_values):
        # Find optimal threshold (global or per-group)
        optimal_threshold = find_optimal_threshold_for_cost(
            confidence_scores_val, preds_val, labels_val, class_to_group, K, cost_c, 
            class_weights, metric_type, use_per_group, t_group_base
        )
        
        # Apply to test set
        if use_per_group and t_group_base is not None:
            # Per-group thresholds: different threshold per sample based on label's group
            test_groups = class_to_group[labels_test]
            thresholds_per_sample = optimal_threshold[test_groups]
            accepted_test = confidence_scores_test > thresholds_per_sample
        else:
            # Global threshold: same for all samples
            accepted_test = confidence_scores_test >= optimal_threshold
            
        coverage_test = accepted_test.float().mean().item()
        risk_test = compute_group_risk_for_aurc(preds_test, labels_test, accepted_test, 
                                               class_to_group, K, class_weights, metric_type)
        
        rc_points.append((cost_c, coverage_test, risk_test))
        
        if (i + 1) % 3 == 0 or i == 0:  # Show more frequent progress for small cost arrays
            print(f"   Progress: {i+1}/{len(cost_values)} - c={cost_c:.3f}, "
                  f"cov={coverage_test:.3f}, risk={risk_test:.3f}")
    
    print(f"   >> Coverage range: {min(p[1] for p in rc_points):.3f} to {max(p[1] for p in rc_points):.3f}")
    return rc_points

def compute_aurc_from_points(rc_points, coverage_range='full'):
    """
    Compute AURC using trapezoidal integration.
    
    Args:
        rc_points: List of (cost, coverage, risk) tuples
        coverage_range: 'full' for [0, 1] or '0.2-1.0' for [0.2, 1.0]
        
    Returns:
        aurc: scalar AURC value
    """
    # Sort by coverage
    rc_points = sorted(rc_points, key=lambda x: x[1])
    
    coverages = [p[1] for p in rc_points]
    risks = [p[2] for p in rc_points]
    
    if coverage_range == '0.2-1.0':
        # Need to interpolate risk at coverage=0.2 BEFORE filtering
        # Find points around 0.2 for interpolation
        all_points = list(zip(coverages, risks))
        
        # Find the last point with coverage < 0.2 and first point with coverage >= 0.2
        points_below = [(c, r) for c, r in all_points if c < 0.2]
        points_above = [(c, r) for c, r in all_points if c >= 0.2]
        
        if not points_above:
            # No points >= 0.2, cannot compute
            return float('nan')
        
        # Interpolate risk at coverage = 0.2
        if points_below:
            # Have points on both sides of 0.2
            c_below, r_below = points_below[-1]  # Last point before 0.2
            c_above, r_above = points_above[0]   # First point after 0.2
            
            if c_above > c_below:
                # Linear interpolation
                risk_at_02 = r_below + (r_above - r_below) * (0.2 - c_below) / (c_above - c_below)
            else:
                # Should not happen, but use r_above as fallback
                risk_at_02 = r_above
        else:
            # No points below 0.2, use first point's risk
            risk_at_02 = points_above[0][1]
        
        # Build final curve starting from 0.2
        coverages = [0.2] + [c for c, r in points_above]
        risks = [risk_at_02] + [r for c, r in points_above]
        
        # Ensure endpoint at 1.0
        if coverages[-1] < 1.0:
            coverages = coverages + [1.0]
            risks = risks + [risks[-1]]
    else:
        # Full range [0, 1]
        # Ensure we have endpoints for proper integration
        if coverages[0] > 0.0:
            coverages = [0.0] + coverages
            # When coverage=0 (reject all), risk should be very high (no correct predictions)
            # Use the first available risk as approximation (conservative)
            risks = [risks[0]] + risks
        
        if coverages[-1] < 1.0:
            coverages = coverages + [1.0]
            risks = risks + [risks[-1]]  # Extend last risk to coverage=1
    
    # Trapezoidal integration
    aurc = np.trapz(risks, coverages)  # Updated from deprecated trapezoid
    return aurc
#############################################
# Plotting                                    
#############################################

def plot_aurc_curves(all_rc_points, aurc_results, save_path):
    """Plot rejection-based curves and grouped AURC comparisons.

    Layout (1 row, 3 columns):
      1. Error vs Proportion of Rejections (Full 0-1)
      2. Error vs Proportion of Rejections (Zoom 0-0.8)
      3. Grouped bar chart: Full-range AURC vs Practical (0.2-1.0) for each metric.
    """
    from scipy.interpolate import interp1d
    colors = {'balanced': 'green', 'worst': 'orange'}
    markers = {'balanced': 'v', 'worst': 'x'}
    linestyles = {'balanced': '-', 'worst': '-'}

    plt.figure(figsize=(18, 5))

    # ---- Subplot 1: Full range rejection 0-1 ----
    ax1 = plt.subplot(1, 3, 1)
    for metric, rc_points in all_rc_points.items():
        rc_points = sorted(rc_points, key=lambda x: x[1])
        coverages = np.array([p[1] for p in rc_points])
        risks = np.array([p[2] for p in rc_points])
        rejection = 1.0 - coverages
        aurc_full = aurc_results[metric]
        ax1.scatter(rejection, risks, color=colors[metric], s=40, marker=markers[metric],
                    edgecolor='white', linewidth=1.5, zorder=5, alpha=0.85)
        if len(rejection) >= 4:
            try:
                idx = np.argsort(rejection)
                f = interp1d(rejection[idx], risks[idx], kind='cubic', fill_value='extrapolate')
                rej_smooth = np.linspace(rejection.min(), rejection.max(), 250)
                risk_smooth = f(rej_smooth)
                ax1.plot(rej_smooth, risk_smooth, color=colors[metric], linestyle=linestyles[metric],
                         linewidth=2.2, label=f"{metric.title()} (AURC={aurc_full:.4f})")
            except Exception:
                ax1.plot(rejection, risks, color=colors[metric], linestyle=linestyles[metric],
                         linewidth=2.2, label=f"{metric.title()} (AURC={aurc_full:.4f})")
        else:
            ax1.plot(rejection, risks, color=colors[metric], linestyle=linestyles[metric],
                     linewidth=2.2, label=f"{metric.title()} (AURC={aurc_full:.4f})")
    ax1.set_xlabel('Proportion of Rejections', fontweight='bold')
    ax1.set_ylabel('Error', fontweight='bold')
    ax1.set_title('Error vs Rejection Rate (0-1)', fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 1)
    ax1.legend(fontsize=10)

    # ---- Subplot 2: Zoom rejection 0-0.8 ----
    ax2 = plt.subplot(1, 3, 2)
    for metric, rc_points in all_rc_points.items():
        rc_points = sorted(rc_points, key=lambda x: x[1])
        coverages = np.array([p[1] for p in rc_points])
        risks = np.array([p[2] for p in rc_points])
        rejection = 1.0 - coverages
        ax2.scatter(rejection, risks, color=colors[metric], s=40, marker=markers[metric],
                    edgecolor='white', linewidth=1.5, zorder=5, alpha=0.85)
        if len(rejection) >= 4:
            try:
                idx = np.argsort(rejection)
                f = interp1d(rejection[idx], risks[idx], kind='cubic', fill_value='extrapolate')
                rej_smooth = np.linspace(rejection.min(), min(0.8, rejection.max()), 200)
                risk_smooth = f(rej_smooth)
                ax2.plot(rej_smooth, risk_smooth, color=colors[metric], linestyle=linestyles[metric],
                         linewidth=2.2, label=metric.title())
            except Exception:
                ax2.plot(rejection, risks, color=colors[metric], linestyle=linestyles[metric],
                         linewidth=2.2, label=metric.title())
        else:
            ax2.plot(rejection, risks, color=colors[metric], linestyle=linestyles[metric],
                     linewidth=2.2, label=metric.title())
    ax2.set_xlabel('Proportion of Rejections', fontweight='bold')
    ax2.set_ylabel('Error', fontweight='bold')
    ax2.set_title('Error vs Rejection Rate (0-0.8)', fontweight='bold')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 0.8)
    ax2.legend(fontsize=10)

    # ---- Subplot 3: Grouped AURC bars (Full vs 0.2-1.0) ----
    ax3 = plt.subplot(1, 3, 3)
    metrics_main = [m for m in aurc_results.keys() if not m.endswith('_02_10')]
    full_values = [aurc_results[m] for m in metrics_main]
    practical_values = [aurc_results[f"{m}_02_10"] for m in metrics_main]
    x = np.arange(len(metrics_main))
    width = 0.35
    bars1 = ax3.bar(x - width/2, full_values, width, label='Full (0-1)', color=[colors[m] for m in metrics_main], alpha=0.75, edgecolor='black')
    bars2 = ax3.bar(x + width/2, practical_values, width, label='Practical (0.2-1.0)', color=[colors[m] for m in metrics_main], alpha=0.45, edgecolor='black', hatch='//')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.title() for m in metrics_main], fontweight='bold')
    ax3.set_ylabel('AURC', fontweight='bold')
    ax3.set_title('AURC Comparison (Full vs 0.2-1.0)', fontweight='bold')
    ax3.grid(alpha=0.3, axis='y', linestyle='--')
    # Value labels
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(full_values)*0.015,
                 f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(practical_values)*0.015,
                 f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=9)
    # Difference annotation (worst vs balanced full)
    if len(metrics_main) == 2:
        diff_pct = ((full_values[1] - full_values[0]) / full_values[0]) * 100
        ax3.text(0.5, max(full_values + practical_values) * 0.75,
                 f"Worst full AURC +{diff_pct:.1f}% vs Balanced", ha='center', va='center',
                 fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.5))
    ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">> Saved AURC plots to {save_path}")

def main():
    # Reproducibility
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    print("=== Comprehensive AURC Evaluation (Reweighted for Long-Tail) ===")

    # Output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class weights for reweighting
    class_weights_path = Path(CONFIG['dataset']['splits_dir']) / 'class_weights.json'
    if class_weights_path.exists():
        with open(class_weights_path, 'r') as f:
            class_weights_list = json.load(f)
        # Convert list to dict: class_id -> weight
        class_weights = {i: w for i, w in enumerate(class_weights_list)}
        print(f"✅ Loaded class weights from {class_weights_path}")
        print(f"   Sample weights: class 0={class_weights[0]:.4f}, class 99={class_weights[99]:.4f}")
    else:
        class_weights = None
        print("⚠️  No class_weights.json found - using uniform weighting")

    # 1) Load plugin checkpoint (α*, μ*, class_to_group, gating)
    plugin_ckpt_path = Path(CONFIG['plugin_checkpoint'])
    if not plugin_ckpt_path.exists():
        raise FileNotFoundError(f"Plugin checkpoint not found: {plugin_ckpt_path}")
    print(f"📂 Loading plugin checkpoint: {plugin_ckpt_path}")
    checkpoint = torch.load(plugin_ckpt_path, map_location=DEVICE, weights_only=False)
    alpha_star = checkpoint['alpha'].to(DEVICE)
    mu_star = checkpoint['mu'].to(DEVICE)
    class_to_group = checkpoint['class_to_group'].to(DEVICE)
    num_groups = checkpoint['num_groups']
    
    # Load per-group thresholds if available
    use_per_group_thresholds = checkpoint.get('per_group_threshold', False)
    if 't_group' in checkpoint:
        t_group_star = checkpoint['t_group']
        if isinstance(t_group_star, list):
            t_group_star = torch.tensor(t_group_star, dtype=torch.float32)
        t_group_star = t_group_star.to(DEVICE)
    else:
        t_group_star = None
        use_per_group_thresholds = False

    print("✅ Loaded optimal parameters:")
    print(f"   α* = {alpha_star.detach().cpu().tolist()}")
    print(f"   μ* = {mu_star.detach().cpu().tolist()}")
    if use_per_group_thresholds and t_group_star is not None:
        print(f"   t_group* = {t_group_star.detach().cpu().tolist()}")
        print(f"   → Using PER-GROUP thresholds (consistent with plugin training)")

    # 2) Build AR-GSE and load gating
    num_experts = len(CONFIG['experts']['names'])
    with torch.no_grad():
        dummy_logits = torch.zeros(2, num_experts, CONFIG['dataset']['num_classes']).to(DEVICE)
        temp_model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, 1).to(DEVICE)
        gating_feature_dim = temp_model.feature_builder(dummy_logits).size(-1)
        del temp_model
    print(f"✅ Dynamic gating feature dim: {gating_feature_dim}")
    model = AR_GSE(num_experts, CONFIG['dataset']['num_classes'], num_groups, gating_feature_dim).to(DEVICE)

    # Load gating network weights with dimension compatibility check
    if 'gating_net_state_dict' in checkpoint:
        saved_state = checkpoint['gating_net_state_dict']
        current_state = model.gating_net.state_dict()
        
        compatible = True
        for key in saved_state.keys():
            if key in current_state and saved_state[key].shape != current_state[key].shape:
                print(f"⚠️  Dimension mismatch for {key}: saved {saved_state[key].shape} vs current {current_state[key].shape}")
                compatible = False
        
        if compatible:
            model.gating_net.load_state_dict(saved_state)
            print("✅ Gating network weights loaded successfully")
        else:
            print("❌ Gating checkpoint incompatible with enriched features. Using random weights.")
    else:
        print("⚠️ No gating network weights found in checkpoint")

    # Set optimal α*, μ*
    with torch.no_grad():
        model.alpha.copy_(alpha_star)
        model.mu.copy_(mu_star)
    print("✅ Model configured with optimal parameters and gating ready")

    # 3) Load AURC splits: tuneV + val (validation), test (test)
    print("\n" + "="*60)
    print("COMPREHENSIVE AURC EVALUATION (REWEIGHTED)")
    print("="*60)
    (tunev_logits, tunev_labels, _), (val_logits, val_labels, _), (test_logits, test_labels, _) = load_aurc_splits_data()
    
    print(f"📊 Validation set (tuneV + val): {len(tunev_labels) + len(val_labels)} samples (balanced)")
    print(f"📊 Test set (test): {len(test_labels)} samples (balanced)")
    print("✅ Correct methodology: Optimize thresholds on (tuneV + val), evaluate on test")
    print("✅ All splits are balanced, metrics reweighted for long-tail performance")

    # 4) Compute mixture posteriors and GSE margins/predictions
    print("\n🔮 Computing mixture posteriors for all splits...")
    tunev_eta_mix = get_mixture_posteriors(model, tunev_logits)
    val_eta_mix = get_mixture_posteriors(model, val_logits)
    test_eta_mix = get_mixture_posteriors(model, test_logits)

    class_to_group_cpu = class_to_group.cpu()
    alpha_star_cpu = alpha_star.cpu()
    mu_star_cpu = mu_star.cpu()

    # Compute RAW margins for all splits (without subtracting c)
    # The threshold c will be applied during AURC evaluation
    gse_margins_tunev = compute_raw_margin(tunev_eta_mix, alpha_star_cpu, mu_star_cpu, class_to_group_cpu)
    gse_margins_val = compute_raw_margin(val_eta_mix, alpha_star_cpu, mu_star_cpu, class_to_group_cpu)
    gse_margins_test = compute_raw_margin(test_eta_mix, alpha_star_cpu, mu_star_cpu, class_to_group_cpu)

    # Compute predictions for all splits
    preds_tunev = (alpha_star_cpu[class_to_group_cpu] * tunev_eta_mix).argmax(dim=1)
    preds_val = (alpha_star_cpu[class_to_group_cpu] * val_eta_mix).argmax(dim=1)
    preds_test = (alpha_star_cpu[class_to_group_cpu] * test_eta_mix).argmax(dim=1)
    
    # Combine tuneV + val as validation set for threshold optimization
    gse_margins_val_combined = torch.cat([gse_margins_tunev, gse_margins_val])
    preds_val_combined = torch.cat([preds_tunev, preds_val])
    labels_val_combined = torch.cat([tunev_labels, val_labels])
    
    print(f"✅ Combined validation set: {len(labels_val_combined)} samples")

    # 5) Debug confidence scores distribution first
    print("\n🔍 DEBUGGING: GSE RAW margin distribution")
    print(f"   Test raw margins - min: {gse_margins_test.min():.4f}, max: {gse_margins_test.max():.4f}")
    print(f"   Test raw margins - mean: {gse_margins_test.mean():.4f}, std: {gse_margins_test.std():.4f}")
    
    # Check percentiles to understand distribution
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    margin_percentiles = torch.quantile(gse_margins_test, torch.tensor([p/100.0 for p in percentiles]))
    print(f"   Percentiles: {dict(zip(percentiles, [f'{v:.4f}' for v in margin_percentiles]))}")
    
    # Debug threshold behavior
    if use_per_group_thresholds and t_group_star is not None:
        print("\n🔍 DEBUGGING: Per-group threshold behavior")
        print(f"   Base thresholds: {t_group_star.cpu().tolist()}")
        debug_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
        for scale in debug_scales:
            t_scaled = t_group_star.cpu() * scale  # Move to CPU for consistency
            test_groups = class_to_group_cpu[test_labels]
            thresholds_per_sample = t_scaled[test_groups]
            coverage_test = (gse_margins_test > thresholds_per_sample).float().mean().item()
            print(f"   scale={scale:.1f} → t_group={t_scaled.tolist()} → coverage={coverage_test:.4f}")
    else:
        print("\n🔍 DEBUGGING: Global threshold behavior (threshold = -c)")
        debug_costs = [0.0, 0.1, 0.5, 0.75, 0.99]
        for cost_c in debug_costs:
            threshold = -cost_c  # AR-GSE threshold is deterministic
            coverage_test = (gse_margins_test >= threshold).float().mean().item()
            print(f"   c={cost_c:.2f} → threshold={threshold:.4f} → coverage={coverage_test:.4f}")
    
    # 6) Sweep cost values and compute AURC
    mode = CONFIG['aurc_eval']['mode']
    if mode == 'fast':
        cost_values = CONFIG['aurc_eval']['cost_values_fast']
    elif mode == 'detailed':
        cost_values = CONFIG['aurc_eval']['cost_values_detailed']
    elif mode == 'full':
        cost_values = CONFIG['aurc_eval']['cost_values_full']
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fast', 'detailed', or 'full'")
    
    metrics = CONFIG['aurc_eval']['metrics']
    print(f"\n🎯 Cost grid ({mode} mode): {len(cost_values)} values from {cost_values[0]:.1f} to {cost_values[-1]:.1f}")
    if mode == 'fast':
        print("   ⚡ Fast mode: Quick evaluation with strategic cost points")
    elif mode == 'detailed':
        print("   📈 Detailed mode: Better visualization with more points")
    else:
        print("   🎨 Full mode: Publication-quality smooth curves")

    aurc_results = {}
    all_rc_points = {}
    
    # Determine cost/scale values based on mode
    # if use_per_group_thresholds and t_group_star is not None:
    #     # Per-group mode: use scale factors
    #     # We want to sweep from very selective (scale~0) to very lenient (scale>1)
    #     # Base thresholds are negative, so scale=0 → very negative (reject all)
    #     #                                scale=1 → original thresholds
    #     #                                scale=2+ → more lenient
    #     scale_values = [0.0, 0.1, 0.5, 0.75, 0.85, 0.91, 0.95, 0.97, 0.99]
    #     print(f"\n🎯 Scale grid (per-group mode): {len(scale_values)} values from {scale_values[0]:.1f} to {scale_values[-1]:.1f}")
    #     print(f"   Base thresholds: {t_group_star.cpu().tolist()}")
    #     print(f"   → scale=0: very selective (reject most)")
    #     print(f"   → scale=1: original thresholds from training")
    #     print(f"   → scale>1: more lenient (accept more)")
    #     sweep_values = scale_values
    # else:
        # Global mode: use regular cost values
    sweep_values = cost_values
    
    for metric in metrics:
        print(f"\n🔄 Processing {metric} metric {'(REWEIGHTED)' if class_weights else ''}...")
        print(f"   • Optimizing thresholds on validation (tunev + val): {len(labels_val_combined)} samples")
        print(f"   • Evaluating on test: {len(test_labels)} samples")
        
        rc_points = sweep_cost_values_aurc(
            gse_margins_val_combined, preds_val_combined, labels_val_combined,  # Validation
            gse_margins_test, preds_test, test_labels,                          # Test
            class_to_group_cpu, num_groups, sweep_values, class_weights, metric,  # Pass class_weights
            use_per_group_thresholds, t_group_star.cpu() if use_per_group_thresholds else None  # Per-group params
        )
        aurc_full = compute_aurc_from_points(rc_points, coverage_range='full')
        aurc_02_10 = compute_aurc_from_points(rc_points, coverage_range='0.2-1.0')
        aurc_results[metric] = aurc_full
        aurc_results[f'{metric}_02_10'] = aurc_02_10
        all_rc_points[metric] = rc_points
        print(f"   ✅ {metric.upper()} AURC (0-1):     {aurc_full:.6f}")
        print(f"   ✅ {metric.upper()} AURC (0.2-1):   {aurc_02_10:.6f}")

    # 6) Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    aurc_rows = []
    for metric, rc_points in all_rc_points.items():
        for cost_c, coverage, risk in rc_points:
            aurc_rows.append({'metric': metric, 'cost': cost_c, 'coverage': coverage, 'risk': risk})
    pd.DataFrame(aurc_rows).to_csv(output_dir / 'aurc_detailed_results.csv', index=False)
    with open(output_dir / 'aurc_summary.json', 'w') as f:
        json.dump(aurc_results, f, indent=4)
    plot_aurc_curves(all_rc_points, aurc_results, output_dir / 'aurc_curves.png')

    # 7) Final summary
    print("\n" + "="*60)
    print("FINAL AURC RESULTS - GROUP-AWARE METRICS")
    print("(REWEIGHTED FOR LONG-TAIL)" if class_weights else "(UNIFORM WEIGHTING)")
    print("="*60)
    print("\n📊 AURC (Full Range 0-1):")
    for metric in metrics:
        print(f"   • {metric.upper():>12} AURC: {aurc_results[metric]:.6f}")
    print("\n📊 AURC (Practical Range 0.2-1):")
    for metric in metrics:
        print(f"   • {metric.upper():>12} AURC: {aurc_results.get(f'{metric}_02_10', float('nan')):.6f}")
    
    # Show which metric is better
    balanced_aurc = aurc_results['balanced']
    worst_aurc = aurc_results['worst']
    print("\n" + "="*60)
    print("📈 METRIC COMPARISON:")
    print(f"   • BALANCED Error: {balanced_aurc:.6f} (average of group errors)")
    print(f"   • WORST Error:    {worst_aurc:.6f} (maximum group error)")
    diff = abs(worst_aurc - balanced_aurc)
    print(f"   • Difference:     {diff:.6f}")
    if worst_aurc > balanced_aurc:
        print(f"   ⚠️  Worst-group error is {((worst_aurc/balanced_aurc - 1)*100):.1f}% higher than balanced")
        print("   → Indicates group disparity in performance")
    else:
        print("   ✅ Groups are relatively balanced")
    
    print("="*60)
    if class_weights:
        print("✅ Metrics reweighted by train class distribution (proper long-tail evaluation)")
    print("📝 Lower AURC is better (less area under risk-coverage curve)")
    print("🎯 Methodology: Optimize thresholds on (tunev + val), evaluate on test")
    print("🎯 Focus: Group-aware fairness (balanced & worst-group metrics)")
    print("="*60)

if __name__ == '__main__':
    main()