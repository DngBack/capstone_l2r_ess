"""
Visualize Gating Network Outputs
Plot distribution of mixture posteriors across 100 labels
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load gating network
print("=" * 70)
print("VISUALIZING GATING NETWORK OUTPUTS")
print("=" * 70)
print("\nLoading gating network...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from src.models.gating_network_map import GatingNetwork

num_experts = 3
num_classes = 100

gating = GatingNetwork(
    num_experts=num_experts, 
    num_classes=num_classes, 
    routing="dense"
).to(DEVICE)

checkpoint_path = "./checkpoints/gating_map/cifar100_lt_if100/final_gating.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
gating.load_state_dict(checkpoint["model_state_dict"])
gating.eval()

print("SUCCESS: Gating loaded")

# Load test data
print("\nLoading test data...")
from train_ltr_plugin import load_expert_logits, load_labels

expert_names = ["ce_baseline", "logitadjust_baseline", "balsoftmax_baseline"]
logits_dir = "./outputs/logits/cifar100_lt_if100/"
splits_dir = "./data/cifar100_lt_if100_splits_fixed"

expert_logits_test = load_expert_logits(expert_names, logits_dir, "test", DEVICE)
labels_test = load_labels(splits_dir, "test", DEVICE)

print(f"SUCCESS: Test data loaded - {len(labels_test)} samples")

# Get gating outputs for some samples
print("\nComputing mixture posteriors...")
import torch.nn.functional as F

# Sample a few examples
num_samples_to_plot = 10
sample_indices = torch.randperm(len(labels_test))[:num_samples_to_plot]

with torch.no_grad():
    # Expert posteriors
    expert_posteriors = F.softmax(expert_logits_test, dim=-1)
    
    # Gating weights
    gating_output = gating(expert_posteriors)
    if isinstance(gating_output, tuple):
        gating_weights = gating_output[0]
    else:
        gating_weights = gating_output
    
    # Mixture posterior
    mixture_posterior = (gating_weights.unsqueeze(-1) * expert_posteriors).sum(dim=1)
    
    print("SUCCESS: Mixture posteriors computed")

# Plot samples
print(f"\nPlotting {num_samples_to_plot} samples...")

fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(16, 2*num_samples_to_plot))
if num_samples_to_plot == 1:
    axes = [axes]

for i, idx in enumerate(sample_indices):
    ax = axes[i]
    
    # Mixture posterior for this sample
    posterior = mixture_posterior[idx].cpu().numpy()
    true_label = labels_test[idx].item()
    
    # Plot all 100 classes
    colors = ['#2E86AB' if c != true_label else '#A23B72' for c in range(100)]
    bars = ax.bar(range(100), posterior, alpha=0.7, edgecolor='black', linewidth=0.5, color=colors)
    
    # Highlight true label (darker)
    ax.bar(true_label, posterior[true_label], color='#F18F01', edgecolor='red', linewidth=3)
    
    # Highlight max
    pred_label = posterior.argmax()
    if pred_label != true_label:
        ax.bar(pred_label, posterior[pred_label], color='green', edgecolor='darkgreen', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Probability')
    ax.set_title(f'Sample {idx}: True={true_label}, Pred={pred_label}, Conf={posterior.max():.3f}')
    ax.set_xlim([0, 99])
    ax.set_ylim([0, max(posterior.max() * 1.2, 0.05)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A23B72', edgecolor='red', label='True Label'),
        Patch(facecolor='green', label='Predicted Label'),
        Patch(facecolor='#2E86AB', label='Other Classes')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
output_dir = Path("./gating_analysis")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'gating_mixture_posteriors_samples.png', dpi=150, bbox_inches='tight')
print(f"SUCCESS: Saved plot to {output_dir / 'gating_mixture_posteriors_samples.png'}")

# Plot distribution statistics
print("\nComputing distribution statistics...")

# Mean posterior per class (across all test samples)
mean_posterior_per_class = mixture_posterior.mean(dim=0).cpu().numpy()

# Plot mean distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Mean posterior per class (full 100 classes)
ax = axes[0, 0]
ax.bar(range(100), mean_posterior_per_class, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Class ID')
ax.set_ylabel('Mean Probability')
ax.set_title('Mean Mixture Posterior per Class (All 100 Classes)')
ax.set_ylim([0, mean_posterior_per_class.max() * 1.1])
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(69.5, color='red', linestyle='--', linewidth=2, label='Head/Tail boundary')
ax.legend()

# 2. Head vs Tail mean
ax = axes[0, 1]
head_mean = mean_posterior_per_class[:70].mean()
tail_mean = mean_posterior_per_class[70:].mean()
bars = ax.bar(['Head (0-69)', 'Tail (70-99)'], [head_mean, tail_mean], 
       color=['skyblue', 'salmon'], edgecolor='black', linewidth=2)
ax.set_ylabel('Mean Probability')
ax.set_title('Mean Mixture Posterior: Head vs Tail')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, [head_mean, tail_mean]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold')

# 3. Distribution of max probabilities
ax = axes[1, 0]
max_probs = mixture_posterior.max(dim=1)[0].cpu().numpy()
ax.hist(max_probs, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
ax.set_xlabel('Max Probability (Confidence)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Max Probabilities (Confidence Scores)')
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean={max_probs.mean():.3f}')
ax.legend()

# 4. Entropy distribution
ax = axes[1, 1]
entropies = -(mixture_posterior * torch.log(mixture_posterior + 1e-8)).sum(dim=1).cpu().numpy()
ax.hist(entropies, bins=50, alpha=0.7, edgecolor='black', color='green')
ax.set_xlabel('Entropy')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Mixture Entropy (Uncertainty)')
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(entropies.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean={entropies.mean():.3f}')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'gating_mixture_statistics.png', dpi=150, bbox_inches='tight')
print(f"SUCCESS: Saved statistics to {output_dir / 'gating_mixture_statistics.png'}")

# Print statistics
print("\n" + "="*70)
print("STATISTICS:")
print("="*70)
print(f"Total test samples: {len(labels_test)}")
print(f"\nMean posterior per class range: [{mean_posterior_per_class.min():.6f}, {mean_posterior_per_class.max():.6f}]")
print(f"Head (0-69) mean: {head_mean:.6f}")
print(f"Tail (70-99) mean: {tail_mean:.6f}")
print(f"Ratio tail/head: {tail_mean/head_mean:.3f}")
print(f"\nConfidence (max prob): mean={max_probs.mean():.4f}, std={max_probs.std():.4f}")
print(f"Entropy: mean={entropies.mean():.4f}, std={entropies.std():.4f}")
print("="*70)

# Plot per-class distribution
print("\nPlotting per-class distribution...")
fig, ax = plt.subplots(1, 1, figsize=(16, 6))

# Plot mean posterior with color coding for head/tail
colors = ['#FF6B6B' if i < 70 else '#4ECDC4' for i in range(100)]
bars = ax.bar(range(100), mean_posterior_per_class, edgecolor='black', linewidth=0.5, color=colors)
ax.set_xlabel('Class ID')
ax.set_ylabel('Mean Mixture Posterior Probability')
ax.set_title('Mean Mixture Posterior per Class - Head vs Tail Distribution')
ax.axvline(69.5, color='red', linestyle='--', linewidth=2, label='Head/Tail boundary')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# Add annotations
head_max_idx = mean_posterior_per_class[:70].argmax()
tail_max_idx = mean_posterior_per_class[70:].argmax() + 70
ax.annotate(f'Max Head: Class {head_max_idx}', xy=(head_max_idx, mean_posterior_per_class[head_max_idx]),
            xytext=(head_max_idx, mean_posterior_per_class[head_max_idx] + 0.002),
            arrowprops=dict(arrowstyle='->', color='orange'), fontweight='bold')
ax.annotate(f'Max Tail: Class {tail_max_idx}', xy=(tail_max_idx, mean_posterior_per_class[tail_max_idx]),
            xytext=(tail_max_idx, mean_posterior_per_class[tail_max_idx] + 0.002),
            arrowprops=dict(arrowstyle='->', color='teal'), fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'gating_mean_per_class.png', dpi=150, bbox_inches='tight')
print(f"SUCCESS: Saved per-class distribution to {output_dir / 'gating_mean_per_class.png'}")

print("\nAll visualizations saved to ./gating_analysis/")
plt.close()

