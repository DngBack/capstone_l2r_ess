"""
Improved Gating Training - Addressing Mismatch Issues
====================================================

Key Fixes:
1. Inverse frequency weighting for head/tail groups
2. Responsibility loss (EM-style alignment)
3. Prior-based regularizer (group-aware)
4. Router temperature annealing
5. Better input normalization

Usage:
    python -m src.train.train_gating_improved --routing dense
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import argparse

from src.models.gating_network_map import GatingNetwork
from src.models.gating_losses import GatingLoss, compute_gating_metrics


# ============================================================================
# IMPROVED LOSS FUNCTIONS
# ============================================================================

def compute_group_inverse_weights(labels: torch.Tensor, group_boundaries: List[int]) -> torch.Tensor:
    """
    Compute inverse frequency weights per group (head/tail).
    
    Formula: w(x) = 1 / freq(G[y]) / E[1/freq(G[y])]
    
    Args:
        labels: [N] tensor
        group_boundaries: [69] means head=0-68, tail=69-99
    
    Returns:
        weights: [N] tensor
    """
    # Assign group for each sample
    groups = torch.zeros_like(labels)
    for i, boundary in enumerate(group_boundaries):
        groups[labels >= boundary] = i + 1
    
    # Count per group
    num_groups = len(group_boundaries) + 1
    group_counts = torch.bincount(groups.long(), minlength=num_groups).float()
    group_freqs = group_counts / group_counts.sum()
    
    # Inverse frequency weight per sample
    inv_freqs = 1.0 / (group_freqs + 1e-8)
    sample_weights = inv_freqs[groups.long()]
    
    # Normalize: divide by mean to keep scale ~1.0
    sample_weights = sample_weights / sample_weights.mean()
    
    return sample_weights


def compute_responsibility_loss(
    posteriors: torch.Tensor,  # [B, E, C]
    weights: torch.Tensor,      # [B, E]
    labels: torch.Tensor,       # [B]
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute EM-style responsibility loss.
    
    Formula: KL(r(x) || w(x)) where
    r_e(x) = w_e(x) * p_e(y|x) / sum_j w_j(x) * p_j(y|x)
    
    Args:
        posteriors: expert posteriors [B, E, C]
        weights: gating weights [B, E]
        labels: true labels [B]
        temperature: softmax temperature
    
    Returns:
        loss: scalar
    """
    B, E, C = posteriors.shape
    
    # Get expert probs for true class: p_e(y|x)
    expert_probs = posteriors[torch.arange(B), :, labels]  # [B, E]
    
    # Compute responsibility: r_e(x) = w_e(x) * p_e(y|x) / Z
    numerator = weights * expert_probs
    responsibility = numerator / (numerator.sum(dim=1, keepdim=True) + 1e-8)
    
    # Soft targets from responsibility
    target_weights = F.softmax(responsibility / temperature, dim=1)
    
    # KL divergence: KL(target_weights || current_weights)
    kl = (target_weights * torch.log(target_weights / (weights + 1e-8))).sum(dim=1).mean()
    
    return kl


def compute_prior_regularizer(
    weights: torch.Tensor,      # [B, E]
    labels: torch.Tensor,       # [B]
    group_boundaries: List[int],
    group_priors: torch.Tensor  # [G, E] - learned priors
) -> torch.Tensor:
    """
    Compute group-aware prior regularization.
    
    Formula: sum_k KL(E[w(x)|group_k] || pi_k)
    
    Args:
        weights: [B, E]
        labels: [B]
        group_boundaries: [69]
        group_priors: [G, E] - empirical priors from tune set
    
    Returns:
        loss: scalar
    """
    # Assign groups
    groups = torch.zeros_like(labels)
    for i, boundary in enumerate(group_boundaries):
        groups[labels >= boundary] = i + 1
    
    num_groups = len(group_boundaries) + 1
    total_kl = 0.0
    
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() == 0:
            continue
        
        # Mean weights for this group: E[w(x)|group_k]
        group_mean_weights = weights[mask].mean(dim=0)  # [E]
        
        # Prior for this group: pi_k
        prior = group_priors[g]  # [E]
        
        # KL divergence
        kl = (group_mean_weights * torch.log(group_mean_weights / (prior + 1e-8))).sum()
        total_kl += kl
    
    return total_kl / num_groups


def estimate_group_priors(
    posteriors: torch.Tensor,  # [N, E, C]
    labels: torch.Tensor,       # [N]
    group_boundaries: List[int]
) -> torch.Tensor:
    """
    Estimate group priors by finding which expert is most correct for each group.
    
    Args:
        posteriors: [N, E, C]
        labels: [N]
        group_boundaries: [69]
    
    Returns:
        priors: [G, E] - each row sums to 1
    """
    num_groups = len(group_boundaries) + 1
    num_experts = posteriors.shape[1]
    
    # Assign groups
    groups = torch.zeros_like(labels)
    for i, boundary in enumerate(group_boundaries):
        groups[labels >= boundary] = i + 1
    
    # For each group, find best expert
    priors = torch.zeros(num_groups, num_experts)
    
    for g in range(num_groups):
        mask = (groups == g)
        if mask.sum() == 0:
            # Uniform prior if no samples
            priors[g] = 1.0 / num_experts
            continue
        
        # Get expert predictions for this group
        expert_preds = posteriors[mask].argmax(dim=-1)  # [B, E]
        labels_g = labels[mask]  # [B]
        
        # Accuracy per expert for this group
        for e in range(num_experts):
            acc = (expert_preds[:, e] == labels_g).float().mean()
            priors[g, e] = acc
        
        # Softmax to make it a probability distribution
        priors[g] = F.softmax(priors[g] / 0.1, dim=0)
    
    return priors


def improved_gating_loss(
    posteriors: torch.Tensor,     # [B, E, C]
    weights: torch.Tensor,        # [B, E]
    labels: torch.Tensor,         # [B]
    config: Dict
) -> Tuple[torch.Tensor, Dict]:
    """
    Improved gating loss with:
    1. Mixture NLL
    2. Responsibility loss (EM-style)
    3. Prior regularizer (group-aware)
    4. Load-balancing
    5. Entropy reg
    
    Returns:
        loss: scalar
        components: dict of loss components
    """
    components = {}
    
    # 1. Mixture NLL
    mixture = (weights.unsqueeze(-1) * posteriors).sum(dim=1)
    nll = F.nll_loss(
        torch.log(mixture + 1e-8),
        labels,
        reduction='mean'
    )
    components['nll'] = nll
    total_loss = nll
    
    # 2. Responsibility loss (EM-style alignment)
    if config['lambda_resp'] > 0:
        resp_loss = compute_responsibility_loss(
            posteriors, weights, labels,
            temperature=config['router_temperature']
        )
        components['responsibility'] = resp_loss
        total_loss = total_loss + config['lambda_resp'] * resp_loss
    
    # 3. Load-balancing (only if sparse routing)
    if config['use_load_balancing']:
        lb_loss = -weights.sum(dim=0).var() / (weights.numel() + 1e-8)
        components['load_balancing'] = lb_loss
        total_loss = total_loss + config['lambda_lb'] * lb_loss
    
    # 4. Entropy regularization
    if config['lambda_entropy'] > 0:
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
        # Maximize entropy (diversity)
        entropy_loss = -entropy
        components['entropy'] = entropy_loss
        total_loss = total_loss + config['lambda_entropy'] * entropy_loss
    
    return total_loss, components


# ============================================================================
# TRAINING LOOP WITH IMPROVEMENTS
# ============================================================================

def train_one_epoch_improved(
    model: GatingNetwork,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Dict,
    epoch: int,
    group_priors: torch.Tensor
) -> Dict[str, float]:
    """Train one epoch with improved losses."""
    model.train()
    
    total_loss = 0.0
    loss_components = {'nll': 0.0, 'responsibility': 0.0, 
                      'load_balancing': 0.0, 'entropy': 0.0, 'prior': 0.0}
    
    all_weights = []
    all_posteriors = []
    all_targets = []
    
    # Anneal router temperature: start 2.0 → end 0.7
    router_temp = config['router_temp_start'] - \
                  (config['router_temp_start'] - config['router_temp_end']) * \
                  (epoch / config['epochs'])
    
    for batch_idx, (logits, targets) in enumerate(train_loader):
        # [B, E, C], [B]
        logits = logits.to(DEVICE)
        targets = targets.to(DEVICE)
        
        posteriors = torch.softmax(logits, dim=-1)
        
        # Forward
        weights, _ = model(posteriors)
        
        # Compute sample weights (group inverse frequency)
        sample_weights = compute_group_inverse_weights(targets, [69])
        
        # Main loss
        loss, components = improved_gating_loss(posteriors, weights, targets, {
            **config,
            'router_temperature': router_temp
        })
        
        # Prior regularizer (if available)
        if config['lambda_prior'] > 0 and group_priors is not None:
            prior_loss = compute_prior_regularizer(weights, targets, [69], group_priors)
            loss = loss + config['lambda_prior'] * prior_loss
            components['prior'] = prior_loss.item()
        
        # Weight by sample_weights (for long-tail)
        if 'use_sample_weights' in config and config['use_sample_weights']:
            loss = (loss * sample_weights).mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if config['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        for k, v in components.items():
            if k in loss_components:
                loss_components[k] += v
        
        all_weights.append(weights.detach())
        all_posteriors.append(posteriors.detach())
        all_targets.append(targets.detach())
    
    # Aggregated metrics
    num_batches = len(train_loader)
    metrics = {
        'loss': total_loss / num_batches,
        'router_temp': router_temp,
    }
    metrics.update({k: v / num_batches for k, v in loss_components.items()})
    
    # Gating metrics
    all_weights = torch.cat(all_weights, dim=0)
    all_posteriors = torch.cat(all_posteriors, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    gating_metrics = compute_gating_metrics(all_weights, all_posteriors, all_targets)
    metrics.update(gating_metrics)
    
    return metrics


@torch.no_grad()
def validate_improved(
    model: GatingNetwork,
    val_loader: DataLoader,
    config: Dict
) -> Dict[str, float]:
    """Validate with improved metrics."""
    model.eval()
    
    all_weights = []
    all_posteriors = []
    all_targets = []
    
    for logits, targets in val_loader:
        logits = logits.to(DEVICE)
        targets = targets.to(DEVICE)
        
        posteriors = torch.softmax(logits, dim=-1)
        weights, _ = model(posteriors)
        
        all_weights.append(weights)
        all_posteriors.append(posteriors)
        all_targets.append(targets)
    
    all_weights = torch.cat(all_weights, dim=0)
    all_posteriors = torch.cat(all_posteriors, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_gating_metrics(all_weights, all_posteriors, all_targets)
    
    # Group accuracies
    mixture_posterior = (all_weights.unsqueeze(-1) * all_posteriors).sum(dim=1)
    predictions = mixture_posterior.argmax(dim=-1)
    
    head_mask = all_targets < 69
    tail_mask = all_targets >= 69
    
    head_acc = (predictions[head_mask] == all_targets[head_mask]).float().mean().item()
    tail_acc = (predictions[tail_mask] == all_targets[tail_mask]).float().mean().item()
    balanced_acc = (head_acc + tail_acc) / 2
    
    metrics.update({
        'head_acc': head_acc,
        'tail_acc': tail_acc,
        'balanced_acc': balanced_acc
    })
    
    return metrics


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

# Import existing functions from train_gating_map.py
from src.train.train_gating_map import load_expert_logits, load_labels, create_dataloaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMPROVED_CONFIG = {
    'epochs': 100,
    'batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 5e-4,  # Stronger WD
    
    # Loss weights
    'lambda_resp': 0.1,      # Responsibility loss
    'lambda_prior': 0.05,    # Prior regularizer
    'lambda_lb': 1e-2,
    'lambda_entropy': 0.01,
    
    # Router temperature annealing
    'router_temp_start': 2.0,
    'router_temp_end': 0.7,
    
    # Training config
    'use_sample_weights': True,  # Group inverse frequency
    'use_load_balancing': False,  # Disable for dense
    'grad_clip': 1.0,
    
    # Other
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'val_interval': 5,
}


def main():
    parser = argparse.ArgumentParser(description='Train Improved Gating')
    parser.add_argument('--routing', type=str, default='dense')
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    # Update config
    IMPROVED_CONFIG['epochs'] = args.epochs
    IMPROVED_CONFIG['routing'] = args.routing
    
    print("="*70)
    print("IMPROVED GATING TRAINING")
    print("="*70)
    print("Key improvements:")
    print("  - Group inverse frequency weighting")
    print("  - Responsibility loss (EM-style)")
    print("  - Prior-based regularizer")
    print("  - Router temperature annealing")
    print("="*70)
    
    # Create dataloaders (reuse existing function)
    # You'll need to implement or import this
    # train_loader, val_loader = create_dataloaders(...)
    
    print("\nTo use this improved training:")
    print("1. Update train_gating_map.py with these loss functions")
    print("2. Add responsibility/prior terms to GatingLoss")
    print("3. Use group_inverse_weights for sample weighting")
    print("4. Enable router temperature annealing")


if __name__ == '__main__':
    main()

