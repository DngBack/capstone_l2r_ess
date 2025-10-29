"""
Test Rejector Formula Verification
=================================

Kiem tra rejector formula mapping (alpha, mu) <-> (lambda*, nu*) theo paper Theorem 1.

Paper Formula (Theorem 1, Equation 8):
    h*(x) = argmax_y (1/lambda*[y] * eta_y(x))
    r*(x) = 1 <-> max_y (1/lambda*[y] * eta_y(x)) < sum_y' (1/lambda*[y'] - nu*[y']) eta_y'(x) - c

Current Implementation:
    h(x) = argmax_y (alpha[y] * eta_y(x))
    r(x) = 1 <-> max_y (alpha[y] * eta_y(x)) < sum_y' ((alpha[y'] - mu[y']) * eta_y'(x)) - c

Verification:
    If alpha = 1/lambda*, then we need: mu[y'] = nu*[y']
    Check if this mapping is correct.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from src.models.ltr_plugin import LtRPlugin, LtRPluginConfig


def create_toy_mixture_posterior(num_samples=1000, num_classes=10, num_groups=2):
    """Create toy mixture posterior with known ground truth."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create bimodal distribution (head and tail)
    # Head classes: 0-4, Tail classes: 5-9
    
    posteriors = []
    true_labels = []
    
    for i in range(num_samples):
        # Determine if sample is from head or tail
        is_head = np.random.rand() < 0.7  # 70% head samples
        
        if is_head:
            true_class = np.random.randint(0, 5)  # Head: 0-4
        else:
            true_class = np.random.randint(5, 10)  # Tail: 5-9
        
        # Create posterior: high confidence in true class
        posterior = np.random.dirichlet([1.0] * num_classes)
        posterior[true_class] += 2.0  # Boost true class
        posterior = posterior / posterior.sum()
        
        posteriors.append(posterior)
        true_labels.append(true_class)
    
    return torch.tensor(posteriors, dtype=torch.float32), torch.tensor(true_labels, dtype=torch.long)


def test_rejector_formula():
    """Test rejector formula correctness."""
    print("=" * 80)
    print("REJECTOR FORMULA VERIFICATION")
    print("=" * 80)
    
    # Create toy data
    print("\n1. Creating toy data...")
    mixture_posterior, true_labels = create_toy_mixture_posterior(
        num_samples=1000, num_classes=10, num_groups=2
    )
    
    # Group mapping: head=0 (0-4), tail=1 (5-9)
    group_boundaries = [5]
    class_to_group = torch.zeros(10, dtype=torch.long)
    class_to_group[5:] = 1
    
    print(f"   Shape: {mixture_posterior.shape}")
    print(f"   True labels: {true_labels.min().item()}-{true_labels.max().item()}")
    print(f"   Head samples: {(true_labels < 5).sum().item()}")
    print(f"   Tail samples: {(true_labels >= 5).sum().item()}")
    
    # Create LtR plugin with group-based params
    print("\n2. Creating LtR plugin...")
    config = LtRPluginConfig(
        num_classes=10,
        num_groups=2,
        group_boundaries=group_boundaries,
        param_mode='group',
        alpha_grid=[0.5, 1.0, 2.0],
        mu_grid=[0.0, 0.5, 1.0],
        cost_grid=[0.5],
        objective='balanced'
    )
    
    plugin = LtRPlugin(config)
    
    # Test different (alpha, mu, c) combinations
    print("\n3. Testing different parameter combinations...")
    print("-" * 80)
    
    test_configs = [
        {'alpha': [1.0, 1.0], 'mu': [0.0, 0.0], 'cost': 0.0, 'name': 'Baseline (neutral)'},
        {'alpha': [1.5, 0.5], 'mu': [0.0, 0.0], 'cost': 0.0, 'name': 'Head boost, tail penalty'},
        {'alpha': [1.0, 1.0], 'mu': [0.5, 0.5], 'cost': 0.0, 'name': 'Uniform mu shift'},
        {'alpha': [1.5, 0.5], 'mu': [0.5, 0.5], 'cost': 0.5, 'name': 'Combined + cost'},
    ]
    
    results = []
    
    for cfg in test_configs:
        # Set parameters
        alpha_tensor = torch.tensor(cfg['alpha'], dtype=torch.float32)
        mu_tensor = torch.tensor(cfg['mu'], dtype=torch.float32)
        plugin.set_parameters(alpha=alpha_tensor, mu=mu_tensor, cost=cfg['cost'])
        
        # Predict
        with torch.no_grad():
            predictions = plugin.predict_class(mixture_posterior)
            reject = plugin.predict_reject(mixture_posterior)
        
        # Compute metrics
        accept = ~reject
        if accept.sum() > 0:
            accuracy = (predictions[accept] == true_labels[accept]).float().mean().item()
            coverage = accept.float().mean().item()
            rejection_rate = 1.0 - coverage
        else:
            accuracy = 0.0
            coverage = 0.0
            rejection_rate = 1.0
        
        # Group-level metrics
        head_mask = true_labels < 5
        tail_mask = true_labels >= 5
        
        head_accuracy = 0.0
        tail_accuracy = 0.0
        head_coverage = 0.0
        tail_coverage = 0.0
        
        if accept.sum() > 0:
            if (accept & head_mask).sum() > 0:
                head_accuracy = (
                    (predictions[accept & head_mask] == true_labels[accept & head_mask]).float().mean().item()
                )
                head_coverage = (accept & head_mask).float().sum() / head_mask.sum().float()
            
            if (accept & tail_mask).sum() > 0:
                tail_accuracy = (
                    (predictions[accept & tail_mask] == true_labels[accept & tail_mask]).float().mean().item()
                )
                tail_coverage = (accept & tail_mask).float().sum() / tail_mask.sum().float()
        
        results.append({
            'name': cfg['name'],
            'alpha': cfg['alpha'],
            'mu': cfg['mu'],
            'cost': cfg['cost'],
            'accuracy': accuracy,
            'coverage': coverage,
            'rejection_rate': rejection_rate,
            'head_accuracy': head_accuracy,
            'tail_accuracy': tail_accuracy,
            'head_coverage': head_coverage,
            'tail_coverage': tail_coverage,
        })
        
        print(f"\n[TEST] {cfg['name']}:")
        print(f"   alpha = {cfg['alpha']}, mu = {cfg['mu']}, c = {cfg['cost']}")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Coverage: {coverage:.3f} (Rejection: {rejection_rate:.3f})")
        print(f"   Head: acc={head_accuracy:.3f}, cov={head_coverage:.3f}")
        print(f"   Tail: acc={tail_accuracy:.3f}, cov={tail_coverage:.3f}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Name':<30} {'alpha':<15} {'mu':<15} {'Acc':<8} {'Cov':<8} {'Head Acc':<10} {'Tail Acc':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<30} {str(r['alpha']):<15} {str(r['mu']):<15} "
              f"{r['accuracy']:<8.3f} {r['coverage']:<8.3f} "
              f"{r['head_accuracy']:<10.3f} {r['tail_accuracy']:<10.3f}")
    
    # Critical check: Formula consistency
    print("\n" + "=" * 80)
    print("FORMULA CONSISTENCY CHECK")
    print("=" * 80)
    
    print("\n[CHECK] Test 1: Classifier formula (h(x) = argmax_y (alpha[y] * eta[y]))")
    
    # Get alpha values
    alpha = plugin.get_alpha()
    
    # Manual computation
    reweighted_manual = mixture_posterior * alpha.unsqueeze(0)
    predictions_manual = reweighted_manual.argmax(dim=-1)
    predictions_plugin = plugin.predict_class(mixture_posterior)
    
    match_rate = (predictions_manual == predictions_plugin).float().mean()
    
    print(f"   Match rate: {match_rate:.4f} (should be 1.0)")
    assert match_rate > 0.999, "Classifier formula mismatch!"
    print("   PASS")
    
    print("\n[CHECK] Test 2: Rejector formula (r(x) vs manual computation)")
    
    # Get parameters
    alpha = plugin.get_alpha()
    mu = plugin.get_mu()
    cost = plugin.cost.item()
    
    # Manual rejector computation
    # Left side: max_y (alpha[y] * eta[y])
    reweighted = mixture_posterior * alpha.unsqueeze(0)
    left_side = reweighted.max(dim=-1)[0]
    
    # Right side: sum_y' ((alpha[y'] - mu[y']) * eta[y']) - c
    threshold_coeff = (alpha - mu).unsqueeze(0)
    right_side = (threshold_coeff * mixture_posterior).sum(dim=-1) - cost
    
    # Rejection decision
    reject_manual = left_side < right_side
    
    # Plugin prediction
    with torch.no_grad():
        reject_plugin = plugin.predict_reject(mixture_posterior)
    
    match_rate = (reject_manual == reject_plugin).float().mean()
    
    print(f"   Match rate: {match_rate:.4f} (should be 1.0)")
    assert match_rate > 0.999, "Rejector formula mismatch!"
    print("   PASS")
    
    print("\n[CHECK] Test 3: Parameter sensitivity")
    
    # Test effect of alpha variation
    print("\n   Testing alpha effect:")
    for alpha_head in [0.5, 1.0, 1.5, 2.0]:
        alpha_test = torch.tensor([alpha_head, 1.0], dtype=torch.float32)
        mu_test = torch.tensor([0.0, 0.0], dtype=torch.float32)
        plugin.set_parameters(alpha=alpha_test, mu=mu_test, cost=0.0)
        
        with torch.no_grad():
            predictions_test = plugin.predict_class(mixture_posterior)
            reject_test = plugin.predict_reject(mixture_posterior)
        
        head_acc = (predictions_test[~reject_test & (true_labels < 5)] == 
                   true_labels[~reject_test & (true_labels < 5)]).float().mean().item()
        tail_acc = (predictions_test[~reject_test & (true_labels >= 5)] == 
                   true_labels[~reject_test & (true_labels >= 5)]).float().mean().item()
        rejection_rate = reject_test.float().mean().item()
        
        print(f"      alpha_head={alpha_head:.1f}: head_acc={head_acc:.3f}, "
              f"tail_acc={tail_acc:.3f}, rejection={rejection_rate:.3f}")
    
    # Test effect of mu variation
    print("\n   Testing mu effect:")
    for mu_val in [0.0, 0.25, 0.5, 1.0]:
        alpha_test = torch.tensor([1.0, 1.0], dtype=torch.float32)
        mu_test = torch.tensor([mu_val, mu_val], dtype=torch.float32)
        plugin.set_parameters(alpha=alpha_test, mu=mu_test, cost=0.5)
        
        with torch.no_grad():
            predictions_test = plugin.predict_class(mixture_posterior)
            reject_test = plugin.predict_reject(mixture_posterior)
        
        rejection_rate = reject_test.float().mean().item()
        coverage = (~reject_test).float().mean().item()
        
        print(f"      mu={mu_val:.2f}: rejection={rejection_rate:.3f}, "
              f"coverage={coverage:.3f}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("[PASS] All formula consistency checks passed!")
    print("\nConclusion:")
    print("  - Current implementation uses (alpha, mu) parameters")
    print("  - Classifier: h(x) = argmax_y (alpha[y] * eta[y])")
    print("  - Rejector: r(x) = 1 if max_y (alpha[y] * eta[y]) < sum_y' ((alpha[y'] - mu[y']) * eta[y']) - c")
    print("  - This is a reparameterization of paper formula with alpha = 1/lambda* and mu = nu*")
    print("  - Formula is mathematically consistent")


if __name__ == "__main__":
    test_rejector_formula()

