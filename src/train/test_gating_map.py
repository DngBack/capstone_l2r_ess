"""
Test Script for Gating Network Implementation
==============================================

Verify các components hoạt động đúng:
1. Feature extraction
2. Gating network forward pass
3. Loss computation
4. Training loop (mini test)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from src.models.gating_network_map import (
    GatingNetwork,
    UncertaintyDisagreementFeatures,
    compute_uncertainty_for_map
)
from src.models.gating_losses import (
    GatingLoss,
    compute_gating_metrics
)


def test_uncertainty_features():
    """Test uncertainty/disagreement feature computation."""
    print("\n" + "="*70)
    print("TEST 1: Uncertainty & Disagreement Features")
    print("="*70)
    
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    
    computer = UncertaintyDisagreementFeatures(num_experts=E)
    features = computer.compute(posteriors)
    
    print(f"✓ Input shape: {posteriors.shape}")
    print(f"✓ Features computed:")
    for name, tensor in features.items():
        print(f"  - {name}: shape={tensor.shape}, range=[{tensor.min():.4f}, {tensor.max():.4f}]")
    
    # Sanity checks
    assert features['expert_entropy'].shape == (B, E)
    assert features['expert_confidence'].shape == (B, E)
    assert features['disagreement_ratio'].shape == (B,)
    assert features['mixture_entropy'].shape == (B,)
    
    print("✅ Uncertainty features test passed!")
    return True


def test_gating_network():
    """Test gating network forward pass."""
    print("\n" + "="*70)
    print("TEST 2: Gating Network Forward Pass")
    print("="*70)
    
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    
    # Dense routing
    print("\n2.1 Dense Routing:")
    gating_dense = GatingNetwork(
        num_experts=E,
        num_classes=C,
        hidden_dims=[128, 64],
        routing='dense'
    )
    
    weights_dense, aux_dense = gating_dense(posteriors)
    print(f"  ✓ Weights shape: {weights_dense.shape}")
    print(f"  ✓ Weights sum: {weights_dense.sum(dim=1).mean():.6f} (should be 1.0)")
    print(f"  ✓ Weights range: [{weights_dense.min():.4f}, {weights_dense.max():.4f}]")
    
    # Sanity check
    assert weights_dense.shape == (B, E)
    assert torch.allclose(weights_dense.sum(dim=1), torch.ones(B), atol=1e-5)
    
    # Top-K routing
    print("\n2.2 Top-K Routing (K=2):")
    gating_topk = GatingNetwork(
        num_experts=E,
        num_classes=C,
        hidden_dims=[128, 64],
        routing='top_k',
        top_k=2
    )
    
    weights_topk, aux_topk = gating_topk(posteriors)
    print(f"  ✓ Weights shape: {weights_topk.shape}")
    print(f"  ✓ Non-zero experts: {(weights_topk > 1e-6).sum(dim=1).float().mean():.2f} (should be ~2)")
    print(f"  ✓ Weights sum: {weights_topk.sum(dim=1).mean():.6f} (should be 1.0)")
    
    # Mixture posterior
    print("\n2.3 Mixture Posterior:")
    mixture = gating_dense.get_mixture_posterior(posteriors, weights_dense)
    print(f"  ✓ Shape: {mixture.shape}")
    print(f"  ✓ Sum: {mixture.sum(dim=1).mean():.6f} (should be 1.0)")
    
    assert mixture.shape == (B, C)
    assert torch.allclose(mixture.sum(dim=1), torch.ones(B), atol=1e-5)
    
    print("\n✅ Gating network test passed!")
    return True


def test_loss_functions():
    """Test loss computation."""
    print("\n" + "="*70)
    print("TEST 3: Loss Functions")
    print("="*70)
    
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    weights = F.softmax(torch.randn(B, E), dim=-1)
    targets = torch.randint(0, C, (B,))
    
    # Combined loss
    print("\n3.1 Combined Gating Loss:")
    loss_fn = GatingLoss(
        lambda_lb=1e-2,
        lambda_h=0.01,
        use_load_balancing=True,
        use_entropy_reg=True,
        top_k=1
    )
    
    loss, components = loss_fn(posteriors, weights, targets, return_components=True)
    print(f"  ✓ Total loss: {loss.item():.4f}")
    print(f"  ✓ Components:")
    for name, value in components.items():
        print(f"    - {name}: {value:.4f}")
    
    assert not torch.isnan(loss)
    assert loss.item() > 0
    
    # With sample weights
    print("\n3.2 With Sample Weights (long-tail):")
    sample_weights = torch.rand(C)  # class weights
    batch_weights = sample_weights[targets]
    
    loss_weighted = loss_fn(posteriors, weights, targets, sample_weights=batch_weights)
    print(f"  ✓ Weighted loss: {loss_weighted.item():.4f}")
    
    print("\n✅ Loss functions test passed!")
    return True


def test_metrics():
    """Test metric computation."""
    print("\n" + "="*70)
    print("TEST 4: Gating Metrics")
    print("="*70)
    
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    weights = F.softmax(torch.randn(B, E), dim=-1)
    targets = torch.randint(0, C, (B,))
    
    metrics = compute_gating_metrics(weights, posteriors, targets)
    
    print("✓ Computed metrics:")
    for name, value in metrics.items():
        print(f"  - {name}: {value:.4f}")
    
    # Sanity checks
    assert 'gating_entropy' in metrics
    assert 'mixture_acc' in metrics
    assert 'effective_experts' in metrics
    assert 0 <= metrics['mixture_acc'] <= 1
    assert 0 <= metrics['effective_experts'] <= E
    
    print("\n✅ Metrics test passed!")
    return True


def test_uncertainty_for_map():
    """Test U(x) computation for MAP margin."""
    print("\n" + "="*70)
    print("TEST 5: Uncertainty for MAP")
    print("="*70)
    
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    weights = F.softmax(torch.randn(B, E), dim=-1)
    
    # Default coefficients
    print("\n5.1 Default coefficients (a=b=d=1.0):")
    U = compute_uncertainty_for_map(posteriors, weights)
    print(f"  ✓ U shape: {U.shape}")
    print(f"  ✓ U range: [{U.min():.4f}, {U.max():.4f}]")
    print(f"  ✓ U mean: {U.mean():.4f}")
    
    assert U.shape == (B,)
    assert not torch.isnan(U).any()
    
    # Custom coefficients
    print("\n5.2 Custom coefficients (emphasize gating entropy):")
    U_custom = compute_uncertainty_for_map(
        posteriors, weights,
        coeffs={'a': 2.0, 'b': 0.5, 'd': 0.5}
    )
    print(f"  ✓ U range: [{U_custom.min():.4f}, {U_custom.max():.4f}]")
    
    print("\n✅ Uncertainty computation test passed!")
    return True


def test_gradient_flow():
    """Test gradient flow through network."""
    print("\n" + "="*70)
    print("TEST 6: Gradient Flow")
    print("="*70)
    
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    targets = torch.randint(0, C, (B,))
    
    # Create model and loss
    model = GatingNetwork(
        num_experts=E,
        num_classes=C,
        hidden_dims=[64],
        routing='dense'
    )
    
    loss_fn = GatingLoss(lambda_lb=1e-2, lambda_h=0.01)
    
    # Forward
    weights, aux = model(posteriors)
    loss = loss_fn(posteriors, weights, targets)
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = False
    max_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            max_grad = max(max_grad, grad_norm)
            print(f"  ✓ {name}: grad_norm={grad_norm:.6f}")
    
    assert has_grad, "No gradients computed!"
    assert max_grad > 0, "All gradients are zero!"
    assert not np.isnan(max_grad), "NaN in gradients!"
    
    print(f"\n  ✓ Max gradient norm: {max_grad:.6f}")
    print("\n✅ Gradient flow test passed!")
    return True


def test_with_real_logits():
    """Test with real expert logits if available."""
    print("\n" + "="*70)
    print("TEST 7: With Real Expert Logits (if available)")
    print("="*70)
    
    logits_dir = Path('./outputs/logits/cifar100_lt_if100')
    
    if not logits_dir.exists():
        print("⚠️  Logits directory not found, skipping this test")
        return True
    
    expert_names = ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline']
    split_name = 'val'
    
    # Try to load
    logits_list = []
    for expert_name in expert_names:
        logits_path = logits_dir / expert_name / f"{split_name}_logits.pt"
        if not logits_path.exists():
            print(f"⚠️  {logits_path} not found, skipping")
            return True
        logits_e = torch.load(logits_path, map_location='cpu').float()
        logits_list.append(logits_e)
    
    # Stack
    logits = torch.stack(logits_list, dim=0).transpose(0, 1)  # [N, E, C]
    posteriors = torch.softmax(logits, dim=-1)
    
    N, E, C = posteriors.shape
    print(f"✓ Loaded real logits: shape={posteriors.shape}")
    print(f"  - Experts: {E}")
    print(f"  - Classes: {C}")
    print(f"  - Samples: {N}")
    
    # Test gating
    print("\n7.1 Forward pass with real data:")
    model = GatingNetwork(num_experts=E, num_classes=C)
    
    # Process in batches
    batch_size = 128
    all_weights = []
    
    for i in range(0, N, batch_size):
        batch = posteriors[i:i+batch_size]
        weights, _ = model(batch)
        all_weights.append(weights)
    
    all_weights = torch.cat(all_weights, dim=0)
    print(f"  ✓ Processed {N} samples")
    print(f"  ✓ Mean gating entropy: {-(all_weights * torch.log(all_weights + 1e-8)).sum(dim=1).mean():.4f}")
    print(f"  ✓ Expert usage: {all_weights.mean(dim=0)}")
    
    print("\n✅ Real logits test passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 GATING NETWORK TEST SUITE")
    print("="*70)
    
    tests = [
        ("Uncertainty Features", test_uncertainty_features),
        ("Gating Network", test_gating_network),
        ("Loss Functions", test_loss_functions),
        ("Metrics", test_metrics),
        ("Uncertainty for MAP", test_uncertainty_for_map),
        ("Gradient Flow", test_gradient_flow),
        ("Real Logits", test_with_real_logits),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "✅ PASSED" if success else "❌ FAILED"))
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"❌ ERROR: {str(e)}"))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    for name, status in results:
        print(f"{status}: {name}")
    
    num_passed = sum(1 for _, status in results if "✅" in status)
    num_total = len(results)
    print(f"\nTotal: {num_passed}/{num_total} passed")
    
    if num_passed == num_total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed!")


if __name__ == '__main__':
    run_all_tests()
