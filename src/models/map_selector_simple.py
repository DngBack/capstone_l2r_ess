"""
Simplified MAP Selector - Confidence-Based Rejection
=====================================================

Uses simple confidence thresholding instead of complex L2R margin:
    margin(x) = max_y(η̃[y]) - threshold - γ·U(x)
    accept if margin ≥ 0

Much more stable and interpretable than full L2R formula.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SimpleMAPConfig:
    """Configuration for simplified MAP selector."""
    num_classes: int = 100
    num_groups: int = 2
    group_boundaries: list = None
    
    # Grid search ranges
    threshold_grid: list = None  # Confidence thresholds
    gamma_grid: list = None      # Uncertainty penalty weights
    
    # Optimization
    objective: str = 'balanced'  # 'balanced' or 'worst'
    rejection_cost: float = 0.0  # c in R(θ,γ;c) = error + c·ρ
    
    def __post_init__(self):
        if self.group_boundaries is None:
            self.group_boundaries = [50]
        
        if self.threshold_grid is None:
            # Confidence thresholds: 0.1 to 0.9
            self.threshold_grid = list(np.linspace(0.1, 0.9, 17))
        
        if self.gamma_grid is None:
            # Uncertainty penalties: 0.0 to 2.0
            self.gamma_grid = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]


class SimpleMAPSelector(nn.Module):
    """
    Simplified MAP selector using confidence thresholding.
    
    Decision rule (Simple version - stable):
        margin(x) = max_y η̃[y] - θ - γ·U(x)
        accept if margin(x) ≥ 0
    
    Note: Full L2R formula (more general but requires α,μ optimization):
        h_α(x) = argmax_y (α[y] · η̃[y])
        r(x) = 1 if max_y (α[y]·η̃[y]) < Σ_y' ((1/α[y']) - μ[y']) · η̃[y'] - c - γ·U(x)
    
    For now, we use Simple version with (θ, γ) instead of (α, μ, γ).
    """
    
    def __init__(self, config: SimpleMAPConfig):
        super().__init__()
        self.config = config
        
        # Parameters (to be optimized)
        self.register_buffer('threshold', torch.tensor(0.5))
        self.register_buffer('gamma', torch.tensor(0.5))
        
        # Class-to-group mapping
        class_to_group = torch.zeros(config.num_classes, dtype=torch.long)
        for g_id, boundary in enumerate(config.group_boundaries):
            class_to_group[boundary:] = g_id + 1
        self.register_buffer('class_to_group', class_to_group)
    
    def set_parameters(self, threshold: float, gamma: float):
        """Set optimized parameters."""
        self.threshold = torch.tensor(threshold, device=self.threshold.device)
        self.gamma = torch.tensor(gamma, device=self.gamma.device)
    
    def compute_confidence(
        self,
        mixture_posterior: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute confidence as max posterior.
        
        Args:
            mixture_posterior: [B, C]
        
        Returns:
            confidence: [B]
        """
        return mixture_posterior.max(dim=-1)[0]
    
    def compute_margin(
        self,
        mixture_posterior: torch.Tensor,
        uncertainty: torch.Tensor,
        threshold: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute margin for rejection decision.
        
        margin(x) = confidence(x) - threshold - γ·U(x)
        
        Args:
            mixture_posterior: [B, C]
            uncertainty: [B]
            threshold: confidence threshold (optional)
            gamma: uncertainty penalty (optional)
        
        Returns:
            margin: [B]
        """
        if threshold is None:
            threshold = self.threshold.item()
        if gamma is None:
            gamma = self.gamma.item()
        
        # Confidence
        confidence = self.compute_confidence(mixture_posterior)
        
        # Adjusted margin
        margin = confidence - threshold - gamma * uncertainty
        
        return margin
    
    def predict_class(
        self,
        mixture_posterior: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict class (argmax on mixture posterior).
        
        Args:
            mixture_posterior: [B, C]
        
        Returns:
            predictions: [B]
        """
        return mixture_posterior.argmax(dim=-1)
    
    def predict_reject(
        self,
        mixture_posterior: torch.Tensor,
        uncertainty: torch.Tensor,
        threshold: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> torch.Tensor:
        """
        Predict rejection decision.
        
        Args:
            mixture_posterior: [B, C]
            uncertainty: [B]
            threshold: optional override
            gamma: optional override
        
        Returns:
            reject: [B] boolean (True = reject)
        """
        margin = self.compute_margin(
            mixture_posterior, uncertainty,
            threshold, gamma
        )
        
        reject = margin < 0  # Reject if margin negative
        
        return reject
    
    def forward(
        self,
        mixture_posterior: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass.
        
        Returns:
            dict with 'predictions', 'reject', 'margin', 'confidence'
        """
        predictions = self.predict_class(mixture_posterior)
        reject = self.predict_reject(mixture_posterior, uncertainty)
        margin = self.compute_margin(mixture_posterior, uncertainty)
        confidence = self.compute_confidence(mixture_posterior)
        
        return {
            'predictions': predictions,
            'reject': reject,
            'margin': margin,
            'confidence': confidence
        }


# ============================================================================
# METRICS
# ============================================================================

def compute_selective_metrics(
    predictions: torch.Tensor,
    reject: torch.Tensor,
    labels: torch.Tensor,
    class_to_group: torch.Tensor,
    num_groups: int,
    sample_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute selective classification metrics.
    
    Args:
        predictions: [N]
        reject: [N] boolean
        labels: [N]
        class_to_group: [C]
        num_groups: K
        sample_weights: [N] optional
    
    Returns:
        metrics dict
    """
    device = predictions.device
    N = predictions.shape[0]
    
    # Accepted samples
    accept = ~reject
    
    # Sample weights
    if sample_weights is None:
        sample_weights = torch.ones(N, device=device)
    
    total_weight = sample_weights.sum()
    
    # Coverage
    coverage = (accept.float() * sample_weights).sum() / total_weight
    
    # Correctness
    correct = (predictions == labels).float()
    
    # Selective error (on accepted samples)
    if accept.sum() > 0:
        errors_on_accepted = (accept & ~correct.bool()).float() * sample_weights
        selective_error = errors_on_accepted.sum() / (accept.float() * sample_weights).sum()
    else:
        # No samples accepted → worst possible error
        selective_error = torch.tensor(1.0, device=device)
    
    # Group-wise metrics
    group_errors = torch.zeros(num_groups, device=device)
    group_coverage = torch.zeros(num_groups, device=device)
    
    # Ensure class_to_group on same device
    class_to_group = class_to_group.to(device)
    label_groups = class_to_group[labels]
    
    for k in range(num_groups):
        group_mask = (label_groups == k)
        group_weights = sample_weights[group_mask]
        group_total = group_weights.sum()
        
        if group_total > 0:
            # Coverage
            group_accept = accept[group_mask].float() * group_weights
            group_coverage[k] = group_accept.sum() / group_total
            
            # Error (on accepted in this group)
            group_correct = correct[group_mask]
            group_errors_accepted = (accept[group_mask] & ~group_correct.bool()).float() * group_weights
            
            if group_accept.sum() > 0:
                group_errors[k] = group_errors_accepted.sum() / group_accept.sum()
            else:
                group_errors[k] = 1.0  # No accepted → worst error
    
    metrics = {
        'coverage': coverage.item(),
        'selective_error': selective_error.item(),
        'group_errors': group_errors.cpu().numpy(),
        'group_coverage': group_coverage.cpu().numpy(),
        'rejection_rate': 1.0 - coverage.item()
    }
    
    return metrics


# ============================================================================
# GRID SEARCH
# ============================================================================

@dataclass
class GridSearchResult:
    """Result from grid search."""
    threshold: float
    gamma: float
    selective_error: float
    coverage: float
    group_errors: np.ndarray
    worst_group_error: float


class SimpleGridSearchOptimizer:
    """
    Grid search over (threshold, γ) for simplified MAP.
    """
    
    def __init__(self, config: SimpleMAPConfig):
        self.config = config
    
    def compute_objective(
        self,
        selector: SimpleMAPSelector,
        posteriors: torch.Tensor,
        uncertainty: torch.Tensor,
        labels: torch.Tensor,
        threshold: float,
        gamma: float,
        beta: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[float, Dict]:
        """
        Compute objective for given parameters.
        
        Returns:
            (score, metrics)
        """
        # Predict reject
        reject = selector.predict_reject(
            posteriors, uncertainty,
            threshold=threshold, gamma=gamma
        )
        
        # Predictions
        predictions = selector.predict_class(posteriors)
        
        # Metrics
        metrics = compute_selective_metrics(
            predictions, reject, labels,
            selector.class_to_group,
            self.config.num_groups,
            sample_weights
        )
        
        # Objective with rejection cost
        # General form: R(θ,γ;c) = error_term + c·ρ
        rejection_rate = metrics['rejection_rate']
        cost_term = self.config.rejection_cost * rejection_rate
        
        if self.config.objective == 'balanced':
            # ĐÚNG: Trung bình theo NHÓM (không phải overall error)
            # R_bal = (1/K) * Σ_k e_k + c·ρ
            errors_per_group = torch.tensor(metrics['group_errors'], device=posteriors.device)
            error_term = errors_per_group.mean().item()  # Mean của group errors
            score = error_term + cost_term
        elif self.config.objective == 'worst':
            errors_per_group = torch.tensor(metrics['group_errors'], device=posteriors.device)
            
            if beta is not None:
                # Ensure beta on same device
                beta = beta.to(posteriors.device)
                # Weighted by beta (for EG-outer): Σ_k β_k·e_k + c·ρ
                error_term = (beta * errors_per_group).sum().item()
            else:
                # Pure worst-group: R_max = max_k e_k + c·ρ
                error_term = errors_per_group.max().item()
            
            score = error_term + cost_term
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")
        
        return score, metrics
    
    def search(
        self,
        selector: SimpleMAPSelector,
        posteriors: torch.Tensor,
        uncertainty: torch.Tensor,
        labels: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> GridSearchResult:
        """
        Grid search to find best (threshold, γ).
        
        Args:
            selector: SimpleMAPSelector
            posteriors: [N, C]
            uncertainty: [N]
            labels: [N]
            beta: [G] optional weights for worst-group
            sample_weights: [N] optional
            verbose: print progress
        
        Returns:
            GridSearchResult with best parameters
        """
        threshold_grid = self.config.threshold_grid
        gamma_grid = self.config.gamma_grid
        
        best_score = float('inf')
        best_result = None
        
        total = len(threshold_grid) * len(gamma_grid)
        count = 0
        
        if verbose:
            print(f"\nGrid search: {len(threshold_grid)} thresholds × {len(gamma_grid)} gammas = {total} combinations")
        
        for threshold in threshold_grid:
            for gamma in gamma_grid:
                count += 1
                
                # Evaluate
                score, metrics = self.compute_objective(
                    selector, posteriors, uncertainty, labels,
                    threshold, gamma, beta, sample_weights
                )
                
                # Update best
                if score < best_score:
                    best_score = score
                    best_result = GridSearchResult(
                        threshold=threshold,
                        gamma=gamma,
                        selective_error=metrics['selective_error'],
                        coverage=metrics['coverage'],
                        group_errors=metrics['group_errors'],
                        worst_group_error=metrics['group_errors'].max()
                    )
                    
                    if verbose:
                        print(f"  [{count}/{total}] New best: threshold={threshold:.3f}, γ={gamma:.3f}, "
                              f"error={score:.4f}, coverage={metrics['coverage']:.3f}")
        
        if verbose:
            print(f"\n✅ Best found: threshold={best_result.threshold:.3f}, γ={best_result.gamma:.3f}")
            print(f"   Selective error: {best_result.selective_error:.4f}")
            print(f"   Coverage: {best_result.coverage:.3f}")
            print(f"   Group errors: {best_result.group_errors}")
        
        return best_result


# ============================================================================
# RC CURVE
# ============================================================================

class RCCurveComputer:
    """Compute Risk-Coverage curves."""
    
    def __init__(self, config: SimpleMAPConfig):
        self.config = config
    
    def compute_rc_curve(
        self,
        selector: SimpleMAPSelector,
        posteriors: torch.Tensor,
        uncertainty: torch.Tensor,
        labels: torch.Tensor,
        gamma: float,
        threshold_grid: Optional[np.ndarray] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute RC curve by sweeping threshold.
        
        Args:
            selector: SimpleMAPSelector
            posteriors: [N, C]
            uncertainty: [N]
            labels: [N]
            gamma: fixed uncertainty penalty
            threshold_grid: thresholds to sweep
            sample_weights: [N] optional
        
        Returns:
            dict with rc curve data
        """
        if threshold_grid is None:
            threshold_grid = np.linspace(0.0, 1.0, 100)
        
        rejection_rates = []
        selective_errors = []
        group_errors_list = []
        
        predictions = selector.predict_class(posteriors)
        
        for threshold in threshold_grid:
            # Reject with this threshold
            reject = selector.predict_reject(
                posteriors, uncertainty,
                threshold=threshold, gamma=gamma
            )
            
            # Metrics
            metrics = compute_selective_metrics(
                predictions, reject, labels,
                selector.class_to_group,
                self.config.num_groups,
                sample_weights
            )
            
            rejection_rates.append(metrics['rejection_rate'])
            
            # Compute error based on objective
            if self.config.objective == 'balanced':
                # Balanced: mean of group errors
                error = np.mean(metrics['group_errors'])
            elif self.config.objective == 'worst':
                # Worst: max of group errors
                error = np.max(metrics['group_errors'])
            else:
                # Default: overall selective error
                error = metrics['selective_error']
            
            selective_errors.append(error)
            group_errors_list.append(metrics['group_errors'])
        
        rejection_rates = np.array(rejection_rates)
        selective_errors = np.array(selective_errors)
        
        # AURC (area under RC curve)
        aurc = np.trapz(selective_errors, rejection_rates)
        
        return {
            'rejection_rates': rejection_rates,
            'selective_errors': selective_errors,
            'group_errors_list': group_errors_list,
            'aurc': aurc,
            'threshold_grid': threshold_grid
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Testing SimpleMAPSelector...")
    
    # Config
    config = SimpleMAPConfig(
        num_classes=100,
        num_groups=2,
        group_boundaries=[50]
    )
    
    selector = SimpleMAPSelector(config)
    
    # Fake data
    B = 1000
    posteriors = torch.randn(B, 100).softmax(dim=-1)
    uncertainty = torch.rand(B)
    labels = torch.randint(0, 100, (B,))
    
    # Set parameters
    selector.set_parameters(threshold=0.5, gamma=0.5)
    
    # Forward
    output = selector(posteriors, uncertainty)
    
    print(f"✓ Predictions shape: {output['predictions'].shape}")
    print(f"✓ Reject shape: {output['reject'].shape}")
    print(f"✓ Margin shape: {output['margin'].shape}")
    print(f"✓ Confidence shape: {output['confidence'].shape}")
    print(f"✓ Rejection rate: {output['reject'].float().mean():.3f}")
    
    # Metrics
    metrics = compute_selective_metrics(
        output['predictions'],
        output['reject'],
        labels,
        selector.class_to_group,
        config.num_groups
    )
    
    print(f"\n✓ Metrics:")
    print(f"  Coverage: {metrics['coverage']:.3f}")
    print(f"  Selective error: {metrics['selective_error']:.3f}")
    print(f"  Group errors: {metrics['group_errors']}")
    
    # Grid search
    print(f"\n✓ Testing grid search...")
    optimizer = SimpleGridSearchOptimizer(config)
    
    result = optimizer.search(
        selector, posteriors, uncertainty, labels,
        verbose=False
    )
    
    print(f"  Best threshold: {result.threshold:.3f}")
    print(f"  Best gamma: {result.gamma:.3f}")
    print(f"  Selective error: {result.selective_error:.3f}")
    
    print("\n✅ All tests passed!")
