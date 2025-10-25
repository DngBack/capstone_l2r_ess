"""
MAP (Mixture-Aware Plug-in) Selector cho L2R với Mixture Posterior
===================================================================

Triển khai:
1. Linear-threshold classifier/rejector trên mixture posterior
2. Margin với uncertainty penalty: m_MAP = m_L2R - γ·U(x)
3. Fixed-point α optimization (S1)
4. Grid search cho (μ, γ, ν) (S2)

References:
- Stutz et al. (2023): Learning to Reject with L2R
- Deep Ensembles: Uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MAPConfig:
    """Configuration cho MAP Plugin."""
    
    # Group settings
    num_classes: int = 100
    num_groups: int = 2
    group_boundaries: list = None  # [50] cho CIFAR-100-LT (head: 0-49, tail: 50-99)
    
    # Cost parameter
    cost_default: float = 0.0  # Will sweep this
    
    # Uncertainty coefficients
    uncertainty_coeff_a: float = 1.0  # H(w_φ)
    uncertainty_coeff_b: float = 1.0  # Disagreement
    uncertainty_coeff_d: float = 1.0  # H(η̃)
    
    # Fixed-point settings (S1)
    fp_iterations: int = 10
    fp_nu: float = 5.0          # Sigmoid slope
    fp_ema: float = 0.7         # EMA coefficient
    fp_alpha_min: float = 0.1   # Minimum alpha (prevent division by zero)
    
    # Grid search settings (S2)
    lambda_grid: list = None    # For K=2: λ = μ_1 - μ_2
    gamma_grid: list = None     # Uncertainty penalty weight
    nu_grid: list = None        # Sigmoid slope
    
    # Optimization objective
    objective: str = 'balanced'  # 'balanced' or 'worst'
    
    # EG-outer settings (worst-group)
    eg_iterations: int = 10
    eg_xi: float = 0.1  # Learning rate for EG
    
    def __post_init__(self):
        if self.group_boundaries is None:
            self.group_boundaries = [self.num_classes // 2]
        
        if self.lambda_grid is None:
            # Default: quét từ -3 đến 3
            self.lambda_grid = list(np.linspace(-3.0, 3.0, 13))
        
        if self.gamma_grid is None:
            # Default: no penalty, small, medium
            self.gamma_grid = [0.0, 0.5, 1.0, 2.0]
        
        if self.nu_grid is None:
            # Default: sigmoid slopes
            self.nu_grid = [2.0, 5.0, 10.0]


# ============================================================================
# MAP SELECTOR
# ============================================================================

class MAPSelector(nn.Module):
    """
    MAP Selector: Linear-threshold classifier + rejector trên mixture posterior.
    
    Quy tắc:
    - Classifier: h_α(x) = argmax_y α[y] · η̃_y(x)
    - Rejector: accept if m_MAP(x) ≥ 0
    
    Margin:
        m_L2R = max_y(α[y]·η̃[y]) - (Σ_y'(α[y'] - μ[y'])·η̃[y'] - c)
        m_MAP = m_L2R - γ·U(x)
    """
    
    def __init__(self, config: MAPConfig):
        super().__init__()
        self.config = config
        
        # Parameters (will be optimized via fixed-point)
        self.register_buffer('alpha', torch.ones(config.num_classes))
        self.register_buffer('mu', torch.zeros(config.num_classes))
        
        # Hyperparameters (selected via grid search)
        self.gamma = 0.0  # Uncertainty penalty
        self.cost = config.cost_default
        
        # Group mapping
        self.class_to_group = self._create_class_to_group_map()
    
    def _create_class_to_group_map(self) -> torch.Tensor:
        """
        Map class to group index.
        
        Returns:
            class_to_group: [C] tensor
        """
        class_to_group = torch.zeros(self.config.num_classes, dtype=torch.long)
        
        boundaries = [0] + self.config.group_boundaries + [self.config.num_classes]
        for g in range(self.config.num_groups):
            start = boundaries[g]
            end = boundaries[g + 1]
            class_to_group[start:end] = g
        
        return class_to_group
    
    def set_parameters(
        self,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        gamma: float,
        cost: float
    ):
        """Set optimized parameters."""
        self.alpha.copy_(alpha)
        self.mu.copy_(mu)
        self.gamma = gamma
        self.cost = cost
    
    def compute_margin_l2r(
        self,
        mixture_posterior: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        cost: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute L2R margin (without uncertainty penalty).
        
        m_L2R = max_y(α[y]·η̃[y]) - (Σ_y'(α[y'] - μ[y'])·η̃[y'] - c)
        
        Args:
            mixture_posterior: [B, C]
            alpha: [C] (optional, use self.alpha if None)
            mu: [C] (optional, use self.mu if None)
            cost: scalar (optional, use self.cost if None)
        
        Returns:
            margin_l2r: [B]
        """
        if alpha is None:
            alpha = self.alpha
        if mu is None:
            mu = self.mu
        if cost is None:
            cost = self.cost
        
        # LHS: max_y(α[y] · η̃[y])
        weighted_posterior = alpha * mixture_posterior  # [B, C]
        lhs = weighted_posterior.max(dim=-1)[0]  # [B]
        
        # RHS: Σ_y'(α[y'] - μ[y']) · η̃[y'] - c
        coeffs = alpha - mu  # [C]
        rhs = torch.sum(coeffs * mixture_posterior, dim=-1) - cost  # [B]
        
        # Margin
        margin = lhs - rhs  # [B]
        
        return margin
    
    def compute_margin_map(
        self,
        mixture_posterior: torch.Tensor,
        uncertainty: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
        cost: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute MAP margin with uncertainty penalty.
        
        m_MAP = m_L2R - γ·U(x)
        
        Args:
            mixture_posterior: [B, C]
            uncertainty: [B]
            alpha, mu, gamma, cost: parameters
        
        Returns:
            margin_map: [B]
        """
        if gamma is None:
            gamma = self.gamma
        
        # L2R margin
        margin_l2r = self.compute_margin_l2r(
            mixture_posterior, alpha, mu, cost
        )
        
        # MAP margin = L2R margin - γ·U
        margin_map = margin_l2r - gamma * uncertainty
        
        return margin_map
    
    def predict_class(
        self,
        mixture_posterior: torch.Tensor,
        alpha: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Classifier: h_α(x) = argmax_y α[y] · η̃[y]
        
        Args:
            mixture_posterior: [B, C]
            alpha: [C]
        
        Returns:
            predictions: [B]
        """
        if alpha is None:
            alpha = self.alpha
        
        weighted_posterior = alpha * mixture_posterior  # [B, C]
        predictions = weighted_posterior.argmax(dim=-1)  # [B]
        
        return predictions
    
    def predict_reject(
        self,
        mixture_posterior: torch.Tensor,
        uncertainty: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        gamma: Optional[float] = None,
        cost: Optional[float] = None
    ) -> torch.Tensor:
        """
        Rejector: accept if m_MAP(x) ≥ 0
        
        Returns:
            reject: [B] boolean (True = reject, False = accept)
        """
        margin = self.compute_margin_map(
            mixture_posterior, uncertainty,
            alpha, mu, gamma, cost
        )
        
        reject = margin < 0  # Reject if margin negative
        
        return reject
    
    def forward(
        self,
        mixture_posterior: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass: prediction + rejection decision.
        
        Args:
            mixture_posterior: [B, C]
            uncertainty: [B]
        
        Returns:
            dict with 'predictions', 'reject', 'margin'
        """
        predictions = self.predict_class(mixture_posterior)
        reject = self.predict_reject(mixture_posterior, uncertainty)
        margin = self.compute_margin_map(mixture_posterior, uncertainty)
        
        return {
            'predictions': predictions,
            'reject': reject,
            'margin': margin,
        }


# ============================================================================
# FIXED-POINT ALPHA OPTIMIZER (S1)
# ============================================================================

class FixedPointAlphaOptimizer:
    """
    Optimize α via fixed-point iteration trên S1.
    
    Update rule:
        α_k ← K · (1/|S1|) · Σ_{(x,y)∈S1} σ(ν·m_MAP(x)) · 1{y∈G_k}
    
    với σ = sigmoid (soft acceptance), EMA + clip để ổn định.
    
    References:
    - L2R: Fixed-point for acceptance rate constraints
    - SelectiveNet: Differentiable surrogate for selection
    """
    
    def __init__(self, config: MAPConfig):
        self.config = config
    
    def optimize(
        self,
        selector: MAPSelector,
        mixture_posteriors: torch.Tensor,
        uncertainties: torch.Tensor,
        labels: torch.Tensor,
        mu: torch.Tensor,
        gamma: float,
        nu: float,
        cost: float,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Run fixed-point iteration cho α.
        
        Args:
            selector: MAPSelector instance
            mixture_posteriors: [N, C] on S1
            uncertainties: [N] on S1
            labels: [N] ground truth
            mu, gamma, nu, cost: hyperparameters
        
        Returns:
            alpha: [C] optimized
        """
        device = mixture_posteriors.device
        C = self.config.num_classes
        K = self.config.num_groups
        
        # Initialize α = 1 (uniform)
        alpha = torch.ones(C, device=device)
        
        # Get group indices
        class_to_group = selector.class_to_group.to(device)
        
        for iteration in range(self.config.fp_iterations):
            # Compute soft acceptance for each sample
            margins = selector.compute_margin_map(
                mixture_posteriors, uncertainties,
                alpha=alpha, mu=mu, gamma=gamma, cost=cost
            )
            
            # Soft acceptance: σ(ν·m)
            soft_accept = torch.sigmoid(nu * margins)  # [N]
            
            # Accumulate per group
            group_accept = torch.zeros(K, device=device)
            group_count = torch.zeros(K, device=device)
            
            for k in range(K):
                mask = class_to_group[labels] == k
                group_accept[k] = soft_accept[mask].sum()
                group_count[k] = mask.sum()
            
            # Compute target acceptance rate per group
            target_accept_rate = group_accept / (group_count + 1e-8)
            target_accept_rate = torch.clamp(
                target_accept_rate,
                min=self.config.fp_alpha_min,
                max=1.0
            )
            
            # Map to per-class α: α[y] = K · target_rate[group(y)]
            alpha_new = torch.zeros(C, device=device)
            for k in range(K):
                mask = class_to_group == k
                alpha_new[mask] = K * target_accept_rate[k]
            
            # EMA update
            alpha = self.config.fp_ema * alpha + (1 - self.config.fp_ema) * alpha_new
            
            # Clip to prevent extreme values
            alpha = torch.clamp(alpha, min=self.config.fp_alpha_min, max=10.0)
            
            if verbose and iteration % 3 == 0:
                print(f"  FP iter {iteration}: α range=[{alpha.min():.3f}, {alpha.max():.3f}], "
                      f"accept_rates={target_accept_rate.cpu().numpy()}")
        
        return alpha


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def lambda_to_mu(lambda_val: float, num_classes: int, group_boundaries: list) -> torch.Tensor:
    """
    Convert λ = μ_1 - μ_2 to per-class μ vector (for K=2).
    
    Args:
        lambda_val: scalar difference
        num_classes: C
        group_boundaries: [boundary] e.g., [50]
    
    Returns:
        mu: [C] tensor
    """
    mu = torch.zeros(num_classes)
    
    # Group 1 (head): 0 to boundary
    boundary = group_boundaries[0]
    mu[:boundary] = lambda_val / 2
    
    # Group 2 (tail): boundary to end
    mu[boundary:] = -lambda_val / 2
    
    return mu


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
        sample_weights: [N] optional (for reweighting)
    
    Returns:
        metrics: dict with errors, coverage, risk
    """
    device = predictions.device
    N = predictions.shape[0]
    
    # Accepted samples
    accept = ~reject
    
    # Overall metrics
    if sample_weights is None:
        sample_weights = torch.ones(N, device=device)
    
    total_weight = sample_weights.sum()
    coverage = (accept.float() * sample_weights).sum() / total_weight
    
    # Errors on accepted samples
    correct = (predictions == labels).float()
    errors_on_accepted = (accept & ~correct.bool()).float() * sample_weights
    
    if accept.sum() > 0:
        selective_error = errors_on_accepted.sum() / (accept.float() * sample_weights).sum()
    else:
        selective_error = torch.tensor(0.0, device=device)
    
    # Group-wise metrics
    group_errors = torch.zeros(num_groups, device=device)
    group_coverage = torch.zeros(num_groups, device=device)
    
    # Ensure class_to_group is on the same device as labels
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
                group_errors[k] = 0.0
    
    metrics = {
        'coverage': coverage.item(),
        'selective_error': selective_error.item(),
        'group_errors': group_errors.cpu().numpy(),
        'group_coverage': group_coverage.cpu().numpy(),
    }
    
    return metrics


if __name__ == '__main__':
    """Test code"""
    print("Testing MAP Selector...")
    
    # Mock data
    B, C = 32, 100
    config = MAPConfig(num_classes=C, num_groups=2, group_boundaries=[50])
    
    mixture_posteriors = F.softmax(torch.randn(B, C), dim=-1)
    uncertainties = torch.rand(B) * 2  # [0, 2]
    labels = torch.randint(0, C, (B,))
    
    # Test selector
    print("\n1. MAP Selector:")
    selector = MAPSelector(config)
    
    # Set dummy parameters
    alpha = torch.ones(C) * 1.2
    mu = lambda_to_mu(1.0, C, config.group_boundaries)
    selector.set_parameters(alpha, mu, gamma=0.5, cost=0.0)
    
    # Forward
    outputs = selector(mixture_posteriors, uncertainties)
    print(f"   Predictions: {outputs['predictions'].shape}")
    print(f"   Reject: {outputs['reject'].sum()}/{B} samples")
    print(f"   Margin range: [{outputs['margin'].min():.3f}, {outputs['margin'].max():.3f}]")
    
    # Test fixed-point optimizer
    print("\n2. Fixed-Point Alpha Optimizer:")
    fp_opt = FixedPointAlphaOptimizer(config)
    
    alpha_opt = fp_opt.optimize(
        selector, mixture_posteriors, uncertainties, labels,
        mu=mu, gamma=0.5, nu=5.0, cost=0.0,
        verbose=True
    )
    print(f"   Optimized α range: [{alpha_opt.min():.3f}, {alpha_opt.max():.3f}]")
    
    # Test metrics
    print("\n3. Selective Metrics:")
    metrics = compute_selective_metrics(
        outputs['predictions'], outputs['reject'], labels,
        selector.class_to_group, config.num_groups
    )
    print(f"   Coverage: {metrics['coverage']:.3f}")
    print(f"   Selective error: {metrics['selective_error']:.3f}")
    print(f"   Group errors: {metrics['group_errors']}")
    
    print("\n✅ All tests passed!")
