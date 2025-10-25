"""
Grid Search (S2) và EG-Outer cho MAP Plugin
=============================================

Triển khai:
1. Grid search cho (μ, γ, ν) trên S2
2. EG-outer cho worst-group optimization
3. RC curve sweeping cost c
4. AURC computation

References:
- L2R: S1/S2 splitting, EG-outer for minimax
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

from src.models.map_selector import (
    MAPSelector,
    MAPConfig,
    FixedPointAlphaOptimizer,
    lambda_to_mu,
    compute_selective_metrics
)


# ============================================================================
# GRID SEARCH (S2)
# ============================================================================

@dataclass
class GridSearchResult:
    """Result from grid search."""
    lambda_val: float
    gamma: float
    nu: float
    alpha: torch.Tensor
    mu: torch.Tensor
    score: float
    metrics: Dict
    

class GridSearchOptimizer:
    """
    Grid search cho (λ, γ, ν) trên S2.
    
    Objective:
    - 'balanced': R_β with β_k = 1/K
    - 'worst': max_k e_k
    - 'aurc': Area Under Risk-Coverage curve
    """
    
    def __init__(self, config: MAPConfig):
        self.config = config
        self.fp_optimizer = FixedPointAlphaOptimizer(config)
    
    def compute_objective(
        self,
        selector: MAPSelector,
        mixture_posteriors: torch.Tensor,
        uncertainties: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        gamma: float,
        cost: float,
        beta: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[float, Dict]:
        """
        Compute objective on S2.
        
        Args:
            beta: [K] group weights (for weighted objective)
            sample_weights: [N] for reweighting
        
        Returns:
            score: scalar (lower is better)
            metrics: dict
        """
        device = mixture_posteriors.device
        K = self.config.num_groups
        
        # Predictions and rejections
        margins = selector.compute_margin_map(
            mixture_posteriors, uncertainties,
            alpha=alpha, mu=mu, gamma=gamma, cost=cost
        )
        predictions = selector.predict_class(mixture_posteriors, alpha=alpha)
        reject = margins < 0
        
        # Compute metrics
        metrics = compute_selective_metrics(
            predictions, reject, labels,
            selector.class_to_group, K,
            sample_weights=sample_weights
        )
        
        # Compute objective based on mode
        if self.config.objective == 'balanced':
            # Balanced risk: Σ_k (1/K) · e_k + c · coverage
            if beta is None:
                beta = torch.ones(K, device=device) / K
            
            group_errors = torch.tensor(metrics['group_errors'], device=device)
            error_term = (beta * group_errors).sum()
            coverage_term = cost * metrics['coverage']
            score = error_term + coverage_term
            
        elif self.config.objective == 'worst':
            # Worst-group: max_k e_k + c · coverage
            if beta is None:
                beta = torch.ones(K, device=device) / K
            
            group_errors = torch.tensor(metrics['group_errors'], device=device)
            error_term = (beta * group_errors).sum()  # Will be updated by EG
            coverage_term = cost * metrics['coverage']
            score = error_term + coverage_term
            
        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")
        
        metrics['score'] = score
        
        return score, metrics
    
    def search(
        self,
        selector: MAPSelector,
        s1_mixture: torch.Tensor,
        s1_uncertainty: torch.Tensor,
        s1_labels: torch.Tensor,
        s2_mixture: torch.Tensor,
        s2_uncertainty: torch.Tensor,
        s2_labels: torch.Tensor,
        cost: float = 0.0,
        beta: Optional[torch.Tensor] = None,
        s1_weights: Optional[torch.Tensor] = None,
        s2_weights: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> GridSearchResult:
        """
        Grid search over (λ, γ, ν).
        
        Args:
            s1_*: S1 data (for fixed-point)
            s2_*: S2 data (for evaluation)
            cost: rejection cost
            beta: [K] group weights (for worst-group)
            s1_weights, s2_weights: sample weights for reweighting
        
        Returns:
            best_result: GridSearchResult
        """
        best_score = float('inf')
        best_result = None
        
        total_configs = len(self.config.lambda_grid) * len(self.config.gamma_grid) * len(self.config.nu_grid)
        
        if verbose:
            print(f"Grid search: {total_configs} configurations")
            print(f"  λ: {len(self.config.lambda_grid)} values")
            print(f"  γ: {len(self.config.gamma_grid)} values")
            print(f"  ν: {len(self.config.nu_grid)} values")
        
        # Progress bar
        pbar = tqdm(total=total_configs, desc="Grid search") if verbose else None
        
        for lambda_val in self.config.lambda_grid:
            # Convert λ to μ (for K=2)
            mu = lambda_to_mu(lambda_val, self.config.num_classes, self.config.group_boundaries)
            mu = mu.to(s1_mixture.device)
            
            for gamma in self.config.gamma_grid:
                for nu in self.config.nu_grid:
                    # S1: Fixed-point for α
                    alpha = self.fp_optimizer.optimize(
                        selector,
                        s1_mixture, s1_uncertainty, s1_labels,
                        mu=mu, gamma=gamma, nu=nu, cost=cost,
                        verbose=False
                    )
                    
                    # S2: Evaluate
                    score, metrics = self.compute_objective(
                        selector,
                        s2_mixture, s2_uncertainty, s2_labels,
                        alpha=alpha, mu=mu, gamma=gamma, cost=cost,
                        beta=beta,
                        sample_weights=s2_weights
                    )
                    
                    # Track best
                    if score < best_score:
                        best_score = score
                        best_result = GridSearchResult(
                            lambda_val=lambda_val,
                            gamma=gamma,
                            nu=nu,
                            alpha=alpha.cpu(),
                            mu=mu.cpu(),
                            score=score,
                            metrics=metrics
                        )
                    
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({
                            'best_score': f'{best_score:.4f}',
                            'λ': f'{lambda_val:.2f}',
                            'γ': f'{gamma:.2f}'
                        })
        
        if pbar is not None:
            pbar.close()
        
        if verbose and best_result is not None:
            print(f"\n✅ Best configuration:")
            print(f"   λ={best_result.lambda_val:.3f}, γ={best_result.gamma:.3f}, ν={best_result.nu:.3f}")
            print(f"   Score: {best_result.score:.4f}")
            print(f"   Coverage: {best_result.metrics['coverage']:.3f}")
            print(f"   Selective error: {best_result.metrics['selective_error']:.3f}")
            print(f"   Group errors: {best_result.metrics['group_errors']}")
        
        return best_result


# ============================================================================
# EG-OUTER FOR WORST-GROUP
# ============================================================================

class EGOuterOptimizer:
    """
    Exponentiated Gradient (EG) outer loop cho worst-group optimization.
    
    Algorithm:
        for t in 1..T:
            # Inner: solve với β^(t)
            (h_t, r_t) = grid_search với β^(t)
            
            # Evaluate group errors
            ê_k = error on group k (on S2)
            
            # Update β
            β_k^(t+1) ∝ β_k^(t) · exp(ξ · ê_k)
            normalize β
    
    References:
    - L2R: EG-outer for minimax
    - Online learning: Multiplicative weights with no-regret
    """
    
    def __init__(self, config: MAPConfig):
        self.config = config
        self.grid_search = GridSearchOptimizer(config)
    
    def optimize(
        self,
        selector: MAPSelector,
        s1_mixture: torch.Tensor,
        s1_uncertainty: torch.Tensor,
        s1_labels: torch.Tensor,
        s2_mixture: torch.Tensor,
        s2_uncertainty: torch.Tensor,
        s2_labels: torch.Tensor,
        cost: float = 0.0,
        s1_weights: Optional[torch.Tensor] = None,
        s2_weights: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Tuple[GridSearchResult, torch.Tensor]:
        """
        Run EG-outer optimization.
        
        Returns:
            best_result: GridSearchResult
            beta: [K] final group weights
        """
        device = s2_mixture.device
        K = self.config.num_groups
        
        # Initialize β = uniform
        beta = torch.ones(K, device=device) / K
        
        best_worst_error = float('inf')
        best_result = None
        best_beta = beta.clone()
        
        if verbose:
            print(f"\n{'='*70}")
            print("EG-OUTER OPTIMIZATION (Worst-Group)")
            print(f"{'='*70}")
        
        for t in range(self.config.eg_iterations):
            if verbose:
                print(f"\nIteration {t+1}/{self.config.eg_iterations}:")
                print(f"  β = {beta.cpu().numpy()}")
            
            # Inner: Grid search với β hiện tại
            result = self.grid_search.search(
                selector,
                s1_mixture, s1_uncertainty, s1_labels,
                s2_mixture, s2_uncertainty, s2_labels,
                cost=cost,
                beta=beta,
                s1_weights=s1_weights,
                s2_weights=s2_weights,
                verbose=False
            )
            
            # Get group errors
            group_errors = torch.tensor(result.metrics['group_errors'], device=device)
            
            # Worst-group error
            worst_error = group_errors.max().item()
            
            if verbose:
                print(f"  Group errors: {group_errors.cpu().numpy()}")
                print(f"  Worst error: {worst_error:.4f}")
            
            # Track best
            if worst_error < best_worst_error:
                best_worst_error = worst_error
                best_result = result
                best_beta = beta.clone()
                if verbose:
                    print(f"  → New best worst-error: {best_worst_error:.4f}")
            
            # Update β via EG
            # β_k ← β_k · exp(ξ · e_k)
            beta = beta * torch.exp(self.config.eg_xi * group_errors)
            beta = beta / beta.sum()  # Normalize
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ EG-Outer completed!")
            print(f"   Best worst-group error: {best_worst_error:.4f}")
            print(f"   Final β: {best_beta.cpu().numpy()}")
        
        return best_result, best_beta


# ============================================================================
# RC CURVE & AURC
# ============================================================================

class RCCurveComputer:
    """
    Compute Risk-Coverage (RC) curve by sweeping cost c.
    
    RC curve:
    - X-axis: 1 - coverage (rejection rate)
    - Y-axis: selective error (on accepted)
    
    AURC: Area Under RC curve (lower is better)
    """
    
    def __init__(self, config: MAPConfig):
        self.config = config
    
    def compute_rc_curve(
        self,
        selector: MAPSelector,
        mixture_posteriors: torch.Tensor,
        uncertainties: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        gamma: float,
        cost_grid: np.ndarray,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute RC curve by sweeping cost.
        
        Args:
            cost_grid: array of cost values to sweep
        
        Returns:
            rc_data: dict with 'rejection_rates', 'selective_errors', 'aurc'
        """
        device = mixture_posteriors.device
        K = self.config.num_groups
        
        rejection_rates = []
        selective_errors = []
        group_errors_list = []
        
        for cost in cost_grid:
            # Compute margins with this cost
            margins = selector.compute_margin_map(
                mixture_posteriors, uncertainties,
                alpha=alpha, mu=mu, gamma=gamma, cost=cost
            )
            
            predictions = selector.predict_class(mixture_posteriors, alpha=alpha)
            reject = margins < 0
            
            # Metrics
            metrics = compute_selective_metrics(
                predictions, reject, labels,
                selector.class_to_group, K,
                sample_weights=sample_weights
            )
            
            rejection_rates.append(1.0 - metrics['coverage'])
            selective_errors.append(metrics['selective_error'])
            group_errors_list.append(metrics['group_errors'])
        
        # Compute AURC (trapezoidal rule)
        rejection_rates = np.array(rejection_rates)
        selective_errors = np.array(selective_errors)
        
        # Sort by rejection rate
        sorted_indices = np.argsort(rejection_rates)
        rejection_rates = rejection_rates[sorted_indices]
        selective_errors = selective_errors[sorted_indices]
        
        # AURC
        aurc = np.trapz(selective_errors, rejection_rates)
        
        rc_data = {
            'rejection_rates': rejection_rates,
            'selective_errors': selective_errors,
            'group_errors_list': group_errors_list,
            'aurc': aurc,
            'cost_grid': cost_grid
        }
        
        return rc_data
    
    def compute_aurc_for_optimization(
        self,
        selector: MAPSelector,
        mixture_posteriors: torch.Tensor,
        uncertainties: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        gamma: float,
        sample_weights: Optional[torch.Tensor] = None,
        num_points: int = 20
    ) -> float:
        """
        Compute AURC for model selection (simplified).
        
        Args:
            num_points: number of cost values to sample
        
        Returns:
            aurc: scalar
        """
        # Sample cost values
        cost_grid = np.linspace(-2.0, 2.0, num_points)
        
        rc_data = self.compute_rc_curve(
            selector, mixture_posteriors, uncertainties, labels,
            alpha, mu, gamma, cost_grid,
            sample_weights=sample_weights
        )
        
        return rc_data['aurc']


if __name__ == '__main__':
    """Test code"""
    print("Testing Grid Search & EG-Outer...")
    
    # Mock data
    N1, N2 = 100, 80
    C = 100
    K = 2
    
    config = MAPConfig(
        num_classes=C,
        num_groups=K,
        group_boundaries=[50],
        lambda_grid=[-1.0, 0.0, 1.0],
        gamma_grid=[0.0, 0.5],
        nu_grid=[2.0, 5.0]
    )
    
    selector = MAPSelector(config)
    
    # S1 data
    s1_mixture = torch.softmax(torch.randn(N1, C), dim=-1)
    s1_uncertainty = torch.rand(N1)
    s1_labels = torch.randint(0, C, (N1,))
    
    # S2 data
    s2_mixture = torch.softmax(torch.randn(N2, C), dim=-1)
    s2_uncertainty = torch.rand(N2)
    s2_labels = torch.randint(0, C, (N2,))
    
    # Test grid search
    print("\n1. Grid Search:")
    grid_opt = GridSearchOptimizer(config)
    result = grid_opt.search(
        selector,
        s1_mixture, s1_uncertainty, s1_labels,
        s2_mixture, s2_uncertainty, s2_labels,
        cost=0.0,
        verbose=True
    )
    
    print(f"\n   Best λ: {result.lambda_val:.3f}")
    print(f"   Best γ: {result.gamma:.3f}")
    print(f"   Best ν: {result.nu:.3f}")
    
    # Test RC curve
    print("\n2. RC Curve:")
    rc_computer = RCCurveComputer(config)
    cost_grid = np.linspace(-1, 1, 10)
    rc_data = rc_computer.compute_rc_curve(
        selector, s2_mixture, s2_uncertainty, s2_labels,
        alpha=result.alpha, mu=result.mu, gamma=result.gamma,
        cost_grid=cost_grid
    )
    print(f"   AURC: {rc_data['aurc']:.4f}")
    print(f"   Rejection rates: {rc_data['rejection_rates'][:3]}...")
    
    # Test EG-outer
    print("\n3. EG-Outer:")
    config.objective = 'worst'
    config.eg_iterations = 3
    eg_opt = EGOuterOptimizer(config)
    
    eg_result, eg_beta = eg_opt.optimize(
        selector,
        s1_mixture, s1_uncertainty, s1_labels,
        s2_mixture, s2_uncertainty, s2_labels,
        cost=0.0,
        verbose=True
    )
    
    print("\n✅ All tests passed!")
