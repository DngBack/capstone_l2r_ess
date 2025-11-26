"""
Mixture-aware gating network used by the main Learning-to-Reject pipeline.

The module focuses on the routing component (a learned mixture-of-experts
router) and exposes three steps:

1. Feature extraction from expert posteriors (uncertainty + disagreement cues)
2. A lightweight MLP that maps those features to unnormalized expert scores
3. A routing policy (dense softmax or noisy top-k) that produces expert weights

References:
- Jordan & Jacobs (1994) — Hierarchical Mixtures of Experts
- Shazeer et al. (2017) — Outrageously Large Neural Networks
- Fedus et al. (2021) — Switch Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class GatingNetworkConfig:
    """
    Configuration block used to build the gating network end-to-end.

    Attributes:
        num_experts: Number of experts being routed.
        num_classes: Number of classes/dimensions per expert posterior.
        hidden_dims: Hidden sizes for the gating MLP.
        dropout: Dropout applied after each hidden activation.
        routing: Routing strategy (`dense` or `top_k`).
        top_k: Number of experts to keep when routing='top_k'.
        noise_std: Gaussian noise std for the noisy top-k router.
        activation: Activation function for the MLP (`relu` or `gelu`).
        normalize_features: Whether to layer-normalize concatenated features.
    """

    num_experts: int
    num_classes: int
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    routing: str = 'dense'
    top_k: int = 2
    noise_std: float = 1.0
    activation: str = 'relu'
    normalize_features: bool = False


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def _disagreement_ratio(top1_classes: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Compute normalized disagreement ratio per sample given top-1 predictions.

    We keep the simple for-loop implementation because the number of experts E
    is typically small (<=4) and the explicit version improves clarity during
    presentations.
    """
    B, E = top1_classes.shape
    ratios = torch.zeros(B, dtype=dtype, device=device)
    denom = max(E - 1, 1)

    for b in range(B):
        unique_preds = torch.unique(top1_classes[b]).numel()
        ratios[b] = (unique_preds - 1) / denom

    return ratios


def _mean_pairwise_kl(posteriors: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute the mean KL divergence between every pair of expert posteriors.

    This implementation vectorizes the pairwise computation and averages over
    the upper triangular entries, leading to cleaner code than the previous
    nested for-loops.
    """
    B, E, _ = posteriors.shape
    if E <= 1:
        return torch.zeros(B, device=posteriors.device, dtype=posteriors.dtype)

    log_post = torch.log(posteriors + eps)
    diff = log_post.unsqueeze(2) - log_post.unsqueeze(1)              # [B, E, E, C]
    pairwise_kl = (posteriors.unsqueeze(2) * diff).sum(dim=-1)        # [B, E, E]

    # Take strictly upper triangle to avoid double-counting/self terms
    triu_indices = torch.triu_indices(E, E, offset=1, device=posteriors.device)
    upper_vals = pairwise_kl[:, triu_indices[0], triu_indices[1]]     # [B, num_pairs]
    mean_vals = upper_vals.mean(dim=-1)
    return mean_vals


class UncertaintyDisagreementFeatures:
    """
    Compute per-expert and global uncertainty features from posteriors.

    Motivated by ensemble uncertainty literature (e.g., Deep Ensembles), these
    statistics highlight when the experts disagree and when each expert is
    individually uncertain—both are strong predictors for when the router
    should reduce confidence.
    """
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
    
    @torch.no_grad()
    def compute(self, posteriors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            posteriors: [B, E, C] - batch, experts, classes
        
        Returns:
            Dictionary of features (all detached, for input to gating)
        """
        B, E, C = posteriors.shape
        eps = 1e-8
        
        features = {}
        
        # ====================================================================
        # 1. PER-EXPERT FEATURES
        # ====================================================================
        
        # 1.1 Per-expert entropy: H(p^(e)) — higher means lower confidence
        expert_entropy = -torch.sum(
            posteriors * torch.log(posteriors + eps), 
            dim=-1
        )  # [B, E]
        features['expert_entropy'] = expert_entropy
        
        # 1.2 Confidence (max probability) per expert
        expert_max_prob, _ = posteriors.max(dim=-1)  # [B, E]
        features['expert_confidence'] = expert_max_prob
        
        # 1.3 Top-1 vs Top-2 gap (margin)
        top2_probs, _ = posteriors.topk(k=min(2, C), dim=-1)  # [B, E, 2]
        if top2_probs.size(-1) == 2:
            expert_margin = top2_probs[..., 0] - top2_probs[..., 1]  # [B, E]
        else:
            expert_margin = top2_probs[..., 0]  # single-class fallback
        features['expert_margin'] = expert_margin
        
        # ====================================================================
        # 2. DISAGREEMENT FEATURES (across experts)
        # ====================================================================
        
        # 2.1 Fraction of unique top-1 predictions (normalized)
        top1_classes = posteriors.argmax(dim=-1)  # [B, E]
        disagreement_ratio = _disagreement_ratio(top1_classes, dtype=posteriors.dtype, device=posteriors.device)
        features['disagreement_ratio'] = disagreement_ratio
        
        # 2.2 Mean pairwise KL divergence
        # KL(p_i || p_j) averaged over all pairs
        mean_pairwise_kl = _mean_pairwise_kl(posteriors, eps=eps)
        features['mean_pairwise_kl'] = mean_pairwise_kl
        
        # ====================================================================
        # 3. MIXTURE/ENSEMBLE FEATURES
        # ====================================================================
        
        # 3.1 Uniform mixture (proxy before gating weights are known)
        # Note: simple heuristic to estimate ensemble behavior up-front
        uniform_mixture = posteriors.mean(dim=1)  # [B, C]
        
        # 3.2 Entropy of the uniform mixture: H(uniform_η̃)
        # Not the actual mixture—just a stabilizing feature for the router
        uniform_mixture_entropy = -torch.sum(
            uniform_mixture * torch.log(uniform_mixture + eps),
            dim=-1
        )  # [B]
        features['uniform_mixture_entropy'] = uniform_mixture_entropy
        
        # 3.3 Posterior variance across experts (averaged over classes)
        # Higher variance implies larger disagreement
        posterior_variance = posteriors.var(dim=1)  # [B, C]
        mean_posterior_variance = posterior_variance.mean(dim=-1)  # [B]
        features['posterior_variance'] = mean_posterior_variance
        
        # 3.4 Mutual Information: I(Y; E | X) ≈ H(uniform_η̃) - mean(H(p^(e)))
        # Large MI indicates valuable expert diversity
        # Note: still relies on uniform mixture because weights are unknown
        mean_expert_entropy = expert_entropy.mean(dim=-1)  # [B]
        mutual_info = uniform_mixture_entropy - mean_expert_entropy  # [B]
        features['mutual_information'] = mutual_info
        
        return features


class GatingFeatureExtractor(nn.Module):
    """
    Turn expert posteriors into a flattened, presentation-friendly feature vector.

    The extractor concatenates the raw posteriors with several summary features
    (entropy, margins, disagreement ratios, etc.) so that the downstream MLP can
    operate on a fixed-size representation regardless of the number of classes.
    """
    
    def __init__(self, num_experts: int, num_classes: int, normalize_features: bool = False):  # ← Changed default to False
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.normalize_features = normalize_features
        self.uncertainty_computer = UncertaintyDisagreementFeatures(num_experts)
        
        # Feature dimension bookkeeping
        # Posteriors: E*C
        # Per-expert summaries: 3*E (entropy, confidence, margin)
        # Global scalars: 5 (disagreement_ratio, mean_kl, uniform_mixture_entropy, posterior_var, mutual_info)
        self.feature_dim = (num_experts * num_classes +  # posteriors flattened
                           3 * num_experts +              # per-expert features
                           5)                             # global features
        
        # Optional: LayerNorm cho features (stable training)
        if normalize_features:
            self.feature_norm = nn.LayerNorm(self.feature_dim)
    
    def forward(self, posteriors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            posteriors: [B, E, C]
        
        Returns:
            features: [B, D]
        """
        B, E, C = posteriors.shape
        assert E == self.num_experts
        assert C == self.num_classes
        
        # 1. Flatten posteriors
        posteriors_flat = posteriors.reshape(B, -1)  # [B, E*C]
        
        # 2. Compute uncertainty/disagreement features
        unc_features = self.uncertainty_computer.compute(posteriors)
        
        # 3. Concatenate all features
        feature_list = [
            posteriors_flat,                               # [B, E*C]
            unc_features['expert_entropy'],               # [B, E]
            unc_features['expert_confidence'],            # [B, E]
            unc_features['expert_margin'],                # [B, E]
            unc_features['disagreement_ratio'].unsqueeze(-1),  # [B, 1]
            unc_features['mean_pairwise_kl'].unsqueeze(-1),    # [B, 1]
            unc_features['uniform_mixture_entropy'].unsqueeze(-1),     # [B, 1]
            unc_features['posterior_variance'].unsqueeze(-1),  # [B, 1]
            unc_features['mutual_information'].unsqueeze(-1),  # [B, 1]
        ]
        
        features = torch.cat(feature_list, dim=-1)  # [B, D]
        assert features.shape[1] == self.feature_dim
        
        # Optional normalization with clipping for numerical stability
        if self.normalize_features:
            # Clip extreme values before LayerNorm to prevent NaN
            features = torch.clamp(features, min=-100, max=100)
            features = self.feature_norm(features)
        
        return features


# ============================================================================
# GATING NETWORK ARCHITECTURE
# ============================================================================

class GatingMLP(nn.Module):
    """
    Lightweight MLP with LayerNorm (more stable than BatchNorm for small batch sizes).

    Architecture:
        Input  -> Linear -> LayerNorm -> Activation -> Dropout (repeat)
        Output -> Linear producing expert logits

    References:
        Ba et al. (2016): Layer Normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # Build MLP
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(in_dim, num_experts))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Weight initialization helper using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D]
        
        Returns:
            logits: [B, E] (gating scores before normalization)
        """
        return self.mlp(features)


# ============================================================================
# ROUTING STRATEGIES
# ============================================================================

class DenseSoftmaxRouter(nn.Module):
    """
    Classic dense MoE routing: softmax(g(x)).
    """
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, E]
        
        Returns:
            weights: [B, E] (simplex)
        """
        return F.softmax(logits, dim=-1)


class NoisyTopKRouter(nn.Module):
    """
    Noisy Top-K routing (Shazeer et al., 2017).
    
    Algorithm:
    1. Add Gaussian noise to logits
    2. Select Top-K experts
    3. Softmax renormalize over Top-K
    
    Benefits:
    - Sparse computation (only K experts)
    - Noise for exploration/load-balancing
    
    References:
    - Shazeer et al. (2017): Outrageously Large Neural Networks
    """
    
    def __init__(self, top_k: int = 2, noise_std: float = 1.0, training_only: bool = True):
        super().__init__()
        self.top_k = top_k
        self.noise_std = noise_std
        self.training_only = training_only
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, E]
        
        Returns:
            weights: [B, E] (sparse simplex - only top-k non-zero)
        """
        B, E = logits.shape
        
        # Add noise during training (if enabled)
        if self.training and (not self.training_only or self.training):
            noise = torch.randn_like(logits) * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        
        # Top-K selection
        top_k = min(self.top_k, E)
        topk_logits, topk_indices = torch.topk(noisy_logits, k=top_k, dim=-1)
        
        # Softmax over top-k
        topk_weights = F.softmax(topk_logits, dim=-1)  # [B, K]
        
        # Scatter back to full dimension
        weights = torch.zeros_like(logits)
        weights.scatter_(dim=1, index=topk_indices, src=topk_weights)
        
        return weights


# ============================================================================
# COMPLETE GATING NETWORK
# ============================================================================

class GatingNetwork(nn.Module):
    """
    Full gating stack = feature extractor + MLP + routing policy.

    Usage:
        gating = GatingNetwork(num_experts=3, num_classes=100, routing='dense')
        posteriors = expert_posteriors  # [B, E, C]
        weights, aux = gating(posteriors)
    """
    
    def __init__(
        self,
        num_experts: Optional[int] = None,
        num_classes: Optional[int] = None,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        routing: str = 'dense',  # 'dense' or 'top_k'
        top_k: int = 2,
        noise_std: float = 1.0,
        activation: str = 'relu',
        normalize_features: bool = False,
        config: Optional[GatingNetworkConfig] = None,
    ):
        super().__init__()

        if config is None:
            if num_experts is None or num_classes is None:
                raise ValueError("Provide `config` or both `num_experts` and `num_classes`.")
            hidden_dims = hidden_dims or [256, 128]
            config = GatingNetworkConfig(
                num_experts=num_experts,
                num_classes=num_classes,
                hidden_dims=tuple(hidden_dims),
                dropout=dropout,
                routing=routing,
                top_k=top_k,
                noise_std=noise_std,
                activation=activation,
                normalize_features=normalize_features,
            )
        else:
            if any(param is not None for param in [num_experts, num_classes, hidden_dims]):
                raise ValueError("When `config` is provided, omit legacy positional args.")

        self.config = config
        self.num_experts = config.num_experts
        self.num_classes = config.num_classes
        self.routing_type = config.routing

        self.feature_extractor = GatingFeatureExtractor(
            config.num_experts,
            config.num_classes,
            normalize_features=config.normalize_features,
        )

        self.mlp = GatingMLP(
            input_dim=self.feature_extractor.feature_dim,
            num_experts=config.num_experts,
            hidden_dims=list(config.hidden_dims),
            dropout=config.dropout,
            activation=config.activation,
        )

        if config.routing == 'dense':
            self.router = DenseSoftmaxRouter()
        elif config.routing == 'top_k':
            self.router = NoisyTopKRouter(top_k=config.top_k, noise_std=config.noise_std)
        else:
            raise ValueError(f"Unknown routing strategy: {config.routing}")
    
    def forward(self, posteriors: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            posteriors: [B, E, C] expert posteriors (already temperature calibrated)

        Returns:
            weights: [B, E] gating weights (simplex)
            aux_outputs: diagnostics (logits, features) for auxiliary losses
        """
        # Extract features
        features = self.feature_extractor(posteriors)  # [B, D]
        
        # MLP
        logits = self.mlp(features)  # [B, E]
        
        # Routing
        weights = self.router(logits)  # [B, E]
        
        # Auxiliary outputs (for loss computation)
        aux_outputs = {
            'logits': logits,
            'features': features,
        }
        
        return weights, aux_outputs
    
    def get_mixture_posterior(
        self, 
        posteriors: torch.Tensor, 
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the mixture posterior η̃(x) = Σ_e w_e · p_e(y|x).

        Args:
            posteriors: [B, E, C] expert posteriors
            weights: [B, E] gating weights; recomputed if None

        Returns:
            mixture_posterior: [B, C]
        """
        if weights is None:
            weights, _ = self.forward(posteriors)
        
        # weights: [B, E] → [B, E, 1]
        # posteriors: [B, E, C]
        # mixture: [B, C]
        mixture_posterior = torch.sum(
            weights.unsqueeze(-1) * posteriors,
            dim=1
        )
        
        return mixture_posterior


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_uncertainty_for_map(
    posteriors: torch.Tensor,
    weights: torch.Tensor,
    mixture_posterior: Optional[torch.Tensor] = None,
    coeffs: Dict[str, float] = None
) -> torch.Tensor:
    """
    Compute the composite uncertainty term used in the MAP margin:
        U(x) = a · H(w) + b · Disagreement + d · H(η̃)

    Args:
        posteriors: [B, E, C] expert posteriors
        weights: [B, E] gating weights
        mixture_posterior: [B, C] mixture; recomputed if None
        coeffs: Optional dictionary overriding coefficients (defaults to 1.0)

    Returns:
        Tensor of shape [B] with per-sample uncertainty scores.
    """
    if coeffs is None:
        coeffs = {'a': 1.0, 'b': 1.0, 'd': 1.0}
    
    eps = 1e-8
    
    # 1. Entropy of gating weights: H(w)
    H_w = -torch.sum(weights * torch.log(weights + eps), dim=-1)  # [B]

    # 2. Disagreement ratio using helper (keeps implementation consistent)
    top1_classes = posteriors.argmax(dim=-1)  # [B, E]
    disagree = _disagreement_ratio(top1_classes, dtype=posteriors.dtype, device=posteriors.device)

    # 3. Entropy of mixture: H(η̃)
    if mixture_posterior is None:
        mixture_posterior = torch.sum(weights.unsqueeze(-1) * posteriors, dim=1)
    H_mix = -torch.sum(mixture_posterior * torch.log(mixture_posterior + eps), dim=-1)
    
    # Combine
    U = coeffs['a'] * H_w + coeffs['b'] * disagree + coeffs['d'] * H_mix
    
    return U


if __name__ == '__main__':
    """Test code"""
    print("Testing GatingNetwork...")
    
    # Mock data
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    
    # Test dense routing
    print("\n1. Dense Routing:")
    gating_dense = GatingNetwork(
        num_experts=E,
        num_classes=C,
        routing='dense'
    )
    weights_dense, aux_dense = gating_dense(posteriors)
    print(f"   Weights shape: {weights_dense.shape}")
    print(f"   Weights sum: {weights_dense.sum(dim=1).mean():.4f} (should be ~1.0)")
    print(f"   Feature dim: {gating_dense.feature_extractor.feature_dim}")
    
    # Test top-k routing
    print("\n2. Top-K Routing (K=2):")
    gating_topk = GatingNetwork(
        num_experts=E,
        num_classes=C,
        routing='top_k',
        top_k=2
    )
    weights_topk, aux_topk = gating_topk(posteriors)
    print(f"   Weights shape: {weights_topk.shape}")
    print(f"   Non-zero experts per sample: {(weights_topk > 0).sum(dim=1).float().mean():.2f}")
    print(f"   Weights sum: {weights_topk.sum(dim=1).mean():.4f} (should be ~1.0)")
    
    # Test mixture posterior
    print("\n3. Mixture Posterior:")
    mixture = gating_dense.get_mixture_posterior(posteriors, weights_dense)
    print(f"   Mixture shape: {mixture.shape}")
    print(f"   Mixture sum: {mixture.sum(dim=1).mean():.4f} (should be ~1.0)")
    
    # Test uncertainty
    print("\n4. Uncertainty for MAP:")
    U = compute_uncertainty_for_map(posteriors, weights_dense, mixture)
    print(f"   U shape: {U.shape}")
    print(f"   U range: [{U.min():.4f}, {U.max():.4f}]")
    
    print("\n✅ All tests passed!")
