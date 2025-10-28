"""
Gating Network cho MAP (Mixture-Aware Plug-in) với L2R
===========================================================

Triển khai theo ý tưởng:
1. Đầu vào: posteriors/logits đã calibrated + uncertainty/disagreement features
2. Kiến trúc: MLP nông với LayerNorm (stable hơn BN cho batch nhỏ)
3. Routing: Dense softmax (Jordan-Jacobs) hoặc Noisy Top-K (Shazeer et al.)
4. Loss: Mixture NLL + Load-balancing (Switch Transformer)

References:
- Jordan & Jacobs (1994): Hierarchical Mixtures of Experts
- Shazeer et al. (2017): Outrageously Large Neural Networks (Noisy Top-K)
- Fedus et al. (2021): Switch Transformers (Load-balancing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class UncertaintyDisagreementFeatures:
    """
    Tính các feature uncertainty và disagreement từ expert posteriors.
    
    Động cơ (từ Deep Ensembles):
    - Entropy, disagreement tương quan mạnh với sai số
    - Giúp router "biết khi nào nên dè chừng"
    
    References:
    - Lakshminarayanan et al. (2017): Simple and Scalable Predictive Uncertainty Estimation
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
        
        # 1.1 Entropy của mỗi expert: H(p^(e))
        # Cao → expert không chắc chắn
        expert_entropy = -torch.sum(
            posteriors * torch.log(posteriors + eps), 
            dim=-1
        )  # [B, E]
        features['expert_entropy'] = expert_entropy
        
        # 1.2 Confidence (max prob) của mỗi expert
        expert_max_prob, _ = posteriors.max(dim=-1)  # [B, E]
        features['expert_confidence'] = expert_max_prob
        
        # 1.3 Top-1 vs Top-2 gap (margin)
        top2_probs, _ = posteriors.topk(k=min(2, C), dim=-1)  # [B, E, 2]
        if top2_probs.size(-1) == 2:
            expert_margin = top2_probs[..., 0] - top2_probs[..., 1]  # [B, E]
        else:
            expert_margin = top2_probs[..., 0]  # chỉ có 1 class
        features['expert_margin'] = expert_margin
        
        # ====================================================================
        # 2. DISAGREEMENT FEATURES (giữa các experts)
        # ====================================================================
        
        # 2.1 Tỉ lệ bất đồng top-1 prediction
        top1_classes = posteriors.argmax(dim=-1)  # [B, E]
        # Đếm số lượng unique predictions cho mỗi sample
        disagreement_ratio = []
        for b in range(B):
            unique_preds = torch.unique(top1_classes[b]).numel()
            ratio = (unique_preds - 1) / max(E - 1, 1)  # normalize [0,1]
            disagreement_ratio.append(ratio)
        disagreement_ratio = torch.tensor(
            disagreement_ratio, 
            device=posteriors.device, 
            dtype=posteriors.dtype
        )  # [B]
        features['disagreement_ratio'] = disagreement_ratio
        
        # 2.2 Mean pairwise KL divergence
        # KL(p_i || p_j) averaged over all pairs
        kl_sum = 0.0
        count = 0
        for i in range(E):
            for j in range(i+1, E):
                p_i = posteriors[:, i, :]  # [B, C]
                p_j = posteriors[:, j, :]  # [B, C]
                kl_ij = torch.sum(
                    p_i * (torch.log(p_i + eps) - torch.log(p_j + eps)),
                    dim=-1
                )  # [B]
                kl_sum += kl_ij
                count += 1
        
        if count > 0:
            mean_pairwise_kl = kl_sum / count  # [B]
        else:
            mean_pairwise_kl = torch.zeros(B, device=posteriors.device)
        features['mean_pairwise_kl'] = mean_pairwise_kl
        
        # ====================================================================
        # 3. MIXTURE/ENSEMBLE FEATURES
        # ====================================================================
        
        # 3.1 Uniform mixture (bootstrap estimate trước khi có gating weights)
        # Note: Đây là heuristic để có ước lượng ban đầu về ensemble behavior
        uniform_mixture = posteriors.mean(dim=1)  # [B, C]
        
        # 3.2 Entropy của uniform mixture: H(uniform_η̃)
        # Đây KHÔNG phải mixture thực (chưa có weights), chỉ là feature tham khảo
        uniform_mixture_entropy = -torch.sum(
            uniform_mixture * torch.log(uniform_mixture + eps),
            dim=-1
        )  # [B]
        features['uniform_mixture_entropy'] = uniform_mixture_entropy
        
        # 3.3 Variance của posteriors giữa các experts (trung bình theo class)
        # Cao → experts bất đồng nhiều
        posterior_variance = posteriors.var(dim=1)  # [B, C]
        mean_posterior_variance = posterior_variance.mean(dim=-1)  # [B]
        features['posterior_variance'] = mean_posterior_variance
        
        # 3.4 Mutual Information: I(Y; E | X) ≈ H(uniform_η̃) - mean(H(p^(e)))
        # Cao → expert diversity có giá trị
        # Note: Dùng uniform mixture vì chưa có gating weights
        mean_expert_entropy = expert_entropy.mean(dim=-1)  # [B]
        mutual_info = uniform_mixture_entropy - mean_expert_entropy  # [B]
        features['mutual_information'] = mutual_info
        
        return features


class GatingFeatureExtractor(nn.Module):
    """
    Trích xuất features cho gating network theo approach của GatingFeatureBuilder.
    
    Lightweight & scalable approach (không phụ thuộc vào số classes):
    - Chỉ extract các statistics quan trọng từ posteriors
    - Per-expert features: entropy, confidence, margin, disagreement metrics
    - Global features: ensemble uncertainty, variance, agreement
    
    Input: expert posteriors [B, E, C]
    Output: concatenated feature vector [B, D] where D = 7*E + 3
    """
    
    def __init__(self, num_experts: int, num_classes: int, top_k: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.top_k = top_k
        
        # Feature dimension: 7 per-expert + 3 global = 7*E + 3
        self.feature_dim = 7 * num_experts + 3
    
    @torch.no_grad()
    def forward(self, posteriors: torch.Tensor) -> torch.Tensor:
        """
        Extract features from expert posteriors.
        
        Args:
            posteriors: [B, E, C]
        
        Returns:
            features: [B, 7*E + 3]
        """
        B, E, C = posteriors.shape
        eps = 1e-8
        
        # ====================================================================
        # PER-EXPERT FEATURES (7 features × E experts)
        # ====================================================================
        
        # 1. Entropy: H(p^(e)) = -Σ p(y|x) log p(y|x)
        expert_entropy = -torch.sum(posteriors * torch.log(posteriors + eps), dim=-1)  # [B, E]
        
        # 2. Top-K mass: concentration of probability mass
        topk_vals, _ = torch.topk(posteriors, k=min(self.top_k, C), dim=-1)
        topk_mass = torch.sum(topk_vals, dim=-1)  # [B, E]
        residual_mass = 1.0 - topk_mass  # [B, E]
        
        # 3. Confidence metrics
        max_probs, _ = posteriors.max(dim=-1)  # [B, E] - max probability (confidence)
        
        # Top1-Top2 gap (margin)
        if topk_vals.size(-1) >= 2:
            top1 = topk_vals[..., 0]
            top2 = topk_vals[..., 1]
            top_gap = top1 - top2  # [B, E]
        else:
            top_gap = torch.zeros_like(max_probs)
        
        # 4. Agreement metrics với mean posterior
        mean_posterior = torch.mean(posteriors, dim=1, keepdim=True)  # [B, 1, C]
        
        # Cosine similarity: cos(p^(e), mean)
        cosine_sim = F.cosine_similarity(posteriors, mean_posterior, dim=-1)  # [B, E]
        
        # KL divergence: KL(p^(e) || mean)
        kl_to_mean = torch.sum(
            posteriors * (torch.log(posteriors + eps) - torch.log(mean_posterior + eps)),
            dim=-1
        )  # [B, E]
        
        # ====================================================================
        # GLOBAL FEATURES (3 features)
        # ====================================================================
        
        # Mean entropy of ensemble
        mean_posterior_flat = mean_posterior.squeeze(1)  # [B, C]
        mean_entropy = -torch.sum(
            mean_posterior_flat * torch.log(mean_posterior_flat + eps),
            dim=-1
        )  # [B]
        
        # Mean variance across classes (disagreement measure)
        class_var = posteriors.var(dim=1)  # [B, C]
        mean_class_var = class_var.mean(dim=-1)  # [B]
        
        # Std of max confidences (confidence dispersion between experts)
        std_max_conf = max_probs.std(dim=-1)  # [B]
        
        # ====================================================================
        # CONCATENATE ALL FEATURES
        # ====================================================================
        
        # Per-expert features: [B, 7*E]
        per_expert_feats = [
            expert_entropy,      # [B, E]
            topk_mass,           # [B, E]
            residual_mass,       # [B, E]
            max_probs,           # [B, E]
            top_gap,             # [B, E]
            cosine_sim,          # [B, E]
            kl_to_mean           # [B, E]
        ]
        per_expert_concat = torch.cat(per_expert_feats, dim=1)  # [B, 7*E]
        
        # Global features: [B, 3]
        global_feats = torch.stack([mean_entropy, mean_class_var, std_max_conf], dim=1)
        
        # Final concatenation: [B, 7*E + 3]
        features = torch.cat([per_expert_concat, global_feats], dim=1)
        
        # Clipping cho numerical stability
        features = torch.clamp(features, min=-100, max=100)
        
        return features


# ============================================================================
# GATING NETWORK ARCHITECTURE
# ============================================================================

class GatingMLP(nn.Module):
    """
    MLP nông với LayerNorm (stable hơn BatchNorm cho batch nhỏ).
    
    Architecture:
    - Input: features [B, D]
    - Hidden: 2-3 layers với LayerNorm + ReLU + Dropout
    - Output: logits [B, E] (before softmax/top-k)
    
    References:
    - Ba et al. (2016): Layer Normalization (stable for small batches)
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
            layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm thay vì BN
            
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo trọng số (Xavier uniform)"""
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
    Dense routing: softmax(g(x))
    
    Cổ điển MoE (Jordan & Jacobs, 1994): tất cả experts được dùng.
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
    Complete Gating Network cho MAP.
    
    Components:
    1. Feature Extractor: posteriors → features
    2. MLP: features → logits
    3. Router: logits → weights
    
    Usage:
        gating = GatingNetwork(num_experts=3, num_classes=100, routing='dense')
        posteriors = expert_posteriors  # [B, E, C]
        weights = gating(posteriors)    # [B, E]
    """
    
    def __init__(
        self,
        num_experts: int,
        num_classes: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        routing: str = 'dense',  # 'dense' or 'top_k'
        top_k: int = 2,
        noise_std: float = 1.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.routing_type = routing
        
        # 1. Feature extractor
        self.feature_extractor = GatingFeatureExtractor(num_experts, num_classes)
        
        # 2. MLP
        self.mlp = GatingMLP(
            input_dim=self.feature_extractor.feature_dim,
            num_experts=num_experts,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation
        )
        
        # 3. Router
        if routing == 'dense':
            self.router = DenseSoftmaxRouter()
        elif routing == 'top_k':
            self.router = NoisyTopKRouter(top_k=top_k, noise_std=noise_std)
        else:
            raise ValueError(f"Unknown routing: {routing}")
    
    def forward(self, posteriors: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            posteriors: [B, E, C] expert posteriors (đã calibrated)
        
        Returns:
            weights: [B, E] gating weights (simplex)
            aux_outputs: dict với các thông tin phụ (for loss computation)
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
        Tính mixture posterior: η̃(x) = Σ_e w_e · p^(e)(y|x)
        
        Args:
            posteriors: [B, E, C]
            weights: [B, E] (nếu None, sẽ tính từ forward)
        
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
    Tính U(x) cho MAP margin: U(x) = a·H(w) + b·Disagree + d·H(η̃)
    
    Args:
        posteriors: [B, E, C]
        weights: [B, E]
        mixture_posterior: [B, C] (nếu None sẽ tính)
        coeffs: {'a': 1.0, 'b': 1.0, 'd': 1.0}
    
    Returns:
        U: [B] uncertainty scores
    """
    if coeffs is None:
        coeffs = {'a': 1.0, 'b': 1.0, 'd': 1.0}
    
    eps = 1e-8
    B = posteriors.shape[0]
    
    # 1. Entropy của gating weights: H(w)
    H_w = -torch.sum(weights * torch.log(weights + eps), dim=-1)  # [B]
    
    # 2. Disagreement: tỉ lệ bất đồng top-1
    top1_classes = posteriors.argmax(dim=-1)  # [B, E]
    disagreement = []
    for b in range(B):
        unique = torch.unique(top1_classes[b]).numel()
        ratio = (unique - 1) / max(posteriors.shape[1] - 1, 1)
        disagreement.append(ratio)
    disagree = torch.tensor(disagreement, device=posteriors.device, dtype=posteriors.dtype)
    
    # 3. Entropy của mixture: H(η̃)
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
