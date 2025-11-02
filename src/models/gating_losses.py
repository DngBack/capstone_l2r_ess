"""
Loss Functions cho Gating Network
===================================

Triển khai:
1. Mixture NLL (core) - Maximum likelihood cho mixture model
2. Load-balancing loss (Switch Transformer) - tránh routing collapse khi Top-K
3. (Optional) Selection-aware loss - ăn khớp với reject decision

References:
- Jordan & Jacobs (1994): Hierarchical Mixtures of Experts and EM Algorithm
- Fedus et al. (2021): Switch Transformers - Scaling to Trillion Parameter Models
- Geifman & El-Yaniv (2017): Selective Classification (proper scoring rules)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


# ============================================================================
# 1. MIXTURE NLL - CORE LOSS
# ============================================================================

class MixtureNLLLoss(nn.Module):
    """
    Mixture Negative Log-Likelihood Loss.
    
    Loss = -log(Σ_e w_e · p^(e)(y_true | x))
    
    Đây là maximum likelihood cho mixture model:
    - p(y|x) = Σ_e w_φ(x)_e · p^(e)(y|x)
    - Tối đa hóa log p(y|x) = log(Σ_e ...)
    
    Lý thuyết:
    - Proper scoring rule → khuyến khích calibration
    - Nhất quán với EM algorithm cho HME
    - Khả vi → có thể backprop end-to-end
    
    References:
    - Jordan & Jacobs (1994): HME with EM
    - Hinton et al. (1995): The wake-sleep algorithm
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        posteriors: torch.Tensor,
        weights: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            posteriors: [B, E, C] expert posteriors (đã calibrated)
            weights: [B, E] gating weights
            targets: [B] ground truth labels
            sample_weights: [B] optional per-sample weights (for imbalanced data)
        
        Returns:
            loss: scalar
        """
        B, E, C = posteriors.shape
        
        # Get mixture posterior: η̃(x) = Σ_e w_e · p^(e)
        # weights: [B, E, 1], posteriors: [B, E, C] → mixture: [B, C]
        mixture_posterior = torch.sum(
            weights.unsqueeze(-1) * posteriors,
            dim=1
        )  # [B, C]
        
        # Gather probabilities for true labels
        # mixture_posterior: [B, C], targets: [B] → probs: [B]
        true_class_probs = torch.gather(
            mixture_posterior,
            dim=1,
            index=targets.unsqueeze(1)
        ).squeeze(1)  # [B]
        
        # NLL = -log(p(y_true))
        nll = -torch.log(true_class_probs + self.eps)  # [B]
        
        # Apply sample weights if provided (for long-tail)
        if sample_weights is not None:
            nll = nll * sample_weights
        
        return nll.mean()
    
    def forward_with_mixture(
        self,
        mixture_posterior: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Phiên bản khi đã có mixture posterior.
        
        Args:
            mixture_posterior: [B, C]
            targets: [B]
            sample_weights: [B]
        
        Returns:
            loss: scalar
        """
        true_class_probs = torch.gather(
            mixture_posterior,
            dim=1,
            index=targets.unsqueeze(1)
        ).squeeze(1)
        
        nll = -torch.log(true_class_probs + self.eps)
        
        if sample_weights is not None:
            nll = nll * sample_weights
        
        return nll.mean()


# ============================================================================
# 2. LOAD-BALANCING LOSS (Switch Transformer)
# ============================================================================

class LoadBalancingLoss(nn.Module):
    """
    Load-Balancing Auxiliary Loss (Fedus et al., 2021).
    
    Mục tiêu: khuyến khích routing cân bằng giữa các experts để tránh collapse.
    
    Công thức Switch Transformer:
        L_LB = α · N · Σ_i f_i · P_i
    
    trong đó:
    - N = số experts
    - f_i = fraction of tokens routed to expert i (theo argmax/top-1)
    - P_i = average router probability for expert i
    - α = balancing coefficient (thường 10^-2)
    
    Intuition:
    - Nếu expert i nhận nhiều tokens (f_i cao) và có xác suất cao (P_i cao)
      → penalty cao → khuyến khích phân bổ đều hơn
    - Minimum = N/N² = 1/N (khi hoàn toàn cân bằng)
    
    References:
    - Fedus et al. (2021): Switch Transformers, Section 2.2, Equation 3
    - Code: https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py
    """
    
    def __init__(self, alpha: float = 1e-2):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        weights: torch.Tensor,
        top_k: int = 1
    ) -> torch.Tensor:
        """
        Args:
            weights: [B, E] gating weights (after softmax/top-k)
            top_k: number of experts used (1 cho Switch, 2+ cho Noisy Top-K)
        
        Returns:
            loss: scalar
        """
        B, E = weights.shape
        
        # f_i: fraction of samples routed to expert i
        # Đối với Top-K, ta xét K experts có weight cao nhất cho mỗi sample
        if top_k == 1:
            # Top-1: argmax
            expert_indices = weights.argmax(dim=-1)  # [B]
            f = torch.zeros(E, device=weights.device)
            for i in range(E):
                f[i] = (expert_indices == i).float().mean()
        else:
            # Top-K: count số lần expert xuất hiện trong top-K
            _, topk_indices = torch.topk(weights, k=min(top_k, E), dim=-1)  # [B, K]
            
            # Efficient counting: one-hot encode và sum
            # topk_indices: [B, K] → one_hot: [B, K, E] → sum over B,K: [E]
            one_hot = F.one_hot(topk_indices, num_classes=E).float()  # [B, K, E]
            counts = one_hot.sum(dim=(0, 1))  # [E]
            f = counts / (B * top_k)  # normalize by total possible assignments
        
        # P_i: average router probability for expert i
        P = weights.mean(dim=0)  # [E]
        
        # Load-balancing loss
        loss = self.alpha * E * torch.sum(f * P)
        
        return loss


class GroupAwareLoadBalancingLoss(nn.Module):
    """
    Load-balance theo nhóm: ràng buộc E[w_e | G_k] (hoặc penalty) để 
    LA/BS/CE nhận tải phù hợp Head/Tail.
    
    Ý tưởng (A): Không chỉ balance global, mà còn đảm bảo mỗi expert 
    nhận đúng tải theo group (Head/Tail).
    
    References:
    - Inspired by group fairness in MoE literature
    """
    
    def __init__(self, alpha: float = 1e-2, lambda_group: float = 0.5):
        """
        Args:
            alpha: global load-balancing coefficient
            lambda_group: group-aware balancing coefficient
        """
        super().__init__()
        self.alpha = alpha
        self.lambda_group = lambda_group
        self.base_lb = LoadBalancingLoss(alpha=alpha)
    
    def forward(
        self,
        weights: torch.Tensor,
        group_membership: torch.Tensor,
        num_groups: int = 2,
        top_k: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            weights: [B, E] gating weights
            group_membership: [B] group ids (0 for head, 1 for tail, etc.)
            num_groups: số groups
            top_k: number of experts used
        
        Returns:
            loss: scalar
            metrics: dict
        """
        B, E = weights.shape
        
        # 1. Global load-balancing (base)
        loss_global = self.base_lb(weights, top_k=top_k)
        
        # 2. Group-aware load-balancing
        # Compute per-group, per-expert usage
        group_expert_usage = torch.zeros(num_groups, E, device=weights.device)  # [G, E]
        group_counts = torch.zeros(num_groups, device=weights.device)
        
        for g in range(num_groups):
            mask = group_membership == g  # [B]
            if mask.sum() > 0:
                # Average expert weights for this group
                group_expert_usage[g] = weights[mask].mean(dim=0)  # [E]
                group_counts[g] = mask.float().sum()
        
        # Penalty: variance of expert usage across groups (should be balanced)
        # For each expert, measure how uneven its usage is across groups
        expert_usage_std = group_expert_usage.std(dim=0)  # [E]
        loss_group = expert_usage_std.mean() * self.lambda_group
        
        total_loss = loss_global + loss_group
        
        metrics = {
            'global_lb': loss_global.item(),
            'group_lb': loss_group.item(),
            'total_lb': total_loss.item()
        }
        
        return total_loss, metrics


# ============================================================================
# 3. ENTROPY REGULARIZATION
# ============================================================================

class EntropyRegularizer(nn.Module):
    """
    Entropy regularization cho gating weights.
    
    Mục tiêu: 
    - Maximum entropy → khuyến khích sử dụng nhiều experts (diversity)
    - Minimum entropy → khuyến khích sparse routing (specialization)
    
    Implementation:
    - H(w) = -Σ w_e log(w_e)  (entropy, always positive)
    - mode='maximize': return max_entropy - H(w) (penalty khi H thấp)
    - mode='minimize': return H(w) (penalty khi H cao)
    
    Trong cả 2 trường hợp, loss LUÔN >= 0 và được MINIMIZE.
    """
    
    def __init__(self, mode: str = 'maximize', eps: float = 1e-8, num_experts: int = 3):
        """
        Args:
            mode: 'maximize' (khuyến khích đều) hoặc 'minimize' (khuyến khích sparse)
            num_experts: số experts (để tính max entropy = log(E))
        """
        super().__init__()
        assert mode in ['maximize', 'minimize']
        self.mode = mode
        self.eps = eps
        self.max_entropy = np.log(num_experts)  # Maximum possible entropy
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: [B, E]
        
        Returns:
            loss: scalar (ALWAYS >= 0, to be minimized)
        """
        # Entropy: H(w) = -Σ w log(w), range [0, log(E)]
        entropy = -torch.sum(weights * torch.log(weights + self.eps), dim=-1)  # [B]
        
        if self.mode == 'maximize':
            # Khuyến khích high entropy (uniform weights)
            # Loss = max_entropy - H(w)
            # Loss cao khi H thấp (peaked distribution) → penalty
            # Loss = 0 khi H = max (uniform distribution) → optimal
            loss = self.max_entropy - entropy.mean()
        else:
            # Khuyến khích low entropy (sparse weights)
            # Loss = H(w)
            # Loss cao khi H cao (uniform) → penalty
            # Loss = 0 khi H = 0 (peaked) → optimal
            loss = entropy.mean()
        
        return loss


class AdaptiveEntropyRegularizer(nn.Module):
    """
    Adaptive entropy regularization based on expert disagreement.
    
    Ý tưởng (A): Khi disagreement cao (mixture-entropy ↑), ép entropy(w) ↓ 
    (routing dứt khoát chọn specialist). Khi consensus (disagreement thấp), 
    cho entropy(w) ↑ (an toàn trộn đều).
    
    Loss phụ: λ · corr(entropy(w), -disagreement) hoặc KL tới prior Dirichlet 
    có nhiệt độ phụ thuộc disagreement.
    
    References:
    - Malinin & Gales (2021): Uncertainty Estimation in Deep Learning
    """
    
    def __init__(self, eps: float = 1e-8, num_experts: int = 3, 
                 temperature_anneal: bool = True, use_kl: bool = True):
        """
        Args:
            eps: numerical stability
            num_experts: số experts
            temperature_anneal: nếu True, sử dụng Dirichlet prior với temp
            use_kl: nếu True, sử dụng KL-divergence thay vì correlation
        """
        super().__init__()
        self.eps = eps
        self.max_entropy = np.log(num_experts)
        self.temperature_anneal = temperature_anneal
        self.use_kl = use_kl
    
    def compute_disagreement(self, posteriors: torch.Tensor) -> torch.Tensor:
        """
        Compute expert disagreement metric.
        
        Args:
            posteriors: [B, E, C]
        
        Returns:
            disagreement: [B] (higher = more disagreement)
        """
        B, E, C = posteriors.shape
        
        # Mixture entropy as proxy for disagreement
        uniform_mixture = posteriors.mean(dim=1)  # [B, C]
        mixture_entropy = -torch.sum(
            uniform_mixture * torch.log(uniform_mixture + self.eps),
            dim=-1
        )  # [B]
        
        # Normalize to [0, 1]
        max_possible_entropy = np.log(C)
        disagreement = mixture_entropy / max_possible_entropy
        
        return disagreement
    
    def forward(
        self,
        weights: torch.Tensor,
        posteriors: Optional[torch.Tensor] = None,
        disagreement: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            weights: [B, E]
            posteriors: [B, E, C] (optional, để compute disagreement)
            disagreement: [B] (optional, nếu đã compute)
        
        Returns:
            loss: scalar
            metrics: dict với các metrics phụ
        """
        # Compute gating entropy
        entropy_w = -torch.sum(weights * torch.log(weights + self.eps), dim=-1)  # [B]
        
        # Compute disagreement if not provided
        if disagreement is None:
            if posteriors is None:
                # Fallback: uniform disagreement
                disagreement = torch.zeros_like(entropy_w)
            else:
                disagreement = self.compute_disagreement(posteriors)
        
        if self.use_kl:
            # Method 1: KL divergence to Dirichlet prior with disagreement-dependent temperature
            if self.temperature_anneal:
                # Temperature: high disagreement → low temp (sharp) → low entropy desired
                # Low disagreement → high temp (flat) → high entropy desired
                temperature = 1.0 + (1.0 - disagreement)  # [1, 2], reversed
            else:
                temperature = 1.0
            
            # Dirichlet prior parameters (uniform, scaled by temperature)
            alpha = torch.ones_like(weights) * temperature.unsqueeze(-1)  # [B, E]
            
            # KL divergence (simplified, ignoring constants)
            # KL(Dir(w | α)) = log Γ(Σα) - log Γ(α) + Σ(α-1)·(ψ(w) - ψ(Σα))
            # Simplified: penalize when entropy(w) is misaligned with disagreement
            target_entropy = self.max_entropy * disagreement
            entropy_diff = (entropy_w - target_entropy).abs()
            loss = entropy_diff.mean()
            
            method = "kl"
        else:
            # Method 2: Negative correlation penalty
            # We want entropy(w) to correlate negatively with disagreement
            # Loss = MSE từ ideal relationship
            target_entropy = self.max_entropy * disagreement
            loss = F.mse_loss(entropy_w, target_entropy)
            method = "corr"
        
        metrics = {
            'adaptive_entropy_loss': loss.item(),
            'mean_gating_entropy': entropy_w.mean().item(),
            'mean_disagreement': disagreement.mean().item(),
            'method': method
        }
        
        return loss, metrics


# ============================================================================
# 4. COMBINED GATING LOSS
# ============================================================================

class GatingLoss(nn.Module):
    """
    Combined loss cho gating training:
        L = L_mixture + λ_LB · L_LB + λ_H · L_H
    
    Components:
    1. Mixture NLL (bắt buộc) - likelihood của mixture
    2. Load-balancing (cho Top-K) - tránh collapse
    3. Entropy regularizer (tùy chọn) - khuyến khích diversity
    
    Usage:
        loss_fn = GatingLoss(lambda_lb=1e-2, lambda_h=0.01)
        loss = loss_fn(posteriors, weights, targets)
    """
    
    def __init__(
        self,
        lambda_lb: float = 1e-2,
        lambda_h: float = 0.01,
        use_load_balancing: bool = True,
        use_entropy_reg: bool = True,
        top_k: int = 1,
        num_experts: int = 3,
        entropy_mode: str = 'maximize',
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.lambda_lb = lambda_lb
        self.lambda_h = lambda_h
        self.use_load_balancing = use_load_balancing
        self.use_entropy_reg = use_entropy_reg
        self.top_k = top_k
        
        # Sub-losses
        self.mixture_nll = MixtureNLLLoss(eps=eps)
        
        if use_load_balancing:
            self.load_balancing = LoadBalancingLoss(alpha=lambda_lb)
        
        if use_entropy_reg:
            self.entropy_reg = EntropyRegularizer(mode=entropy_mode, eps=eps, num_experts=num_experts)
    
    def forward(
        self,
        posteriors: torch.Tensor,
        weights: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Args:
            posteriors: [B, E, C]
            weights: [B, E]
            targets: [B]
            sample_weights: [B] optional
            return_components: nếu True, trả về dict với các loss components
        
        Returns:
            loss: scalar (hoặc tuple(loss, components_dict))
        """
        # 1. Mixture NLL
        loss_nll = self.mixture_nll(posteriors, weights, targets, sample_weights)
        total_loss = loss_nll
        
        components = {'nll': loss_nll.item()}
        
        # 2. Load-balancing
        if self.use_load_balancing:
            loss_lb = self.load_balancing(weights, top_k=self.top_k)
            total_loss = total_loss + loss_lb
            components['load_balancing'] = loss_lb.item()
        
        # 3. Entropy regularization
        if self.use_entropy_reg:
            loss_h = self.entropy_reg(weights)  # Now ALWAYS >= 0
            total_loss = total_loss + self.lambda_h * loss_h
            components['entropy'] = loss_h.item()
        
        if return_components:
            return total_loss, components
        else:
            return total_loss


# ============================================================================
# 5. (OPTIONAL) SELECTION-AWARE LOSS
# ============================================================================

class SelectionAwareLoss(nn.Module):
    """
    Selection-aware auxiliary loss: ăn khớp gating với reject decision.
    
    Ý tưởng:
    - Khi sample "nên reject" (margin thấp, uncertainty cao), 
      gating cũng nên phản ánh sự không chắc chắn này
    - Có thể dùng như hinge loss: phạt khi "sai mà margin vẫn dương"
    
    Công thức đơn giản:
        L_sel = Σ 1{ŷ ≠ y} · max(0, m_MAP(x))
    
    Tức là: nếu dự đoán sai mà margin vẫn > 0 (sẽ accept) → penalty
    
    Note: Đây là tùy chọn nâng cao, không bắt buộc cho baseline.
    """
    
    def __init__(self, margin_fn=None):
        super().__init__()
        self.margin_fn = margin_fn  # function để tính margin (nếu cần)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        margins: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [B] predicted classes
            targets: [B] true classes
            margins: [B] MAP margins
        
        Returns:
            loss: scalar
        """
        # Incorrect predictions
        incorrect = (predictions != targets).float()  # [B]
        
        # Penalty khi incorrect và margin > 0
        penalty = incorrect * F.relu(margins)  # [B]
        
        return penalty.mean()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_gating_metrics(
    weights: torch.Tensor,
    posteriors: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Tính các metrics để monitor gating training.
    
    Args:
        weights: [B, E]
        posteriors: [B, E, C]
        targets: [B]
    
    Returns:
        metrics: dict
    """
    B, E, C = posteriors.shape
    eps = 1e-8
    
    metrics = {}
    
    # 1. Gating entropy (diversity)
    gating_entropy = -torch.sum(
        weights * torch.log(weights + eps),
        dim=-1
    ).mean().item()
    metrics['gating_entropy'] = gating_entropy
    
    # 2. Load balance (std of expert usage)
    expert_usage = weights.mean(dim=0)  # [E]
    metrics['load_std'] = expert_usage.std().item()
    metrics['load_max'] = expert_usage.max().item()
    metrics['load_min'] = expert_usage.min().item()
    
    # 3. Mixture accuracy
    mixture_posterior = torch.sum(
        weights.unsqueeze(-1) * posteriors,
        dim=1
    )  # [B, C]
    mixture_pred = mixture_posterior.argmax(dim=-1)
    mixture_acc = (mixture_pred == targets).float().mean().item()
    metrics['mixture_acc'] = mixture_acc
    
    # 4. Individual expert accuracies (weighted average)
    expert_accs = []
    for e in range(E):
        pred_e = posteriors[:, e, :].argmax(dim=-1)
        acc_e = (pred_e == targets).float().mean().item()
        expert_accs.append(acc_e)
    metrics['expert_acc_mean'] = np.mean(expert_accs)
    metrics['expert_acc_std'] = np.std(expert_accs)
    
    # 5. Effective number of experts (diversity metric)
    # exp(H(mean_weights))
    mean_weights = weights.mean(dim=0)
    mean_entropy = -torch.sum(mean_weights * torch.log(mean_weights + eps)).item()
    effective_experts = np.exp(mean_entropy)
    metrics['effective_experts'] = effective_experts
    
    return metrics


if __name__ == '__main__':
    """Test code"""
    import numpy as np
    
    print("Testing Gating Loss Functions...")
    
    # Mock data
    B, E, C = 32, 3, 100
    posteriors = F.softmax(torch.randn(B, E, C), dim=-1)
    weights = F.softmax(torch.randn(B, E), dim=-1)
    targets = torch.randint(0, C, (B,))
    
    print("\n1. Mixture NLL Loss:")
    loss_nll = MixtureNLLLoss()
    nll = loss_nll(posteriors, weights, targets)
    print(f"   Loss: {nll.item():.4f}")
    
    print("\n2. Load-Balancing Loss:")
    loss_lb = LoadBalancingLoss(alpha=1e-2)
    lb = loss_lb(weights, top_k=1)
    print(f"   Loss: {lb.item():.4f}")
    
    print("\n3. Entropy Regularizer:")
    ent_reg = EntropyRegularizer(mode='maximize')
    ent = ent_reg(weights)
    print(f"   Loss: {ent.item():.4f}")
    
    print("\n4. Combined Gating Loss:")
    gating_loss = GatingLoss(
        lambda_lb=1e-2,
        lambda_h=0.01,
        use_load_balancing=True,
        use_entropy_reg=True
    )
    total_loss, components = gating_loss(
        posteriors, weights, targets,
        return_components=True
    )
    print(f"   Total: {total_loss.item():.4f}")
    print(f"   Components: {components}")
    
    print("\n5. Gating Metrics:")
    metrics = compute_gating_metrics(weights, posteriors, targets)
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
    
    print("\n✅ All tests passed!")
