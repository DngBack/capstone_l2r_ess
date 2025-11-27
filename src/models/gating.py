# src/models/gating.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """
    Configuration for selecting which features to include in GatingFeatureBuilder.

    Per-expert features (each produces [B, E] tensor):
    - entropy: Entropy of each expert's prediction distribution
    - topk_mass: Top-k probability mass per expert
    - residual_mass: Remaining probability mass (1 - topk_mass)
    - max_probs: Maximum probability (confidence) per expert
    - top_gap: Gap between top-1 and top-2 probabilities
    - cosine_sim: Cosine similarity to ensemble mean (agreement proxy)
    - kl_to_mean: KL divergence to ensemble mean (disagreement measure)

    Global features (each produces [B] tensor):
    - mean_entropy: Entropy of ensemble mean posterior
    - mean_class_var: Mean variance across classes (expert disagreement)
    - std_max_conf: Std of expert max probabilities (confidence dispersion)
    """

    # Per-expert features
    use_entropy: bool = True
    use_topk_mass: bool = True
    use_residual_mass: bool = True
    use_max_probs: bool = True
    use_top_gap: bool = True
    use_cosine_sim: bool = True
    use_kl_to_mean: bool = True

    # Global features
    use_mean_entropy: bool = True
    use_mean_class_var: bool = True
    use_std_max_conf: bool = True

    # Hyperparameters
    top_k: int = 5

    @property
    def enabled_per_expert_features(self) -> list:
        """Return list of enabled per-expert feature names."""
        features = []
        if self.use_entropy:
            features.append("entropy")
        if self.use_topk_mass:
            features.append("topk_mass")
        if self.use_residual_mass:
            features.append("residual_mass")
        if self.use_max_probs:
            features.append("max_probs")
        if self.use_top_gap:
            features.append("top_gap")
        if self.use_cosine_sim:
            features.append("cosine_sim")
        if self.use_kl_to_mean:
            features.append("kl_to_mean")
        return features

    @property
    def enabled_global_features(self) -> list:
        """Return list of enabled global feature names."""
        features = []
        if self.use_mean_entropy:
            features.append("mean_entropy")
        if self.use_mean_class_var:
            features.append("mean_class_var")
        if self.use_std_max_conf:
            features.append("std_max_conf")
        return features

    def compute_feature_dim(self, num_experts: int) -> int:
        """Compute output feature dimension based on enabled features."""
        per_expert_count = len(self.enabled_per_expert_features)
        global_count = len(self.enabled_global_features)
        return per_expert_count * num_experts + global_count

    def __repr__(self) -> str:
        per_expert = ", ".join(self.enabled_per_expert_features) or "none"
        global_feats = ", ".join(self.enabled_global_features) or "none"
        return f"FeatureConfig(per_expert=[{per_expert}], global=[{global_feats}], top_k={self.top_k})"


# Predefined feature configurations for common experiments
FEATURE_PRESETS = {
    "all": FeatureConfig(),  # All features enabled (default)
    # Per-expert only
    "per_expert_only": FeatureConfig(
        use_mean_entropy=False,
        use_mean_class_var=False,
        use_std_max_conf=False,
    ),
    # Global only
    "global_only": FeatureConfig(
        use_entropy=False,
        use_topk_mass=False,
        use_residual_mass=False,
        use_max_probs=False,
        use_top_gap=False,
        use_cosine_sim=False,
        use_kl_to_mean=False,
    ),
    # Uncertainty-based features
    "uncertainty_only": FeatureConfig(
        use_entropy=True,
        use_topk_mass=False,
        use_residual_mass=False,
        use_max_probs=True,
        use_top_gap=False,
        use_cosine_sim=False,
        use_kl_to_mean=False,
        use_mean_entropy=True,
        use_mean_class_var=False,
        use_std_max_conf=True,
    ),
    # Agreement/disagreement features
    "agreement_only": FeatureConfig(
        use_entropy=False,
        use_topk_mass=False,
        use_residual_mass=False,
        use_max_probs=False,
        use_top_gap=False,
        use_cosine_sim=True,
        use_kl_to_mean=True,
        use_mean_entropy=False,
        use_mean_class_var=True,
        use_std_max_conf=False,
    ),
    # Minimal (most important features)
    "minimal": FeatureConfig(
        use_entropy=True,
        use_topk_mass=False,
        use_residual_mass=False,
        use_max_probs=True,
        use_top_gap=False,
        use_cosine_sim=False,
        use_kl_to_mean=False,
        use_mean_entropy=False,
        use_mean_class_var=False,
        use_std_max_conf=False,
    ),
    # Confidence-based only
    "confidence_only": FeatureConfig(
        use_entropy=False,
        use_topk_mass=True,
        use_residual_mass=True,
        use_max_probs=True,
        use_top_gap=True,
        use_cosine_sim=False,
        use_kl_to_mean=False,
        use_mean_entropy=False,
        use_mean_class_var=False,
        use_std_max_conf=True,
    ),
}


class GatingFeatureBuilder:
    """
    Builds scalable, class-count-independent features from expert posteriors/logits.

    Features are selected via FeatureConfig, allowing flexible ablation studies.
    """

    def __init__(self, feature_config: Optional[FeatureConfig] = None, top_k: int = 5):
        """
        Args:
            feature_config: FeatureConfig specifying which features to include.
                           If None, uses default FeatureConfig (all features).
            top_k: Number of top probabilities to consider (for topk_mass feature).
        """
        if feature_config is None:
            self.config = FeatureConfig(top_k=top_k)
        else:
            self.config = feature_config
            if top_k != 5:
                self.config.top_k = top_k

        self.top_k = self.config.top_k

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (requires num_experts to compute)."""
        raise RuntimeError(
            "feature_dim requires num_experts. Use compute_feature_dim(num_experts)."
        )

    def compute_feature_dim(self, num_experts: int) -> int:
        """Compute output feature dimension based on enabled features."""
        return self.config.compute_feature_dim(num_experts)

    @torch.no_grad()
    def __call__(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_logits: Tensor of shape [B, E, C] (Batch, Experts, Classes)

        Returns:
            A feature tensor of shape [B, D] where D is the feature dimension.
        """
        """
        Args:
            expert_logits: Tensor of shape [B, E, C] (Batch, Experts, Classes)
        
        Returns:
            A feature tensor of shape [B, D] where D depends on enabled features.
        """
        # Ensure input is float32 for stable calculations
        expert_logits = expert_logits.float()
        B, E, C = expert_logits.shape

        # Use posteriors for probability-based features
        expert_posteriors = torch.softmax(expert_logits, dim=-1)

        # Track which computations we need for enabled features
        need_topk = (
            self.config.use_topk_mass
            or self.config.use_residual_mass
            or self.config.use_top_gap
        )
        need_max_probs = self.config.use_max_probs or self.config.use_std_max_conf
        need_mean_posterior = (
            self.config.use_cosine_sim
            or self.config.use_kl_to_mean
            or self.config.use_mean_entropy
            or self.config.use_mean_class_var
        )

        # Compute shared quantities only if needed
        mean_posterior = None
        if need_mean_posterior:
            mean_posterior = torch.mean(expert_posteriors, dim=1)  # [B, C]

        topk_vals = None
        if need_topk:
            topk_vals, _ = torch.topk(
                expert_posteriors, k=min(self.top_k, expert_posteriors.size(-1)), dim=-1
            )

        max_probs = None
        if need_max_probs:
            max_probs, _ = expert_posteriors.max(dim=-1)  # [B, E]

        # Build per-expert features list
        per_expert_feats = []

        # Feature 1: Entropy
        if self.config.use_entropy:
            entropy = -torch.sum(
                expert_posteriors * torch.log(expert_posteriors + 1e-8), dim=-1
            )  # [B, E]
            per_expert_feats.append(entropy)

        # Feature 2: Top-k probability mass
        if self.config.use_topk_mass:
            if topk_vals is None:
                topk_vals, _ = torch.topk(
                    expert_posteriors,
                    k=min(self.top_k, expert_posteriors.size(-1)),
                    dim=-1,
                )
            topk_mass = torch.sum(topk_vals, dim=-1)  # [B, E]
            per_expert_feats.append(topk_mass)

        # Feature 3: Residual mass
        if self.config.use_residual_mass:
            if topk_vals is None:
                topk_vals, _ = torch.topk(
                    expert_posteriors,
                    k=min(self.top_k, expert_posteriors.size(-1)),
                    dim=-1,
                )
                topk_mass = torch.sum(topk_vals, dim=-1)
            else:
                topk_mass = torch.sum(topk_vals, dim=-1)
            residual_mass = 1.0 - topk_mass  # [B, E]
            per_expert_feats.append(residual_mass)

        # Feature 4: Max probability (confidence)
        if self.config.use_max_probs:
            if max_probs is None:
                max_probs, _ = expert_posteriors.max(dim=-1)
            per_expert_feats.append(max_probs)

        # Feature 5: Top-1 - Top-2 gap
        if self.config.use_top_gap:
            if topk_vals is None:
                topk_vals, _ = torch.topk(
                    expert_posteriors,
                    k=min(self.top_k, expert_posteriors.size(-1)),
                    dim=-1,
                )
            if topk_vals.size(-1) >= 2:
                top1 = topk_vals[..., 0]
                top2 = topk_vals[..., 1]
                top_gap = top1 - top2  # [B, E]
            else:
                if max_probs is None:
                    max_probs, _ = expert_posteriors.max(dim=-1)
                top_gap = torch.zeros_like(max_probs)
            per_expert_feats.append(top_gap)

        # Feature 6: Cosine similarity to mean (agreement)
        if self.config.use_cosine_sim:
            if mean_posterior is None:
                mean_posterior = torch.mean(expert_posteriors, dim=1)
            cosine_sim = F.cosine_similarity(
                expert_posteriors, mean_posterior.unsqueeze(1), dim=-1
            )  # [B, E]
            per_expert_feats.append(cosine_sim)

        # Feature 7: KL divergence to mean (disagreement)
        if self.config.use_kl_to_mean:
            if mean_posterior is None:
                mean_posterior = torch.mean(expert_posteriors, dim=1)
            kl_to_mean = torch.sum(
                expert_posteriors
                * (
                    torch.log(expert_posteriors + 1e-8)
                    - torch.log(mean_posterior.unsqueeze(1) + 1e-8)
                ),
                dim=-1,
            )  # [B, E]
            per_expert_feats.append(kl_to_mean)

        # Concatenate per-expert features
        if per_expert_feats:
            per_expert_concat = torch.cat(
                per_expert_feats, dim=1
            )  # [B, num_per_expert*E]
        else:
            per_expert_concat = torch.empty(B, 0, device=expert_logits.device)

        # Build global features list
        global_feats = []

        # Global Feature 1: Mean posterior entropy
        if self.config.use_mean_entropy:
            if mean_posterior is None:
                mean_posterior = torch.mean(expert_posteriors, dim=1)
            mean_entropy = -torch.sum(
                mean_posterior * torch.log(mean_posterior + 1e-8), dim=-1
            )  # [B]
            global_feats.append(mean_entropy)

        # Global Feature 2: Mean class variance (expert disagreement)
        if self.config.use_mean_class_var:
            if E > 1:
                class_var = expert_posteriors.var(dim=1, unbiased=False)  # [B, C]
                mean_class_var = class_var.mean(dim=-1)  # [B]
            else:
                # Need shape for empty case
                if mean_posterior is None:
                    mean_posterior = torch.mean(expert_posteriors, dim=1)
                mean_class_var = torch.zeros(B, device=expert_logits.device)
            global_feats.append(mean_class_var)

        # Global Feature 3: Std of max probabilities (confidence dispersion)
        if self.config.use_std_max_conf:
            if max_probs is None:
                max_probs, _ = expert_posteriors.max(dim=-1)
            if E > 1:
                std_max_conf = max_probs.std(dim=-1, unbiased=False)  # [B]
            else:
                std_max_conf = torch.zeros(B, device=expert_logits.device)
            global_feats.append(std_max_conf)

        # Concatenate global features
        if global_feats:
            global_feats_tensor = torch.stack(global_feats, dim=1)  # [B, num_global]
        else:
            global_feats_tensor = torch.empty(B, 0, device=expert_logits.device)

        # Final concatenation
        if per_expert_concat.size(1) > 0 and global_feats_tensor.size(1) > 0:
            features = torch.cat([per_expert_concat, global_feats_tensor], dim=1)
        elif per_expert_concat.size(1) > 0:
            features = per_expert_concat
        elif global_feats_tensor.size(1) > 0:
            features = global_feats_tensor
        else:
            raise ValueError("At least one feature must be enabled in FeatureConfig!")

        return features


class GatingNet(nn.Module):
    """
    A simple MLP that takes gating features and outputs expert weights.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list = [128, 64],
        num_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Gating features of shape [B, D]

        Returns:
            Expert weights (before softmax) of shape [B, E]
        """
        return self.net(x)
