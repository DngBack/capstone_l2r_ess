# Feature Ablation Study Guide

Hướng dẫn sử dụng hệ thống flexible feature selection và ablation study cho Gating Network.

## Tổng quan

Hệ thống cho phép bạn:

1. **Bật/tắt từng feature** trong `GatingFeatureBuilder`
2. **Sử dụng preset configurations** (all, minimal, uncertainty_only, etc.)
3. **Chạy ablation study** tự động để so sánh các feature combinations

## Features Available

### Per-Expert Features (mỗi feature tạo [B, E] tensor)

1. **`entropy`**: Entropy của prediction distribution của mỗi expert
2. **`topk_mass`**: Top-k probability mass
3. **`residual_mass`**: Probability mass còn lại (1 - topk_mass)
4. **`max_probs`**: Maximum probability (confidence)
5. **`top_gap`**: Gap giữa top-1 và top-2 probabilities
6. **`cosine_sim`**: Cosine similarity tới ensemble mean (agreement proxy)
7. **`kl_to_mean`**: KL divergence tới ensemble mean (disagreement measure)

### Global Features (mỗi feature tạo [B] tensor)

1. **`mean_entropy`**: Entropy của ensemble mean posterior
2. **`mean_class_var`**: Mean variance across classes (expert disagreement)
3. **`std_max_conf`**: Std của expert max probabilities (confidence dispersion)

## Feature Presets

Các preset có sẵn trong `FEATURE_PRESETS`:

| Preset             | Description                         | Features Included                                          |
| ------------------ | ----------------------------------- | ---------------------------------------------------------- |
| `all`              | Tất cả features (default)           | 7 per-expert + 3 global = 10 total                         |
| `minimal`          | Chỉ features quan trọng nhất        | entropy, max_probs                                         |
| `uncertainty_only` | Chỉ uncertainty-based features      | entropy, max_probs, mean_entropy, std_max_conf             |
| `agreement_only`   | Chỉ agreement/disagreement features | cosine_sim, kl_to_mean, mean_class_var                     |
| `confidence_only`  | Chỉ confidence-based features       | topk_mass, residual_mass, max_probs, top_gap, std_max_conf |
| `per_expert_only`  | Chỉ per-expert features             | 7 per-expert features, không có global                     |
| `global_only`      | Chỉ global features                 | 3 global features, không có per-expert                     |

## Usage

### 1. Training với Feature Preset

Sử dụng `--feature-preset` argument:

```bash
# Sử dụng preset "minimal"
python src/train/train_gating_map.py \
    --dataset cifar100_lt_if100 \
    --routing dense \
    --feature-preset minimal

# Sử dụng preset "uncertainty_only"
python src/train/train_gating_map.py \
    --dataset cifar100_lt_if100 \
    --feature-preset uncertainty_only
```

### 2. Training với Custom Feature Config

Tạo custom config trong code:

```python
from src.models.gating import FeatureConfig, GatingFeatureBuilder

# Custom config: chỉ entropy và max_probs
custom_config = FeatureConfig(
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
)

feature_builder = GatingFeatureBuilder(feature_config=custom_config)
```

Hoặc update CONFIG dict trong `train_gating_map.py`:

```python
CONFIG["gating"]["feature_config"] = {
    "use_entropy": True,
    "use_max_probs": True,
    "use_topk_mass": False,
    # ... other features
}
```

### 3. Ablation Study Script

Chạy ablation study tự động để test nhiều feature combinations:

```bash
# Test tất cả presets
python scripts/ablation_feature_study.py \
    --dataset cifar100_lt_if100 \
    --presets all minimal uncertainty_only agreement_only

# Test với custom combinations
python scripts/ablation_feature_study.py \
    --dataset cifar100_lt_if100 \
    --presets all minimal \
    --include-custom \
    --epochs 50

# Test với top_k routing
python scripts/ablation_feature_study.py \
    --dataset cifar100_lt_if100 \
    --routing top_k \
    --presets all minimal confidence_only
```

**Kết quả sẽ được lưu:**

- CSV: `./results/ablation_feature_study/ablation_results_{dataset}_{timestamp}.csv`
- JSON: `./results/ablation_feature_study/ablation_results_{dataset}_{timestamp}.json`
- Logs: `./results/ablation_feature_study/logs/`

### 4. Analyzing Results

Load và phân tích kết quả:

```python
import pandas as pd

# Load results
df = pd.read_csv("./results/ablation_feature_study/ablation_results_cifar100_lt_if100_20241201_120000.csv")

# Sort by balanced accuracy
df_sorted = df.sort_values("balanced_acc", ascending=False)
print(df_sorted[["preset", "balanced_acc", "val_loss", "feature_dim"]].head(10))

# Compare feature dimensions vs accuracy
import matplotlib.pyplot as plt
plt.scatter(df["feature_dim"], df["balanced_acc"])
plt.xlabel("Feature Dimension")
plt.ylabel("Balanced Accuracy")
plt.title("Feature Dimension vs Performance")
plt.show()
```

## Feature Dimension Calculation

Feature dimension được tính tự động:

```python
feature_config = FeatureConfig(...)
num_experts = 3  # 3 experts
feature_dim = feature_config.compute_feature_dim(num_experts)
# = (số per-expert features enabled) * num_experts + (số global features enabled)
```

Ví dụ:

- `all` với 3 experts: 7 \* 3 + 3 = 24
- `minimal` với 3 experts: 2 \* 3 + 0 = 6
- `uncertainty_only` với 3 experts: 2 \* 3 + 2 = 8

## Best Practices

1. **Bắt đầu với presets**: Sử dụng các preset có sẵn để có baseline
2. **Incremental ablation**: Bắt đầu với `all`, sau đó loại bỏ từng nhóm features
3. **Document experiments**: Ghi lại feature config cho mỗi experiment
4. **Monitor feature dimension**: Feature dimension ảnh hưởng đến model capacity

## Custom Presets

Thêm custom preset vào `FEATURE_PRESETS` trong `src/models/gating.py`:

```python
FEATURE_PRESETS["my_custom"] = FeatureConfig(
    use_entropy=True,
    use_max_probs=True,
    # ... other features
)
```

## Example: Complete Ablation Workflow

```bash
# 1. Baseline với all features
python src/train/train_gating_map.py \
    --dataset cifar100_lt_if100 \
    --feature-preset all \
    --epochs 100 \
    --log-file logs/baseline_all.log

# 2. Test minimal features
python src/train/train_gating_map.py \
    --dataset cifar100_lt_if100 \
    --feature-preset minimal \
    --epochs 100 \
    --log-file logs/baseline_minimal.log

# 3. Run full ablation study
python scripts/ablation_feature_study.py \
    --dataset cifar100_lt_if100 \
    --presets all minimal uncertainty_only agreement_only confidence_only \
    --epochs 50

# 4. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('./results/ablation_feature_study/ablation_results_*.csv')
print(df.sort_values('balanced_acc', ascending=False))
"
```

## Troubleshooting

**Lỗi: Feature dimension mismatch**

- Đảm bảo `feature_config` được pass đúng vào `GatingFeatureBuilder`
- Check rằng MLP input_dim match với feature_dim

**Lỗi: Empty features**

- Phải có ít nhất 1 feature enabled
- Check `FeatureConfig` có ít nhất 1 flag = True

**Lỗi: Preset not found**

- Check preset name trong `FEATURE_PRESETS.keys()`
- Hoặc sử dụng `None` để dùng default (all features)
