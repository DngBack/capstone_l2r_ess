# Feature Selection Implementation Summary

## Tổng quan

Đã refactor `GatingFeatureBuilder` để hỗ trợ flexible feature selection và tạo hệ thống ablation study để đánh giá ảnh hưởng của từng feature đến kết quả.

## Các thay đổi chính

### 1. Refactor `GatingFeatureBuilder` (`src/models/gating.py`)

**Thêm:**

- `FeatureConfig` dataclass: Config để bật/tắt từng feature
- `FEATURE_PRESETS`: Dictionary chứa các preset configurations phổ biến
- Flexible feature computation: Chỉ compute features được enable

**Features có thể control:**

- **Per-expert** (7 features): entropy, topk_mass, residual_mass, max_probs, top_gap, cosine_sim, kl_to_mean
- **Global** (3 features): mean_entropy, mean_class_var, std_max_conf

**Presets có sẵn:**

- `all`: Tất cả features (default)
- `minimal`: Chỉ entropy + max_probs
- `uncertainty_only`: Uncertainty-based features
- `agreement_only`: Agreement/disagreement features
- `confidence_only`: Confidence-based features
- `per_expert_only`: Chỉ per-expert features
- `global_only`: Chỉ global features

### 2. Update Training Script (`src/train/train_gating_map.py`)

**Thêm:**

- `_get_feature_config()` helper function để parse feature config từ CONFIG dict
- Support `feature_preset` và `feature_config` trong CONFIG
- CLI argument `--feature-preset` để chọn preset từ command line
- Dynamic feature dimension calculation

**Thay đổi:**

- Feature dimension được tính động dựa trên enabled features
- Feature builder được tạo với config từ CONFIG dict

### 3. Ablation Study Script (`scripts/ablation_feature_study.py`)

**Tính năng:**

- Chạy training với nhiều feature presets
- Tự động parse kết quả từ log files
- Export results ra CSV và JSON
- Hỗ trợ custom feature combinations
- Summary report với top configurations

**Usage:**

```bash
python scripts/ablation_feature_study.py \
    --dataset cifar100_lt_if100 \
    --presets all minimal uncertainty_only \
    --epochs 50
```

### 4. Documentation (`docs/FEATURE_ABLATION_GUIDE.md`)

Hướng dẫn chi tiết về:

- Cách sử dụng feature presets
- Tạo custom feature configs
- Chạy ablation studies
- Phân tích kết quả

## File Structure

```
src/models/gating.py                    # Refactored với FeatureConfig
src/train/train_gating_map.py          # Updated để support feature config
scripts/ablation_feature_study.py      # New: Ablation study script
docs/FEATURE_ABLATION_GUIDE.md         # New: User guide
```

## Example Usage

### Single experiment với preset

```bash
python src/train/train_gating_map.py \
    --dataset cifar100_lt_if100 \
    --feature-preset minimal
```

### Custom feature config trong code

```python
from src.models.gating import FeatureConfig

custom_config = FeatureConfig(
    use_entropy=True,
    use_max_probs=True,
    use_topk_mass=False,
    # ... other features
)

feature_builder = GatingFeatureBuilder(feature_config=custom_config)
```

### Ablation study

```bash
python scripts/ablation_feature_study.py \
    --dataset cifar100_lt_if100 \
    --presets all minimal uncertainty_only agreement_only
```

## Kết quả mong đợi

Sau khi chạy ablation study, bạn sẽ có:

1. **CSV file** với kết quả của tất cả experiments:

   - Balanced accuracy
   - Validation loss
   - Feature dimension
   - Enabled features

2. **JSON file** với full details

3. **Log files** cho mỗi experiment

4. **Summary report** với top configurations

## Next Steps

1. Chạy baseline với `all` features
2. Test các presets khác nhau
3. Chạy ablation study để so sánh
4. Phân tích kết quả để tìm optimal feature set
5. Fine-tune với custom combinations dựa trên insights

## Lưu ý kỹ thuật

- Feature dimension được tính tự động: `(per_expert_count * num_experts) + global_count`
- MLP input dimension phải match với feature dimension
- Chỉ compute features được enable để tối ưu performance
- Feature config được lưu trong checkpoint để reproducibility
