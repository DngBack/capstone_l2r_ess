# Feature Ablation Study Results Summary

## Dataset: CIFAR-100 Long-Tail (IF=100)

## Giải thích Thí nghiệm (Tiếng Việt)

### Mục đích của thí nghiệm

Thí nghiệm này được thực hiện để đánh giá tác động của từng nhóm features đầu vào đến hiệu suất của mạng Gating. Bằng cách loại bỏ từng nhóm features một cách có hệ thống, chúng ta có thể xác định được:

- Nhóm features nào quan trọng nhất cho hiệu suất
- Nhóm features nào có thể bỏ qua mà không ảnh hưởng nhiều
- Sự cân bằng giữa số lượng features và hiệu suất

### Cấu trúc Features

Features đầu vào cho Gating Network được chia thành 2 loại:

1. **Per-Expert Features** (Tính cho từng expert riêng biệt):

   - `entropy`: Entropy của phân phối xác suất từ expert đó
   - `topk_mass`: Tổng xác suất của top-k classes
   - `residual_mass`: Khối lượng xác suất còn lại
   - `max_probs`: Xác suất lớn nhất (confidence)
   - `top_gap`: Khoảng cách giữa top-1 và top-2 probabilities
   - `cosine_sim`: Cosine similarity giữa expert đó và mean của tất cả experts
   - `kl_to_mean`: KL divergence từ expert đó đến mean

2. **Global Features** (Tính chung cho tất cả experts):
   - `mean_entropy`: Entropy trung bình của tất cả experts
   - `mean_class_var`: Phương sai trung bình theo từng class
   - `std_max_conf`: Độ lệch chuẩn của max confidence qua các experts

### Mô tả các Preset

#### 1. `all` (Bản gốc - Baseline)

**Chứa gì:**

- Tất cả 7 per-expert features (tính cho 3 experts → 21 features)
- Tất cả 3 global features
- **Tổng cộng: 24 features**

**Đây là baseline đầy đủ nhất, sử dụng tất cả thông tin có sẵn.**

---

#### 2. `per_expert_only` (Chỉ Per-Expert Features)

**Chứa gì:**

- Tất cả 7 per-expert features: `entropy, topk_mass, residual_mass, max_probs, top_gap, cosine_sim, kl_to_mean`
- Tổng cộng: 21 features (7 features × 3 experts)

**Bỏ đi:**

- ❌ Tất cả 3 global features: `mean_entropy, mean_class_var, std_max_conf`

**Mục đích:** Đánh giá xem việc chỉ sử dụng thông tin từ từng expert riêng lẻ (không tổng hợp) có đủ không.

**Kết quả:** Đạt hiệu suất tốt nhất, cho thấy per-expert features là quan trọng nhất.

---

#### 3. `global_only` (Chỉ Global Features)

**Chứa gì:**

- Tất cả 3 global features: `mean_entropy, mean_class_var, std_max_conf`
- Tổng cộng: 3 features

**Bỏ đi:**

- ❌ Tất cả 7 per-expert features

**Mục đích:** Đánh giá xem chỉ với thông tin tổng hợp (không có thông tin riêng từ từng expert) có đủ không.

**Kết quả:** Hiệu suất kém, chứng tỏ per-expert features là rất quan trọng.

---

#### 4. `uncertainty_only` (Chỉ Uncertainty Features)

**Chứa gì:**

- 2 per-expert features về uncertainty: `entropy, max_probs`
- 2 global features về uncertainty: `mean_entropy, std_max_conf`
- Tổng cộng: 8 features (2 per-expert × 3 experts + 2 global)

**Bỏ đi:**

- ❌ `topk_mass, residual_mass, top_gap` (confidence features)
- ❌ `cosine_sim, kl_to_mean` (agreement features)
- ❌ `mean_class_var` (global agreement feature)

**Mục đích:** Đánh giá xem chỉ dùng uncertainty (độ không chắc chắn) có đủ không.

---

#### 5. `agreement_only` (Chỉ Agreement Features)

**Chứa gì:**

- 2 per-expert features về sự đồng thuận: `cosine_sim, kl_to_mean`
- 1 global feature: `mean_class_var`
- Tổng cộng: 7 features (2 per-expert × 3 experts + 1 global)

**Bỏ đi:**

- ❌ Tất cả uncertainty features: `entropy, max_probs, mean_entropy, std_max_conf`
- ❌ Tất cả confidence features: `topk_mass, residual_mass, top_gap`

**Mục đích:** Đánh giá xem chỉ dùng thông tin về sự đồng thuận giữa các experts có đủ không.

**Kết quả:** Hiệu suất kém nhất, cho thấy chỉ dựa vào agreement không đủ.

---

#### 6. `minimal` (Tối giản - Chỉ 2 Features Quan Trọng Nhất)

**Chứa gì:**

- 2 per-expert features: `entropy, max_probs`
- Tổng cộng: 6 features (2 features × 3 experts)

**Bỏ đi:**

- ❌ Tất cả global features
- ❌ 5 per-expert features khác: `topk_mass, residual_mass, top_gap, cosine_sim, kl_to_mean`

**Mục đích:** Tìm bộ features tối thiểu nhất nhưng vẫn đạt hiệu suất tốt.

**Kết quả:** Đạt hạng 2 cho balanced performance, rất hiệu quả về số lượng features (chỉ 6 features so với 24 của bản gốc).

---

#### 7. `confidence_only` (Chỉ Confidence Features)

**Chứa gì:**

- 4 per-expert features về confidence: `topk_mass, residual_mass, max_probs, top_gap`
- 1 global feature: `std_max_conf`
- Tổng cộng: 13 features (4 per-expert × 3 experts + 1 global)

**Bỏ đi:**

- ❌ Uncertainty features: `entropy, mean_entropy`
- ❌ Agreement features: `cosine_sim, kl_to_mean, mean_class_var`

**Mục đích:** Đánh giá xem chỉ dùng thông tin về độ tin cậy (confidence) có đủ không.

---

### Tóm tắt So sánh với Bản gốc (`all`)

| Preset             | Số Features | So với `all`    | Ghi chú                                |
| ------------------ | ----------- | --------------- | -------------------------------------- |
| `all`              | 24          | 100% (baseline) | Bản gốc đầy đủ                         |
| `per_expert_only`  | 21          | -3 features     | **Bỏ toàn bộ global features**         |
| `global_only`      | 3           | -21 features    | **Bỏ toàn bộ per-expert features**     |
| `uncertainty_only` | 8           | -16 features    | Chỉ giữ uncertainty features           |
| `agreement_only`   | 7           | -17 features    | Chỉ giữ agreement features             |
| `minimal`          | 6           | -18 features    | **Chỉ giữ 2 features quan trọng nhất** |
| `confidence_only`  | 13          | -11 features    | Chỉ giữ confidence features            |

---

### Top Configurations by Balanced Plugin AURC (Balanced)

_Lower AURC is better_

| Rank | Preset             | Balanced AURC | Worst-Group AURC | Baseline Balanced Error | Feature Dim |
| ---- | ------------------ | ------------- | ---------------- | ----------------------- | ----------- |
| 1    | `per_expert_only`  | 0.260810      | 0.303058         | 0.5355                  | 21          |
| 2    | `minimal`          | 0.261076      | 0.303966         | 0.5324                  | 6           |
| 3    | `all`              | 0.262064      | 0.305644         | 0.5378                  | 24          |
| 4    | `confidence_only`  | 0.263376      | 0.308098         | 0.5326                  | 13          |
| 5    | `uncertainty_only` | 0.264786      | 0.309688         | 0.5362                  | 8           |
| 6    | `global_only`      | 0.265446      | 0.313011         | 0.5391                  | 3           |
| 7    | `agreement_only`   | 0.266197      | 0.312501         | 0.5593                  | 7           |

**Key Findings:**

- Best performance: `per_expert_only` preset with AURC = 0.260810
- Minimal preset (6 features) achieves second-best performance, demonstrating efficiency
- Full feature set (`all`, 24 features) ranks 3rd, showing that more features don't always guarantee better performance
- `agreement_only` performs worst, suggesting agreement features alone are insufficient

---

### Top Configurations by Worst Plugin AURC (Worst-Group)

_Lower AURC is better_

| Rank | Preset             | Worst-Group AURC | Balanced AURC | Baseline Balanced Error | Feature Dim |
| ---- | ------------------ | ---------------- | ------------- | ----------------------- | ----------- |
| 1    | `per_expert_only`  | 0.229808         | 0.218643      | 0.5355                  | 21          |
| 2    | `all`              | 0.230616         | 0.218512      | 0.5378                  | 24          |
| 3    | `confidence_only`  | 0.233423         | 0.220604      | 0.5326                  | 13          |
| 4    | `agreement_only`   | 0.237212         | 0.223545      | 0.5593                  | 7           |
| 5    | `global_only`      | 0.263641         | 0.233834      | 0.5391                  | 3           |
| 6    | `uncertainty_only` | 0.264590         | 0.235791      | 0.5362                  | 8           |
| 7    | `minimal`          | 0.265863         | 0.234639      | 0.5324                  | 6           |

**Key Findings:**

- `per_expert_only` again achieves the best worst-group performance (AURC = 0.229808)
- Full feature set (`all`) ranks 2nd, showing consistency across metrics
- `minimal` preset drops to 7th place for worst-group performance, indicating it may not capture worst-group characteristics as well
- Worst-group AURC values are generally lower than balanced AURC values, suggesting better performance when optimizing for worst-group

---

## Comparative Analysis

### Balanced Plugin vs Worst Plugin Performance

| Preset             | Balanced Plugin (Balanced AURC) | Balanced Plugin (Worst-Group AURC) | Worst Plugin (Worst-Group AURC) | Worst Plugin (Balanced AURC) |
| ------------------ | ------------------------------- | ---------------------------------- | ------------------------------- | ---------------------------- |
| `per_expert_only`  | **0.260810**                    | 0.303058                           | **0.229808**                    | 0.218643                     |
| `all`              | 0.262064                        | 0.305644                           | **0.230616**                    | 0.218512                     |
| `minimal`          | **0.261076**                    | 0.303966                           | 0.265863                        | 0.234639                     |
| `confidence_only`  | 0.263376                        | 0.308098                           | **0.233423**                    | 0.220604                     |
| `uncertainty_only` | 0.264786                        | 0.309688                           | 0.264590                        | 0.235791                     |
| `global_only`      | 0.265446                        | 0.313011                           | 0.263641                        | 0.233834                     |
| `agreement_only`   | 0.266197                        | 0.312501                           | **0.237212**                    | 0.223545                     |

### Observations

1. **Worst Plugin achieves better worst-group performance** than Balanced Plugin across all presets

   - Worst Plugin worst-group AURC: 0.229808 - 0.265863
   - Balanced Plugin worst-group AURC: 0.303058 - 0.313011
   - This is expected as Worst Plugin explicitly optimizes for worst-group error

2. **Feature efficiency**

   - `per_expert_only` (21 features) performs best overall
   - `minimal` (6 features) is efficient for balanced performance but less so for worst-group
   - Full feature set (`all`, 24 features) provides consistent performance but with higher dimensionality

3. **Preset rankings differ** between balanced and worst-group optimization
   - `minimal`: Ranks 2nd for balanced, 7th for worst-group
   - `uncertainty_only`: Ranks 5th for balanced, 6th for worst-group
   - Suggests different features are important for different objectives

---

## Recommendations

1. **For balanced performance**: Use `per_expert_only` or `minimal` preset
2. **For worst-group performance**: Use `per_expert_only` or `all` preset with Worst Plugin
3. **For efficiency**: Consider `minimal` preset (6 features) if balanced performance is the primary concern
4. **For robustness**: Use `all` or `per_expert_only` preset for consistent performance across metrics

---

_Generated from ablation study results on CIFAR-100 Long-Tail (IF=100)_
_Date: November 27, 2025_
