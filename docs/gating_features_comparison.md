# Gating Features Comparison

## 📊 Tổng quan thay đổi

Đã refactor `GatingFeatureExtractor` trong `src/models/gating_network_map.py` để sử dụng approach nhẹ và hiệu quả hơn, tương tự như `GatingFeatureBuilder` trong `src/models/gating.py`.

---

## 🔄 Comparison

### **Trước (Old Implementation)**

**Feature dimension:** `314` (3 experts × 100 classes)

```python
Features = [
    Flattened posteriors:    300 dims (E×C)  # Toàn bộ posteriors
    Per-expert:                9 dims (3×3)
    Global:                    5 dims
]
```

**Nhược điểm:**
- ❌ Phụ thuộc vào số classes → không scalable
- ❌ Quá nhiều redundant information (300/314 = 95% là posteriors)
- ❌ Overfit risk cao với dữ liệu nhỏ
- ❌ Computational expensive

---

### **Sau (New Implementation)**

**Feature dimension:** `24` (3 experts)

```python
Features = [
    Per-expert (7 × E):      21 dims
        - Entropy
        - Top-K mass
        - Residual mass  
        - Max confidence
        - Top1-Top2 gap
        - Cosine similarity to mean
        - KL divergence to mean
    
    Global (3):              3 dims
        - Mean entropy
        - Mean class variance
        - Std of max confidences
]
```

**Ưu điểm:**
- ✅ **Không phụ thuộc vào số classes** → scalable cho mọi dataset
- ✅ Chỉ extract statistics quan trọng → informative & compact
- ✅ **92% reduction** (24 vs 314) → ít overfitting, training nhanh hơn
- ✅ Tính toán nhanh hơn

---

## 📈 Impact

| Metric | Old | New | Change |
|--------|-----|-----|--------|
| Feature dim | 314 | 24 | **-92%** |
| Computations | High | Low | **Fast** |
| Scalability | No | Yes | ✅ |
| Info density | Low (95% redundant) | High | ✅ |

---

## 🎯 Features Extracted

### **Per-Expert Features (7 × E)**

1. **Entropy** [B, E]
   ```python
   H(p^e) = -Σ p(y|x) log p(y|x)
   ```
   - Cao → expert không chắc chắn

2. **Top-K mass** [B, E]
   ```python
   Σ_{i=1}^K p_i (top-K probabilities)
   ```
   - Concentration của probability mass

3. **Residual mass** [B, E]
   ```python
   1 - top-K mass
   ```
   - Long-tail probability

4. **Max confidence** [B, E]
   ```python
   max_y p(y|x)
   ```
   - Confidence của expert

5. **Top1-Top2 gap** [B, E]
   ```python
   p_1 - p_2
   ```
   - Margin/quyết định rõ ràng

6. **Cosine similarity** [B, E]
   ```python
   cos(p^e, mean(p))
   ```
   - Đồng thuận với ensemble

7. **KL divergence** [B, E]
   ```python
   KL(p^e || mean(p))
   ```
   - Disagreement với ensemble

### **Global Features (3)**

1. **Mean entropy** [B]
   ```python
   H(mean(p))
   ```
   - Ensemble uncertainty

2. **Mean class variance** [B]
   ```python
   mean(var(p, dim=experts))
   ```
   - Disagreement giữa experts

3. **Std of max confidences** [B]
   ```python
   std([max(p^1), ..., max(p^E)])
   ```
   - Confidence dispersion

---

## 💻 Usage

```python
# Usage không đổi
model = GatingNetwork(
    num_experts=3,
    num_classes=100,
    hidden_dims=[256, 128],
    routing='dense'  # or 'top_k'
)

# Forward pass
posteriors = torch.softmax(logits, dim=-1)  # [B, E, C]
weights, aux = model(posteriors)  # [B, E]

# Features được extract tự động bên trong
```

---

## ✅ Benefits

1. **Lightweight:** 24 dims vs 314 dims
2. **Scalable:** Không phụ thuộc C (num_classes)
3. **Informative:** Chỉ giữ statistics quan trọng
4. **Fast:** Ít computation hơn nhiều
5. **Stable:** Ít overfitting, numerical stable hơn

---

## 📝 References

- Approach từ `src/models/gating.py` (`GatingFeatureBuilder`)
- Được test trong `src/train/train_gating_only.py`
- Literature: Switch Transformers (Fedus et al., 2021)

