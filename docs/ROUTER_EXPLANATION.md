# Giải Thích Cách Hoạt Động Của Router

Router là thành phần cuối cùng trong Gating Network, có nhiệm vụ chuyển đổi **gating logits** thành **expert weights** (trọng số cho từng expert).

## Tổng Quan

```
MLP Output (logits) → Router → Expert Weights (simplex)
[B, E]              →        → [B, E] (tổng = 1)
```

## 1. DenseSoftmaxRouter (Dense Routing)

### Cách hoạt động:
```python
weights = softmax(logits, dim=-1)
```

### Chi tiết:
- **Input**: `logits [B, E]` - gating scores từ MLP (chưa normalize)
- **Output**: `weights [B, E]` - trọng số cho mỗi expert (tổng = 1, tất cả experts đều được dùng)

### Ví dụ:
```python
# Giả sử có 3 experts (E=3)
logits = torch.tensor([[2.0, 1.0, 0.5],    # Sample 1
                       [0.1, 3.0, 1.5]])   # Sample 2

# Softmax
weights = F.softmax(logits, dim=-1)
# weights = [[0.576, 0.212, 0.128],   # Sample 1: Expert 0 được ưu tiên
#            [0.090, 0.665, 0.245]]   # Sample 2: Expert 1 được ưu tiên

# Tổng mỗi hàng = 1.0 ✓
```

### Đặc điểm:
- ✅ **Dense**: Tất cả experts đều nhận trọng số > 0
- ✅ **Đơn giản**: Chỉ cần softmax
- ✅ **Phù hợp**: Khi muốn combine tất cả experts (mixture model)

---

## 2. NoisyTopKRouter (Sparse Routing)

### Cách hoạt động:
```python
1. Thêm Gaussian noise vào logits (chỉ khi training)
2. Chọn Top-K experts có logits cao nhất
3. Softmax chỉ trên Top-K experts
4. Các experts còn lại nhận weight = 0
```

### Chi tiết từng bước:

#### Bước 1: Thêm Noise (Training only)
```python
if self.training:
    noise = torch.randn_like(logits) * noise_std  # Gaussian noise
    noisy_logits = logits + noise
else:
    noisy_logits = logits  # Không thêm noise khi inference
```

**Mục đích**: 
- **Exploration**: Giúp model khám phá các expert khác nhau
- **Load balancing**: Tránh một expert bị overused

#### Bước 2: Top-K Selection
```python
top_k = min(self.top_k, E)  # Ví dụ: top_k = 2
topk_logits, topk_indices = torch.topk(noisy_logits, k=top_k, dim=-1)
```

**Ví dụ**:
```python
# Giả sử có 3 experts, top_k = 2
noisy_logits = [[2.1, 1.2, 0.3],   # Sample 1
                [0.2, 3.1, 1.4]]   # Sample 2

# Top-2 selection
topk_logits = [[2.1, 1.2],         # Sample 1: chọn expert 0, 1
               [3.1, 1.4]]         # Sample 2: chọn expert 1, 2

topk_indices = [[0, 1],            # Sample 1: indices của expert 0, 1
                [1, 2]]             # Sample 2: indices của expert 1, 2
```

#### Bước 3: Softmax trên Top-K
```python
topk_weights = F.softmax(topk_logits, dim=-1)  # [B, K]
```

**Ví dụ**:
```python
topk_weights = [[0.711, 0.289],    # Sample 1: Expert 0 (71%), Expert 1 (29%)
                [0.838, 0.162]]    # Sample 2: Expert 1 (84%), Expert 2 (16%)
```

#### Bước 4: Scatter về full dimension
```python
weights = torch.zeros_like(logits)  # [B, E] - tất cả = 0
weights.scatter_(dim=1, index=topk_indices, src=topk_weights)
```

**Ví dụ**:
```python
# Kết quả cuối cùng
weights = [[0.711, 0.289, 0.000],  # Sample 1: Expert 0, 1 được dùng, Expert 2 = 0
           [0.000, 0.838, 0.162]]   # Sample 2: Expert 1, 2 được dùng, Expert 0 = 0
```

### Đặc điểm:
- ✅ **Sparse**: Chỉ K experts được dùng (K < E)
- ✅ **Hiệu quả**: Giảm computation (chỉ cần forward K experts)
- ✅ **Noise**: Giúp exploration và load balancing
- ⚠️ **Phức tạp hơn**: Cần thêm noise và top-k selection

---

## So Sánh 2 Router

| Đặc điểm | DenseSoftmaxRouter | NoisyTopKRouter |
|----------|-------------------|-----------------|
| **Số experts dùng** | Tất cả (E) | Chỉ Top-K |
| **Computation** | O(E) | O(K) - nhanh hơn |
| **Sparsity** | Dense (tất cả > 0) | Sparse (chỉ K > 0) |
| **Noise** | Không | Có (training) |
| **Use case** | Mixture model | Sparse MoE |

---

## Luồng Hoạt Động Tổng Thể

```
Expert Posteriors [B, E, C]
    ↓
Feature Extractor
    ↓
Features [B, D]
    ↓
MLP (GatingMLP)
    ↓
Gating Logits [B, E]  ← Đây là input của Router
    ↓
Router (DenseSoftmaxRouter hoặc NoisyTopKRouter)
    ↓
Expert Weights [B, E] (simplex: tổng = 1)
    ↓
Mixture Posterior = Σ(weights[i] × posteriors[i])
```

---

## Code Example

```python
from src.models.gating_network_map import GatingNetwork

# Tạo gating network với Dense routing
gating_dense = GatingNetwork(
    num_experts=3,
    num_classes=100,
    routing='dense'  # DenseSoftmaxRouter
)

# Tạo gating network với Top-K routing
gating_topk = GatingNetwork(
    num_experts=3,
    num_classes=100,
    routing='top_k',  # NoisyTopKRouter
    top_k=2,         # Chỉ dùng 2 experts
    noise_std=1.0    # Độ lớn của noise
)

# Forward pass
posteriors = expert_posteriors  # [B, 3, 100]
weights, aux = gating_dense(posteriors)  # weights: [B, 3]

# weights sẽ có dạng:
# Dense: [[0.4, 0.3, 0.3], ...]  (tất cả > 0)
# TopK:  [[0.7, 0.3, 0.0], ...]  (chỉ 2 > 0)
```

---

## Tại Sao Cần Router?

1. **Normalization**: Logits từ MLP chưa được normalize → Router chuyển thành probability distribution
2. **Sparsity**: Top-K router giúp giảm computation (chỉ dùng K experts thay vì tất cả)
3. **Exploration**: Noise trong Top-K giúp model khám phá các expert khác nhau
4. **Load Balancing**: Noise giúp phân bố workload đều hơn giữa các experts



