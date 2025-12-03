# Giải Thích Chi Tiết: Balanced Error và Plugin (Balance Mode & Worst Mode)

## 📋 Mục Lục
1. [Balanced Error - Công Thức và Cách Tính](#1-balanced-error---công-thức-và-cách-tính)
2. [Code Tính Balanced Error](#2-code-tính-balanced-error)
3. [Plugin - Balance Mode (Algorithm 1)](#3-plugin---balance-mode-algorithm-1)
4. [Plugin - Worst Mode (Algorithm 2)](#4-plugin---worst-mode-algorithm-2)
5. [So Sánh Hai Modes](#5-so-sánh-hai-modes)

---

## 1. Balanced Error - Công Thức và Cách Tính

### 1.1. Định Nghĩa từ Paper

Theo paper "Learning to Reject Meets Long-Tail Learning" (ICLR 2024), **Balanced Error** được định nghĩa như sau:

Cho phân loại với **Learning to Reject (L2R)**:

```
R^rej_bal(h, r) = (1/K) * Σ_k P(y ≠ h(x) | r(x) = 0, y ∈ G_k) + c · P(r(x) = 1)
```

Trong đó:
- `K`: Số lượng groups (ví dụ: K=2 cho head/tail)
- `G_k`: Group thứ k (ví dụ: G_0=head, G_1=tail)
- `h(x)`: Classifier prediction
- `r(x)`: Rejector (0=accept, 1=reject)
- `c`: Rejection cost

**Phần chính của balanced error** (không tính cost term):
```
Balanced Error = (1/K) * Σ_k P(y ≠ h(x) | r(x) = 0, y ∈ G_k)
```

Đây là **trung bình của các conditional error rates** trên từng group, chỉ tính trên các samples **được accept** (không reject).

### 1.2. Ý Nghĩa

**Tại sao gọi là "Balanced"?**
- Không giống standard accuracy (bị ảnh hưởng bởi class imbalance)
- Balanced error **đối xử công bằng** với mỗi group: mỗi group có weight = 1/K
- Ví dụ với 2 groups: 
  - Head error: 10%
  - Tail error: 30%
  - **Balanced error = (10% + 30%) / 2 = 20%**

**Tại sao chỉ tính trên accepted samples?**
- Chỉ quan tâm đến **chất lượng dự đoán** trên các samples mà model tự tin (accept)
- Rejected samples không được sử dụng để tính error

---

## 2. Code Tính Balanced Error

### 2.1. Vị Trí Code Chính

Code tính balanced error nằm ở nhiều nơi, nhưng **implementation chính** có trong:

1. **`run_balanced_plugin_gating.py`** - dòng 434-504: Hàm `compute_metrics()`
2. **`src/models/ltr_plugin.py`** - dòng 309-399: Hàm `compute_selective_metrics()`
3. **`run_worst_plugin_gating.py`** - dòng 329-383: Hàm `compute_metrics()`

### 2.2. Chi Tiết Implementation

Dưới đây là code từ `run_balanced_plugin_gating.py` (dòng 434-504):

```python
@torch.no_grad()
def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    reject: torch.Tensor,
    class_to_group: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    # Bước 1: Tách accepted/rejected samples
    accept = ~reject
    if accept.sum() == 0:
        # Nếu reject hết, return worst case
        return {
            "balanced_error": 1.0,
            "worst_group_error": 1.0,
            ...
        }
    
    # Chỉ lấy accepted samples
    preds_a = preds[accept]
    labels_a = labels[accept]
    errors = (preds_a != labels_a).float()  # [N_accept]
    
    # Bước 2: Tính error cho từng group
    groups = class_to_group[labels_a]  # Group của mỗi accepted sample
    num_groups = int(class_to_group.max().item() + 1)
    
    group_errors = []
    
    # Tính conditional error rate cho từng group
    for g in range(num_groups):
        mask = groups == g  # Samples thuộc group g
        
        if mask.sum() == 0:
            # Không có accepted samples từ group này
            group_errors.append(1.0)  # Worst case
        else:
            # P(y ≠ h(x) | r(x) = 0, y ∈ G_k)
            # = số lỗi trong group g / tổng samples trong group g (accepted)
            num_errors_in_group = errors[mask].sum().item()
            num_accepted_in_group = mask.sum().item()
            conditional_error = num_errors_in_group / num_accepted_in_group
            group_errors.append(conditional_error)
    
    # Bước 3: Balanced Error = trung bình của group errors
    balanced_error = float(np.mean(group_errors))
    
    # Worst-group Error = max của group errors
    worst_group_error = float(np.max(group_errors))
    
    return {
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
        "group_errors": group_errors,
        ...
    }
```

### 2.3. Ví Dụ Cụ Thể

Giả sử có:
- **Head group (G_0)**: 1000 accepted samples, 100 errors → error = 10%
- **Tail group (G_1)**: 500 accepted samples, 150 errors → error = 30%

**Balanced Error** = (10% + 30%) / 2 = **20%**

**Worst-group Error** = max(10%, 30%) = **30%**

---

## 3. Plugin - Balance Mode (Algorithm 1)

### 3.1. Mục Tiêu

Optimize **Balanced Error** với constraint về rejection rate:

```
Minimize: R^rej_bal(h, r) = (1/K) * Σ_k P(y ≠ h(x) | r(x) = 0, y ∈ G_k) + c · P(r(x) = 1)
```

### 3.2. Decision Rules (Theorem 1)

Theo paper, Bayes-optimal classifier và rejector có dạng:

**Classifier:**
```
h*(x) = argmax_y (1/α[y]) · η_y(x)
```

**Rejector:**
```
r*(x) = 1  nếu  max_y(1/α[y]·η_y(x)) < Σ_y'(1/α[y'] - μ[y'])·η_y'(x) - c
```

Trong đó:
- `η_y(x)`: Mixture posterior probability của class y (từ gating network)
- `α[y]`: Group-level reweighting parameter (α_head, α_tail)
- `μ[y]`: Group-level threshold adjustment parameter
- `c`: Rejection cost

### 3.3. Algorithm 1: Power Iteration

**Code location**: `src/models/ltr_plugin.py` - class `LtRPowerIterOptimizer` (dòng 420-644)

**Thuật toán:**

```
For mỗi μ trong grid search:
    α^(0) ← khởi tạo (dựa trên class priors)
    
    For m = 0 to M-1:  # Power iteration
        # Bước 1: Construct classifier và rejector với α^(m)
        h^(m+1)(x) = argmax_y (1/α^(m)[y]) · p_y(x)
        r^(m+1)(x) = 1 nếu max_y(...) < threshold
        
        # Bước 2: Update α dựa trên empirical coverage
        α^(m+1)_k = K * P(y ∈ G_k, r^(m+1)(x) = 0)
        # Tức là: α_k = K * (tỷ lệ samples từ group k được accept)
    
    # Bước 3: Evaluate objective với (h^(M), r^(M))
    objective = balanced_error + c * (1 - coverage)
    
# Trả về best (α, μ, c) có objective thấp nhất
```

### 3.4. Chi Tiết Code Implementation

#### 3.4.1. Initialize Alpha

```python
def _initialize_alpha(self, labels, class_to_group, sample_weights):
    """Khởi tạo α^(0) dựa trên class priors."""
    num_groups = class_to_group.max().item() + 1
    alpha = np.zeros(num_groups)
    
    for g in range(num_groups):
        # Tính tỷ lệ samples thuộc group g
        group_mask = class_to_group[labels] == g
        proportion = group_mask.sum().float().item() / len(labels)
        
        # α_k = K * proportion (để đảm bảo α ∈ (0, K))
        alpha[g] = num_groups * proportion
    
    return alpha
```

#### 3.4.2. Update Alpha from Coverage

```python
def _update_alpha_from_coverage(self, reject, labels, class_to_group):
    """Update α dựa trên empirical coverage."""
    num_groups = class_to_group.max().item() + 1
    alpha = np.zeros(num_groups)
    accept = ~reject
    N = len(labels)
    
    for g in range(num_groups):
        # Tìm samples từ group g
        in_group = class_to_group[labels] == g
        
        # Tìm samples từ group g được accept
        accepted_in_group = accept & in_group
        
        # α_k^(m+1) = K * P(y ∈ G_k, r(x) = 0)
        # = K * (số samples accepted từ group g / tổng samples)
        empirical_coverage = accepted_in_group.sum().float().item() / N
        alpha[g] = num_groups * empirical_coverage
    
    return alpha
```

#### 3.4.3. Power Iteration Loop

```python
def search(self, plugin, mixture_posterior, labels, ...):
    """Power iteration để tìm optimal (α, μ, c)."""
    
    # Grid search over μ và c
    for mu, cost in search_grid:
        # Khởi tạo α
        alpha = self._initialize_alpha(labels, class_to_group)
        
        # Power iteration
        for m in range(self.num_iters):
            # Set parameters
            plugin.set_parameters(alpha=alpha, mu=mu, cost=cost)
            
            # Construct (h, r)
            predictions = plugin.predict_class(mixture_posterior)
            reject = plugin.predict_reject(mixture_posterior)
            
            # Update α
            alpha_new = self._update_alpha_from_coverage(
                reject, labels, class_to_group
            )
            
            # Damping để ổn định
            alpha = (1 - damping) * alpha + damping * alpha_new
            
            # Check convergence
            if np.abs(alpha_new - alpha).max() < 1e-4:
                break
        
        # Evaluate objective
        metrics = compute_selective_metrics(...)
        objective = metrics['balanced_error'] + cost * (1 - metrics['coverage'])
        
        # Track best
        if objective < best_objective:
            best_result = (alpha, mu, cost)
    
    return best_result
```

### 3.5. Ví Dụ Workflow

**Input:**
- Mixture posterior từ gating network: `η(x) = [0.6, 0.3, 0.1]` (3 classes)
- Labels: `y = 0`
- Group mapping: `[0=head, 1=head, 2=tail]`

**Initialization:**
- `α^(0) = [1.5, 0.5]` (head có nhiều samples hơn tail)

**Iteration 1:**
- Classifier: `h(x) = argmax_y (1/α[y]) * η_y = argmax([0.4, 0.2, 2.0]) = 2`
- Rejector: tính threshold và quyết định reject hay không
- Update α dựa trên coverage thực tế

**Convergence:**
- α hội tụ về `[1.2, 0.8]` (tail được up-weight để balance error)

---

## 4. Plugin - Worst Mode (Algorithm 2)

### 4.1. Mục Tiêu

Optimize **Worst-group Error** (minimize maximum error across groups):

```
Minimize: R^rej_wst(h, r) = max_k P(y ≠ h(x) | r(x) = 0, y ∈ G_k) + c · P(r(x) = 1)
```

### 4.2. Algorithm 2: Exponentiated Gradient

**Code location**: `src/models/ltr_plugin.py` - class `LtRWorstGroupOptimizer` (dòng 1065-1278)

**Thuật toán:**

```
β^(0) ← uniform (1/K cho mỗi group)

For t = 0 to T-1:
    # Bước 1: Solve cost-sensitive problem với β^(t)
    # Gọi Algorithm 1 với weighted objective: Σ_k β_k * e_k
    (h^(t), r^(t)) ← Algorithm1(β^(t), c)
    
    # Bước 2: Compute group errors trên validation set
    e_k^(t) ← P(y ≠ h^(t)(x) | r^(t)(x) = 0, y ∈ G_k)
    
    # Bước 3: Update β bằng exponentiated gradient
    β^(t+1)_k ∝ β^(t)_k * exp(ξ * e_k^(t))
    β^(t+1) ← normalize về simplex

# Trả về (h, r) có worst-group error thấp nhất
```

### 4.3. Chi Tiết Code Implementation

#### 4.3.1. Generalized Plugin với β

Trong worst mode, plugin sử dụng **β weights** để up-weight groups có error cao:

```python
class GeneralizedLtRPlugin(nn.Module):
    """Plugin với β weights cho worst-group optimization."""
    
    def _u_class(self) -> torch.Tensor:
        """u[y] = β[y] / α[y] - dùng cho classifier."""
        u_group = self.beta_group / self.alpha_group.clamp(min=eps)
        return u_group[self.class_to_group]
    
    def predict(self, posterior: torch.Tensor):
        """h(x) = argmax_y u[y] * p_y(x) = argmax_y (β[y]/α[y]) * p_y(x)"""
        u = self._u_class().unsqueeze(0)
        return (posterior * u).argmax(dim=-1)
```

#### 4.3.2. Exponentiated Gradient Update

```python
def search(self, plugin, posterior_s1, labels_s1, posterior_s2, labels_s2, ...):
    """Worst-group optimization với exponentiated gradient."""
    
    # Khởi tạo β uniform
    num_groups = self.config.num_groups
    beta = np.ones(num_groups) / num_groups  # [0.5, 0.5] cho 2 groups
    
    best_result = None
    best_worst_error = float('inf')
    
    # Outer loop: Exponentiated gradient
    for t in range(self.num_outer_iters):
        # Bước 1: Inner optimization với β^(t)
        # Gọi Algorithm 1 với weighted objective
        result = self.inner_optimizer.search(
            plugin, posterior_s1, labels_s1,
            beta=torch.tensor(beta, ...),  # Pass β weights
            ...
        )
        
        # Bước 2: Compute group errors trên S2
        plugin.set_parameters(
            alpha=result.alpha,
            mu=result.mu,
            cost=result.cost
        )
        
        predictions_s2 = plugin.predict_class(posterior_s2)
        reject_s2 = plugin.predict_reject(posterior_s2)
        
        # Tính group errors trên accepted samples
        group_errors = []
        for g in range(num_groups):
            group_mask = class_to_group[labels_s2] == g
            accepted_in_group = group_mask & (~reject_s2)
            
            if accepted_in_group.sum() > 0:
                errors = (predictions_s2[accepted_in_group] != 
                         labels_s2[accepted_in_group]).sum()
                group_error = errors.float() / accepted_in_group.sum().float()
            else:
                group_error = 1.0  # Worst case
            
            group_errors.append(group_error.item())
        
        worst_error = max(group_errors)
        
        # Track best
        if worst_error < best_worst_error:
            best_worst_error = worst_error
            best_result = result
        
        # Bước 3: Update β bằng exponentiated gradient
        # β^(t+1)_k ∝ β^(t)_k * exp(ξ * e_k^(t))
        beta_old = beta.copy()
        beta = beta * np.exp(self.learning_rate * np.array(group_errors))
        
        # Normalize về simplex
        beta = beta / beta.sum()
        
        # Early stopping nếu β hội tụ
        if np.abs(beta - beta_old).max() < 1e-6:
            break
    
    return best_result
```

### 4.4. Ví Dụ Workflow

**Iteration 0:**
- `β^(0) = [0.5, 0.5]` (uniform)
- Algorithm 1 tối ưu weighted objective: `0.5 * e_head + 0.5 * e_tail`
- Group errors: `e_head = 0.15`, `e_tail = 0.40`
- Worst error: `max(0.15, 0.40) = 0.40`

**Iteration 1:**
- Update β: `β^(1) ∝ [0.5, 0.5] * exp([0.15, 0.40]) = [0.58, 0.74]`
- Normalize: `β^(1) = [0.44, 0.56]` (tail được up-weight)
- Algorithm 1 tối ưu với `β^(1)`: tập trung vào tail error
- Group errors: `e_head = 0.18`, `e_tail = 0.35`
- Worst error: `max(0.18, 0.35) = 0.35` ✓ (tốt hơn!)

**Iteration 2:**
- Update β: `β^(2) ∝ [0.44, 0.56] * exp([0.18, 0.35]) = [0.52, 0.78]`
- Normalize: `β^(2) = [0.40, 0.60]` (tail còn được up-weight hơn)
- ...

**Convergence:**
- β hội tụ về `[0.35, 0.65]` (tail được up-weight nhiều hơn)
- Worst-group error giảm xuống `0.32`

---

## 5. So Sánh Hai Modes

### 5.1. Bảng So Sánh

| Tiêu chí | Balance Mode (Algorithm 1) | Worst Mode (Algorithm 2) |
|----------|---------------------------|-------------------------|
| **Objective** | Minimize balanced error: `(1/K) * Σ_k e_k` | Minimize worst-group error: `max_k e_k` |
| **Parameters** | `α` (coverage), `μ` (threshold) | `α`, `μ`, `β` (group weights) |
| **Algorithm** | Power iteration trên `α` | Exponentiated gradient trên `β` + Power iteration |
| **Focus** | Cân bằng error giữa các groups | Tập trung vào group có error cao nhất |
| **Use Case** | Khi muốn fair performance | Khi muốn guarantee cho worst-case |

### 5.2. Ví Dụ Số

**Giả sử:**
- Head error: 15%
- Tail error: 35%

**Balance Mode:**
- Balanced error = (15% + 35%) / 2 = **25%**
- Có thể hy sinh một chút head error để giảm tail error

**Worst Mode:**
- Worst-group error = max(15%, 35%) = **35%**
- Tập trung giảm tail error (35%) xuống, có thể head error tăng lên 18%
- Kết quả: worst-group error = **32%** (tốt hơn!)

### 5.3. Khi Nào Dùng Mode Nào?

**Dùng Balance Mode khi:**
- Muốn fair performance across all groups
- Chấp nhận trade-off: một số groups tốt hơn, một số groups kém hơn
- Đánh giá bằng balanced error

**Dùng Worst Mode khi:**
- Cần guarantee cho worst-case scenario
- Không thể chấp nhận một group có error quá cao
- Đánh giá bằng worst-group error (ví dụ: fairness constraints)

---

## 6. Tóm Tắt

### 6.1. Balanced Error

- **Công thức**: `(1/K) * Σ_k P(y ≠ h(x) | r(x) = 0, y ∈ G_k)`
- **Tính chất**: Trung bình của conditional error rates trên từng group
- **Code location**: `compute_metrics()` trong các file plugin

### 6.2. Balance Mode (Algorithm 1)

- **Mục tiêu**: Minimize balanced error
- **Algorithm**: Power iteration để tìm optimal `α`
- **Key idea**: Update `α` dựa trên empirical coverage
- **Code**: `LtRPowerIterOptimizer` trong `src/models/ltr_plugin.py`

### 6.3. Worst Mode (Algorithm 2)

- **Mục tiêu**: Minimize worst-group error
- **Algorithm**: Exponentiated gradient trên `β` + Power iteration trên `α`
- **Key idea**: Up-weight groups có error cao để tập trung optimize
- **Code**: `LtRWorstGroupOptimizer` trong `src/models/ltr_plugin.py`

---

## 7. References

- Paper: "Learning to Reject Meets Long-Tail Learning" (ICLR 2024)
- Code files:
  - `run_balanced_plugin_gating.py`
  - `run_worst_plugin_gating.py`
  - `src/models/ltr_plugin.py`
  - `src/metrics/reweighted_metrics.py`

