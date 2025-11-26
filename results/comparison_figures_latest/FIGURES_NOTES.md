# Ghi chú về các Biểu đồ So sánh (Comparison Figures)

Tài liệu này mô tả ý nghĩa và mục đích của từng biểu đồ so sánh các phương pháp LtR Plugin.

---

## Tổng quan

Các biểu đồ được chia thành 4 nhóm chính:

1. **Head & Tail Error** (4 biểu đồ): So sánh lỗi của từng nhóm (head/tail) theo rejection rate
2. **Head/Tail Ratio** (2 biểu đồ): So sánh tỉ lệ lỗi giữa head và tail
3. **Risk - Balanced Mode** (2 biểu đồ): So sánh risk (error) cho các phương pháp balanced
4. **Risk - Worst-group Mode** (2 biểu đồ): So sánh risk (error) cho các phương pháp worst-group

---

## Nhóm 1: Head & Tail Error (4 biểu đồ)

### Figure 1: Balanced Mode - Head Error vs Rejection Rate

**File:** `fig_balanced_head_error.png`

**Mục đích:**

- So sánh lỗi của nhóm **Head** (các lớp có nhiều dữ liệu) khi sử dụng các phương pháp **Balanced mode**
- Xem phương pháp nào giảm head error hiệu quả hơn khi tăng rejection rate

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- Đường càng thấp càng tốt → phương pháp đó có head error thấp hơn
- Độ dốc của đường cho thấy tốc độ giảm error khi reject thêm samples
- So sánh giữa single expert (CE, LogitAdjust, BalSoftmax) và ensemble methods (Uniform, MoE)

---

### Figure 2: Balanced Mode - Tail Error vs Rejection Rate

**File:** `fig_balanced_tail_error.png`

**Mục đích:**

- So sánh lỗi của nhóm **Tail** (các lớp có ít dữ liệu) khi sử dụng các phương pháp **Balanced mode**
- Đánh giá khả năng xử lý long-tail distribution của từng phương pháp

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- Tail error thường cao hơn head error do ít dữ liệu training
- Phương pháp tốt sẽ có tail error thấp và giảm nhanh khi reject
- Ensemble methods (Uniform, MoE) thường có tail error thấp hơn single experts

---

### Figure 3: Worst-group Mode - Head Error vs Rejection Rate

**File:** `fig_worst_head_error.png`

**Mục đích:**

- So sánh lỗi của nhóm **Head** khi sử dụng các phương pháp **Worst-group mode**
- Xem phương pháp nào tối ưu cho worst-group có head error như thế nào

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- Worst-group mode tập trung vào giảm worst-group error, có thể hy sinh head error
- So sánh với balanced mode để thấy trade-off
- Head error trong worst-group mode có thể cao hơn balanced mode

---

### Figure 4: Worst-group Mode - Tail Error vs Rejection Rate

**File:** `fig_worst_tail_error.png`

**Mục đích:**

- So sánh lỗi của nhóm **Tail** khi sử dụng các phương pháp **Worst-group mode**
- Đánh giá hiệu quả của worst-group optimization cho tail classes

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- Worst-group mode nên có tail error thấp hơn balanced mode
- Đây là mục tiêu chính của worst-group optimization
- So sánh với Figure 2 để thấy sự cải thiện

---

## Nhóm 2: Head/Tail Ratio (2 biểu đồ)

### Figure 5: Balanced Mode - Head/Tail Error Ratio vs Rejection Rate

**File:** `fig_balanced_head_tail_ratio.png`

**Mục đích:**

- Đo lường **sự công bằng (fairness)** giữa head và tail groups
- Ratio = Head Error / Tail Error
- Ratio = 1.0 nghĩa là head và tail có error bằng nhau (công bằng nhất)

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- **Ratio < 1.0**: Tail error cao hơn head error (không công bằng, tail bị thiệt)
- **Ratio = 1.0**: Công bằng hoàn hảo (đường tham chiếu)
- **Ratio > 1.0**: Head error cao hơn tail error (hiếm xảy ra)
- Đường càng gần 1.0 càng tốt → phương pháp đó công bằng hơn
- Balanced mode nên có ratio gần 1.0 hơn worst-group mode

---

### Figure 6: Worst-group Mode - Head/Tail Error Ratio vs Rejection Rate

**File:** `fig_worst_head_tail_ratio.png`

**Mục đích:**

- Đo lường sự công bằng cho các phương pháp **Worst-group mode**
- Xem worst-group optimization có làm mất cân bằng giữa head và tail không

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- Worst-group mode có thể hy sinh head error để giảm tail error
- Ratio có thể < 1.0 (tail error cao) hoặc > 1.0 (head error cao)
- So sánh với Figure 5 để thấy trade-off giữa fairness và worst-group performance

---

## Nhóm 3: Risk - Balanced Mode (2 biểu đồ)

### Figure 7a: Balanced Mode - Balanced Risk vs Rejection Rate

**File:** `fig_balanced_balanced_risk.png`

**Mục đích:**

- So sánh **Balanced Error** (trung bình error của tất cả groups) cho các phương pháp **Balanced mode**
- Đây là metric chính để đánh giá hiệu suất tổng thể

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- **Balanced Error = (Head Error + Tail Error) / 2**
- Đường càng thấp càng tốt → phương pháp có balanced error thấp hơn
- Độ dốc cho thấy tốc độ cải thiện khi reject thêm samples
- Ensemble methods (Uniform, MoE) thường có balanced error thấp hơn single experts
- Đây là biểu đồ quan trọng nhất để so sánh overall performance

---

### Figure 7b: Balanced Mode - Worst Risk vs Rejection Rate

**File:** `fig_balanced_worst_risk.png`

**Mục đích:**

- So sánh **Worst-group Error** (error của group tệ nhất) cho các phương pháp **Balanced mode**
- Xem balanced optimization có giảm worst-group error không

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- **Worst-group Error = max(Head Error, Tail Error)**
- Đường càng thấp càng tốt
- So sánh với Figure 7a để thấy gap giữa balanced và worst-group error
- Gap nhỏ → phương pháp công bằng hơn
- Balanced mode có thể không tối ưu worst-group error bằng worst-group mode

---

## Nhóm 4: Risk - Worst-group Mode (2 biểu đồ)

### Figure 8a: Worst-group Mode - Balanced Risk vs Rejection Rate

**File:** `fig_worst_balanced_risk.png`

**Mục đích:**

- So sánh **Balanced Error** cho các phương pháp **Worst-group mode**
- Xem worst-group optimization có ảnh hưởng đến balanced error như thế nào

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- Worst-group mode tập trung vào giảm worst-group error, có thể hy sinh balanced error
- So sánh với Figure 7a để thấy trade-off
- Balanced error trong worst-group mode có thể cao hơn balanced mode
- Nhưng worst-group error (Figure 8b) sẽ thấp hơn

---

### Figure 8b: Worst-group Mode - Worst Risk vs Rejection Rate

**File:** `fig_worst_worst_risk.png`

**Mục đích:**

- So sánh **Worst-group Error** cho các phương pháp **Worst-group mode**
- Đây là metric chính để đánh giá hiệu quả của worst-group optimization

**Phương pháp được so sánh:**

- CE-only
- LogitAdjust-only
- BalSoftmax-only
- Uniform 3-Experts
- MoE (Gating)

**Ý nghĩa:**

- **Đây là mục tiêu chính của worst-group mode**: giảm worst-group error
- Đường càng thấp càng tốt
- So sánh với Figure 7b để thấy sự cải thiện
- Worst-group mode nên có worst-group error thấp hơn balanced mode
- Ensemble methods (Uniform, MoE) thường tốt hơn single experts

---

## Tổng kết và So sánh

### So sánh Balanced vs Worst-group Mode:

1. **Balanced Error:**

   - Balanced mode (Figure 7a) thường có balanced error thấp hơn worst-group mode (Figure 8a)
   - Vì balanced mode tối ưu trực tiếp cho balanced error

2. **Worst-group Error:**

   - Worst-group mode (Figure 8b) thường có worst-group error thấp hơn balanced mode (Figure 7b)
   - Vì worst-group mode tối ưu trực tiếp cho worst-group error

3. **Fairness (Head/Tail Ratio):**
   - Balanced mode (Figure 5) thường có ratio gần 1.0 hơn (công bằng hơn)
   - Worst-group mode (Figure 6) có thể hy sinh fairness để giảm worst-group error

### So sánh Single Expert vs Ensemble:

1. **Single Experts (CE-only, LogitAdjust-only, BalSoftmax-only):**

   - Đơn giản, dễ triển khai
   - Có thể có error cao hơn, đặc biệt cho tail classes
   - LogitAdjust-only có cả balanced và worst-group mode

2. **Ensemble Methods (Uniform 3-Experts, MoE Gating):**
   - Phức tạp hơn nhưng thường có error thấp hơn
   - Uniform: đơn giản, trung bình hóa các experts
   - MoE (Gating): thông minh hơn, học cách chọn expert phù hợp

### Khuyến nghị sử dụng:

- **Nếu quan tâm overall performance**: Dùng **Balanced mode** với **MoE (Gating)** hoặc **Uniform 3-Experts**
- **Nếu quan tâm worst-case performance**: Dùng **Worst-group mode** với **MoE (Gating)** hoặc **Uniform 3-Experts**
- **Nếu cần đơn giản**: Dùng **Balanced mode** với **BalSoftmax-only** hoặc **LogitAdjust-only**

---

## Lưu ý kỹ thuật

- Tất cả các biểu đồ sử dụng **test set** để đánh giá
- Rejection rate từ 0.0 (không reject) đến 1.0 (reject tất cả)
- Error metrics được tính trên các samples được **accept** (không reject)
- Head/Tail được định nghĩa dựa trên số lượng training samples (tail ≤ 20 samples)

---

_Tài liệu được tạo tự động từ script `scripts/plot_comparison_figures.py`_
