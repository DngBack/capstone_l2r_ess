# Demo: So Sánh Phương Pháp MoE + Plugin vs Paper Baseline

## 📋 Mục Đích

Demo này so sánh phương pháp của bạn (3 Experts + Gating + Plugin) với paper baseline (CE-only với Chow's rule) trên một ảnh tail class.

## 🚀 Cách Sử Dụng

### Option 1: Chạy Script Python (Nhanh nhất)

```bash
# Chạy với random tail class
python demo_single_image_comparison.py

# Chọn class cụ thể (ví dụ: class 95 - một tail class)
python demo_single_image_comparison.py --class-idx 95

# Thay đổi rejection threshold cho Chow's rule
python demo_single_image_comparison.py --rejection-threshold 0.3

# Kết hợp các options
python demo_single_image_comparison.py --class-idx 95 --rejection-threshold 0.5 --seed 42
```

**Output:**
- Visualization: `./results/demo_single_image/demo_comparison_class_{class_idx}.png`
- Results JSON: `./results/demo_single_image/demo_comparison_results_class_{class_idx}.json`

### Option 2: Chạy Jupyter Notebook

1. **Mở notebook:**
   ```bash
   jupyter notebook demo_comparison_single_image.ipynb
   ```

2. **Chạy tất cả cells:**
   - Cell 1: Setup imports
   - Cell 2: Configuration (có thể thay đổi `class_idx` và `rejection_threshold`)
   - Cell 3-5: Load data và models
   - Cell 6-7: Run inference
   - Cell 8-9: Visualization và comparison

3. **Kết quả:**
   - Visualization hiển thị ngay trong notebook
   - Files được save vào `./results/demo_single_image/`

## 📊 Nội Dung Demo

Demo sẽ:

1. **Load Models:**
   - CE expert (paper baseline)
   - 3 experts (CE, LogitAdjust, BalancedSoftmax)
   - Gating network
   - Plugin parameters (từ optimized results)

2. **Chọn Sample:**
   - Tự động chọn một ảnh từ tail class
   - Hoặc bạn có thể specify class index

3. **Chạy Inference:**

   **Paper Baseline:**
   - Forward pass qua CE expert
   - Chow's rule: `reject if max_prob < 1 - c`

   **Our Method:**
   - Forward pass qua 3 experts
   - Gating network để combine
   - Plugin với optimized (α, μ, c) parameters

4. **So Sánh và Visualize:**
   - Top-5 predictions của cả 2 methods
   - Expert contributions và gating weights
   - Confidence comparison
   - Prediction accuracy
   - Rejection decisions
   - Probability distributions

## 📈 Metrics Hiển Thị

- **Prediction:** Class được predict bởi mỗi method
- **Confidence:** Max probability
- **Rejection Decision:** Accept hay reject
- **Correctness:** Có đúng hay không
- **Top-5 Predictions:** Top 5 classes với probability cao nhất
- **Expert Predictions:** Predictions từ 3 experts
- **Gating Weights:** Trọng số của từng expert
- **Plugin Parameters:** α, μ, cost được sử dụng

## 🎯 Ví Dụ Kết Quả

```
📊 PAPER BASELINE (CE + Chow's Rule)
Prediction: Class 42 (beaver)
Confidence: 0.4523
Reject: YES
Correct: ❌

🚀 OUR METHOD (MoE + Gating + Plugin)
Expert Predictions: [95, 95, 94] (CE, LogitAdjust, BalancedSoftmax)
Gating Weights: [0.2, 0.3, 0.5]
Plugin Prediction: Class 95 (willow_tree)
Confidence: 0.6234
Reject: NO
Correct: ✅
```

## 💡 Giải Thích

**Tại sao phương pháp của bạn tốt hơn:**

1. **MoE (Mixture of Experts):**
   - Kết hợp 3 experts với chiến lược khác nhau
   - Mỗi expert có điểm mạnh riêng (head vs tail)

2. **Gating Network:**
   - Học cách weight từng expert dựa trên uncertainty/disagreement
   - Tự động điều chỉnh contribution của từng expert

3. **Plugin với Balanced Error:**
   - Optimize cho balanced error (fair với tail classes)
   - Parameters (α, μ) được tối ưu để balance head/tail performance
   - Rejection rule phù hợp với long-tail distribution

**So với Paper Baseline:**
- Paper baseline chỉ dùng 1 expert (CE)
- Chow's rule không tối ưu cho balanced error
- Không có cơ chế để handle tail classes đặc biệt

## 📁 Files

- `demo_single_image_comparison.py`: Script Python chính
- `demo_comparison_single_image.ipynb`: Jupyter notebook
- `create_demo_notebook.py`: Script để tạo notebook
- `DEMO_README.md`: File này

## 🔧 Requirements

Đảm bảo bạn đã:
1. Train experts và gating network
2. Run plugin optimization (có file `results/ltr_plugin/{dataset}/ltr_plugin_gating_balanced.json`)
3. Có checkpoints ở:
   - `checkpoints/experts/{dataset}/best_ce_baseline.pth`
   - `checkpoints/experts/{dataset}/best_logitadjust_baseline.pth`
   - `checkpoints/experts/{dataset}/best_balsoftmax_baseline.pth`
   - `checkpoints/gating_map/{dataset}/final_gating.pth`

## 🐛 Troubleshooting

**Lỗi: "Plugin results not found"**
- Chạy `python run_balanced_plugin_gating.py` trước để generate plugin parameters

**Lỗi: "Checkpoint not found"**
- Đảm bảo đã train experts và gating network
- Kiểm tra đường dẫn trong code

**Lỗi: "No test samples found for class X"**
- Class đó không có trong test set
- Thử class khác hoặc để None để random

## 📝 Notes

- Demo sử dụng các functions có sẵn từ project
- Visualization được save tự động
- Results được export sang JSON để phân tích sau
- Có thể chạy nhiều lần với different seeds để xem các samples khác nhau

