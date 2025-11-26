# iNaturalist 2018 - Full Pipeline Commands

Tài liệu này chứa tất cả các câu lệnh cần thiết để chạy dự án từ đầu với dataset iNaturalist 2018.

## 📋 Tổng quan các bước

1. **Setup môi trường và download data**
2. **Tạo dataset splits**
3. **Train Expert Models** (CE, LogitAdjust, BalancedSoftmax)
4. **Train Gating Network**
5. **Run Plugin Methods** (Balanced & Worst-group)

---

## 🚀 Các câu lệnh chi tiết

### Bước 1: Setup thư mục và download data

```bash
# Di chuyển vào thư mục project
cd /path/to/capstone_l2r_ess

# Tạo thư mục data
mkdir -p data
cd data
```

#### Option A: Sử dụng aria2c (khuyến nghị - nhanh hơn)

```bash
# Download với aria2c (16 connections, nhanh hơn)
aria2c -x 16 -s 16 \
    https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz \
    https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz \
    https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz
```

#### Option B: Sử dụng wget (mặc định)

```bash
# Download với wget
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz
```

### Bước 2: Extract và cleanup

```bash
# Extract các file tar.gz
tar -xvzf train_val2018.tar.gz
tar -xvzf train2018.json.tar.gz
tar -xvzf val2018.json.tar.gz

# Xóa các file nén để tiết kiệm dung lượng
rm train_val2018.tar.gz train2018.json.tar.gz val2018.json.tar.gz

# Quay lại thư mục project root
cd ..
```

### Bước 3: Tạo dataset splits

```bash
# Tạo thư mục logs nếu chưa có
mkdir -p logs

# Chạy script tạo splits
python scripts/create_inaturalist_splits.py \
    --train-json data/train2018.json \
    --val-json data/val2018.json \
    --data-dir data/inaturalist2018/train_val2018 \
    --output-dir data/inaturalist2018_splits \
    --seed 42 \
    --expert-ratio 0.9 \
    --log-file logs/inaturalist2018_splits_$(date +%Y%m%d_%H%M%S).log
```

**Giải thích tham số:**
- `--train-json`: Đường dẫn đến file train2018.json
- `--val-json`: Đường dẫn đến file val2018.json
- `--data-dir`: Thư mục chứa ảnh (train_val2018)
- `--output-dir`: Thư mục output cho các splits
- `--seed`: Random seed (42)
- `--expert-ratio`: Tỷ lệ train cho expert (0.9 = 90%)
- `--log-file`: File log (tùy chọn)

### Bước 4: Train Expert Models

Train cả 3 experts (CE, LogitAdjust, BalancedSoftmax):

```bash
python train_experts.py \
    --dataset inaturalist2018 \
    --expert all \
    --log-file logs/experts_inaturalist2018_$(date +%Y%m%d_%H%M%S).log
```

**Train từng expert riêng lẻ (nếu cần):**

```bash
# Train CE expert
python train_experts.py \
    --dataset inaturalist2018 \
    --expert ce \
    --log-file logs/expert_ce_inat.log

# Train LogitAdjust expert
python train_experts.py \
    --dataset inaturalist2018 \
    --expert logitadjust \
    --log-file logs/expert_logitadjust_inat.log

# Train BalancedSoftmax expert
python train_experts.py \
    --dataset inaturalist2018 \
    --expert balsoftmax \
    --log-file logs/expert_balsoftmax_inat.log
```

**Quick test (2 epochs, batch size nhỏ hơn):**

```bash
python train_experts.py \
    --dataset inaturalist2018 \
    --expert ce \
    --epochs 2 \
    --batch-size 512 \
    --log-file logs/inat_test.log
```

**Override các tham số:**

```bash
python train_experts.py \
    --dataset inaturalist2018 \
    --expert all \
    --epochs 200 \
    --lr 0.4 \
    --batch-size 1024 \
    --log-file logs/experts_custom.log
```

### Bước 5: Train Gating Network

```bash
python -m src.train.train_gating_map \
    --dataset inaturalist2018 \
    --routing dense \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-3 \
    --lambda_lb 1e-2 \
    --log-file logs/gating_inaturalist2018_$(date +%Y%m%d_%H%M%S).log
```

**Các tùy chọn routing:**

```bash
# Dense routing (tất cả experts)
python -m src.train.train_gating_map \
    --dataset inaturalist2018 \
    --routing dense

# Top-k routing (chọn k experts tốt nhất)
python -m src.train.train_gating_map \
    --dataset inaturalist2018 \
    --routing top_k \
    --top_k 2
```

### Bước 6: Run Plugin Methods

#### 6a. Balanced Plugin với Gating (3 experts)

```bash
python run_balanced_plugin_gating.py --dataset inaturalist2018
```

#### 6b. Worst-group Plugin với Gating (3 experts)

```bash
python run_worst_plugin_gating.py --dataset inaturalist2018
```

#### 6c. Balanced Plugin CE-only (baseline, 1 expert)

```bash
python run_balanced_plugin_ce_only.py --dataset inaturalist2018
```

#### 6d. Worst-group Plugin CE-only (baseline, 1 expert)

```bash
python run_worst_plugin_ce_only.py --dataset inaturalist2018
```

---

## 📁 Cấu trúc thư mục sau khi chạy

```
capstone_l2r_ess/
├── data/
│   ├── inaturalist2018/
│   │   └── train_val2018/          # Ảnh dataset
│   ├── train2018.json             # Train annotations
│   ├── val2018.json                # Val annotations
│   └── inaturalist2018_splits/     # Generated splits
│       ├── train_indices.json
│       ├── expert_indices.json
│       ├── gating_indices.json
│       ├── val_indices.json
│       ├── test_indices.json
│       ├── tunev_indices.json
│       └── train_class_counts.json
├── checkpoints/
│   ├── experts/
│   │   └── inaturalist2018/
│   │       ├── best_ce_baseline.pth
│   │       ├── best_logitadjust_baseline.pth
│   │       ├── best_balsoftmax_baseline.pth
│   │       └── final_calibrated_*.pth
│   └── gating_map/
│       └── inaturalist2018/
│           └── final_gating.pth
├── outputs/
│   └── logits/
│       └── inaturalist2018/
│           ├── ce_baseline/
│           ├── logitadjust_baseline/
│           └── balsoftmax_baseline/
└── results/
    └── ltr_plugin/
        └── inaturalist2018/
            ├── ltr_plugin_gating_balanced.json
            ├── ltr_plugin_gating_worst.json
            ├── ltr_plugin_ce_only_balanced.json
            ├── ltr_plugin_ce_only_worst.json
            └── *.png (plots)
```

---

## ⚡ Chạy tự động với script

Để chạy tất cả các bước tự động, sử dụng script shell:

```bash
# Cấp quyền thực thi
chmod +x run_inaturalist2018_full_pipeline.sh

# Chạy script
bash run_inaturalist2018_full_pipeline.sh
```

Hoặc:

```bash
./run_inaturalist2018_full_pipeline.sh
```

---

## 🔍 Kiểm tra kết quả

### Kiểm tra experts đã train xong:

```bash
ls -lh checkpoints/experts/inaturalist2018/
```

### Kiểm tra logits đã export:

```bash
ls -lh outputs/logits/inaturalist2018/*/
```

### Kiểm tra gating model:

```bash
ls -lh checkpoints/gating_map/inaturalist2018/
```

### Kiểm tra plugin results:

```bash
ls -lh results/ltr_plugin/inaturalist2018/
```

---

## ⚠️ Lưu ý

1. **Dung lượng disk**: Dataset iNaturalist 2018 rất lớn (~50GB+ sau khi extract). Đảm bảo có đủ dung lượng.

2. **Thời gian training**: 
   - Experts: ~10-20 giờ mỗi expert (tùy GPU)
   - Gating: ~1-2 giờ
   - Plugin: ~30 phút - 1 giờ

3. **GPU memory**: 
   - ResNet-50 với batch size 1024 cần GPU có ít nhất 16GB VRAM
   - Có thể giảm batch size nếu thiếu memory

4. **Resume training**: Các script tự động lưu checkpoint, có thể resume nếu bị gián đoạn.

5. **Log files**: Tất cả logs được lưu trong thư mục `logs/` để dễ debug.

---

## 🐛 Troubleshooting

### Lỗi "Out of memory":
```bash
# Giảm batch size
python train_experts.py --dataset inaturalist2018 --batch-size 512
```

### Lỗi "File not found":
- Kiểm tra đường dẫn đến data files
- Đảm bảo đã chạy bước tạo splits trước

### Lỗi "CUDA out of memory":
- Giảm batch size hoặc sử dụng CPU
- Thêm `--device cpu` nếu cần

---

## 📊 Monitoring Training

Xem log real-time:

```bash
# Tail log file
tail -f logs/experts_inaturalist2018_*.log

# Hoặc với less
less logs/experts_inaturalist2018_*.log
```







