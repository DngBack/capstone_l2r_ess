# ImageNet-LT Setup Guide

Hướng dẫn thiết lập và sử dụng ImageNet-LT dataset trong dự án.

## Cấu trúc dữ liệu

ImageNet-LT dataset cần được đặt trong `data/imagenet_lt/` với cấu trúc sau:

```
data/imagenet_lt/
├── ImageNet_LT_train.txt      # File label cho train set
├── ImageNet_LT_test.txt        # File label cho val set
├── train/                      # Thư mục chứa ảnh train
│   └── n01440764/
│       └── ...
└── val/                        # Thư mục chứa ảnh val (nếu có)
    └── ...
```

Format của file label: `path/to/image class_id`
Ví dụ: `train/n01440764/n01440764_190.JPEG 0`

## Bước 1: Tạo splits

Chạy script để tạo các splits cho ImageNet-LT:

```bash
python scripts/create_imagenet_lt_splits.py \
  --data-dir data/imagenet_lt \
  --train-label-file ImageNet_LT_train.txt \
  --val-label-file ImageNet_LT_test.txt \
  --output-dir data/imagenet_lt_splits \
  --seed 42 \
  --expert-ratio 0.9
```

Script này sẽ:
- Phân tích distribution của dataset
- Chia train thành expert (90%) và gating (10%) với cùng imbalance ratio
- Chia val thành val/test/tunev với tỉ lệ 3:1:1 (vì val có ~20k samples, mỗi class ~40 samples)
- Lưu tất cả splits vào `data/imagenet_lt_splits/`

## Bước 2: Train Experts

Train 3 experts (CE, LogitAdjust, BalancedSoftmax):

```bash
python train_experts.py --dataset imagenet_lt --expert all
```

Hoặc train từng expert riêng:

```bash
python train_experts.py --dataset imagenet_lt --expert ce
python train_experts.py --dataset imagenet_lt --expert logitadjust
python train_experts.py --dataset imagenet_lt --expert balsoftmax
```

## Bước 3: Train Gating Network

Train gating network để combine 3 experts:

```bash
python -m src.train.train_gating_map --dataset imagenet_lt --routing dense
```

## Bước 4: Run LtR Plugin

Chạy balanced plugin với gating:

```bash
python run_balanced_plugin_gating.py --dataset imagenet_lt
```

Hoặc worst-group plugin:

```bash
python run_worst_plugin_gating.py --dataset imagenet_lt
```

## Cấu hình Dataset

ImageNet-LT được cấu hình với:
- **Num classes**: 1000
- **Num groups**: 2 (head/tail với threshold = 20 samples)
- **Backbone**: ResNet-50
- **Batch size**: 512
- **Epochs**: 200 (với cosine annealing)
- **Learning rate**: 0.4 (với warmup 5 epochs)

## Lưu ý

1. **Val split**: ImageNet-LT val được chia thành 8:1:1 (test:val:tunev) theo paper:
   - Val có ~50,000 samples
   - 1000 classes → ~50 samples/class
   - Test: 80% = 40,000 samples (40 per class)
   - Val: 10% = 5,000 samples (5 per class)
   - TuneV: 10% = 5,000 samples (5 per class)

2. **Train split**: Train được chia 9:1 (expert:gating) giống như CIFAR và iNaturalist

3. **File paths**: Đảm bảo đường dẫn trong label files đúng với cấu trúc thư mục thực tế

4. **Memory**: ImageNet-LT lớn hơn CIFAR, cần đủ RAM/GPU memory

## Troubleshooting

### Lỗi: File not found
- Kiểm tra đường dẫn trong label files có đúng không
- Đảm bảo ảnh nằm đúng vị trí như trong label files

### Lỗi: Out of memory
- Giảm batch size: `--batch-size 256`
- Giảm số workers trong DataLoader

### Lỗi: Missing splits
- Chạy lại script tạo splits
- Kiểm tra output directory có đúng không

### Tất cả run scripts:

# Tạo splits
python scripts/create_imagenet_lt_splits.py --log-file logs/imagenet_lt_splits.log

# Train experts
python train_experts.py --dataset imagenet_lt --log-file logs/imagenet_lt_experts.log

# Train gating
python -m src.train.train_gating_map --dataset imagenet_lt --log-file logs/imagenet_lt_gating.log

# Run balanced plugin
python run_balanced_plugin_gating.py --dataset imagenet_lt --log-file logs/imagenet_lt_balanced.log

# Run worst plugin
python run_worst_plugin_gating.py --dataset imagenet_lt --log-file logs/imagenet_lt_worst.log