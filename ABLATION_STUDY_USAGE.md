# Feature Ablation Study - Complete Usage Guide

## Tổng quan

Script `ablation_feature_study_complete.py` tích hợp toàn bộ pipeline để tạo file CSV ablation study đầy đủ:

1. **Train gating network** với từng feature preset
2. **Run plugin evaluation** với từng trained gating
3. **Tổng hợp kết quả** vào một CSV file duy nhất

## Usage

### Basic usage - Test tất cả presets

```bash
python scripts/ablation_feature_study_complete.py --dataset cifar100_lt_if100
```

### Test một số presets cụ thể

```bash
python scripts/ablation_feature_study_complete.py \
    --dataset cifar100_lt_if100 \
    --presets all minimal uncertainty_only agreement_only
```

### Custom parameters

```bash
python scripts/ablation_feature_study_complete.py \
    --dataset cifar100_lt_if100 \
    --presets all minimal uncertainty_only \
    --epochs 100 \
    --batch_size 128 \
    --lr 1e-3 \
    --routing dense
```

### Skip training (chỉ chạy plugin evaluation)

Nếu đã train rồi, có thể skip training:

```bash
python scripts/ablation_feature_study_complete.py \
    --dataset cifar100_lt_if100 \
    --skip-training
```

### Chỉ train, không chạy plugin

```bash
python scripts/ablation_feature_study_complete.py \
    --dataset cifar100_lt_if100 \
    --skip-plugin
```

## Output

Script sẽ tạo ra:

### CSV File

`./results/ablation_feature_study_complete/ablation_complete_{dataset}_{timestamp}.csv`

**Columns bao gồm:**

- **Identification:**

  - `preset`: Tên feature preset
  - `status`: Trạng thái (success/error)
  - `feature_dim`: Feature dimension
  - `total_features`: Tổng số features
  - `per_expert_count`, `global_count`: Số lượng từng loại features
  - `per_expert_features`, `global_features`: Danh sách features

- **Training Metrics:**

  - `balanced_acc`: Best balanced accuracy từ validation
  - `val_loss`: Best validation loss

- **Baseline Metrics (without rejection):**

  - `baseline_balanced_error`: Balanced error của gating (r=0)
  - `baseline_head_error`: Head group error
  - `baseline_tail_error`: Tail group error

- **Plugin Metrics (with rejection):**
  - `plugin_aurc_balanced`: AURC cho balanced error
  - `plugin_aurc_worst_group`: AURC cho worst-group error
  - `plugin_test_balanced_error_r0`: Balanced error tại r=0
  - `plugin_test_head_error_r0`: Head error tại r=0
  - `plugin_test_tail_error_r0`: Tail error tại r=0

### JSON File

`./results/ablation_feature_study_complete/ablation_complete_{dataset}_{timestamp}.json`

Chứa full details của tất cả experiments.

### Log Files

`./results/ablation_feature_study_complete/logs/`

- `train_{dataset}_{preset}_{timestamp}.log`: Training logs
- `plugin_{dataset}_{preset}_{timestamp}.log`: Plugin evaluation logs

### Checkpoints

`./checkpoints/gating_map/{dataset}/`

- `final_gating_{preset}.pth`: Final checkpoint cho mỗi preset
- `best_gating_{preset}.pth`: Best checkpoint cho mỗi preset

## Workflow

```
For each preset:
  1. Train gating network
     ├── Save checkpoint: final_gating_{preset}.pth
     ├── Parse training metrics
     └── Continue if successful

  2. Run plugin evaluation
     ├── Load checkpoint: final_gating_{preset}.pth
     ├── Compute mixture posteriors
     ├── Run balanced plugin algorithm
     ├── Evaluate on test set
     └── Parse plugin metrics

  3. Combine results
     └── Add to results DataFrame

Save final CSV + JSON
```

## Example Output CSV

```csv
preset,status,feature_dim,total_features,balanced_acc,val_loss,baseline_balanced_error,plugin_aurc_balanced,plugin_test_balanced_error_r0
all,success,24,10,0.8234,0.4521,0.3456,0.1234,0.3456
minimal,success,6,2,0.8100,0.4689,0.3512,0.1289,0.3512
uncertainty_only,success,8,4,0.8156,0.4612,0.3489,0.1256,0.3489
```

## Tips

1. **Chạy với ít epochs trước** để test:

   ```bash
   python scripts/ablation_feature_study_complete.py --epochs 10
   ```

2. **Resume nếu một preset fail**: Script sẽ tiếp tục với preset tiếp theo

3. **Analyze CSV**: Mở CSV trong Excel/Python để so sánh:

   ```python
   import pandas as pd
   df = pd.read_csv("results/ablation_feature_study_complete/ablation_complete_*.csv")
   print(df.sort_values("plugin_aurc_balanced").head(10))
   ```

4. **Feature importance**: So sánh `total_features` vs `plugin_aurc_balanced` để tìm optimal feature set

## Troubleshooting

- **Checkpoint not found**: Đảm bảo training đã hoàn thành trước khi skip-training
- **Feature dimension mismatch**: Kiểm tra feature config trong checkpoint
- **Plugin evaluation fails**: Kiểm tra log file để xem lỗi chi tiết

## Next Steps

Sau khi có CSV results:

1. **Analyze feature importance**: So sánh các presets
2. **Find optimal preset**: Chọn preset có lowest AURC
3. **Fine-tune**: Train với preset tốt nhất và nhiều epochs hơn
4. **Create custom preset**: Dựa trên insights từ ablation study
