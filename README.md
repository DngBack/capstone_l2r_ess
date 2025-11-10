# Learning to Reject Meets MoE in Long-Tail Learning

**Capstone Project: Combining Mixture of Experts with Learning to Reject for Selective Classification in Long-Tail Scenarios**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

## 📋 Overview

This project implements a novel approach to **Learning to Reject (L2R)** in long-tail classification by combining:

- **3 Expert Models**: Cross-Entropy (CE), LogitAdjust, and BalancedSoftmax
- **Gating Router**: Mixture of Experts (MoE) network that learns to combine expert predictions
- **Plug-in Rejector**: Theoretical optimal rejection rule for balanced/worst-group error metrics

### Key Contributions

This implementation addresses three critical challenges in selective classification for long-tail learning:

1. **Bias & Miscalibration in Tail Classes**: Traditional methods favor head classes
2. **Head/Tail Tradeoff with Global Threshold**: Single threshold fails to balance head and tail performance
3. **Instability with Few Samples**: Tail classes have insufficient data for reliable confidence estimation

### Theoretical Foundation

Based on the paper: **"Learning to Reject Meets Long-Tail Learning"** (ICLR 2024) by Narasimhan et al.

- Derives Bayes-optimal classifier and rejector for balanced error metric
- Proposes plug-in approach that mimics optimal solution without retraining base models
- Extends to general evaluation metrics (worst-group error, etc.)

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd capstone_l2r_ess
```

2. **Create conda environment** (Python 3.11):
```bash
conda create -n l2r_ess python=3.11
conda activate l2r_ess
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependency list

---

## 📊 Supported Datasets

### 1. CIFAR-100-LT

Long-tail version of CIFAR-100 with imbalance factor 100.

- **Model**: CIFARResNet-32 (CIFAR-optimized)
- **Batch Size**: 128
- **Classes**: 100
- **Imbalance Factor**: 100

### 2. iNaturalist 2018

Large-scale fine-grained classification with 8,000+ classes and inherent long-tail distribution.

- **Model**: ResNet-50 (ImageNet architecture)
- **Batch Size**: 1024
- **Classes**: 8,142
- **Scheduler**: Cosine annealing with warmup

### 3. ImageNet-LT

Long-tail version of ImageNet with 1,000 classes.

- **Model**: ResNet-50
- **Batch Size**: 512
- **Classes**: 1,000
- **Epochs**: 200 (with cosine annealing)

---

## 🔧 Dataset Setup

### CIFAR-100-LT

Generate CIFAR-100-LT splits:

```bash
python src/data/balanced_test_splits.py
```

### iNaturalist 2018

Generate iNaturalist 2018 splits from JSON files:

```bash
python scripts/create_inaturalist_splits.py \
  --train-json /path/to/train.json \
  --val-json /path/to/val.json \
  --data-dir data/inaturalist2018/train_val2018 \
  --log-file logs/inaturalist2018_splits_$(date +%Y%m%d_%H%M%S).log
```

### ImageNet-LT

Generate ImageNet-LT splits:

```bash
python scripts/create_imagenet_lt_splits.py \
  --data-dir data/imagenet_lt \
  --train-label-file ImageNet_LT_train.txt \
  --val-label-file ImageNet_LT_test.txt \
  --output-dir data/imagenet_lt_splits \
  --seed 42 \
  --expert-ratio 0.9
```

For detailed setup instructions, see:
- `IMAGENET_LT_SETUP.md` for ImageNet-LT
- `docs/inaturalist2018_setup.md` for iNaturalist 2018

---

## 🎯 Training Pipeline

The complete pipeline consists of three sequential steps:

### Step 1: Train Expert Models

Train 3 experts with different long-tail strategies:

**CIFAR-100-LT:**
```bash
python train_experts.py --dataset cifar100_lt_if100 --log-file logs/experts_cifar.log
```

**iNaturalist 2018:**
```bash
python train_experts.py --dataset inaturalist2018 --log-file logs/experts_inat.log
```

**ImageNet-LT:**
```bash
python train_experts.py --dataset imagenet_lt --expert all --log-file logs/experts_imagenet.log
```

**Quick Test** (2 epochs, reduced batch size):
```bash
python train_experts.py --dataset inaturalist2018 --expert ce --epochs 2 --batch-size 512 --log-file logs/inat_test.log
```

**Train Individual Experts:**
```bash
# Cross-Entropy
python train_experts.py --dataset <dataset> --expert ce

# LogitAdjust
python train_experts.py --dataset <dataset> --expert logitadjust

# BalancedSoftmax
python train_experts.py --dataset <dataset> --expert balsoftmax
```

### Step 2: Train Gating Network

Train the Mixture of Experts (MoE) router that learns to combine expert predictions:

```bash
python -m src.train.train_gating_map --dataset <dataset> --routing dense
```

**Options:**
- `--dataset`: `cifar100_lt_if100`, `inaturalist2018`, or `imagenet_lt`
- `--routing`: `dense` (all experts) or `sparse` (top-k experts)

### Step 3: Train L2R Plugin (Main Method)

This is the core contribution: plug-in rejection rules optimized for balanced/worst-group error.

#### Balanced Objective (Algorithm 1 - Power Iteration)

Optimizes balanced error across all classes:

```bash
python run_balanced_plugin_gating.py --dataset <dataset>
```

**CE-only baseline** (single expert, no gating):
```bash
python run_balanced_plugin_ce_only.py --dataset <dataset>
```

#### Worst-Group Objective (Algorithm 2 - Exponentiated Gradient)

Optimizes worst-group error (minimizes maximum error across head/tail groups):

```bash
python run_worst_plugin_gating.py --dataset <dataset>
```

**CE-only baseline**:
```bash
python run_worst_plugin_ce_only.py --dataset <dataset>
```

---

## 📁 Project Structure

```
capstone_l2r_ess/
├── data/                          # Dataset files and splits
│   ├── imagenet_lt/              # ImageNet-LT labels
│   └── ...
├── docs/                          # Paper and documentation
│   ├── 6368_Learning_to_Reject_Meets.pdf
│   └── extracted_content/        # Paper sections
├── results/                       # Training results and visualizations
│   ├── gating_map/               # Gating network results
│   ├── ltr_plugin/               # L2R plugin results
│   └── moe_analysis/             # MoE analysis plots
├── scripts/                       # Utility scripts
│   ├── create_imagenet_lt_splits.py
│   ├── create_inaturalist_splits.py
│   └── moe_*.py                  # Analysis scripts
├── src/
│   ├── data/                     # Dataset loaders and splits
│   │   ├── datasets.py
│   │   ├── imagenet_lt_splits.py
│   │   └── ...
│   ├── models/                   # Model implementations
│   │   ├── experts.py            # Expert models
│   │   ├── gating.py             # Gating network
│   │   ├── gating_network_map.py # MoE router
│   │   └── ltr_plugin.py         # L2R plugin (core)
│   ├── train/                    # Training scripts
│   │   ├── train_expert.py
│   │   ├── train_gating_map.py
│   │   └── ...
│   ├── metrics/                  # Evaluation metrics
│   │   ├── selective_metrics.py
│   │   ├── reweighted_metrics.py
│   │   └── ...
│   └── visualize/                # Visualization tools
│       ├── comprehensive_analysis.py
│       └── ...
├── train_experts.py              # Main expert training script
├── run_balanced_plugin_gating.py # Balanced L2R with gating
├── run_worst_plugin_gating.py    # Worst-group L2R with gating
├── run_balanced_plugin_ce_only.py # Balanced L2R (CE-only)
├── run_worst_plugin_ce_only.py   # Worst-group L2R (CE-only)
└── requirements.txt              # Python dependencies
```

---

## 🔬 Key Features

### 1. Mixture of Experts (MoE)

- Combines predictions from 3 specialized experts
- Gating network learns optimal expert combination
- Handles uncertainty and disagreement between experts

### 2. Plug-in Rejection Rules

- **Balanced Error**: Power iteration algorithm to find optimal α, μ parameters
- **Worst-Group Error**: Exponentiated gradient for group-wise optimization
- No retraining required - works with pre-trained models

### 3. Group-Based Parameterization

- Reduces parameters for tail classes with few samples
- More stable optimization
- Configurable group boundaries (head/tail split)

### 4. Comprehensive Evaluation

- Balanced error, worst-group error
- Per-group metrics (head/tail)
- Rejection-coverage curves
- Calibration analysis

---

## 📈 Reproducing Results

### Paper Baseline (CE-only)

Train a single CE expert for comparison:

```bash
python -m src.train.train_ce_expert_paper_final --dataset <dataset>
```

### Full Pipeline Example

Complete workflow for CIFAR-100-LT:

```bash
# 1. Setup data
python src/data/balanced_test_splits.py

# 2. Train experts
python train_experts.py --dataset cifar100_lt_if100

# 3. Train gating
python -m src.train.train_gating_map --dataset cifar100_lt_if100 --routing dense

# 4. Run balanced plugin
python run_balanced_plugin_gating.py --dataset cifar100_lt_if100

# 5. Run worst-group plugin
python run_worst_plugin_gating.py --dataset cifar100_lt_if100
```

---

## 📚 References

**Main Paper:**
- Narasimhan, H., Menon, A. K., Jitkrittum, W., Gupta, N., & Kumar, S. (2024). Learning to Reject Meets Long-Tail Learning. *ICLR 2024*.

**Related Work:**
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*.
- Menon, A. K., et al. (2021). Long-tail learning via logit adjustment. *ICLR 2021*.

---

## 📝 Notes

- All experiments use fixed random seeds for reproducibility
- Results are saved in `results/` directory with timestamps
- Log files are saved to `logs/` directory
- GPU is recommended for training (especially for iNaturalist and ImageNet-LT)

---

## 🤝 Contributing

This is a capstone project. For questions or issues, please refer to the documentation in `docs/` or check the paper implementation details.

---

## 📄 License

See `LICENSE` file for details.
