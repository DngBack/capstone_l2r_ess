# capstone_l2r_ess

**Learning To Reject meets MoE in Long Tail Learning**

## 🎯 Vision

Combining 3 experts (CE, LogitAdjust, BalancedSoftmax) + gating router + plug-in rejector rule to solve selective classification challenges in long-tail learning: (i) bias & miscalibration in tail, (ii) head/tail tradeoff with global threshold, (iii) instability with few samples.

---

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

```bash
# Run complete paper reproduction pipeline
python run_paper_reproduction.py

# Or use quick start helper
python quick_start.py all
```

**Quick Start Commands**:

```bash
python quick_start.py check          # Check if expert files exist
python quick_start.py expert         # Train CE expert only
python quick_start.py plugin-balanced # Train plugin (balanced) only
python quick_start.py plugin-worst   # Train plugin (worst) only
python quick_start.py all            # Run complete pipeline
python quick_start.py clean          # Delete expert files (force retrain)
```

### Option 2: Step-by-Step Manual Execution

#### Step 1: Train 3 Experts

Train 3 experts with different long-tail strategies:

```bash
python train_experts.py
```

This will train:

- **CE Baseline**: Standard cross-entropy expert
- **LogitAdjust**: Prior-based adjustment expert (Menon et al., ICML 2021)
- **BalancedSoftmax**: Frequency-based adjustment expert (Ren et al., NeurIPS 2020)

**Outputs**:

- Checkpoints: `checkpoints/experts/cifar100_lt_if100/*.pth`
- Logits: `outputs/logits/cifar100_lt_if100/{expert_name}/*.pt`

#### Step 2: Train Gating Network

Train Mixture of Experts (MoE) router:

```bash
python -m src.train.train_gating_map --routing dense
```

**Options**:

- `--routing dense`: Dense routing (all experts used)
- `--routing top_k`: Top-K routing (sparse, faster)

**Outputs**:

- Checkpoint: `checkpoints/gating_map/cifar100_lt_if100/final_gating.pth`

#### Step 3: Train LtR Plugin (Main Method)

**Balanced Objective** (Algorithm 1 - Power Iteration):

```bash
python train_ltr_plugin.py --objective balanced --optimizer power_iter --cost_sweep
```

**Worst-group Objective** (Algorithm 2 - Exponentiated Gradient):

```bash
python train_ltr_plugin.py --objective worst --optimizer worst_group --cost_sweep
```

**Other Options**:

- `--optimizer grid`: Grid search baseline
- `--cost_sweep`: Generate full RC curve
- `--no_reweight`: Disable importance weighting

**Outputs**:

- Results: `results/ltr_plugin/cifar100_lt_if100/ltr_plugin_cost_sweep_{objective}.json`
- Plots: `results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_*_test.png`

### Option 3: Standalone Plugins (Alternative Methods)

**CE-only Plugin** (Single expert, no gating):

```bash
python run_balanced_plugin_ce_only.py
```

- Uses only CE baseline expert
- No gating network needed
- Simpler but potentially lower performance

**Gating-based Plugin** (3 experts + gating):

```bash
python run_balanced_plugin_gating.py
```

- Uses all 3 experts combined via gating
- Requires trained gating network
- Better performance through ensemble

**Outputs**:

- `results/ltr_plugin/cifar100_lt_if100/ltr_plugin_{ce_only|gating}_balanced.json`
- `results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_*_test.png`

---

## 📊 Pipeline Overview

### **1. Experts Training**

Train 3 experts with different long-tail handling:

- **CE Baseline**: Standard cross-entropy
- **LogitAdjust**: Menon et al. (ICML 2021) - prior-based adjustment
- **BalancedSoftmax**: Ren et al. (NeurIPS 2020) - frequency-based adjustment

### **2. Gating Network**

Router for Mixture of Experts:

- **Input**: Expert posteriors `[B, E, C]`
- **Architecture**: Feature extractor → MLP → Router
- **Output**: Gating weights `[B, E]` → Mixture posterior `η̃(x) = Σ_e w_e · η^(e)(x)`
- **Loss**: Mixture NLL + Load-balancing (Switch Transformer) + Entropy regularization

### **3. LtR Plugin**

Plug-in decision rule for selective classification:

- **Classifier**: `h_α(x) = argmax_y (α[y] · η̃[y])`
- **Rejector**: `r(x) = 1{max_y(α[y]·η̃[y]) < ⟨μ, η̃⟩ - c}`
- **Optimizers**:
  - Algorithm 1: Power-iteration (balanced objective)
  - Algorithm 2: Worst-group (exponentiated gradient)
- **Evaluation**: RC curves + AURC (Area Under Risk-Coverage)

---

## 📈 Results

Results saved in:

- `results/ltr_plugin/cifar100_lt_if100/ltr_plugin_cost_sweep_{objective}.json`
- `results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_*_test.png`

## 📊 Comprehensive Analysis

### Visualization Scripts

**Generate All Analyses** (recommended):

```bash
python scripts/visualize_all.py
```

Generates 40+ visualizations covering:

- Routing patterns, expert disagreement, ensemble benefits
- Calibration analysis, ablation study
- Full class analysis, per-sample analysis
- RC curves and risk-coverage analysis

**Individual Visualization Scripts**:

```bash
# Gating network outputs visualization
python scripts/visualize_gating_outputs.py

# Comparison plots between methods
python scripts/generate_comparison_plots.py
```

**Outputs**: All saved to `results/comprehensive_analysis/`

### Analysis & Testing Scripts

**Data Distribution Analysis**:

```bash
python scripts/analysis/analyze_data_distribution_paper_final.py
python scripts/analysis/analyze_train_distribution.py
```

**Verification & Testing**:

```bash
# Check expert files exist
python scripts/analysis/check_expert.py

# Verify training statistics
python scripts/analysis/verify_training_stats.py

# Test implementations
python scripts/analysis/test_importance_weights.py
python scripts/analysis/test_rejector_verification.py
python scripts/analysis/test_ce_only_implementation.py
```

**Inference**:

```bash
# Run inference to generate expert logits
python scripts/run_infer_ce_expert_logits.py
```

## 📁 Project Structure

```
capstone_l2r_ess/
│
├── src/                    # Main source code
│   ├── data/              # Dataset utilities, splits, groups
│   ├── models/            # Model definitions (experts, gating, plugins)
│   ├── train/             # Training scripts
│   ├── metrics/           # Metrics and evaluation
│   └── visualize/         # Visualization utilities
│
├── scripts/                # Utility scripts
│   ├── analysis/          # Analysis and test scripts (13 files)
│   ├── archive/           # Old/duplicate files (for reference)
│   ├── visualize_all.py   # Comprehensive visualization
│   ├── visualize_gating_outputs.py
│   ├── generate_comparison_plots.py
│   └── run_infer_ce_expert_logits.py
│
├── data/                   # Dataset and splits
│   └── cifar100_lt_if100_splits_fixed/
│
├── checkpoints/            # Trained models
│   ├── experts/
│   └── gating_map/
│
├── outputs/                # Model outputs (logits, predictions)
│   └── logits/
│
├── results/                # Results and visualizations
│   ├── ltr_plugin/        # LtR plugin results
│   ├── comprehensive_analysis/  # Analysis visualizations
│   └── gating_map/        # Gating results
│
├── train_experts.py        # Main: Train 3 experts
├── train_ltr_plugin.py     # Main: Train LtR plugin
├── run_balanced_plugin_ce_only.py    # Standalone: CE-only plugin
├── run_balanced_plugin_gating.py     # Standalone: Gating-based plugin
├── run_paper_reproduction.py         # Paper reproduction pipeline
└── quick_start.py          # Quick start helper
```

### Main Scripts Location

**Root Directory** (main entry points):

- `train_experts.py` - Train all 3 experts
- `train_ltr_plugin.py` - Main LtR plugin training (supports balanced/worst objectives)
- `run_balanced_plugin_ce_only.py` - Standalone CE-only plugin
- `run_balanced_plugin_gating.py` - Standalone gating-based plugin
- `run_paper_reproduction.py` - Complete pipeline automation
- `quick_start.py` - Quick helper commands

**Scripts Directory**:

- `scripts/visualize_*.py` - Visualization scripts
- `scripts/analysis/*.py` - Analysis and testing scripts
- `scripts/run_infer_ce_expert_logits.py` - Inference script

**Outputs** (40+ visualizations in `results/comprehensive_analysis/`):

**Core Analysis:**

- Routing patterns (9 subplots) - Load balance, expert usage, per-class routing
- Expert disagreement (4 subplots) - Diversity analysis, pairwise agreement
- Ensemble benefits (6 subplots) - Single vs mixture comparison
- Calibration analysis (4 subplots) - ECE, Brier score, reliability diagrams
- Ablation study (4 subplots) - Component importance, per-class improvements

**Enhanced Analysis:**

- Full-class analysis (12 subplots) - **ALL 100 classes** with head/tail breakdown
- Softmax distribution (6 subplots) - Probability distributions, confidence, entropy
- Expert contribution (15 subplots) - Contribution heatmaps, smoothing effect, per-sample analysis
- **Per-sample analysis** (10 samples × 4 plots) - Expert posteriors, gating weights, mixture, summary
- RC curves (3 plots) - Risk-coverage analysis from LtR plugin

**Key Visualizations for Paper:**

- `per_sample_analysis.png` - **10 samples**: Expert posteriors → Gating weights → Mixture distribution
- `full_class_analysis.png` - Complete 100-class performance
- `expert_contribution_detail.png` - Per-sample contribution breakdown
- `softmax_contribution.png` - Distribution comparison
- `mixture_smoothing_comparison.png` - Entropy/confidence analysis

See [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) for interpretation guide.

---

## 📚 References

- **MoE**: Jordan & Jacobs (1994) - Hierarchical Mixtures of Experts
- **Switch Transformer**: Fedus et al. (2021) - Load-balancing
- **LogitAdjust**: Menon et al. (ICML 2021) - Long-tail classification
- **BalancedSoftmax**: Ren et al. (NeurIPS 2020) - Balanced loss
- **LtR Theory**: ICLR 2024 - Learning to Reject meets Long-tail Learning
- **Selective Classification**: Geifman & El-Yaniv (2017) - RC curves

---

## 📝 See Also

- [CODE_ANALYSIS.md](./CODE_ANALYSIS.md) - Detailed code analysis
- [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) - How to generate & interpret visualizations
- [SUMMARY.md](./SUMMARY.md) - Complete summary of visualization setup
