# capstone_l2r_ess

**Learning To Reject meets MoE in Long Tail Learning**

## 🎯 Vision

Combining 3 experts (CE, LogitAdjust, BalancedSoftmax) + gating router + plug-in rejector rule to solve selective classification challenges in long-tail learning: (i) bias & miscalibration in tail, (ii) head/tail tradeoff with global threshold, (iii) instability with few samples.

---

## 🚀 Quick Start

```bash
# Step 1: Train 3 experts with different long-tail strategies
python train_experts.py

# Step 2: Train gating network (Mixture of Experts)
python -m src.train.train_gating_map --routing dense

# Step 3: Train LtR Plugin - Balanced objective
python train_ltr_plugin.py --objective balanced --optimizer power_iter --cost_sweep

# Step 4: Train LtR Plugin - Worst-group objective
python train_ltr_plugin.py --objective worst --optimizer worst_group --cost_sweep

# Alternative: Use standalone plugins (CE-only or Gating-based)
python run_balanced_plugin_ce_only.py      # CE expert only
python run_balanced_plugin_gating.py       # 3 experts + gating
```

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

Generate visualizations to validate the method:

```bash
# Generate all analyses (routing, ensemble, calibration, ablation)
python scripts/visualize_all.py

# Or use other visualization scripts
python scripts/visualize_gating_outputs.py
python scripts/generate_comparison_plots.py
```

## 📁 Project Structure

```
├── src/                    # Main source code
│   ├── data/              # Dataset utilities, splits, groups
│   ├── models/            # Model definitions (experts, gating, plugins)
│   ├── train/             # Training scripts
│   ├── metrics/           # Metrics and evaluation
│   └── visualize/         # Visualization utilities
│
├── scripts/                # Utility scripts
│   ├── analysis/          # Analysis and test scripts
│   └── archive/           # Old/duplicate files (for reference)
│
├── train_experts.py        # Main: Train 3 experts
├── train_ltr_plugin.py     # Main: Train LtR plugin
├── run_balanced_plugin_ce_only.py    # Standalone: CE-only plugin
├── run_balanced_plugin_gating.py     # Standalone: Gating-based plugin
├── run_paper_reproduction.py         # Paper reproduction pipeline
└── quick_start.py          # Quick start helper
```

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
