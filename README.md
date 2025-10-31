# capstone_l2r_ess

**Learning To Reject meets MoE in Long Tail Learning**

## 🎯 Vision

Combining 3 experts (CE, LogitAdjust, BalancedSoftmax) + gating router + plug-in rejector rule to solve selective classification challenges in long-tail learning: (i) bias & miscalibration in tail, (ii) head/tail tradeoff with global threshold, (iii) instability with few samples.

## Run pipeline

#### Step 1: Train 3 Experts

Train 3 experts with different long-tail strategies:

```bash
python train_experts.py
```

#### Step 2: Train Gating Network

Train Mixture of Experts (MoE) router:

```bash
python -m src.train.train_gating_map --routing dense
```

#### Step 3: Train LtR Plugin (Main Method)

**Balanced Objective** (Algorithm 1 - Power Iteration):

```bash
python run_balanced_plugin_gating.py
```

**Worst-group Objective** (Algorithm 2 - Exponentiated Gradient):

```bash
python run_worst_plugin_gating.py
```

### Reproduce paper results

**CE-only Plugin** (Single expert, no gating):

```bash
python run_balanced_plugin_ce_only.py
```

```bash
python run_worst_plugin_ce_only.py
```

#### Note

Run with only CE

```
python -m src.train.train_ce_expert_paper_final
```
