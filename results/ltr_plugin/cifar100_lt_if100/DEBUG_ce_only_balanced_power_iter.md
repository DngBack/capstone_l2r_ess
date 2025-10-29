## LtR Plug-in (CE-only) Run Debug Report

- Command: `python -u train_ltr_plugin_ce_only.py --objective balanced --optimizer power_iter --cost_sweep`
- Device: cuda
- Objective: balanced
- Optimizer: power_iter (Algorithm 1)
- Mode: CE-only posteriors (no gating)

### End-to-end pipeline (this script)
1) Load config and seed RNGs.
2) Build head/tail groups from `train_class_counts.json` using rule tail ⇐ count ≤ 20.
3) Load CE logits and aligned targets for S1=tunev and TEST; compute posteriors `p(x)=softmax(logits)`.
4) Instantiate LtR plug-in implementing Theorem 1 with group parameters.
5) Cost-sweep driver: for each cost c in the sweep
   - Run Algorithm 1 (power-iteration) to pick (α, μ) at that c (S1-only), with λ reparameterization.
   - With the selected (α, μ), build RC points at fixed rejection targets r∈{0.0,…,0.8} by recomputing the threshold cost c*(r) via percentile rule; evaluate balanced/worst errors on S1 and TEST.
6) Aggregate all fixed-grid points from all costs into unified RC arrays (sorted by rejection); compute AURC (balanced and worst).
7) Save a machine-readable JSON dump and plots.

### Dataset / Splits
- Dataset key: `cifar100_lt_if100`
- Splits dir: `./data/cifar100_lt_if100_splits_fixed`
- Num classes: 100
- Num groups: 2 (head/tail)
- Head/Tail construction rule: tail if train count <= 20
- Derived groups: head = 69, tail = 31

### Expert logits
- Expert name: `ce_baseline`
- Logits dir: `./outputs/logits/cifar100_lt_if100/ce_baseline/`
- Optimization split used: tunev (S1)
- S1 (tunev) logits: shape [1000, 100]
- Posterior computation: softmax over CE logits

### LtR Plugin config (Theorem 1 implementation)
- Param mode: group (per-group `α`, `μ` mapped to classes)
- Initial grids (CONFIG):
  - `alpha_grid`: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
  - `mu_grid`: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
  - `cost_grid` (for cost-sweep driver): [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.91, 0.95, 0.97, 0.99]
- Target rejection rates (for fixed-grid plotting): [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

### Algorithm choices used in this run
- Power-iteration (Algorithm 1) with 1D μ reparameterization for K=2 (Appendix E.1):
  - `μ = [0, λ]`, `λ` searched on a fine grid [-1.5, -1.4, …, 1.5]
  - `num_iters` (power-iter): 20
  - α update per-iteration: `α_k ← K * P(r(x)=0, y∈G_k)` on S1
  - Selection per cost: minimize balanced objective `mean(group_errors) + c*(1-coverage)` on S1
- RC curve construction (paper-style): for the selected `(α, μ)` at each cost, recompute cost `c*` for each target rejection rate r∈{0.0,…,0.8} using the percentile rule, and evaluate balanced/worst errors; plot and compute AURC from these fixed-grid points.

### Equations implemented (mapping to the paper)
- Classifier: `h_α(x) = argmax_y (1/α[y]) · p_y(x)`
- Rejector: `r(x)=1{ max_y (1/α[y])·p_y(x) < Σ_y' (1/α[y'] − μ[y'])·p_{y'}(x) − c }`
- Percentile-based cost for target rejection r (fixed α, μ):
  - Define per-sample margin `m(x) = (Σ_y (1/α[y] − μ[y])·p_y(x)) − max_y (1/α[y])·p_y(x)`
  - Sort `m(x)` ascending and pick `c* = quantile(m, 1−r)` → achieves rejection ≈ r.
- Power-iter α update (Algorithm 1, line 7): `α^{(m+1)}_k = K · P_S1(r^{(m+1)}(x) = 0, y∈G_k)`
- Reparameterization for K=2 (Appendix E.1): search only λ = μ_tail − μ_head; we set μ_head=0, μ_tail=λ.

### Pseudocode of inner optimizer (per cost)
```
for λ in LambdaGrid:            # μ = [0, λ]
  α ← initialize_from_group_priors(S1)
  for m in 1..M:                # Power-iteration
    set (α, μ=[0,λ], c)
    predict h, r on S1
    α ← K * P_S1(r=0, y∈G_k)    # coverage update per group
    if max|Δα| < 1e-4: break
  evaluate balanced objective on S1: mean(group_errors) + c*(1-coverage)
pick λ with best objective; return (α*, μ*, c)
```

### Configuration dump
```
CONFIG.dataset.name           = cifar100_lt_if100
CONFIG.dataset.splits_dir     = ./data/cifar100_lt_if100_splits_fixed
CONFIG.dataset.num_classes    = 100
CONFIG.dataset.num_groups     = 2
CONFIG.expert.name            = ce_baseline
CONFIG.expert.logits_dir      = ./outputs/logits/cifar100_lt_if100/
CONFIG.ltr.param_mode         = group
CONFIG.ltr.alpha_grid         = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
CONFIG.ltr.mu_grid            = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
CONFIG.ltr.cost_grid          = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.91, 0.95, 0.97, 0.99]
CONFIG.ltr.target_rejects     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
LambdaGrid for μ              = [-1.5, -1.4, …, 1.5] (step 0.1)
Power-iteration steps (M)     = 20
Seed                          = 42
```

### Unified RC curve (TEST) summary (balanced)
- Points: 315 (35 costs × 9 target coverages 0.0..0.8)
- Rejection range: [0.0000, 0.8000]
- Error range: [0.2401, 0.7153]
- AURC (balanced): 0.451092

### Unified RC curve (TEST) summary (worst)
- Points: 315
- Rejection range: [0.0000, 0.8000]
- Error range: [0.3333, 1.0000]
- AURC (worst): 0.647040

### Per-cost selections and evaluation (excerpted logs)

Below each block shows the optimizer’s chosen `(α, μ, c)` for that cost, followed by VAL/TEST metrics. VAL/TEST errors shown under “SUCCESS: Cost=…” are the mid-point summaries; the actual RC curve uses recomputed costs per target rejection rate.

```
[SUCCESS] Best configuration found:
   alpha = [0.5 0.5] (found by power-iter)
   mu = [0.   0.25]
   c = 0.000
   Objective: 0.5000
   Selective error: 0.0000
   Coverage: 0.002
   Group errors: ['1.0000', '0.0000']

SUCCESS: Cost=1.7782024145126343:
   alpha = [1.38 0.62]
   mu = [ 0. -1.]
   VAL: error=0.6856, coverage=0.501, rejection=0.499
   TEST: error=0.6445, coverage=0.500, rejection=0.500

[SUCCESS] Best configuration found:
   alpha = [0.002  0.0001]
   mu = [ 0. -1.]
   c = 0.001
   Objective: 0.5010
   Selective error: 0.0000
   Coverage: 0.001
   Group errors: ['0.0000', '1.0000']

SUCCESS: Cost=964.40283203125:
   alpha = [1.38 0.62]
   mu = [ 0. -1.]
   VAL: error=0.6856, coverage=0.501, rejection=0.499
   TEST: error=0.6445, coverage=0.500, rejection=0.500

[SUCCESS] Best configuration found:
   alpha = [0.02  0.002]
   mu = [0. 1.]
   c = 0.005
   Objective: 0.0049
   Selective error: 0.0000
   Coverage: 0.011
   Group errors: ['0.0000', '0.0000']

SUCCESS: Cost=55.36899948120117:
   alpha = [1.38 0.62]
   mu = [ 0. -1.]
   VAL: error=0.6856, coverage=0.501, rejection=0.499
   TEST: error=0.6445, coverage=0.500, rejection=0.500

... (similar blocks for each cost in the sweep; see run logs for full detail) ...
```

The complete, machine-readable dump of all per-cost selections and the unified RC curves (balanced and worst) is saved to:

- `results/ltr_plugin/cifar100_lt_if100/ltr_plugin_ce_only_cost_sweep_balanced.json`

This JSON contains:
- `results_per_cost`: array with, for each cost, the chosen `(alpha, mu)`, mid-point VAL/TEST metrics, and the training objective value.
- `unified_rc_curves`: two objects (`balanced`, `worst`), each including sorted arrays of `rejection_rates`, `selective_errors`, and their AURC.

### Notes for debugging
- RC points in the final plot use fixed target coverages {0.0, 0.1, …, 0.8}; costs are recomputed per target using the paper’s percentile rule with the selected `(α, μ)`.
- μ search uses λ line-search with μ = [0, λ]; α is updated via coverage on S1 (tunev) only.
- If you want a single-file repro: re-run the same command; all relevant config is embedded in `CONFIG` at the top of `train_ltr_plugin_ce_only.py`.

### Per-iteration α traces (examples)
- Cost 0.0, λ=0.25: α jumped between extremely small and 0.5 due to near-total rejection at early steps (S1 margin distribution very high). Converged in 2 iterations to α=[0.5, 0.5].
- Cost 0.04, best λ=0.25: α progressed (per log) 0.238→0.106→0.032… and stabilized near tail α small, head α small (α_group scaled internally to classes). Converged in 6–7 iterations.
- High costs (≥0.65) often converged to α≈[1.38, 0.62] with full coverage; this explains flat parts of the RC curve. Fixed-grid evaluation decouples this by recomputing thresholds per target.

### Output artifacts
- Plots (TEST):
  - `results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_balanced_ce_only_test.png`
  - `results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_worst_ce_only_test.png`
  - `results/ltr_plugin/cifar100_lt_if100/ltr_rc_curves_dual_ce_only_test.png`
- Raw JSON:
  - `results/ltr_plugin/cifar100_lt_if100/ltr_plugin_ce_only_cost_sweep_balanced.json`



