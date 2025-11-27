# Detailed Flow Analysis: train_gating_map.py

## Overview

This script trains a Gating Network for Maximum A Posteriori (MAP) routing in a Mixture-of-Experts system. It learns to combine expert predictions optimally using a learned routing network.

---

## 1. INITIALIZATION & CONFIGURATION (Lines 1-167)

### 1.1 Imports (Lines 16-30)

- PyTorch components for training
- Model components: `GatingNetwork`, `GatingMLP`, `GatingFeatureBuilder`
- Loss functions: `GatingLoss`, `compute_gating_metrics`

### 1.2 Improved Loss Functions (Lines 34-80)

#### `compute_responsibility_loss()` (Lines 38-52)

- **Purpose**: EM-style alignment between gating weights and expert responsibilities
- **Formula**: KL divergence between target responsibilities and current weights
- **Input**:
  - `posteriors`: [B, E, C] expert posteriors
  - `weights`: [B, E] gating weights
  - `labels`: [B] ground truth
  - `temperature`: Softmax temperature for annealing
- **Process**:
  1. Extract expert probabilities for true labels: `expert_probs = posteriors[:, :, labels]`
  2. Compute responsibilities: `(weights * expert_probs) / sum(...)`
  3. Target weights via softmax with temperature
  4. KL divergence between target and current weights

#### `estimate_group_priors()` (Lines 55-80)

- **Purpose**: Estimate which expert works best for each group (head/tail classes)
- **Process**:
  1. Split labels into groups using boundaries (e.g., class 69 for CIFAR-100)
  2. For each group, compute expert accuracies
  3. Convert accuracies to softmax priors with temperature=0.1
  4. Returns [num_groups, num_experts] prior matrix

### 1.3 Configuration (Lines 84-165)

#### Dataset Configs (Lines 88-113)

```python
DATASET_CONFIGS_GATING = {
    "cifar100_lt_if100": {...},
    "inaturalist2018": {...},
    "imagenet_lt": {...}
}
```

- Each dataset specifies: split directory, logits directory, num_classes, expert names

#### Main CONFIG (Lines 115-165)

Key sections:

- **Dataset**: name, splits_dir, num_classes, num_groups
- **Experts**: names, logits_dir
- **Gating**:
  - Architecture: hidden_dims=[256, 128], dropout=0.1
  - Routing: 'dense' (softmax) or 'top_k' (sparse)
  - Training: epochs=100, batch_size=128, lr=1e-3
  - Loss weights: λ_lb, λ_h, λ_resp, λ_prior
  - Features: load-balancing, entropy reg, responsibility loss, prior regularizer

---

## 2. DATA LOADING (Lines 170-372)

### 2.1 `load_expert_logits()` (Lines 175-205)

**Purpose**: Load pre-computed expert logits for a split

**Flow**:

1. Iterate over expert names (e.g., ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'])
2. Load logits from `{logits_dir}/{expert_name}/{split_name}_logits.pt`
3. Stack logits: [E, N, C] → transpose → [N, E, C]
4. Convert float16 → float32 if needed

**Returns**: `[N, E, C]` tensor

### 2.2 `load_labels()` (Lines 208-259)

**Purpose**: Load ground truth labels for a split

**Flow**:

- **For iNaturalist**: Load directly from `{split_name}_targets.json`
- **For CIFAR**:
  1. Load indices from `{split_name}_indices.json`
  2. Determine if train or test split
  3. Load CIFAR-100 dataset
  4. Extract labels using indices

**Returns**: `[N]` tensor of class labels

### 2.3 `load_class_weights()` (Lines 262-293)

**Purpose**: Load frequency-based class weights for reweighting loss (handles long-tail)

**Flow**:

1. Load from `class_weights.json` (if exists)
2. Normalize weights (sum = num_classes)
3. Fallback to uniform weights if file not found

**Returns**: `[C]` tensor

### 2.4 `create_dataloaders()` (Lines 296-372)

**Purpose**: Create train and validation DataLoaders

**Flow**:

#### Train Split (Lines 313-331)

1. Load 'gating' split (10% of train data, same long-tail distribution)
   - **Why separate?** Experts trained on 90% ('expert' split), so gating needs separate data to avoid overfitting
2. Load expert logits: [N, E, C]
3. Load labels: [N]
4. **Size mismatch check**: Truncate to minimum if logits and labels differ
5. Create `TensorDataset(logits, labels)`

#### Validation Split (Lines 333-349)

1. Load 'val' split (balanced subset)
2. Same size mismatch handling

#### Create DataLoaders (Lines 356-370)

- Train: `shuffle=True`, batch_size from config
- Val: `shuffle=False`

**Returns**: `(train_loader, val_loader)`

---

## 3. TRAINING FUNCTIONS (Lines 375-609)

### 3.1 `train_one_epoch()` (Lines 380-512)

**Purpose**: Train gating network for one epoch with all loss components

**Flow**:

#### Setup (Lines 397-419)

1. Set model to train mode
2. Initialize metrics trackers
3. **Router temperature annealing**:
   ```python
   router_temp = temp_start - (temp_start - temp_end) * (epoch / epochs)
   ```
   - Starts at 2.0, ends at 0.7 (soft → hard routing)
4. Create `GatingFeatureBuilder()` (reused across batches)

#### Batch Loop (Lines 421-490)

For each batch `(logits, targets)`:

1. **Move to device** (Lines 423-424)

   - `logits`: [B, E, C] → GPU
   - `targets`: [B] → GPU

2. **Convert to posteriors** (Line 427)

   - `posteriors = torch.softmax(logits, dim=-1)` → [B, E, C]

3. **Build features & compute gating weights** (Lines 429-432)

   - `features = feature_builder(logits)` → [B, D] where D = 7\*E + 3
   - `gating_logits = model.mlp(features)` → [B, E]
   - `weights = model.router(gating_logits)` → [B, E] (softmax/top-k)

4. **Compute base loss** (Lines 435-445)

   - Prepare sample weights (class frequency reweighting)
   - Call `loss_fn(posteriors, weights, targets, ...)`
   - Returns `(total_loss, components_dict)`

5. **Add responsibility loss** (Lines 448-453)

   - If `use_responsibility=True`:
     - Compute EM-style alignment: KL(responsibility || weights)
     - Add `λ_resp * responsibility_loss`

6. **Add prior regularizer** (Lines 456-469)

   - If `use_prior_reg=True` and group_priors provided:
     - Compute mean weights per batch: [E]
     - KL divergence between mean weights and target prior
     - Add `λ_prior * prior_loss`

7. **Backward pass** (Lines 472-479)

   - Zero gradients
   - `loss.backward()`
   - Gradient clipping (norm ≤ 1.0)
   - `optimizer.step()`

8. **Track metrics** (Lines 482-490)
   - Accumulate loss components
   - Collect weights, posteriors, targets for epoch-level metrics

#### Epoch Metrics (Lines 493-511)

1. Average loss components
2. Concatenate all batch outputs: `[N, E, C]` posteriors, `[N, E]` weights, `[N]` targets
3. Compute gating metrics:
   - Mixture accuracy
   - Effective experts (entropy-based diversity)
   - Load balance statistics
   - Expert accuracies

**Returns**: `metrics` dict

### 3.2 `validate()` (Lines 515-575)

**Purpose**: Validate gating network performance

**Flow**:

1. Set model to eval mode (`@torch.no_grad()`)
2. For each batch:
   - Same forward pass as training (features → MLP → router)
   - Compute loss with components
   - Collect outputs
3. Aggregate all batches
4. Compute metrics:
   - Standard gating metrics
   - **Group-wise accuracies** (head/tail via `compute_group_accuracies()`)

**Returns**: `metrics` dict with validation performance

### 3.3 `compute_group_accuracies()` (Lines 578-609)

**Purpose**: Compute head/tail class accuracies

**Flow**:

1. Compute mixture posterior: `Σ_e w_e * p_e` → [N, C]
2. Get predictions: `argmax(mixture_posterior)`
3. Define groups:
   - Head: `targets < 50` (classes 0-49)
   - Tail: `targets >= 50` (classes 50-99)
4. Compute accuracies:
   - `head_acc`, `tail_acc`, `balanced_acc = (head_acc + tail_acc) / 2`

**Returns**: `{'head_acc', 'tail_acc', 'balanced_acc'}`

---

## 4. MAIN TRAINING LOOP (Lines 617-891)

### 4.1 `train_gating()` (Lines 617-891)

**Purpose**: Orchestrate entire training pipeline

#### Phase 1: Setup (Lines 624-641)

1. **Seed random generators** (Lines 625-626)
2. **Load data** (Line 629)
   - `train_loader, val_loader = create_dataloaders(config)`
3. **Load class weights** (Lines 632-640)
   - For reweighting loss to handle class imbalance

#### Phase 2: Model Creation (Lines 642-675)

1. **Create GatingNetwork** (Lines 643-660)
   ```python
   model = GatingNetwork(
       num_experts=3,
       num_classes=100,
       hidden_dims=[256, 128],
       routing='dense' or 'top_k',
       ...
   )
   ```
2. **Replace MLP** (Lines 665-675)
   - Replace with lightweight version matching `GatingFeatureBuilder` output
   - Input dim: `7*E + 3` (7 features per expert + 3 global features)

#### Phase 3: Loss & Optimizer Setup (Lines 677-734)

1. **Loss function** (Lines 677-711)

   ```python
   loss_fn = GatingLoss(
       lambda_lb=1e-2,  # Load-balancing (only for top_k)
       lambda_h=0.01,   # Entropy regularization
       use_load_balancing=True (if top_k),
       ...
   )
   ```

2. **Optimizer** (Lines 713-726)

   - AdamW (default) or SGD
   - LR: 1e-3, weight_decay: 1e-4

3. **Scheduler** (Lines 728-734)
   - CosineAnnealingLR (default) or None

#### Phase 4: Group Priors Estimation (Lines 749-766)

If `use_prior_reg=True`:

1. Collect all training posteriors
2. Estimate group priors using `estimate_group_priors()`
3. Move to device

#### Phase 5: Training Loop (Lines 777-855)

For each epoch:

1. **Warmup LR** (Lines 779-782)

   - Linear warmup for first `warmup_epochs` epochs

2. **Train** (Lines 785-795)

   - Call `train_one_epoch()` with all loss components

3. **Update scheduler** (Lines 798-799)

   - Step after warmup

4. **Print progress** (Lines 802-816)

   - Train loss, NLL, responsibility, prior
   - Mixture accuracy, effective experts

5. **Validate** (Lines 819-831)

   - Every `val_interval` epochs or final epoch
   - Compute validation metrics
   - Print head/tail/balanced accuracies

6. **Save best model** (Lines 834-850)

   - Based on `balanced_acc` (average of head and tail)
   - Save checkpoint with: epoch, model state, optimizer state, metrics, config

7. **Save history** (Lines 853-855)
   - Append to `results_history`

#### Phase 6: Finalization (Lines 857-890)

1. **Save final model** (Lines 858-867)

   - Always save final epoch checkpoint

2. **Save training history** (Lines 870-884)

   - Serialize to JSON: `training_history.json`
   - Convert all tensors to Python floats

3. **Print summary** (Lines 886-889)
   - Best balanced accuracy, validation loss, checkpoint location

**Returns**: `(model, results_history)`

---

## 5. MAIN ENTRY POINT (Lines 899-999)

### 5.1 `main()` (Lines 899-999)

**Purpose**: CLI interface and orchestration

**Flow**:

1. **Parse arguments** (Lines 900-932)

   - `--dataset`: cifar100_lt_if100, inaturalist2018, imagenet_lt
   - `--routing`: dense or top_k
   - `--top_k`: K for top-k routing
   - `--epochs`, `--batch_size`, `--lr`
   - `--lambda_lb`, `--lambda_h`
   - `--log-file`: Optional log file path

2. **Setup logging** (Lines 936-963)

   - If `--log-file` provided:
     - Create `TeeOutput` class to write to both stdout and file
     - Redirect stdout
     - Log start time

3. **Update config** (Lines 966-983)

   - Load dataset config from `DATASET_CONFIGS_GATING`
   - Override with command-line arguments
   - Update CONFIG dict

4. **Train** (Line 986)

   - Call `train_gating(CONFIG)`

5. **Cleanup** (Lines 990-995)
   - Restore stdout
   - Close log file if opened

---

## KEY ARCHITECTURAL DECISIONS

### 1. Feature Extraction

- Uses lightweight `GatingFeatureBuilder` (not full feature extractor)
- Features are class-count independent: `7*E + 3` dimensions
- Captures: entropy, top-k mass, confidence, disagreement, ensemble statistics

### 2. Routing Strategies

- **Dense**: Softmax over all experts (full mixture)
- **Top-K**: Sparse routing with noisy top-k (encourages specialization)

### 3. Loss Components

1. **Mixture NLL** (core): Maximum likelihood of mixture model
2. **Load-balancing**: Prevents routing collapse (only for top_k)
3. **Entropy regularization**: Encourages diversity in expert usage
4. **Responsibility loss**: EM-style alignment (soft matching)
5. **Prior regularizer**: Group-aware prior knowledge (head/tail specialization)

### 4. Data Splits

- **Expert split**: 90% of train (used for expert training)
- **Gating split**: 10% of train (used for gating training) - **prevents overfitting**
- **Val split**: Balanced subset for validation
- **Test split**: Final evaluation (not used here)

### 5. Long-Tail Handling

- Class weights for loss reweighting
- Group priors for head/tail specialization
- Balanced accuracy metric (equal weight to head and tail)

---

## TRAINING PIPELINE SUMMARY

```
1. Load expert logits (pre-computed, calibrated)
   ↓
2. Create GatingNetwork (feature extractor + MLP + router)
   ↓
3. For each epoch:
   a. Build features from logits → posteriors
   b. Forward through MLP → gating logits
   c. Route (softmax/top-k) → expert weights
   d. Compute mixture posterior = Σ w_e * p_e
   e. Compute losses (NLL + LB + Entropy + Resp + Prior)
   f. Backward + optimize
   ↓
4. Validate periodically
   ↓
5. Save best model (by balanced accuracy)
   ↓
6. Export checkpoints and history
```

---

## OUTPUT FILES

- **Checkpoints**: `./checkpoints/gating_map/{dataset}/best_gating.pth`
- **Training history**: `./results/gating_map/{dataset}/training_history.json`
- **Log file**: If `--log-file` specified, all output is logged there

---

## USAGE EXAMPLES

```bash
# Dense routing (full mixture)
python train_gating_map.py --routing dense --dataset cifar100_lt_if100

# Top-K routing (sparse)
python train_gating_map.py --routing top_k --top_k 2 --dataset cifar100_lt_if100

# With logging
python train_gating_map.py --routing dense --log-file logs/gating_dense.log
```
