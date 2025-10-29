#!/usr/bin/env python3
"""
Diagnostics for CE Baseline Logits
==================================

- Loads logits/targets from outputs/logits/cifar100_lt_if100/ce_baseline
- Sanity checks: shape, NaNs/Inf, softmax sums
- Metrics on val/test: overall accuracy, head/tail accuracy, balanced error
- Calibration: ECE (15 bins), NLL
- Confidence & entropy stats per split
- Writes report JSON to results/ltr_plugin/cifar100_lt_if100/ce_baseline_diagnostics.json
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

SPLITS = ["val", "test"]
ROOT = Path("./outputs/logits/cifar100_lt_if100/ce_baseline")
SPLITS_DIR = Path("./data/cifar100_lt_if100_splits_fixed")
OUT_DIR = Path("./results/ltr_plugin/cifar100_lt_if100")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_logits_targets(split: str):
    logits = torch.load(ROOT / f"{split}_logits.pt", map_location=DEVICE).float()
    targets_path = ROOT / f"{split}_targets.pt"
    if targets_path.exists():
        targets = torch.load(targets_path, map_location=DEVICE).long()
    else:
        # fallback to indices-based reconstruction
        import torchvision
        with open(SPLITS_DIR / f"{split}_indices.json", "r", encoding="utf-8") as f:
            indices = json.load(f)
        ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=False)
        targets = torch.tensor([ds.targets[i] for i in indices], dtype=torch.long, device=DEVICE)
    return logits, targets


def build_class_to_group(num_classes=100, tail_leq=20):
    with open(SPLITS_DIR / "train_class_counts.json", "r", encoding="utf-8") as f:
        counts = json.load(f)
    if isinstance(counts, dict):
        counts = [counts[str(i)] for i in range(num_classes)]
    counts = np.array(counts)
    tail = counts <= tail_leq
    ctg = np.zeros(num_classes, dtype=np.int64)
    ctg[tail] = 1
    return torch.tensor(ctg, dtype=torch.long, device=DEVICE)


def ece_score(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc = (pred[mask] == labels[mask]).float().mean()
            avg_conf = conf[mask].mean()
            ece += (mask.float().mean()) * torch.abs(avg_conf - acc)
    return float(ece.item())


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, class_to_group: torch.Tensor) -> Dict:
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    acc = (pred == labels).float().mean().item()

    # head/tail
    groups = class_to_group[labels]
    head_mask = groups == 0
    tail_mask = groups == 1
    head_acc = (pred[head_mask] == labels[head_mask]).float().mean().item() if head_mask.any() else float("nan")
    tail_acc = (pred[tail_mask] == labels[tail_mask]).float().mean().item() if tail_mask.any() else float("nan")

    # balanced error across head/tail
    head_err = 1.0 - head_acc if not np.isnan(head_acc) else 1.0
    tail_err = 1.0 - tail_acc if not np.isnan(tail_acc) else 1.0
    balanced_error = float((head_err + tail_err) / 2.0)

    # calibration
    ece = ece_score(probs, labels, n_bins=15)
    nll = -F.log_softmax(logits, dim=-1)[torch.arange(len(labels), device=logits.device), labels].mean().item()

    # confidence & entropy
    conf = probs.max(dim=-1)[0]
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    return {
        "num_samples": int(len(labels)),
        "accuracy": acc,
        "head_accuracy": head_acc,
        "tail_accuracy": tail_acc,
        "balanced_error": balanced_error,
        "ece_15": ece,
        "nll": nll,
        "conf_mean": float(conf.mean().item()),
        "conf_std": float(conf.std().item()),
        "entropy_mean": float(entropy.mean().item()),
        "entropy_std": float(entropy.std().item()),
    }


def sanity_checks(logits: torch.Tensor) -> Dict:
    report = {}
    report["shape"] = list(logits.shape)
    report["has_nan"] = bool(torch.isnan(logits).any().item())
    report["has_inf"] = bool(torch.isinf(logits).any().item())
    probs = F.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    report["softmax_sum_mean"] = float(sums.mean().item())
    report["softmax_sum_std"] = float(sums.std().item())
    return report


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    class_to_group = build_class_to_group()

    report = {"splits": {}}
    for split in SPLITS:
        print(f"Analyzing split: {split}")
        logits, targets = load_logits_targets(split)
        report["splits"][split] = {
            "sanity": sanity_checks(logits),
            "metrics": compute_metrics(logits, targets, class_to_group),
        }
        print(f"  shape={tuple(logits.shape)}  acc={report['splits'][split]['metrics']['accuracy']:.4f}  "
              f"bal_err={report['splits'][split]['metrics']['balanced_error']:.4f}  ece={report['splits'][split]['metrics']['ece_15']:.3f}")

    out_path = OUT_DIR / "ce_baseline_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved diagnostics to: {out_path}")


if __name__ == "__main__":
    main()
