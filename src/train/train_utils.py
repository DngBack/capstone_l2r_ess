"""
Training utilities and metrics functions for paper-compliant training.
Contains helper functions from train_ce_expert_paper_final.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


def evaluate_metrics(model, dataloader, class_to_group, device):
    """
    Compute balanced and worst-group errors in the paper-defined manner.

    Args:
        model: torch.nn.Module to evaluate.
        dataloader: Dataloader that yields (inputs, labels).
        class_to_group: Tensor mapping class indices to group ids (0=head, 1=tail).
        device: torch device to run evaluation on.

    Returns:
        dict with balanced_error, worst_group_error, per-group errors and accuracy.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Class-wise errors (paper definition)
    class_errors = []
    for cls in range(100):
        cls_indices = all_targets == cls
        if cls_indices.sum() > 0:
            cls_error = (
                (all_preds[cls_indices] != all_targets[cls_indices]).float().mean()
            )
            class_errors.append(cls_error.item())
        else:
            class_errors.append(0.0)

    # Balanced Error = average of class-wise errors
    balanced_error = np.mean(class_errors)

    # Worst-group Error = max of class-wise errors
    worst_group_error = np.max(class_errors)

    # Head/Tail errors (for diagnostics)
    head_indices = class_to_group[all_targets] == 0
    tail_indices = class_to_group[all_targets] == 1

    head_error = (
        (all_preds[head_indices] != all_targets[head_indices]).float().mean()
        if head_indices.sum() > 0
        else torch.tensor(0.0)
    )
    tail_error = (
        (all_preds[tail_indices] != all_targets[tail_indices]).float().mean()
        if tail_indices.sum() > 0
        else torch.tensor(0.0)
    )

    # Standard accuracy
    standard_acc = (all_preds == all_targets).float().mean()

    return {
        "balanced_error": balanced_error,
        "worst_group_error": worst_group_error,
        "head_error": head_error.item() if isinstance(head_error, torch.Tensor) else head_error,
        "tail_error": tail_error.item() if isinstance(tail_error, torch.Tensor) else tail_error,
        "standard_acc": standard_acc.item(),
    }


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE) using max probability per sample.

    Args:
        probs: Tensor (N, num_classes) with softmax outputs.
        labels: Tensor (N,) containing ground-truth class indices.
        n_bins: Number of histogram bins (default 15).

    Returns:
        float ECE value.
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * (avg_conf_in_bin - accuracy_in_bin).abs()
    return ece.item()


def get_probs_and_labels(model, dataloader, device):
    """
    Get probabilities and labels from model and dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device to use
    
    Returns:
        probs: Tensor (N, num_classes) - softmax probabilities
        labels: Tensor (N,) - ground truth labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(target)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_probs, all_labels

