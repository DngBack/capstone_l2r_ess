#!/usr/bin/env python3
"""
Export CE expert logits for tunev, val, and test splits
=======================================================

- Loads checkpoint: ./checkpoints/experts/cifar100_lt_if100/ce_expert_best.pth
- Uses CIFAR-100 data and split indices from: ./data/cifar100_lt_if100_splits_fixed
- Saves logits to: ./outputs/logits/cifar100_lt_if100/ce_baseline/{split}_logits.pt
- Also saves targets alongside when available: {split}_targets.pt

This prepares inputs required by run_balanced_plugin_ce_only.py
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import sys
sys.path.append('./src')
from models.experts import Expert  # type: ignore


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SPLITS_DIR = Path('./data/cifar100_lt_if100_splits_fixed')
# Default checkpoint (can be overridden via --ckpt)
CKPT_PATH = Path('./checkpoints/experts/cifar100_lt_if100/ce_expert_best.pth')
OUT_DIR = Path('./outputs/logits/cifar100_lt_if100/ce_baseline')


def build_model(num_classes: int = 100) -> nn.Module:
    # Match the exact structure used in train_ce_expert_paper_final.py
    # which uses nn.Sequential(backbone, nn.Linear(...))
    from models.backbones.resnet_cifar import CIFARResNet32
    backbone = CIFARResNet32(dropout_rate=0.0, init_weights=True)
    model = nn.Sequential(
        backbone,
        nn.Linear(backbone.get_feature_dim(), num_classes)
    )
    return model


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str = 'module.') -> Dict[str, Any]:
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path, device: str) -> None:
    raw = torch.load(ckpt_path, map_location=device)
    # Handle various formats: direct state_dict or wrapped
    if isinstance(raw, dict) and 'state_dict' in raw and isinstance(raw['state_dict'], dict):
        state_dict = raw['state_dict']
    elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        state_dict = raw
    else:
        # Try common nesting keys
        for key in ['model', 'model_state', 'ema_state', 'net']:
            if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
                state_dict = raw[key]
                break
        else:
            raise RuntimeError(f"Unsupported checkpoint format: keys={list(raw.keys()) if isinstance(raw, dict) else type(raw)}")

    # Strip DistributedDataParallel prefixes
    state_dict = _strip_prefix_if_present(state_dict, 'module.')
    # Some trainings may save calibration temperature; ignore unexpected keys safely
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys: {unexpected}")


def get_transforms():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return transform_test


def load_cifar_test() -> torchvision.datasets.CIFAR100:
    transform = get_transforms()
    ds = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return ds


def load_indices(split: str) -> Tuple[torch.Tensor, bool]:
    # The repo uses names: tunev_indices.json, val_indices.json, test_indices.json
    path = SPLITS_DIR / f"{split}_indices.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split indices: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    return torch.tensor(idx, dtype=torch.long), True


@torch.no_grad()
def infer_split(model: nn.Module, ds: torchvision.datasets.CIFAR100, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    targets_list = []
    for i in indices.tolist():
        x, y = ds[i]
        x = x.unsqueeze(0).to(DEVICE)
        out = model(x)
        logits_list.append(out.squeeze(0).cpu())
        targets_list.append(torch.tensor(y, dtype=torch.long))
    logits = torch.stack(logits_list, dim=0)
    targets = torch.stack(targets_list, dim=0)
    return logits, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=str(CKPT_PATH), help='Path to CE baseline checkpoint .pth')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading model...")
    model = build_model(num_classes=100).to(DEVICE)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint from: {ckpt_path}")
    load_checkpoint_into_model(model, ckpt_path, DEVICE)

    print("Loading CIFAR-100 test set and split indices...")
    cifar_test = load_cifar_test()

    for split in ["tunev", "val", "test"]:
        print(f"\n=== Inference split: {split} ===")
        indices, _ = load_indices(split)
        logits, targets = infer_split(model, cifar_test, indices)
        torch.save(logits.to(torch.float32), OUT_DIR / f"{split}_logits.pt")
        torch.save(targets.to(torch.long), OUT_DIR / f"{split}_targets.pt")
        print(f"Saved: {(OUT_DIR / f'{split}_logits.pt').as_posix()}  shape={tuple(logits.shape)}")

    print("\nAll splits exported. Ready for run_balanced_plugin_ce_only.py")


if __name__ == "__main__":
    main()


