#!/usr/bin/env python3
"""
Script để tạo Jupyter notebook từ demo_single_image_comparison.py

Usage:
    python create_demo_notebook.py
"""

import json
from pathlib import Path

def create_notebook():
    """Tạo notebook từ demo script."""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Cell 1: Markdown - Title
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Demo: So Sánh Phương Pháp MoE + Plugin vs Paper Method (CE + Plugin)\n",
            "\n",
            "Notebook này demo việc so sánh phương pháp của bạn (3 Experts + Gating + Plugin) với paper method (CE + Plugin) trên một ảnh tail class.\n",
            "\n",
            "## Nội dung:\n",
            "1. Load models và data\n",
            "2. Chọn một ảnh tail class\n",
            "3. Chạy qua cả 2 pipelines\n",
            "4. So sánh kết quả và visualize"
        ]
    })
    
    # Cell 2: Setup imports
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup imports\n",
            "import sys\n",
            "from pathlib import Path\n",
            "import json\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn.functional as F\n",
            "import matplotlib.pyplot as plt\n",
            "import torchvision\n",
            "from torchvision import transforms\n",
            "from PIL import Image\n",
            "import seaborn as sns\n",
            "from typing import Dict, List, Tuple, Optional\n",
            "\n",
            "# Add project root to path\n",
            "project_root = Path.cwd()\n",
            "if str(project_root) not in sys.path:\n",
            "    sys.path.insert(0, str(project_root))\n",
            "\n",
            "# Import project modules\n",
            "from src.models.experts import Expert\n",
            "from src.models.gating_network_map import GatingNetwork, GatingMLP\n",
            "from src.models.gating import GatingFeatureBuilder\n",
            "from src.data.datasets import get_eval_augmentations\n",
            "\n",
            "# Import demo functions\n",
            "from demo_single_image_comparison import (\n",
            "    load_class_to_group,\n",
            "    load_test_sample_with_image,\n",
            "    load_ce_expert,\n",
            "    load_all_experts,\n",
            "    load_gating_network,\n",
            "    load_plugin_params,\n",
            "    paper_method_pipeline,\n",
            "    our_method_pipeline,\n",
            "    visualize_comparison,\n",
            "    DATASET, DEVICE, OUTPUT_DIR, EXPERT_DISPLAY_NAMES\n",
            ")\n",
            "\n",
            "sns.set_style(\"whitegrid\")\n",
            "plt.rcParams['figure.figsize'] = (16, 10)\n",
            "\n",
            "print(\"✅ Imports successful!\")"
        ]
    })
    
    # Cell 3: Configuration
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Configuration"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configuration\n",
            "class_idx = None  # None = random tail class, or specify class index\n",
            "seed = 42\n",
            "\n",
            "# Set random seed\n",
            "torch.manual_seed(seed)\n",
            "np.random.seed(seed)\n",
            "\n",
            "# Create output directory\n",
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(f\"📁 Dataset: {DATASET}\")\n",
            "print(f\"📁 Device: {DEVICE}\")"
        ]
    })
    
    # Cell 4: Load data
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Models và Data"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load class-to-group mapping\n",
            "class_to_group = load_class_to_group()\n",
            "\n",
            "# Load CIFAR-100 class names\n",
            "dataset = torchvision.datasets.CIFAR100(root=\"./data\", train=False, download=False)\n",
            "class_names = dataset.classes\n",
            "\n",
            "# Load a tail class sample\n",
            "image_tensor, true_label, image_array, class_name = load_test_sample_with_image(class_idx=class_idx)\n",
            "\n",
            "# Check if it's tail\n",
            "is_tail = class_to_group[true_label].item() == 1\n",
            "print(f\"\\n{'✅ Tail class' if is_tail else '⚠️  Not tail class'} - Group: {class_to_group[true_label].item()}\")"
        ]
    })
    
    # Cell 5: Load models
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load models\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"Loading Models...\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "ce_expert = load_ce_expert()\n",
            "experts = load_all_experts()\n",
            "gating = load_gating_network()\n",
            "\n",
            "# Load plugin parameters for both methods\n",
            "ce_plugin_alpha, ce_plugin_mu, ce_plugin_cost = load_plugin_params(method=\"ce_only\", mode=\"balanced\")\n",
            "moe_plugin_alpha, moe_plugin_mu, moe_plugin_cost = load_plugin_params(method=\"moe\", mode=\"worst\")"
        ]
    })
    
    # Cell 6: Run inference
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Run Inference trên Cả 2 Methods"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Paper Method (CE + Plugin)\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"📊 PAPER METHOD (CE + Plugin)\")\n",
            "print(\"=\"*70)\n",
            "baseline_result = paper_method_pipeline(\n",
            "    image_tensor, \n",
            "    ce_expert, \n",
            "    class_to_group,\n",
            "    ce_plugin_alpha, \n",
            "    ce_plugin_mu, \n",
            "    ce_plugin_cost\n",
            ")\n",
            "\n",
            "print(f\"\\nPrediction: Class {baseline_result['prediction']} ({class_names[baseline_result['prediction']]})\")\n",
            "print(f\"Max Probability: {baseline_result['confidence']:.4f}\")\n",
            "if 'plugin_confidence' in baseline_result:\n",
            "    print(f\"Plugin Reweighted Score: {baseline_result['plugin_confidence']:.4f}\")\n",
            "print(f\"Reject: {'YES' if baseline_result['reject'] else 'NO'}\")\n",
            "is_correct = baseline_result['prediction'] == true_label\n",
            "print(f\"Correct: {'[YES]' if is_correct else '[NO]'}\")"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Our method\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"🚀 OUR METHOD (MoE + Gating + Plugin)\")\n",
            "print(\"=\"*70)\n",
            "our_result = our_method_pipeline(image_tensor)\n",
            "\n",
            "print(f\"\\nExpert Predictions: {our_result['expert_predictions']} ({EXPERT_DISPLAY_NAMES})\")\n",
            "print(f\"Gating Weights: {our_result['gating_weights']}\")\n",
            "print(f\"Plugin Prediction: Class {our_result['prediction']} ({class_names[our_result['prediction']]})\")\n",
            "print(f\"Plugin Confidence: {our_result['confidence']:.4f}\")\n",
            "print(f\"Reject: {'YES' if our_result['reject'] else 'NO'}\")\n",
            "print(f\"Correct: {'✅' if our_result['prediction'] == true_label else '❌'}\")"
        ]
    })
    
    # Cell 7: Visualization
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Visualization và So Sánh"
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create comprehensive visualization\n",
            "fig = visualize_comparison(image_array, true_label, baseline_result, our_result, class_names, class_to_group)\n",
            "plt.show()\n",
            "\n",
            "# Save figure\n",
            "output_path = OUTPUT_DIR / f\"demo_comparison_class_{true_label}.png\"\n",
            "fig.savefig(output_path, dpi=150, bbox_inches='tight')\n",
            "print(f\"\\n💾 Saved visualization to {output_path}\")"
        ]
    })
    
    # Cell 8: Detailed comparison
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Print detailed comparison\n",
            "print(\"\\n\" + \"=\"*70)\n",
            "print(\"📊 DETAILED COMPARISON\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "print(f\"\\n📷 Sample Info:\")\n",
            "print(f\"   True Label: {true_label} ({class_names[true_label]})\")\n",
            "print(f\"   Group: {'Tail' if is_tail else 'Head'}\")\n",
            "\n",
            "print(f\"\\n{'='*70}\")\n",
            "print(f\"{'Metric':<30} | {'Paper Baseline':<20} | {'Our Method':<20}\")\n",
            "print(f\"{'='*70}\")\n",
            "\n",
            "baseline_pred = baseline_result['prediction']\n",
            "our_pred = our_result['prediction']\n",
            "baseline_correct = '✅' if baseline_pred == true_label else '❌'\n",
            "our_correct = '✅' if our_pred == true_label else '❌'\n",
            "\n",
            "print(f\"{'Prediction':<30} | {baseline_correct} {baseline_pred} ({class_names[baseline_pred][:15]):<15} | {our_correct} {our_pred} ({class_names[our_pred][:15]):<15}\")\n",
            "print(f\"{'Confidence':<30} | {baseline_result['confidence']:.4f}{' '*15} | {our_result['confidence']:.4f}{' '*15}\")\n",
            "\n",
            "baseline_rej = 'REJECT' if baseline_result['reject'] else 'ACCEPT'\n",
            "our_rej = 'REJECT' if our_result['reject'] else 'ACCEPT'\n",
            "print(f\"{'Rejection Decision':<30} | {baseline_rej:<20} | {our_rej:<20}\")\n",
            "\n",
            "print(f\"\\n{'='*70}\")\n",
            "print(f\"\\n📈 SUMMARY:\")\n",
            "if baseline_pred == true_label and our_pred != true_label:\n",
            "    print(\"   ⚠️  Paper baseline is correct, our method is wrong\")\n",
            "elif baseline_pred != true_label and our_pred == true_label:\n",
            "    print(\"   ✅ Our method is correct, paper baseline is wrong!\")\n",
            "elif baseline_pred == true_label and our_pred == true_label:\n",
            "    print(\"   ✅ Both methods are correct\")\n",
            "    if our_result['confidence'] > baseline_result['confidence']:\n",
            "        print(\"   💡 Our method has higher confidence\")\n",
            "else:\n",
            "    print(\"   ❌ Both methods are wrong\")\n",
            "    if our_result['confidence'] < baseline_result['confidence']:\n",
            "        print(\"   💡 Our method has lower confidence (better uncertainty estimation)\")"
        ]
    })
    
    # Save notebook
    output_path = Path("demo_comparison_single_image.ipynb")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created notebook: {output_path}")
    print("\nTo use the notebook:")
    print("1. Make sure demo_single_image_comparison.py is in the same directory")
    print("2. Open demo_comparison_single_image.ipynb in Jupyter")
    print("3. Run all cells")


if __name__ == "__main__":
    create_notebook()

