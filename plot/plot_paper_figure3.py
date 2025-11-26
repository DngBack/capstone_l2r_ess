#!/usr/bin/env python3
"""
Vẽ biểu đồ Balanced and Worst-group Errors như Figure 3 trong paper
"Learning to Reject Meets Long-Tail Learning"

Figure 3: Balanced and worst-group errors as functions of proportion of rejections.
So sánh các phương pháp: Chow, CSS, Chow [BCE], Plug-in [Balanced], Chow [DRO], Plug-in [Worst]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


# ================================
# Data structures
# ================================

class MethodData:
    """Lưu trữ dữ liệu cho một phương pháp."""
    def __init__(self, name: str, color: str, marker: str, linestyle: str = '-'):
        self.name = name
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.rejection_rates: List[float] = []
        self.balanced_errors: List[float] = []
        self.worst_group_errors: List[float] = []
    
    def add_point(self, rejection_rate: float, balanced_error: float, worst_group_error: float):
        """Thêm một điểm dữ liệu."""
        self.rejection_rates.append(rejection_rate)
        self.balanced_errors.append(balanced_error)
        self.worst_group_errors.append(worst_group_error)
    
    def get_sorted_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trả về dữ liệu đã sắp xếp theo rejection rate."""
        if len(self.rejection_rates) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Sort by rejection rate
        indices = np.argsort(self.rejection_rates)
        r = np.array(self.rejection_rates)[indices]
        e_bal = np.array(self.balanced_errors)[indices]
        e_wst = np.array(self.worst_group_errors)[indices]
        return r, e_bal, e_wst


# ================================
# Data loading
# ================================

def load_from_json(json_path: Path, method_name: str) -> Optional[MethodData]:
    """Load dữ liệu từ JSON file (từ run_balanced_plugin_ce_only.py hoặc tương tự)."""
    if not json_path.exists():
        print(f"Warning: {json_path} not found, skipping {method_name}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    method = MethodData(method_name, 'blue', 'o')
    
    # Extract RC curve data from test split
    if 'rc_curve' in data and 'test' in data['rc_curve']:
        rc_data = data['rc_curve']['test']
        rejection_rates = rc_data.get('rejection_rates', [])
        balanced_errors = rc_data.get('balanced_errors', [])
        worst_group_errors = rc_data.get('worst_group_errors', [])
        
        for r, e_bal, e_wst in zip(rejection_rates, balanced_errors, worst_group_errors):
            method.add_point(float(r), float(e_bal), float(e_wst))
    
    # Alternative: extract from results_per_cost
    elif 'results_per_cost' in data:
        for result in data['results_per_cost']:
            test_metrics = result.get('test_metrics', {})
            rejection_rate = 1.0 - test_metrics.get('coverage', 0.0)
            balanced_error = test_metrics.get('balanced_error', 1.0)
            worst_group_error = test_metrics.get('worst_group_error', 1.0)
            method.add_point(rejection_rate, balanced_error, worst_group_error)
    
    return method


def create_manual_data(dataset: str = 'CIFAR-100') -> Dict[str, MethodData]:
    """
    Tạo dữ liệu thủ công - bạn có thể chỉnh sửa số liệu ở đây.
    Format: (rejection_rate, balanced_error, worst_group_error)
    
    Số liệu mẫu từ paper Table 2 và Figure 3 - bạn có thể thay thế bằng số liệu thực tế của mình.
    """
    methods = {}
    
    # Chow method
    chow = MethodData("Chow", 'black', 'o', '-')
    # Số liệu mẫu từ paper (CIFAR-100) - THAY ĐỔI THEO SỐ LIỆU CỦA BẠN
    if dataset == 'CIFAR-100':
        chow.add_point(0.0, 0.509, 0.883)
        chow.add_point(0.1, 0.450, 0.800)
        chow.add_point(0.2, 0.400, 0.720)
        chow.add_point(0.3, 0.350, 0.650)
        chow.add_point(0.4, 0.300, 0.580)
        chow.add_point(0.5, 0.250, 0.500)
        chow.add_point(0.6, 0.200, 0.420)
        chow.add_point(0.7, 0.150, 0.350)
        chow.add_point(0.8, 0.100, 0.280)
    methods['Chow'] = chow
    
    # CSS method
    css = MethodData("CSS", 'gray', 's', '-')
    if dataset == 'CIFAR-100':
        css.add_point(0.0, 0.483, 0.785)
        css.add_point(0.1, 0.430, 0.700)
        css.add_point(0.2, 0.380, 0.620)
        css.add_point(0.3, 0.330, 0.550)
        css.add_point(0.4, 0.280, 0.480)
    methods['CSS'] = css
    
    # Chow [BCE]
    chow_bce = MethodData("Chow [BCE]", 'orange', '^', '-')
    if dataset == 'CIFAR-100':
        chow_bce.add_point(0.0, 0.359, 0.570)
        chow_bce.add_point(0.1, 0.320, 0.500)
        chow_bce.add_point(0.2, 0.280, 0.430)
        chow_bce.add_point(0.3, 0.240, 0.370)
        chow_bce.add_point(0.4, 0.200, 0.310)
        chow_bce.add_point(0.5, 0.160, 0.250)
        chow_bce.add_point(0.6, 0.120, 0.200)
        chow_bce.add_point(0.7, 0.090, 0.150)
        chow_bce.add_point(0.8, 0.060, 0.100)
    methods['Chow [BCE]'] = chow_bce
    
    # Chow [DRO]
    chow_dro = MethodData("Chow [DRO]", 'purple', 'v', '-')
    if dataset == 'CIFAR-100':
        chow_dro.add_point(0.0, 0.325, 0.333)
        chow_dro.add_point(0.1, 0.290, 0.300)
        chow_dro.add_point(0.2, 0.255, 0.270)
        chow_dro.add_point(0.3, 0.220, 0.240)
        chow_dro.add_point(0.4, 0.185, 0.210)
        chow_dro.add_point(0.5, 0.150, 0.180)
        chow_dro.add_point(0.6, 0.120, 0.150)
        chow_dro.add_point(0.7, 0.090, 0.120)
        chow_dro.add_point(0.8, 0.065, 0.095)
    methods['Chow [DRO]'] = chow_dro
    
    # Plug-in [Balanced] và Plug-in [Worst] sẽ được load từ JSON
    # Nếu không có JSON, bạn có thể thêm số liệu ở đây:
    
    return methods


# ================================
# Plotting
# ================================

def plot_figure3_comparison(
    methods: Dict[str, MethodData],
    dataset_name: str,
    save_path: Path,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Vẽ biểu đồ giống Figure 3 trong paper.
    
    Args:
        methods: Dictionary chứa MethodData cho các phương pháp
        dataset_name: Tên dataset (CIFAR-100, ImageNet, iNaturalist)
        save_path: Đường dẫn để lưu biểu đồ
        figsize: Kích thước figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Subplot 1: Balanced Error
    ax1 = axes[0]
    # Subplot 2: Worst-group Error
    ax2 = axes[1]
    
    # Colors và markers theo paper (có thể điều chỉnh)
    method_configs = {
        'Chow': {'color': 'black', 'marker': 'o', 'linestyle': '-'},
        'CSS': {'color': 'gray', 'marker': 's', 'linestyle': '-'},
        'Chow [BCE]': {'color': 'orange', 'marker': '^', 'linestyle': '-'},
        'Plug-in [Balanced]': {'color': 'green', 'marker': 'd', 'linestyle': '-'},
        'Chow [DRO]': {'color': 'purple', 'marker': 'v', 'linestyle': '-'},
        'Plug-in [Worst]': {'color': 'blue', 'marker': 'p', 'linestyle': '-'},
    }
    
    # Vẽ từng phương pháp
    for method_name, method_data in methods.items():
        r, e_bal, e_wst = method_data.get_sorted_data()
        
        if len(r) == 0:
            continue
        
        config = method_configs.get(method_name, {
            'color': 'black',
            'marker': 'o',
            'linestyle': '-'
        })
        
        # Plot balanced error
        ax1.plot(
            r, e_bal,
            marker=config['marker'],
            color=config['color'],
            linestyle=config['linestyle'],
            label=method_name,
            linewidth=2,
            markersize=6,
            alpha=0.8
        )
        
        # Plot worst-group error
        ax2.plot(
            r, e_wst,
            marker=config['marker'],
            color=config['color'],
            linestyle=config['linestyle'],
            label=method_name,
            linewidth=2,
            markersize=6,
            alpha=0.8
        )
    
    # Format subplot 1: Balanced Error
    ax1.set_xlabel('Proportion of Rejections', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Balanced Error', fontsize=12, fontweight='bold')
    ax1.set_title(f'Balanced Error\n({dataset_name})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9, loc='best')
    ax1.set_xlim([0, 1])
    ax1.set_ylim(bottom=0)
    
    # Format subplot 2: Worst-group Error
    ax2.set_xlabel('Proportion of Rejections', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Worst-group Error', fontsize=12, fontweight='bold')
    ax2.set_title(f'Worst-group Error\n({dataset_name})', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9, loc='best')
    ax2.set_xlim([0, 1])
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure to: {save_path}")


# ================================
# Main
# ================================

def main():
    parser = argparse.ArgumentParser(
        description="Vẽ biểu đồ Balanced và Worst-group Errors như Figure 3 trong paper"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR-100',
        choices=['CIFAR-100', 'ImageNet', 'iNaturalist'],
        help='Dataset name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/paper_figures',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--plugin-balanced-json',
        type=str,
        default=None,
        help='Path to Plug-in [Balanced] JSON results (from run_balanced_plugin_ce_only.py)'
    )
    parser.add_argument(
        '--plugin-worst-json',
        type=str,
        default=None,
        help='Path to Plug-in [Worst] JSON results'
    )
    parser.add_argument(
        '--manual-data',
        action='store_true',
        help='Use manual data instead of loading from JSON'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = {}
    
    # Load Plug-in [Balanced] từ JSON nếu có
    if args.plugin_balanced_json:
        plugin_balanced = load_from_json(
            Path(args.plugin_balanced_json),
            "Plug-in [Balanced]"
        )
        if plugin_balanced:
            methods['Plug-in [Balanced]'] = plugin_balanced
    
    # Load Plug-in [Worst] từ JSON nếu có
    if args.plugin_worst_json:
        plugin_worst = load_from_json(
            Path(args.plugin_worst_json),
            "Plug-in [Worst]"
        )
        if plugin_worst:
            methods['Plug-in [Worst]'] = plugin_worst
    
    # Nếu không có JSON, thử load từ default paths
    if not args.manual_data:
        dataset_path_map = {
            'CIFAR-100': 'cifar100_lt_if100',
            'ImageNet': 'imagenet_lt',
            'iNaturalist': 'inaturalist2018'
        }
        
        dataset_path = dataset_path_map.get(args.dataset, 'cifar100_lt_if100')
        default_plugin_path = Path(f'./results/ltr_plugin/{dataset_path}/ltr_plugin_ce_only_balanced.json')
        
        if default_plugin_path.exists() and 'Plug-in [Balanced]' not in methods:
            plugin_balanced = load_from_json(default_plugin_path, "Plug-in [Balanced]")
            if plugin_balanced:
                methods['Plug-in [Balanced]'] = plugin_balanced
                print(f"✓ Loaded Plug-in [Balanced] from {default_plugin_path}")
    
    # Load hoặc tạo dữ liệu cho các phương pháp khác
    if args.manual_data:
        manual_methods = create_manual_data(args.dataset)
        methods.update(manual_methods)
        print("✓ Using manual data from create_manual_data() function")
    else:
        # Load manual data cho các baseline methods
        manual_methods = create_manual_data(args.dataset)
        # Chỉ thêm các methods chưa có
        for name, method_data in manual_methods.items():
            if name not in methods:
                methods[name] = method_data
        print("✓ Loaded baseline methods from create_manual_data()")
        print("  Note: Edit create_manual_data() to update baseline numbers")
    
    # Đảm bảo có ít nhất một phương pháp
    if len(methods) == 0:
        print("⚠️  No methods loaded! Creating example with manual data...")
        methods = create_manual_data()
    
    # Vẽ biểu đồ
    save_path = output_dir / f'figure3_{args.dataset.lower().replace(" ", "_")}.png'
    plot_figure3_comparison(methods, args.dataset, save_path)
    
    print(f"\n✓ Completed! Plot saved to: {save_path}")
    print(f"\n📝 To add more methods:")
    print(f"   1. Edit create_manual_data() function in this script")
    print(f"   2. Or provide JSON files using --plugin-balanced-json, --plugin-worst-json")


if __name__ == '__main__':
    main()

