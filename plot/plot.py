import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# --- 1. Load dữ liệu từ JSON ---
def load_gating_data(json_path: str = './results/ltr_plugin/cifar100_lt_if100/ltr_plugin_gating_balanced.json'):
    """Load dữ liệu từ JSON file của gating method."""
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Warning: {json_path} not found. Skipping Gating data.")
        return None, None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract test data
    if 'rc_curve' in data and 'test' in data['rc_curve']:
        test_data = data['rc_curve']['test']
        rejection_rates = test_data.get('rejection_rates', [])
        balanced_errors = test_data.get('balanced_errors', [])
        
        # Convert to numpy arrays
        rejection_rates = np.array(rejection_rates)
        balanced_errors = np.array(balanced_errors)
        
        return rejection_rates, balanced_errors
    
    return None, None

def load_gating_worst_data(json_path: str = './results/ltr_plugin/cifar100_lt_if100/ltr_plugin_gating_worst.json'):
    """Load dữ liệu từ JSON file của gating worst-group method."""
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Warning: {json_path} not found. Skipping Gating Worst data.")
        return None, None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract RC curve data
    if 'rc_curve' in data:
        rc_data = data['rc_curve']
        rejection_rates = rc_data.get('rejection_rates', [])
        worst_group_errors = rc_data.get('worst_group_errors', [])
        
        # Convert to numpy arrays
        rejection_rates = np.array(rejection_rates)
        worst_group_errors = np.array(worst_group_errors)
        
        return rejection_rates, worst_group_errors
    
    return None, None

# Load dữ liệu từ JSON
gating_rejections, gating_balanced = load_gating_data()
gating_worst_rejections, gating_worst_errors = load_gating_worst_data()

# In thông tin để xác nhận đã load dữ liệu
if gating_rejections is not None and gating_balanced is not None:
    print(f"✓ Loaded Gating Balanced data from JSON: {len(gating_rejections)} points")
if gating_worst_rejections is not None and gating_worst_errors is not None:
    print(f"✓ Loaded Gating Worst data from JSON: {len(gating_worst_rejections)} points")

# --- 2. Tạo dữ liệu baseline ---
proportions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# Balanced Error data
data_mean_balanced = {
    'Proportion of Rejections': proportions,
    'Chow': [0.59, 0.56, 0.54, 0.50, 0.47, 0.47, 0.47, 0.48, 0.50],
    'CSS': [0.59, 0.57, 0.54, 0.52, 0.49, 0.45, 0.41, 0.38, 0.39],
    'Chow [BCE]': [0.52, 0.49, 0.44, 0.39, 0.35, 0.31, 0.27, 0.25, 0.26],
    'Plug-in [Balanced]': [0.52, 0.48, 0.43, 0.36, 0.29, 0.23, 0.17, 0.13, 0.09],
}

# Worst-group Error data
data_mean_worst = {
    'Proportion of Rejections': proportions,
    'Chow': [0.85, 0.85, 0.86, 0.86, 0.87, 0.88, 0.90, 0.93, 0.98],
    'CSS': [0.85, 0.84, 0.84, 0.81, 0.79, 0.76, 0.72, 0.68, 0.70],
    'Chow [DRO]': [0.55, 0.49, 0.46, 0.42, 0.38, 0.35, 0.30, 0.22, 0.13],
    'Plug-in [Worst]': [0.55, 0.49, 0.44, 0.39, 0.36, 0.33, 0.29, 0.21, 0.12],
}

# Thêm dữ liệu Gating Balanced nếu có
if gating_rejections is not None and gating_balanced is not None:
    if len(gating_rejections) > 0:
        gating_data_balanced = {
            'Proportion of Rejections': gating_rejections,
            'ARE + Plug-in [Balanced]': gating_balanced
        }
    else:
        gating_data_balanced = None
else:
    gating_data_balanced = None

# Thêm dữ liệu Gating Worst nếu có
if gating_worst_rejections is not None and gating_worst_errors is not None:
    if len(gating_worst_rejections) > 0:
        gating_data_worst = {
            'Proportion of Rejections': gating_worst_rejections,
            'ARE + Plug-in [Worst]': gating_worst_errors
        }
    else:
        gating_data_worst = None
else:
    gating_data_worst = None

# Độ lệch chuẩn
data_std_balanced = {
    'Chow': [0.01] * len(proportions),
    'CSS': [0.015] * len(proportions),
    'Chow [BCE]': [0.02, 0.02, 0.03, 0.03, 0.03, 0.035, 0.035, 0.04, 0.04],
    'Plug-in [Balanced]': [0.01] * len(proportions),
}

data_std_worst = {
    'Chow': [0.01] * len(proportions),
    'CSS': [0.015] * len(proportions),
    'Chow [DRO]': [0.02, 0.02, 0.025, 0.025, 0.03, 0.03, 0.03, 0.03, 0.03],
    'Plug-in [Worst]': [0.01] * len(proportions),
}

# Chuẩn bị dữ liệu Balanced Error
df_mean_balanced = pd.DataFrame(data_mean_balanced)
df_mean_balanced_melted = df_mean_balanced.melt(
    id_vars=['Proportion of Rejections'],
    var_name='Method',
    value_name='Balanced Error'
)

if gating_data_balanced is not None:
    df_gating_balanced = pd.DataFrame(gating_data_balanced)
    df_gating_balanced_melted = df_gating_balanced.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Balanced Error'
    )
    df_mean_balanced_melted = pd.concat([df_mean_balanced_melted, df_gating_balanced_melted], ignore_index=True)

df_std_balanced = pd.DataFrame({'Proportion of Rejections': proportions, **data_std_balanced})
df_std_balanced_melted = df_std_balanced.melt(
    id_vars=['Proportion of Rejections'],
    var_name='Method',
    value_name='Std Dev'
)

if gating_data_balanced is not None:
    gating_std_balanced = pd.DataFrame({
        'Proportion of Rejections': gating_rejections,
        'ARE + Plug-in [Balanced]': [0.01] * len(gating_rejections)
    })
    gating_std_balanced_melted = gating_std_balanced.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Std Dev'
    )
    df_std_balanced_melted = pd.concat([df_std_balanced_melted, gating_std_balanced_melted], ignore_index=True)

df_balanced = pd.merge(df_mean_balanced_melted, df_std_balanced_melted, on=['Proportion of Rejections', 'Method'], how='outer')
df_balanced['Std Dev'] = df_balanced['Std Dev'].fillna(0.01)

# Chuẩn bị dữ liệu Worst-group Error
df_mean_worst = pd.DataFrame(data_mean_worst)
df_mean_worst_melted = df_mean_worst.melt(
    id_vars=['Proportion of Rejections'],
    var_name='Method',
    value_name='Worst-group Error'
)

if gating_data_worst is not None:
    df_gating_worst = pd.DataFrame(gating_data_worst)
    df_gating_worst_melted = df_gating_worst.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Worst-group Error'
    )
    df_mean_worst_melted = pd.concat([df_mean_worst_melted, df_gating_worst_melted], ignore_index=True)

df_std_worst = pd.DataFrame({'Proportion of Rejections': proportions, **data_std_worst})
df_std_worst_melted = df_std_worst.melt(
    id_vars=['Proportion of Rejections'],
    var_name='Method',
    value_name='Std Dev'
)

if gating_data_worst is not None:
    gating_std_worst = pd.DataFrame({
        'Proportion of Rejections': gating_worst_rejections,
        'ARE + Plug-in [Worst]': [0.01] * len(gating_worst_rejections)
    })
    gating_std_worst_melted = gating_std_worst.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Std Dev'
    )
    df_std_worst_melted = pd.concat([df_std_worst_melted, gating_std_worst_melted], ignore_index=True)

df_worst = pd.merge(df_mean_worst_melted, df_std_worst_melted, on=['Proportion of Rejections', 'Method'], how='outer')
df_worst['Std Dev'] = df_worst['Std Dev'].fillna(0.01)

# --- 3. Thiết lập Biểu đồ ---
sns.set_theme(style="darkgrid")

# Định nghĩa markers và colors
markers_balanced = {
    'Chow': 'o',
    'CSS': '+',
    'Chow [BCE]': 'X',
    'Plug-in [Balanced]': 's',
    'ARE + Plug-in [Balanced]': 'D'
}

markers_worst = {
    'Chow': 'o',
    'CSS': '+',
    'Chow [DRO]': 'X',
    'Plug-in [Worst]': 's',
    'ARE + Plug-in [Worst]': 'D'
}

colors = {
    'ARE + Plug-in [Balanced]': '#9370DB',  # Tím
    'ARE + Plug-in [Worst]': '#9370DB',  # Tím
}

# Tạo figure với 2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- 4. Vẽ Balanced Error (subplot 1) ---
for method in df_balanced['Method'].unique():
    subset = df_balanced[df_balanced['Method'] == method]
    marker = markers_balanced.get(method, 'o')
    color = colors.get(method, None)
    
    sns.lineplot(
        data=subset,
        x='Proportion of Rejections',
        y='Balanced Error',
        label=method,
        marker=marker,
        color=color,
        markersize=10,
        markeredgecolor='white' if method in ['Chow [BCE]', 'Plug-in [Balanced]', 'ARE + Plug-in [Balanced]'] else None,
        markeredgewidth=2 if method in ['Chow [BCE]', 'Plug-in [Balanced]', 'ARE + Plug-in [Balanced]'] else None,
        linewidth=3,
        zorder=3,
        err_style=None,
        ci=None,
        dashes=False,
        ax=ax1
    )

ax1.set_ylim(0.0, 0.65)
ax1.set_xlim(0.0, 0.8)
ax1.set_yticks(np.arange(0.0, 0.7, 0.1))
ax1.set_xlabel('Proportion of Rejections', fontsize=18)
ax1.set_ylabel('Balanced Error', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(title=None, loc='lower left', fontsize=14, frameon=True, edgecolor='black')
ax1.grid(True, axis='both', linestyle='-', alpha=0.5)

# --- 5. Vẽ Worst-group Error (subplot 2) ---
for method in df_worst['Method'].unique():
    subset = df_worst[df_worst['Method'] == method]
    marker = markers_worst.get(method, 'o')
    color = colors.get(method, None)
    
    sns.lineplot(
        data=subset,
        x='Proportion of Rejections',
        y='Worst-group Error',
        label=None,  # Bỏ chú thích
        marker=marker,
        color=color,
        markersize=10,
        markeredgecolor='white' if method in ['Chow [DRO]', 'Plug-in [Worst]', 'ARE + Plug-in [Worst]'] else None,
        markeredgewidth=2 if method in ['Chow [DRO]', 'Plug-in [Worst]', 'ARE + Plug-in [Worst]'] else None,
        linewidth=3,
        zorder=3,
        err_style='band',
        ci=None,
        dashes=False,
        ax=ax2
    )

ax2.set_ylim(0.0, 1.05)
ax2.set_xlim(0.0, 0.8)
ax2.set_yticks(np.arange(0.0, 1.1, 0.2))
ax2.set_xlabel('Proportion of Rejections', fontsize=18)
ax2.set_ylabel('Worst-group Error', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.grid(True, axis='both', linestyle='-', alpha=0.5)

plt.tight_layout()

# Lưu biểu đồ
output_path = './results/paper_figures/combined_error_comparison.png'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_path}")

plt.show()
