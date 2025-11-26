import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# --- 1. Load dữ liệu từ JSON (Gating Worst method) ---
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

# Load dữ liệu Gating Worst từ JSON
gating_worst_rejections, gating_worst_errors = load_gating_worst_data()

# In thông tin để xác nhận đã load dữ liệu
if gating_worst_rejections is not None and gating_worst_errors is not None:
    print(f"✓ Loaded Gating Worst data from JSON: {len(gating_worst_rejections)} points")
    print(f"  Rejection rates range: [{float(gating_worst_rejections.min()):.3f}, {float(gating_worst_rejections.max()):.3f}]")
    print(f"  Worst-group errors range: [{float(gating_worst_errors.min()):.3f}, {float(gating_worst_errors.max()):.3f}]")
    # Tính AURC để kiểm tra
    if len(gating_worst_rejections) > 1:
        aurc = 0.0
        for i in range(len(gating_worst_rejections) - 1):
            width = float(gating_worst_rejections[i+1]) - float(gating_worst_rejections[i])
            avg_height = (float(gating_worst_errors[i]) + float(gating_worst_errors[i+1])) / 2.0
            aurc += width * avg_height
        print(f"  AURC = {aurc:.4f}")
else:
    print("⚠️  No Gating Worst data loaded from JSON")

# --- 2. Tạo dữ liệu giả lập (Mock Data) cho các phương pháp baseline ---
# Dữ liệu này được tạo để có xu hướng tương tự như trong biểu đồ của bạn.

# Trục X: Tỷ lệ từ chối
proportions = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# Worst-group Error (Giá trị trung bình) cho 4 mô hình/phương pháp
# Mỗi mảng chứa 9 điểm dữ liệu tương ứng với 9 điểm trên trục X
data_mean = {
    'Proportion of Rejections': proportions,
    'Chow': [0.85, 0.85, 0.86, 0.86, 0.87, 0.88, 0.90, 0.93, 0.98],
    'CSS': [0.85, 0.84, 0.84, 0.81, 0.79, 0.76, 0.72, 0.68, 0.70],
    'Chow [DRO]': [0.55, 0.49, 0.46, 0.42, 0.38, 0.35, 0.30, 0.22, 0.13],
    'Plug-in [Worst]': [0.55, 0.49, 0.44, 0.39, 0.36, 0.33, 0.29, 0.21, 0.12], # Điều chỉnh một chút để cuối cùng giống hơn
}

# Thêm dữ liệu Gating Worst nếu có
if gating_worst_rejections is not None and gating_worst_errors is not None:
    if len(gating_worst_rejections) > 0:
        # Tạo DataFrame riêng cho Gating Worst để merge sau
        gating_worst_data = {
            'Proportion of Rejections': gating_worst_rejections,
            'ARE + Plug-in [Worst]': gating_worst_errors
        }
        print(f"✓ Prepared Gating Worst data: {len(gating_worst_rejections)} points")
    else:
        gating_worst_data = None
else:
    gating_worst_data = None

# Độ lệch chuẩn (Standard Deviation) giả lập cho dải lỗi (shaded area)
# Các giá trị nhỏ hơn nhiều so với giá trị trung bình
data_std = {
    'Chow': [0.01] * len(proportions),
    'CSS': [0.015] * len(proportions),
    'Chow [DRO]': [0.02, 0.02, 0.025, 0.025, 0.03, 0.03, 0.03, 0.03, 0.03],
    'Plug-in [Worst]': [0.01] * len(proportions),
}

# Chuyển dữ liệu trung bình sang định dạng "long format" của Pandas cho Seaborn
df_mean = pd.DataFrame(data_mean)
df_mean_melted = df_mean.melt(
    id_vars=['Proportion of Rejections'],
    var_name='Method',
    value_name='Worst-group Error' # Tên trục Y đã thay đổi
)

# Thêm dữ liệu Gating Worst nếu có
if gating_worst_data is not None:
    df_gating_worst = pd.DataFrame(gating_worst_data)
    df_gating_worst_melted = df_gating_worst.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Worst-group Error'
    )
    # Gộp với dữ liệu baseline
    df_mean_melted = pd.concat([df_mean_melted, df_gating_worst_melted], ignore_index=True)

# Chuyển dữ liệu độ lệch chuẩn sang định dạng long format và gán vào df_mean_melted
df_std = pd.DataFrame({'Proportion of Rejections': proportions, **data_std})
df_std_melted = df_std.melt(
    id_vars=['Proportion of Rejections'],
    var_name='Method',
    value_name='Std Dev'
)

# Thêm std cho Gating Worst (nếu có)
if gating_worst_data is not None:
    gating_worst_std = pd.DataFrame({
        'Proportion of Rejections': gating_worst_rejections,
        'ARE + Plug-in [Worst]': [0.01] * len(gating_worst_rejections)  # Giả sử std nhỏ
    })
    gating_worst_std_melted = gating_worst_std.melt(
        id_vars=['Proportion of Rejections'],
        var_name='Method',
        value_name='Std Dev'
    )
    df_std_melted = pd.concat([df_std_melted, gating_worst_std_melted], ignore_index=True)

# Gộp dữ liệu trung bình và độ lệch chuẩn
df = pd.merge(df_mean_melted, df_std_melted, on=['Proportion of Rejections', 'Method'], how='outer')
# Fill NaN cho std nếu thiếu
df['Std Dev'] = df['Std Dev'].fillna(0.01)

# --- 2. Thiết lập Biểu đồ (Plotting Setup) ---
sns.set_theme(style="darkgrid") # Sử dụng theme darkgrid như trong biểu đồ

# Định nghĩa các marker và màu sắc tùy chỉnh để khớp với biểu đồ gốc
# Quan sát: Chow có marker 'o', CSS có marker '+', Chow[DRO] có marker 'x', Plug-in có marker 's'
markers = {
    'Chow': 'o',
    'CSS': '+',
    'Chow [DRO]': 'X',  # Dùng 'X' thay vì 'x' để có marker lớn hơn
    'Plug-in [Worst]': 's',  # Dùng 's' (square)
    'ARE + Plug-in [Worst]': 'D'  # Diamond marker cho ARE method
}

# Định nghĩa màu sắc - chỉ định màu tím cho phương pháp ARE
colors = {
    'ARE + Plug-in [Worst]': '#9370DB',  # Tím cho phương pháp ARE
}

plt.figure(figsize=(8, 7))

# --- 3. Vẽ Biểu đồ (The Plot) ---
for method in df['Method'].unique():
    subset = df[df['Method'] == method]
    
    # Lấy marker và màu
    marker = markers.get(method, 'o')
    color = colors.get(method, None)  # Chỉ định màu cho ARE, các phương pháp khác để None để seaborn tự chọn
    
    # Vẽ đường trung bình
    sns.lineplot(
        data=subset,
        x='Proportion of Rejections',
        y='Worst-group Error', # Tên trục Y đã thay đổi
        label=None,  # Bỏ chú thích
        marker=marker,
        color=color,  # Đặt màu cụ thể
        markersize=10,
        markeredgecolor='white' if method in ['Chow [DRO]', 'Plug-in [Worst]', 'ARE + Plug-in [Worst]'] else None,
        markeredgewidth=2 if method in ['Chow [DRO]', 'Plug-in [Worst]', 'ARE + Plug-in [Worst]'] else None,
        linewidth=3,
        zorder=3,
        err_style='band',
        ci=None,
        dashes=False,
    )
    
    # Tái tạo dải lỗi bằng cách sử dụng độ lệch chuẩn (Std Dev) giả lập
    # lower_bound = mean - std
    # upper_bound = mean + std

# --- 4. Tùy chỉnh Cuối cùng (Final Customizations) ---

# Đặt giới hạn trục Y
plt.ylim(0.0, 1.05) # Thay đổi giới hạn Y để khớp với biểu đồ mới
# Đặt giới hạn trục X (từ 0.0 đến 0.8)
plt.xlim(0.0, 0.8)

# Đặt ticks trên trục Y
plt.yticks(np.arange(0.0, 1.1, 0.2)) # Thay đổi ticks Y để khớp với biểu đồ mới

# Đặt tên trục
plt.xlabel('Proportion of Rejections', fontsize=18)
plt.ylabel('Worst-group Error', fontsize=18) # Tên trục Y đã thay đổi

# Tăng kích thước font cho ticks
plt.tick_params(axis='both', which='major', labelsize=14)

# Bỏ legend (chú giải)
# plt.legend(
#     title=None,
#     loc='lower left', # Vị trí chú giải
#     fontsize=14,
#     frameon=True,
#     edgecolor='black' # Thêm viền đen cho hộp chú giải
# )

# Hiển thị biểu đồ
plt.grid(True, axis='both', linestyle='-', alpha=0.5) # Làm nổi bật grid lines
plt.tight_layout()

# Lưu biểu đồ
output_path = './results/paper_figures/worst_group_error_comparison.png'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_path}")

plt.show()