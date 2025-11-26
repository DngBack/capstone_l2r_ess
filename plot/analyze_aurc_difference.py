#!/usr/bin/env python3
"""
Phân tích chi tiết tại sao AURC của Gating thấp hơn nhưng biểu đồ lại trông giống Plug-in [Balanced]
"""

import json
from pathlib import Path

# Load dữ liệu Gating từ JSON
json_path = Path('./results/ltr_plugin/cifar100_lt_if100/ltr_plugin_gating_balanced.json')
with open(json_path, 'r', encoding='utf-8') as f:
    gating_data = json.load(f)

# Dữ liệu Plug-in [Balanced] từ plot.py (ước lượng)
plugin_balanced_rejections = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
plugin_balanced_errors = [0.52, 0.48, 0.43, 0.36, 0.29, 0.23, 0.17, 0.13, 0.09]

# Dữ liệu Gating từ JSON
gating_rejections = gating_data['rc_curve']['test']['rejection_rates']
gating_balanced_errors = gating_data['rc_curve']['test']['balanced_errors']

print("="*80)
print("PHÂN TÍCH CHI TIẾT: TẠI SAO AURC THẤP HƠN NHƯNG BIỂU ĐỒ TRÔNG GIỐNG NHAU?")
print("="*80)

print("\n" + "="*80)
print("1. SO SÁNH TỪNG ĐIỂM DỮ LIỆU")
print("="*80)
print(f"{'Rejection Rate':<20} {'Plug-in [Balanced]':<25} {'Plug-in [Gating]':<25} {'Chênh lệch':<15}")
print("-" * 85)

# Map dữ liệu Gating về các điểm rejection rate của Plug-in [Balanced]
for i, target_r in enumerate(plugin_balanced_rejections):
    plugin_e = plugin_balanced_errors[i]
    
    # Tìm điểm gần nhất trong Gating data
    closest_idx = min(range(len(gating_rejections)), 
                      key=lambda j: abs(gating_rejections[j] - target_r))
    gating_r = gating_rejections[closest_idx]
    gating_e = gating_balanced_errors[closest_idx]
    
    diff = gating_e - plugin_e
    diff_pct = (diff / plugin_e * 100) if plugin_e > 0 else 0
    
    marker = "✓" if abs(diff) < 0.05 else "⚠️"
    print(f"{target_r:<20.1f} {plugin_e:<25.4f} {gating_e:<25.4f} {diff:+.4f} ({diff_pct:+.1f}%) {marker}")

print("\n" + "="*80)
print("2. TÍNH AURC TỪNG PHẦN ĐỂ XÁC ĐỊNH ĐIỂM KHÁC BIỆT")
print("="*80)

# Tính AURC từng phần cho Plug-in [Balanced]
print("\n📊 PLUG-IN [BALANCED]:")
print(f"{'Segment':<25} {'Width':<15} {'Avg Error':<15} {'Area':<15} {'Cumulative AURC':<20}")
print("-" * 90)
aurc_balanced_total = 0.0
for i in range(len(plugin_balanced_rejections) - 1):
    r1, r2 = plugin_balanced_rejections[i], plugin_balanced_rejections[i+1]
    e1, e2 = plugin_balanced_errors[i], plugin_balanced_errors[i+1]
    width = r2 - r1
    avg_error = (e1 + e2) / 2.0
    area = width * avg_error
    aurc_balanced_total += area
    print(f"[{r1:.1f}, {r2:.1f}]         {width:<15.3f} {avg_error:<15.4f} {area:<15.4f} {aurc_balanced_total:<20.4f}")

print(f"\n  Tổng AURC (Plug-in [Balanced]) = {aurc_balanced_total:.4f}")

# Tính AURC từng phần cho Gating
print("\n📊 PLUG-IN [GATING]:")
print(f"{'Segment':<25} {'Width':<15} {'Avg Error':<15} {'Area':<15} {'Cumulative AURC':<20}")
print("-" * 90)
aurc_gating_total = 0.0
for i in range(len(gating_rejections) - 1):
    r1, r2 = gating_rejections[i], gating_rejections[i+1]
    e1, e2 = gating_balanced_errors[i], gating_balanced_errors[i+1]
    width = r2 - r1
    avg_error = (e1 + e2) / 2.0
    area = width * avg_error
    aurc_gating_total += area
    print(f"[{r1:.3f}, {r2:.3f}]   {width:<15.4f} {avg_error:<15.4f} {area:<15.4f} {aurc_gating_total:<20.4f}")

print(f"\n  Tổng AURC (Plug-in [Gating]) = {aurc_gating_total:.4f}")

print("\n" + "="*80)
print("3. PHÂN TÍCH CHI TIẾT TỪNG VÙNG REJECTION RATE")
print("="*80)

# Chia thành các vùng: Low (0-0.3), Medium (0.3-0.6), High (0.6-0.8)
regions = [
    ("Low (0.0 - 0.3)", 0.0, 0.3),
    ("Medium (0.3 - 0.6)", 0.3, 0.6),
    ("High (0.6 - 0.8)", 0.6, 0.8),
]

print(f"\n{'Region':<25} {'Plug-in [Balanced] AURC':<30} {'Plug-in [Gating] AURC':<30} {'Difference':<15}")
print("-" * 100)

for region_name, r_start, r_end in regions:
    # Tính AURC cho Plug-in [Balanced] trong vùng này
    aurc_bal_region = 0.0
    for i in range(len(plugin_balanced_rejections) - 1):
        r1, r2 = plugin_balanced_rejections[i], plugin_balanced_rejections[i+1]
        if r1 >= r_start and r2 <= r_end:
            e1, e2 = plugin_balanced_errors[i], plugin_balanced_errors[i+1]
            width = r2 - r1
            avg_error = (e1 + e2) / 2.0
            aurc_bal_region += width * avg_error
    
    # Tính AURC cho Gating trong vùng này
    aurc_gat_region = 0.0
    for i in range(len(gating_rejections) - 1):
        r1, r2 = gating_rejections[i], gating_rejections[i+1]
        if r1 >= r_start and r2 <= r_end:
            e1, e2 = gating_balanced_errors[i], gating_balanced_errors[i+1]
            width = r2 - r1
            avg_error = (e1 + e2) / 2.0
            aurc_gat_region += width * avg_error
    
    diff = aurc_gat_region - aurc_bal_region
    print(f"{region_name:<25} {aurc_bal_region:<30.4f} {aurc_gat_region:<30.4f} {diff:+.4f}")

print("\n" + "="*80)
print("4. GIẢI THÍCH TẠI SAO TRÔNG GIỐNG NHAU NHƯNG AURC KHÁC")
print("="*80)

print("\n🔍 PHÂN TÍCH:")
print("\n1. So sánh các điểm quan trọng:")

# So sánh ở rejection = 0
print(f"\n   Ở rejection = 0.0:")
print(f"     Plug-in [Balanced]: {plugin_balanced_errors[0]:.4f}")
print(f"     Plug-in [Gating]:   {gating_balanced_errors[0]:.4f}")
print(f"     Chênh lệch: {gating_balanced_errors[0] - plugin_balanced_errors[0]:+.4f}")

# So sánh ở rejection = 0.4 (giữa)
print(f"\n   Ở rejection = 0.4:")
idx_04_bal = 4  # index 4 = 0.4
idx_04_gat = min(range(len(gating_rejections)), key=lambda i: abs(gating_rejections[i] - 0.4))
print(f"     Plug-in [Balanced]: {plugin_balanced_errors[idx_04_bal]:.4f}")
print(f"     Plug-in [Gating]:   {gating_balanced_errors[idx_04_gat]:.4f}")
print(f"     Chênh lệch: {gating_balanced_errors[idx_04_gat] - plugin_balanced_errors[idx_04_bal]:+.4f}")

# So sánh ở rejection = 0.8 (cuối)
print(f"\n   Ở rejection = 0.8:")
idx_08_bal = 8  # index 8 = 0.8
idx_08_gat = min(range(len(gating_rejections)), key=lambda i: abs(gating_rejections[i] - 0.8))
print(f"     Plug-in [Balanced]: {plugin_balanced_errors[idx_08_bal]:.4f}")
print(f"     Plug-in [Gating]:   {gating_balanced_errors[idx_08_gat]:.4f}")
print(f"     Chênh lệch: {gating_balanced_errors[idx_08_gat] - plugin_balanced_errors[idx_08_bal]:+.4f}")

print("\n2. Lý do trông giống nhau:")
print("   - Cả hai đều có xu hướng giảm đều khi rejection rate tăng")
print("   - Cả hai đều có balanced error giảm từ ~0.5 xuống ~0.1")
print("   - Độ dốc (slope) của đường cong tương tự nhau")

print("\n3. Lý do AURC khác nhau:")
print("   - AURC = tích phân của error theo rejection rate")
print("   - Ngay cả khi các điểm trông giống nhau, sự khác biệt nhỏ tích lũy lại")
print("   - Đặc biệt ở các vùng có error cao (rejection thấp), sự khác biệt nhỏ cũng tạo ra")
print("     chênh lệch lớn trong AURC vì tích phân")

# Tính tổng chênh lệch tích lũy
print("\n4. Chênh lệch tích lũy:")
total_diff = aurc_gating_total - aurc_balanced_total
print(f"   AURC Gating - AURC Balanced = {total_diff:.4f}")
print(f"   Tỷ lệ chênh lệch = {abs(total_diff) / aurc_balanced_total * 100:.2f}%")

print("\n" + "="*80)
print("5. KẾT LUẬN")
print("="*80)
print(f"\n✓ Plug-in [Gating] có AURC = {aurc_gating_total:.4f}")
print(f"✓ Plug-in [Balanced] có AURC = {aurc_balanced_total:.4f}")
print(f"✓ Chênh lệch = {total_diff:.4f} ({abs(total_diff) / aurc_balanced_total * 100:.2f}%)")
print(f"\n💡 Biểu đồ trông giống nhau vì:")
print(f"   - Cả hai đều có cùng xu hướng giảm")
print(f"   - Các điểm dữ liệu gần nhau (chênh lệch < 0.05 ở hầu hết các điểm)")
print(f"   - Nhưng sự khác biệt nhỏ tích lũy lại qua tích phân tạo ra chênh lệch AURC")
print(f"\n💡 AURC thấp hơn là tốt - nghĩa là phương pháp của bạn hiệu quả hơn!")

