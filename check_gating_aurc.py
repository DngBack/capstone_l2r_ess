#!/usr/bin/env python3
"""
Kiểm tra AURC của phương pháp Gating từ JSON file
"""

import json
from pathlib import Path

json_path = Path('./results/ltr_plugin/cifar100_lt_if100/ltr_plugin_gating_balanced.json')

if not json_path.exists():
    print(f"Error: {json_path} not found!")
    exit(1)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*70)
print("KIỂM TRA AURC CỦA PLUG-IN [GATING]")
print("="*70)

# Lấy AURC từ JSON (nếu có)
if 'rc_curve' in data and 'test' in data['rc_curve']:
    test_data = data['rc_curve']['test']
    
    # AURC đã được tính sẵn trong JSON
    aurc_balanced = test_data.get('aurc_balanced', None)
    aurc_worst_group = test_data.get('aurc_worst_group', None)
    aurc_balanced_08 = test_data.get('aurc_balanced_coverage_ge_0_2', None)
    
    rejection_rates = test_data.get('rejection_rates', [])
    balanced_errors = test_data.get('balanced_errors', [])
    worst_group_errors = test_data.get('worst_group_errors', [])
    
    print(f"\n📊 Dữ liệu từ JSON:")
    print(f"  Số điểm dữ liệu: {len(rejection_rates)}")
    print(f"\n  Rejection rates: {rejection_rates}")
    print(f"\n  Balanced errors: {balanced_errors}")
    print(f"\n  Worst-group errors: {worst_group_errors}")
    
    print(f"\n{'='*70}")
    print(f"📈 AURC ĐÃ ĐƯỢC TÍNH SẴN TRONG JSON:")
    print(f"{'='*70}")
    if aurc_balanced is not None:
        print(f"  AURC (Balanced Error) = {aurc_balanced:.4f}")
    if aurc_worst_group is not None:
        print(f"  AURC (Worst-group Error) = {aurc_worst_group:.4f}")
    if aurc_balanced_08 is not None:
        print(f"  AURC (Balanced, coverage >= 0.2) = {aurc_balanced_08:.4f}")
    
    # Tính lại AURC để xác nhận
    print(f"\n{'='*70}")
    print(f"🔍 TÍNH LẠI AURC ĐỂ XÁC NHẬN:")
    print(f"{'='*70}")
    
    if len(rejection_rates) > 1 and len(balanced_errors) > 1:
        # Tính AURC bằng trapezoidal integration
        aurc_calculated = 0.0
        print(f"\n  Chi tiết tính toán từng segment:")
        print(f"  {'Segment':<20} {'Width':<15} {'Avg Height':<15} {'Area':<15}")
        print(f"  {'-'*65}")
        
        for i in range(len(rejection_rates) - 1):
            r1, r2 = rejection_rates[i], rejection_rates[i+1]
            e1, e2 = balanced_errors[i], balanced_errors[i+1]
            width = r2 - r1
            avg_height = (e1 + e2) / 2.0
            area = width * avg_height
            aurc_calculated += area
            print(f"  [{r1:.3f}, {r2:.3f}]   {width:<15.4f} {avg_height:<15.4f} {area:<15.4f}")
        
        print(f"\n  AURC (tính lại) = {aurc_calculated:.4f}")
        
        if aurc_balanced is not None:
            diff = abs(aurc_calculated - aurc_balanced)
            print(f"  AURC (từ JSON) = {aurc_balanced:.4f}")
            print(f"  Chênh lệch = {diff:.6f}")
            if diff < 0.0001:
                print(f"  ✓ Khớp với giá trị trong JSON!")
            else:
                print(f"  ⚠️  Có sự khác biệt nhỏ (có thể do làm tròn)")
    
    # Tính AURC cho worst-group
    if len(rejection_rates) > 1 and len(worst_group_errors) > 1:
        aurc_worst_calculated = 0.0
        for i in range(len(rejection_rates) - 1):
            r1, r2 = rejection_rates[i], rejection_rates[i+1]
            e1, e2 = worst_group_errors[i], worst_group_errors[i+1]
            width = r2 - r1
            avg_height = (e1 + e2) / 2.0
            aurc_worst_calculated += width * avg_height
        
        print(f"\n  AURC Worst-group (tính lại) = {aurc_worst_calculated:.4f}")
        if aurc_worst_group is not None:
            print(f"  AURC Worst-group (từ JSON) = {aurc_worst_group:.4f}")

print(f"\n{'='*70}")
print(f"✅ KẾT LUẬN:")
print(f"{'='*70}")
if aurc_balanced is not None:
    print(f"  AURC của Plug-in [Gating] (Balanced Error) = {aurc_balanced:.4f}")
if aurc_worst_group is not None:
    print(f"  AURC của Plug-in [Gating] (Worst-group Error) = {aurc_worst_group:.4f}")

