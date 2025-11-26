#!/usr/bin/env python3
"""
Tính AURC (Area Under Risk-Coverage curve) cho Plug-in [Balanced] từ dữ liệu trong plot.py
"""

# Dữ liệu Plug-in [Balanced] từ plot.py
rejection_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
balanced_errors = [0.52, 0.48, 0.43, 0.36, 0.29, 0.23, 0.17, 0.13, 0.09]

# Tính AURC bằng trapezoidal integration
# AURC = Σ[(r_{i+1} - r_i) * (e_i + e_{i+1}) / 2]
total_aurc = 0.0

print("="*60)
print("TÍNH AURC CHO PLUG-IN [BALANCED]")
print("="*60)
print(f"\n{'Rejection Rate':<20} {'Balanced Error':<20}")
print("-" * 40)
for r, e in zip(rejection_rates, balanced_errors):
    print(f"{r:<20.1f} {e:<20.2f}")

print(f"\n{'='*60}")
print("Chi tiết tính toán từng segment:")
print("-" * 60)

for i in range(len(rejection_rates) - 1):
    r1, r2 = rejection_rates[i], rejection_rates[i+1]
    e1, e2 = balanced_errors[i], balanced_errors[i+1]
    # Trapezoidal area = (r2 - r1) * (e1 + e2) / 2
    width = r2 - r1
    avg_height = (e1 + e2) / 2.0
    area = width * avg_height
    total_aurc += area
    print(f"Segment [{r1:.1f}, {r2:.1f}]: width={width:.1f}, avg_height={avg_height:.3f}, area={area:.4f}")

print(f"\n{'='*60}")
print(f"AURC (Area Under Risk-Coverage curve) = {total_aurc:.4f}")
print(f"Expected AURC (from paper Table 2) = 0.292")
print(f"Difference = {abs(total_aurc - 0.292):.4f}")
print(f"{'='*60}")

# So sánh với giá trị trong paper
paper_aurc = 0.292
if abs(total_aurc - paper_aurc) < 0.01:
    print(f"\n✓ AURC khớp với giá trị trong paper (sai số < 0.01)")
else:
    print(f"\n⚠️  AURC khác với giá trị trong paper")
    print(f"   Có thể cần điều chỉnh số liệu để khớp với paper")
