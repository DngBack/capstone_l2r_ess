"""
Beautiful Gating Network Architecture Diagram
Similar to the sample but more polished and detailed.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'normal'

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Color scheme
color_feature = '#FFE5E5'  # Light pink/red
color_mlp = '#E5F5E5'      # Light green
color_router = '#E5E5FF'   # Light blue
color_text = '#2C3E50'     # Dark blue-gray
color_arrow = '#34495E'    # Dark gray

# Helper function to create rounded boxes
def create_box(x, y, width, height, text, color, text_color=color_text, 
               fontsize=10, fontweight='normal', alpha=0.8):
    """Create a rounded rectangle box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        facecolor=color, 
        edgecolor='black',
        linewidth=1.5,
        alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=fontsize, fontweight=fontweight, color=text_color)
    return box

# Helper function to create arrows
def create_arrow(x1, y1, x2, y2, color=color_arrow, style='->', linewidth=2):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, 
        color=color, 
        linewidth=linewidth,
        zorder=1
    )
    ax.add_patch(arrow)
    return arrow

# Helper function to create diamond (decision point)
def create_diamond(x, y, width, height, text, color, text_color=color_text, fontsize=10):
    """Create a diamond shape"""
    # Create diamond points
    points = np.array([
        [x, y + height/2],      # top
        [x + width/2, y],       # right
        [x, y - height/2],      # bottom
        [x - width/2, y]        # left
    ])
    diamond = mpatches.Polygon(points, closed=True, 
                               facecolor=color, edgecolor='black', 
                               linewidth=1.5, alpha=0.8)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color=text_color)
    return diamond

# ============================================================================
# FEATURE EXTRACTION LAYER (Top Section)
# ============================================================================
y_start = 15
x_center = 5

# Input: Expert Posteriors
create_box(x_center, y_start, 4, 0.8, 
           "Expert Posteriors\n[B, E, C]", 
           color_feature, fontsize=9, fontweight='bold')

# Feature Extractor
y = y_start - 1.2
create_box(x_center, y, 4.5, 1.0,
           "Feature Extractor\n(Posteriors + Uncertainty Features)",
           color_feature, fontsize=9, fontweight='bold')

# Feature details (smaller text)
y = y - 0.8
feature_text = "• Flattened Posteriors: [B, E×C]\n• Expert Features: [B, 3×E]\n  (entropy, confidence, margin)\n• Global Features: [B, 5]\n  (disagreement, KL, variance, ...)"
ax.text(x_center, y, feature_text, ha='center', va='top', 
        fontsize=8, color=color_text, style='italic')

# Output: Features
y = y - 1.5
create_box(x_center, y, 4, 0.8,
           "Features\n[B, D] where D = E×C + 3×E + 5",
           color_feature, fontsize=9, fontweight='bold')

# ============================================================================
# MLP LAYER (Middle Section)
# ============================================================================
y = y - 1.2

# Layer 1: FC(D → 256) + LayerNorm + ReLU + Dropout
create_box(x_center, y, 4.5, 1.0,
           "FC (D → 256) + LayerNorm + ReLU",
           color_mlp, fontsize=9, fontweight='bold')

y = y - 0.5
create_box(x_center, y, 3, 0.6,
           "Dropout (p = 0.1)",
           color_mlp, fontsize=8)

# Layer 2: FC(256 → 128) + LayerNorm + ReLU + Dropout
y = y - 1.0
create_box(x_center, y, 4.5, 1.0,
           "FC (256 → 128) + LayerNorm + ReLU",
           color_mlp, fontsize=9, fontweight='bold')

y = y - 0.5
create_box(x_center, y, 3, 0.6,
           "Dropout (p = 0.1)",
           color_mlp, fontsize=8)

# Layer 3: FC(128 → E) - Output layer
y = y - 1.0
create_box(x_center, y, 3.5, 0.8,
           "FC (128 → E)",
           color_mlp, fontsize=9, fontweight='bold')

# Output: Logits
y = y - 1.0
create_box(x_center, y, 3.5, 0.8,
           "Gating Logits\n[B, E]",
           color_mlp, fontsize=9, fontweight='bold')

# ============================================================================
# ROUTER LAYER (Bottom Section)
# ============================================================================
y = y - 1.2

# Router decision diamond
diamond = create_diamond(x_center, y, 2.5, 1.2,
                         "Router\nStrategy",
                         color_router, fontsize=9)

# Router options (side by side)
y_router = y - 1.5
x_left = x_center - 2.5
x_right = x_center + 2.5

# Dense Softmax Router
create_box(x_left, y_router, 2.2, 1.0,
           "Dense\nSoftmax",
           color_router, fontsize=9, fontweight='bold')
ax.text(x_left, y_router - 0.7, "σ(g(x))", ha='center', va='center',
        fontsize=8, style='italic', color=color_text)

# Top-K Router (alternative)
create_box(x_right, y_router, 2.2, 1.0,
           "Top-K\nNoisy",
           color_router, fontsize=9, fontweight='bold')
ax.text(x_right, y_router - 0.7, "TopK + σ", ha='center', va='center',
        fontsize=8, style='italic', color=color_text)

# Final output
y = y_router - 1.5
create_box(x_center, y, 3.5, 0.8,
           "Expert Weights\n[B, E] (Simplex)",
           color_router, fontsize=9, fontweight='bold')

# ============================================================================
# ARROWS
# ============================================================================
# Vertical arrows
arrow_y_start = y_start - 0.4
arrow_y_end = y + 0.4

# Main vertical flow
create_arrow(x_center, arrow_y_start, x_center, arrow_y_end, 
             color=color_arrow, linewidth=2.5)

# Router branches
create_arrow(x_center, y - 0.6, x_left, y_router + 0.5, 
             color=color_arrow, linewidth=1.5)
create_arrow(x_center, y - 0.6, x_right, y_router + 0.5, 
             color=color_arrow, linewidth=1.5)

# Router to output
create_arrow(x_left, y_router - 0.5, x_center, y + 0.4, 
             color=color_arrow, linewidth=1.5)
create_arrow(x_right, y_router - 0.5, x_center, y + 0.4, 
             color=color_arrow, linewidth=1.5)

# ============================================================================
# TITLE AND ANNOTATIONS
# ============================================================================
# Title
ax.text(5, 15.8, "Gating Network Architecture", 
        ha='center', va='center', fontsize=16, fontweight='bold', color=color_text)

# Section labels
ax.text(1.5, 13.5, "Feature\nExtraction", ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_text,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color_feature, linewidth=2))

ax.text(1.5, 8, "MLP\nNetwork", ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_text,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color_mlp, linewidth=2))

ax.text(1.5, 3, "Routing\nStrategy", ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_text,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color_router, linewidth=2))

# Legend/Notes
notes = "Notes:\n• E = Number of Experts\n• C = Number of Classes\n• D = E×C + 3×E + 5\n• LayerNorm used instead of BatchNorm\n• Xavier uniform initialization"
ax.text(8.5, 8, notes, ha='left', va='center', fontsize=8, 
        color=color_text, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', 
                 edgecolor='gray', linewidth=1, alpha=0.7))

plt.tight_layout()
output_path_png = 'gating_architecture.png'
output_path_pdf = 'gating_architecture.pdf'
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.savefig(output_path_pdf, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
print(f"✓ Saved {output_path_png} and {output_path_pdf}")
# Don't show on Windows to avoid blocking
# plt.show()

