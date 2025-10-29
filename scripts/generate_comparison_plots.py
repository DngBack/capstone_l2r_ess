#!/usr/bin/env python3
"""
Comparison Plots Generator - Paper vs Our Results
=================================================

Generates comparison plots between paper results and our implementation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paper results (from Figure 3a)
PAPER_RESULTS = {
    'balanced': {
        'chow': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.52, 0.48, 0.46, 0.46, 0.48]
        },
        'css': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.59, 0.55, 0.50, 0.45, 0.41]
        },
        'chow_bce': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.52, 0.45, 0.38, 0.32, 0.26]
        },
        'plugin_balanced': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.53, 0.45, 0.35, 0.25, 0.12]
        }
    },
    'worst': {
        'chow': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.85, 0.88, 0.92, 0.95, 0.98]
        },
        'css': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.85, 0.82, 0.78, 0.75, 0.72]
        },
        'chow_dro': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.55, 0.50, 0.42, 0.35, 0.28]
        },
        'plugin_worst': {
            'rejection_rates': [0.0, 0.2, 0.4, 0.6, 0.8],
            'errors': [0.55, 0.45, 0.35, 0.25, 0.12]
        }
    }
}

def load_our_results(results_dir):
    """Load our results from JSON files."""
    results = {}
    
    # Load balanced results
    balanced_path = Path(results_dir) / 'ltr_plugin_balanced_paper.json'
    if balanced_path.exists():
        with open(balanced_path, 'r') as f:
            balanced_data = json.load(f)
        
        results['balanced'] = {
            'rejection_rates': [1.0 - r['coverage'] for r in balanced_data],
            'errors': [r['balanced_error'] for r in balanced_data]
        }
    
    # Load worst results
    worst_path = Path(results_dir) / 'ltr_plugin_worst_paper.json'
    if worst_path.exists():
        with open(worst_path, 'r') as f:
            worst_data = json.load(f)
        
        results['worst'] = {
            'rejection_rates': [1.0 - r['coverage'] for r in worst_data],
            'errors': [r['worst_group_error'] for r in worst_data]
        }
    
    return results

def plot_comparison(paper_results, our_results, output_dir):
    """Plot comparison between paper and our results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Balanced Error
    ax1.plot(paper_results['balanced']['plugin_balanced']['rejection_rates'],
             paper_results['balanced']['plugin_balanced']['errors'],
             'o-', linewidth=3, markersize=8, color='blue',
             label='Paper: Plug-in [Balanced]', alpha=0.8)
    
    if 'balanced' in our_results:
        ax1.plot(our_results['balanced']['rejection_rates'],
                 our_results['balanced']['errors'],
                 's-', linewidth=3, markersize=8, color='red',
                 label='Our: Plug-in [Balanced]', alpha=0.8)
    
    ax1.set_xlabel('Proportion of Rejections')
    ax1.set_ylabel('Balanced Error')
    ax1.set_title('Balanced Error Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # Plot 2: Worst Error
    ax2.plot(paper_results['worst']['plugin_worst']['rejection_rates'],
             paper_results['worst']['plugin_worst']['errors'],
             'o-', linewidth=3, markersize=8, color='blue',
             label='Paper: Plug-in [Worst]', alpha=0.8)
    
    if 'worst' in our_results:
        ax2.plot(our_results['worst']['rejection_rates'],
                 our_results['worst']['errors'],
                 's-', linewidth=3, markersize=8, color='red',
                 label='Our: Plug-in [Worst]', alpha=0.8)
    
    ax2.set_xlabel('Proportion of Rejections')
    ax2.set_ylabel('Worst-group Error')
    ax2.set_title('Worst-group Error Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'paper_vs_our_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")

def calculate_aurc(rejection_rates, errors):
    """Calculate Area Under Risk-Coverage curve."""
    return np.trapz(errors, rejection_rates)

def print_comparison_summary(paper_results, our_results):
    """Print comparison summary."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Balanced Error AURC
    paper_aurc_balanced = calculate_aurc(
        paper_results['balanced']['plugin_balanced']['rejection_rates'],
        paper_results['balanced']['plugin_balanced']['errors']
    )
    
    if 'balanced' in our_results:
        our_aurc_balanced = calculate_aurc(
            our_results['balanced']['rejection_rates'],
            our_results['balanced']['errors']
        )
        
        print(f"Balanced Error AURC:")
        print(f"  Paper: {paper_aurc_balanced:.4f}")
        print(f"  Our:   {our_aurc_balanced:.4f}")
        print(f"  Diff:  {abs(our_aurc_balanced - paper_aurc_balanced):.4f}")
    
    # Worst Error AURC
    paper_aurc_worst = calculate_aurc(
        paper_results['worst']['plugin_worst']['rejection_rates'],
        paper_results['worst']['plugin_worst']['errors']
    )
    
    if 'worst' in our_results:
        our_aurc_worst = calculate_aurc(
            our_results['worst']['rejection_rates'],
            our_results['worst']['errors']
        )
        
        print(f"\nWorst Error AURC:")
        print(f"  Paper: {paper_aurc_worst:.4f}")
        print(f"  Our:   {our_aurc_worst:.4f}")
        print(f"  Diff:  {abs(our_aurc_worst - paper_aurc_worst):.4f}")

def main():
    """Main function."""
    print("="*80)
    print("COMPARISON PLOTS GENERATOR")
    print("="*80)
    
    # Load our results
    results_dir = "./results/ltr_plugin_paper"
    our_results = load_our_results(results_dir)
    
    if not our_results:
        print("❌ No results found. Please run the training pipeline first.")
        return
    
    # Create output directory
    output_dir = Path(results_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    plot_comparison(PAPER_RESULTS, our_results, output_dir)
    
    # Print summary
    print_comparison_summary(PAPER_RESULTS, our_results)
    
    print(f"\n✅ Comparison plots generated successfully!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == '__main__':
    main()
