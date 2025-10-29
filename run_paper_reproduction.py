#!/usr/bin/env python3
"""
Complete Pipeline Script - Paper Reproduction
=============================================

Runs the complete pipeline to reproduce paper results:
1. Train CE expert (paper-compliant)
2. Train LtR plugin (balanced objective)
3. Train LtR plugin (worst objective)
4. Generate comparison plots

Paper Reference: "Learning to Reject Meets Long-Tail Learning" (ICLR 2024)
"""

import subprocess
import sys
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        print(f"✅ SUCCESS: {description}")
        print(f"⏱️  Time: {elapsed_time:.1f}s")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout[-500:])  # Last 500 chars
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print(f"❌ FAILED: {description}")
        print(f"⏱️  Time: {elapsed_time:.1f}s")
        print(f"Exit code: {e.returncode}")
        
        if e.stdout:
            print("STDOUT:")
            print(e.stdout[-500:])
        
        if e.stderr:
            print("STDERR:")
            print(e.stderr[-500:])
        
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def main():
    """Main pipeline function."""
    print("=" * 80)
    print("COMPLETE PIPELINE - PAPER REPRODUCTION")
    print("=" * 80)
    print("Paper: Learning to Reject Meets Long-Tail Learning (ICLR 2024)")
    print("Dataset: CIFAR-100-LT")
    print("=" * 80)
    
def check_expert_files():
    """Check if expert files exist."""
    expert_files = [
        "./checkpoints/experts_paper/best_ce_expert.pth",
        "./outputs/logits_paper/ce_expert_train_logits.pth",
        "./outputs/logits_paper/ce_expert_test_logits.pth",
        "./outputs/logits/cifar100_lt_if100/ce_baseline/test_logits.pt"
    ]
    
    all_exist = True
    for filepath in expert_files:
        if not check_file_exists(filepath, "Expert file"):
            all_exist = False
    
    return all_exist

def main():
    """Main pipeline function."""
    print("=" * 80)
    print("COMPLETE PIPELINE - PAPER REPRODUCTION")
    print("=" * 80)
    print("Paper: Learning to Reject Meets Long-Tail Learning (ICLR 2024)")
    print("Dataset: CIFAR-100-LT")
    print("=" * 80)
    
    # Check if expert already exists
    if check_expert_files():
        print("\n✅ Expert files already exist. Skipping expert training.")
        print("   If you want to retrain expert, delete the files first.")
    else:
        # Step 1: Train CE Expert
        print("\n🔧 STEP 1: Training CE Expert (Paper Compliant)")
        success = run_command(
            "python train_ce_expert_paper.py",
            "CE Expert Training"
        )
        
        if not success:
            print("❌ CE Expert training failed. Stopping pipeline.")
            return False
        
        # Check expert outputs
        if not check_expert_files():
            print("❌ Expert files missing. Stopping pipeline.")
            return False
    
    # Step 2: Train LtR Plugin (Balanced)
    print("\n🔧 STEP 2: Training LtR Plugin (Balanced Objective)")
    success = run_command(
        "python train_ltr_plugin_paper.py --objective balanced --cost_sweep",
        "LtR Plugin Training (Balanced)"
    )
    
    if not success:
        print("❌ LtR Plugin (Balanced) training failed. Stopping pipeline.")
        return False
    
    # Step 3: Train LtR Plugin (Worst)
    print("\n🔧 STEP 3: Training LtR Plugin (Worst Objective)")
    success = run_command(
        "python train_ltr_plugin_paper.py --objective worst --cost_sweep",
        "LtR Plugin Training (Worst)"
    )
    
    if not success:
        print("❌ LtR Plugin (Worst) training failed. Stopping pipeline.")
        return False
    
    # Step 4: Generate Comparison Plots
    print("\n🔧 STEP 4: Generating Comparison Plots")
    success = run_command(
        "python generate_comparison_plots.py",
        "Comparison Plots Generation"
    )
    
    if not success:
        print("⚠️  Comparison plots generation failed, but continuing...")
    
    # Final Summary
    print(f"\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    # Check final outputs
    final_files = [
        "./results/ltr_plugin_paper/ltr_plugin_balanced_paper.json",
        "./results/ltr_plugin_paper/ltr_plugin_worst_paper.json",
        "./results/ltr_plugin_paper/plots/ltr_rc_curves_balanced_paper.png",
        "./results/ltr_plugin_paper/plots/ltr_rc_curves_worst_paper.png"
    ]
    
    print("\n📊 FINAL OUTPUTS:")
    for filepath in final_files:
        check_file_exists(filepath, "Result file")
    
    print(f"\n🎉 PAPER REPRODUCTION COMPLETED!")
    print(f"📁 Results directory: ./results/ltr_plugin_paper/")
    print(f"📁 Plots directory: ./results/ltr_plugin_paper/plots/")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
