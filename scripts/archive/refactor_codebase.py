#!/usr/bin/env python3
"""
Refactor codebase: Organize files into proper structure.
"""
import shutil
from pathlib import Path
from typing import List, Tuple

# Project root - use current working directory
import os
ROOT = Path(os.getcwd())
if not (ROOT / "src").exists():
    # Try to find project root
    ROOT = Path(__file__).parent

# Directory structure
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis"
ARCHIVE_DIR = SCRIPTS_DIR / "archive"

# Main scripts (keep in root, just organize)
MAIN_SCRIPTS = [
    "train_experts.py",
    "train_ltr_plugin.py",
    "run_balanced_plugin_ce_only.py",
    "run_balanced_plugin_gating.py",
    "run_paper_reproduction.py",
    "quick_start.py",
]

# Scripts to move to scripts/
SCRIPTS_TO_MOVE = [
    "run_infer_ce_expert_logits.py",
    "visualize_all.py",
    "visualize_gating_outputs.py",
    "generate_comparison_plots.py",
]

# Analysis/test scripts to move to scripts/analysis/
ANALYSIS_SCRIPTS = [
    "analyze_data_distribution.py",
    "analyze_data_distribution_paper.py",
    "analyze_data_distribution_paper_final.py",
    "analyze_train_distribution.py",
    "test_importance_weights.py",
    "test_rejector_verification.py",
    "test_ce_only_implementation.py",
    "check_expert.py",
    "verify_training_stats.py",
    "calculate_total_samples.py",
    "manual_calculation.py",
    "find_duplicate_samples.py",
    "quick_train_analysis.py",
]

# Archive old/duplicate versions (keep most recent)
ARCHIVE_FILES = [
    # Keep: train_ce_expert_paper_final.py, Archive: train_ce_expert_paper.py
    "train_ce_expert_paper.py",
    # Keep: analyze_data_distribution_paper_final.py, Archive old ones
    # (already moved to analysis)
    # Keep: train_ltr_plugin.py, Archive old versions
    "train_ltr_plugin_paper.py",
    "train_ltr_plugin_ce_only.py",
]


def create_directories():
    """Create directory structure."""
    print("Creating directory structure...")
    SCRIPTS_DIR.mkdir(exist_ok=True)
    ANALYSIS_DIR.mkdir(exist_ok=True)
    ARCHIVE_DIR.mkdir(exist_ok=True)
    print(f"[OK] Created: {SCRIPTS_DIR}, {ANALYSIS_DIR}, {ARCHIVE_DIR}")


def move_files(file_list: List[str], dest_dir: Path, file_type: str):
    """Move files to destination directory."""
    moved = 0
    skipped = 0
    
    for filename in file_list:
        src = ROOT / filename
        dst = dest_dir / filename
        
        if not src.exists():
            print(f"  [SKIP] {filename} (not found)")
            skipped += 1
            continue
        
        if dst.exists():
            print(f"  [SKIP] {filename} (already exists in destination)")
            skipped += 1
            continue
        
        try:
            shutil.move(str(src), str(dst))
            print(f"  [OK] Moved {filename} -> {dest_dir.name}/")
            moved += 1
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            skipped += 1
    
    print(f"\n{file_type}: {moved} moved, {skipped} skipped\n")
    return moved, skipped


def main():
    """Main refactoring function."""
    print("=" * 70)
    print("REFACTORING CODEBASE")
    print("=" * 70)
    
    # Create directories
    create_directories()
    print()
    
    # Move scripts
    print("Moving scripts to scripts/...")
    move_files(SCRIPTS_TO_MOVE, SCRIPTS_DIR, "Scripts")
    
    # Move analysis scripts
    print("Moving analysis/test scripts to scripts/analysis/...")
    move_files(ANALYSIS_SCRIPTS, ANALYSIS_DIR, "Analysis scripts")
    
    # Archive old files
    print("Archiving old/duplicate files...")
    move_files(ARCHIVE_FILES, ARCHIVE_DIR, "Archived files")
    
    print("=" * 70)
    print("REFACTORING COMPLETE!")
    print("=" * 70)
    print("\nNew structure:")
    print(f"  scripts/         - Utility scripts")
    print(f"  scripts/analysis/ - Analysis and test scripts")
    print(f"  scripts/archive/  - Old/duplicate files (for reference)")
    print("\nMain scripts remain in root:")
    for script in MAIN_SCRIPTS:
        if (ROOT / script).exists():
            print(f"  [OK] {script}")


if __name__ == "__main__":
    main()

