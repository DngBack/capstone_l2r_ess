#!/usr/bin/env python3
"""
Test script for CE Only LtR Plugin Training
============================================

Simple test to verify the implementation works correctly.
"""

import subprocess
import sys
from pathlib import Path

def test_ce_only_implementation():
    """Test the CE only implementation with a simple run."""
    
    print("=" * 60)
    print("TESTING CE ONLY LtR PLUGIN IMPLEMENTATION")
    print("=" * 60)
    
    # Check if the script exists
    script_path = Path("train_ltr_plugin_ce_only.py")
    if not script_path.exists():
        print("❌ ERROR: train_ltr_plugin_ce_only.py not found!")
        return False
    
    print("✅ Script found: train_ltr_plugin_ce_only.py")
    
    # Test with a simple configuration (no cost sweep for faster testing)
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Run with basic parameters
        cmd = [
            sys.executable, "train_ltr_plugin_ce_only.py",
            "--objective", "balanced",
            "--optimizer", "power_iter",
            # Note: Not using --cost_sweep for faster testing
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print("✅ Basic test PASSED!")
            print("Output preview:")
            print("-" * 40)
            print(result.stdout[-500:])  # Last 500 chars
            print("-" * 40)
            return True
        else:
            print("❌ Basic test FAILED!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_cost_sweep():
    """Test the cost sweep functionality."""
    
    print("\n🧪 Testing cost sweep functionality...")
    
    try:
        # Run with cost sweep
        cmd = [
            sys.executable, "train_ltr_plugin_ce_only.py",
            "--objective", "balanced",
            "--optimizer", "power_iter",
            "--cost_sweep",
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            print("✅ Cost sweep test PASSED!")
            print("Output preview:")
            print("-" * 40)
            print(result.stdout[-500:])  # Last 500 chars
            print("-" * 40)
            return True
        else:
            print("❌ Cost sweep test FAILED!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Cost sweep test timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Cost sweep test failed with exception: {e}")
        return False

def check_output_files():
    """Check if output files are created correctly."""
    
    print("\n📁 Checking output files...")
    
    results_dir = Path("results/ltr_plugin/cifar100_lt_if100")
    
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return False
    
    # Check for expected files
    expected_files = [
        "ltr_plugin_ce_only_balanced.json",
        "ltr_plugin_ce_only_cost_sweep_balanced.json",
        "ltr_rc_curves_balanced_ce_only_test.png",
        "ltr_rc_curves_dual_ce_only_test.png",
    ]
    
    found_files = []
    for file_name in expected_files:
        file_path = results_dir / file_name
        if file_path.exists():
            found_files.append(file_name)
            print(f"✅ Found: {file_name}")
        else:
            print(f"❌ Missing: {file_name}")
    
    if len(found_files) >= 2:  # At least basic files should exist
        print(f"✅ Found {len(found_files)}/{len(expected_files)} expected files")
        return True
    else:
        print(f"❌ Only found {len(found_files)}/{len(expected_files)} expected files")
        return False

def main():
    """Main test function."""
    
    print("Starting comprehensive test of CE Only LtR Plugin...")
    
    # Test 1: Basic functionality
    basic_test_passed = test_ce_only_implementation()
    
    if not basic_test_passed:
        print("\n❌ Basic test failed. Stopping here.")
        return False
    
    # Test 2: Cost sweep functionality
    cost_sweep_passed = test_cost_sweep()
    
    # Test 3: Check output files
    files_check_passed = check_output_files()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"Cost sweep: {'✅ PASS' if cost_sweep_passed else '❌ FAIL'}")
    print(f"Output files: {'✅ PASS' if files_check_passed else '❌ FAIL'}")
    
    all_passed = basic_test_passed and cost_sweep_passed and files_check_passed
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! CE Only implementation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

