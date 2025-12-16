#!/usr/bin/env python3
"""
Test script to verify train/val/test split implementation.

This script tests that:
1. get_dataloaders_fixed returns 3 loaders (train, val, test)
2. Validation loader is used for training decisions
3. Test loader is used only for final evaluation
4. No data leakage between splits
"""

import sys
import ast
import re

def test_function_signature():
    """Test that get_dataloaders_fixed returns 3 values."""
    print("\n" + "=" * 80)
    print("TEST 1: Function Signature")
    print("=" * 80)
    
    with open('updated_main_gaze.py', 'r') as f:
        content = f.read()
    
    # Check that function returns 3 loaders
    pattern = r'return train_loader, val_loader, test_loader,'
    if re.search(pattern, content):
        print("✓ get_dataloaders_fixed returns train_loader, val_loader, test_loader")
        return True
    else:
        print("✗ get_dataloaders_fixed does not return 3 loaders")
        return False

def test_main_unpacking():
    """Test that main() unpacks 3 loaders."""
    print("\n" + "=" * 80)
    print("TEST 2: Main Function Unpacking")
    print("=" * 80)
    
    with open('updated_main_gaze.py', 'r') as f:
        content = f.read()
    
    # Check that main() unpacks 3 loaders
    pattern = r'train_loader, val_loader, test_loader, gaze_stats = get_dataloaders_fixed'
    if re.search(pattern, content):
        print("✓ main() correctly unpacks 3 loaders")
        return True
    else:
        print("✗ main() does not unpack 3 loaders correctly")
        return False

def test_validation_usage():
    """Test that validation loader is used for training decisions."""
    print("\n" + "=" * 80)
    print("TEST 3: Validation Loader Usage")
    print("=" * 80)
    
    with open('updated_main_gaze.py', 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check 1: Validation loader used for evaluation
    pattern = r'evaluate_model_comprehensive\s*\(\s*model,\s*val_loader,\s*device,\s*stats_tracker,\s*"val"'
    if re.search(pattern, content):
        print("✓ val_loader used for evaluation")
        checks.append(True)
    else:
        print("✗ val_loader not used for evaluation")
        checks.append(False)
    
    # Check 2: Validation loss used for scheduler
    if re.search(r"metric_for_sched = val_stats\['loss'\]", content):
        print("✓ val_stats['loss'] used for learning rate scheduler")
        checks.append(True)
    else:
        print("✗ val_stats['loss'] not used for scheduler")
        checks.append(False)
    
    # Check 3: Validation loss used for checkpoint saving
    if re.search(r"if val_stats\['loss'\] < best_loss:", content):
        print("✓ val_stats['loss'] used for checkpoint saving decisions")
        checks.append(True)
    else:
        print("✗ val_stats['loss'] not used for checkpoint saving")
        checks.append(False)
    
    return all(checks)

def test_test_usage():
    """Test that test loader is used only for final evaluation."""
    print("\n" + "=" * 80)
    print("TEST 4: Test Loader Usage (Final Evaluation Only)")
    print("=" * 80)
    
    with open('updated_main_gaze.py', 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check 1: Test loader evaluation happens after training loop
    # Look for test evaluation after the training loop (after "Load best model checkpoint")
    pattern = r'Load best model checkpoint.*?evaluate_model_comprehensive.*test_loader'
    if re.search(pattern, content, re.DOTALL):
        print("✓ test_loader used for evaluation after training completes")
        checks.append(True)
    else:
        print("✗ test_loader evaluation not found after training")
        checks.append(False)
    
    # Check 2: Test loader NOT used in training loop
    # Extract training loop section
    training_loop_pattern = r'for epoch in range\(hyps\[\'epochs\'\]\):(.*?)# Load best model checkpoint'
    training_loop_match = re.search(training_loop_pattern, content, re.DOTALL)
    
    if training_loop_match:
        training_loop_content = training_loop_match.group(1)
        if 'test_loader' not in training_loop_content:
            print("✓ test_loader NOT used inside training loop")
            checks.append(True)
        else:
            print("✗ test_loader is being used inside training loop (DATA LEAKAGE!)")
            checks.append(False)
    else:
        print("? Could not locate training loop to verify")
        checks.append(False)
    
    # Check 3: Test statistics not used for training decisions
    if 'test_stats' not in training_loop_content if training_loop_match else True:
        print("✓ test_stats NOT used for training decisions")
        checks.append(True)
    else:
        print("✗ test_stats used in training loop (DATA LEAKAGE!)")
        checks.append(False)
    
    return all(checks)

def test_statistics_tracking():
    """Test that statistics track train/val/test separately."""
    print("\n" + "=" * 80)
    print("TEST 5: Statistics Tracking")
    print("=" * 80)
    
    with open('updated_main_gaze.py', 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check record_epoch signature
    if re.search(r'def record_epoch\(self, epoch, train_stats, val_stats, test_stats=None\):', content):
        print("✓ record_epoch accepts train_stats, val_stats, test_stats")
        checks.append(True)
    else:
        print("✗ record_epoch signature not updated correctly")
        checks.append(False)
    
    # Check that val_ prefix is used instead of eval_
    if re.search(r"'val_acc':", content) and re.search(r"'val_loss':", content):
        print("✓ Statistics use 'val_' prefix for validation metrics")
        checks.append(True)
    else:
        print("✗ Statistics don't use proper naming convention")
        checks.append(False)
    
    # Check that test statistics are recorded
    test_acc_pattern = r"('test_acc':|last_epoch\['test_acc'\])"
    test_loss_pattern = r"('test_loss':|last_epoch\['test_loss'\])"
    if re.search(test_acc_pattern, content) and re.search(test_loss_pattern, content):
        print("✓ Test statistics are tracked separately")
        checks.append(True)
    else:
        print("✗ Test statistics not tracked")
        checks.append(False)
    
    return all(checks)

def test_documentation():
    """Test that documentation reflects the changes."""
    print("\n" + "=" * 80)
    print("TEST 6: Documentation")
    print("=" * 80)
    
    with open('updated_main_gaze.py', 'r') as f:
        content = f.read()
    
    checks = []
    
    # Check function docstring
    if 'Returns 3 loaders: train_loader, val_loader, test_loader' in content:
        print("✓ Function docstring mentions 3 loaders")
        checks.append(True)
    else:
        print("✗ Function docstring not updated")
        checks.append(False)
    
    # Check that it mentions proper usage
    if 'val_loader for: LR scheduling, checkpoint saving' in content:
        print("✓ Documentation mentions proper val_loader usage")
        checks.append(True)
    else:
        print("✗ Documentation doesn't mention val_loader usage")
        checks.append(False)
    
    if 'test_loader ONLY for: Final evaluation' in content:
        print("✓ Documentation mentions test_loader is for final evaluation only")
        checks.append(True)
    else:
        print("✗ Documentation doesn't mention test_loader usage")
        checks.append(False)
    
    return all(checks)

def test_no_use_train_val_split_parameter():
    """Test that use_train_val_split parameter was removed."""
    print("\n" + "=" * 80)
    print("TEST 7: Removed Unnecessary Parameters")
    print("=" * 80)
    
    with open('updated_main_gaze.py', 'r') as f:
        content = f.read()
    
    # Check that use_train_val_split is not in main() signature
    main_signature = re.search(r'def main\([^)]+\):', content)
    if main_signature and 'use_train_val_split' not in main_signature.group(0):
        print("✓ use_train_val_split parameter removed from main()")
        return True
    else:
        print("✗ use_train_val_split parameter still exists in main()")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TRAIN/VAL/TEST SPLIT IMPLEMENTATION TESTS")
    print("=" * 80)
    
    results = []
    
    results.append(test_function_signature())
    results.append(test_main_unpacking())
    results.append(test_validation_usage())
    results.append(test_test_usage())
    results.append(test_statistics_tracking())
    results.append(test_documentation())
    results.append(test_no_use_train_val_split_parameter())
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe implementation correctly:")
        print("  1. Returns 3 loaders (train, val, test)")
        print("  2. Uses val_loader for training decisions (LR scheduling, checkpointing)")
        print("  3. Uses test_loader ONLY for final evaluation")
        print("  4. Tracks train/val/test statistics separately")
        print("  5. Has proper documentation")
        print("\n✓ NO DATA LEAKAGE - Test set not used for training decisions!")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print(f"\nFailed: {total - passed} test(s)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
