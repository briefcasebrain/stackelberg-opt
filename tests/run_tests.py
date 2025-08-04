#!/usr/bin/env python3
"""Simple test runner that doesn't require pytest installation."""

import sys
import os
import traceback
import importlib.util
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_test_file(test_file):
    """Run tests from a single test file."""
    print(f"\n{'='*60}")
    print(f"Running tests from: {test_file.name}")
    print('='*60)
    
    # Load the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    test_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(test_module)
    except Exception as e:
        print(f"ERROR loading module: {e}")
        traceback.print_exc()
        return 0, 1
    
    # Run all test functions
    passed = 0
    failed = 0
    
    for name in dir(test_module):
        if name.startswith('test_') and callable(getattr(test_module, name)):
            test_func = getattr(test_module, name)
            try:
                print(f"\n  Running {name}...", end=' ')
                test_func()
                print("✓ PASSED")
                passed += 1
            except Exception as e:
                print("✗ FAILED")
                print(f"    Error: {e}")
                traceback.print_exc()
                failed += 1
    
    return passed, failed

def main():
    """Run all tests without pytest."""
    print("Running stackelberg-opt tests (without pytest)")
    print("Note: This is a simplified test runner. Install pytest for full functionality.")
    
    tests_dir = Path(__file__).parent / "tests"
    test_files = list(tests_dir.glob("test_*.py"))
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        passed, failed = run_test_file(test_file)
        total_passed += passed
        total_failed += failed
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total_failed} tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())