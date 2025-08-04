#!/usr/bin/env python3
"""
Comprehensive test runner for stackelberg-opt.

This script provides multiple test execution modes:
1. Quick structure tests (no dependencies)
2. Basic tests with mocked dependencies
3. Full test suite with pytest (requires dependencies)
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Change to project root for tests
os.chdir(Path(__file__).parent.parent)


def run_structure_tests():
    """Run structure tests that don't require dependencies."""
    print("=" * 60)
    print("Running Structure Tests")
    print("=" * 60)
    
    try:
        from tests.test_structure import main as structure_main
        result = structure_main()
        return result == 0
    except Exception as e:
        print(f"Structure tests failed: {e}")
        return False


def run_basic_tests():
    """Run basic tests without external dependencies."""
    print("\n" + "=" * 60)
    print("Running Basic Tests (No Dependencies)")
    print("=" * 60)
    
    try:
        from tests.test_no_deps import main as no_deps_main
        result = no_deps_main()
        return result == 0
    except Exception as e:
        print(f"Basic tests failed: {e}")
        return False


def run_mocked_tests():
    """Run tests with mocked dependencies."""
    print("\n" + "=" * 60)
    print("Running Tests with Mocked Dependencies")
    print("=" * 60)
    
    try:
        from tests.test_with_mocks import run_test_suite
        result = run_test_suite()
        return result == 0
    except Exception as e:
        print(f"Mocked tests failed: {e}")
        return False


def run_pytest():
    """Run full test suite with pytest."""
    print("\n" + "=" * 60)
    print("Running Full Test Suite with pytest")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"pytest failed: {e}")
        return False
    except ModuleNotFoundError:
        print("pytest not installed. Install with: pip install pytest")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['numpy', 'scipy', 'litellm', 'networkx']
    missing = []
    
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    return missing


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run stackelberg-opt tests')
    parser.add_argument(
        '--mode', 
        choices=['quick', 'basic', 'mocked', 'full', 'all'],
        default='all',
        help='Test mode to run'
    )
    parser.add_argument(
        '--no-deps-check',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    # Stay in project root (already changed above)
    
    # Check dependencies
    if not args.no_deps_check:
        missing = check_dependencies()
        if missing:
            print(f"Warning: Missing dependencies: {', '.join(missing)}")
            print("Some tests may fail. Install with: pip install -e ..")
            print()
    
    results = {}
    
    # Run tests based on mode
    if args.mode in ['quick', 'all']:
        results['structure'] = run_structure_tests()
    
    if args.mode in ['basic', 'all']:
        results['basic'] = run_basic_tests()
    
    if args.mode in ['mocked', 'all']:
        results['mocked'] = run_mocked_tests()
    
    if args.mode in ['full', 'all']:
        results['pytest'] = run_pytest()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize()}: {status}")
    
    print(f"\nTotal: {total_tests} test suites")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())