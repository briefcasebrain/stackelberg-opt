#!/usr/bin/env python3
"""Test the library structure without requiring dependencies."""

import os
from pathlib import Path

def test_directory_structure():
    """Verify the directory structure is correct."""
    print("\nTesting directory structure...")
    
    base_dir = Path(__file__).parent
    required_dirs = [
        "stackelberg_opt",
        "stackelberg_opt/core",
        "stackelberg_opt/components", 
        "stackelberg_opt/utils",
        "stackelberg_opt/examples",
        "tests",
        "docs"
    ]
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        assert full_path.exists(), f"Missing directory: {dir_path}"
        print(f"  ✓ {dir_path}")
    
    print("✓ Directory structure is correct")

def test_core_files():
    """Verify core module files exist."""
    print("\nTesting core module files...")
    
    base_dir = Path(__file__).parent
    core_files = [
        "stackelberg_opt/core/__init__.py",
        "stackelberg_opt/core/module.py",
        "stackelberg_opt/core/candidate.py",
        "stackelberg_opt/core/optimizer.py"
    ]
    
    for file_path in core_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        assert full_path.stat().st_size > 0, f"Empty file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("✓ Core module files exist")

def test_component_files():
    """Verify component files exist."""
    print("\nTesting component files...")
    
    base_dir = Path(__file__).parent
    component_files = [
        "stackelberg_opt/components/__init__.py",
        "stackelberg_opt/components/mutator.py",
        "stackelberg_opt/components/evaluator.py",
        "stackelberg_opt/components/feedback.py",
        "stackelberg_opt/components/equilibrium.py",
        "stackelberg_opt/components/stability.py",
        "stackelberg_opt/components/constraints.py",
        "stackelberg_opt/components/dependencies.py",
        "stackelberg_opt/components/population.py"
    ]
    
    for file_path in component_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        assert full_path.stat().st_size > 0, f"Empty file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("✓ Component files exist")

def test_utility_files():
    """Verify utility files exist."""
    print("\nTesting utility files...")
    
    base_dir = Path(__file__).parent
    util_files = [
        "stackelberg_opt/utils/__init__.py",
        "stackelberg_opt/utils/cache.py",
        "stackelberg_opt/utils/checkpoint.py",
        "stackelberg_opt/utils/visualization.py"
    ]
    
    for file_path in util_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        assert full_path.stat().st_size > 0, f"Empty file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("✓ Utility files exist")

def test_setup_files():
    """Verify setup and configuration files exist."""
    print("\nTesting setup files...")
    
    base_dir = Path(__file__).parent
    setup_files = [
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
        ".gitignore",
        "CHANGELOG.md",
        "CONTRIBUTING.md"
    ]
    
    for file_path in setup_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        assert full_path.stat().st_size > 0, f"Empty file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("✓ Setup files exist")

def test_test_files():
    """Verify test files exist."""
    print("\nTesting test files...")
    
    base_dir = Path(__file__).parent
    test_files = [
        "tests/__init__.py",
        "tests/test_optimizer.py",
        "tests/test_components.py",
        "tests/test_utils.py"
    ]
    
    for file_path in test_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        assert full_path.stat().st_size > 0, f"Empty file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("✓ Test files exist")

def test_example_files():
    """Verify example files exist."""
    print("\nTesting example files...")
    
    base_dir = Path(__file__).parent
    example_files = [
        "stackelberg_opt/examples/__init__.py",
        "stackelberg_opt/examples/multi_hop_qa.py",
        "stackelberg_opt/examples/simple_optimization.py"
    ]
    
    for file_path in example_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        assert full_path.stat().st_size > 0, f"Empty file: {file_path}"
        print(f"  ✓ {file_path}")
    
    print("✓ Example files exist")

def test_file_content():
    """Test that key files have expected content."""
    print("\nTesting file content...")
    
    base_dir = Path(__file__).parent
    
    # Check setup.py has package name
    with open(base_dir / "setup.py", 'r') as f:
        content = f.read()
        assert 'name="stackelberg-opt"' in content
        print("  ✓ setup.py contains correct package name")
    
    # Check pyproject.toml
    with open(base_dir / "pyproject.toml", 'r') as f:
        content = f.read()
        assert 'name = "stackelberg-opt"' in content
        print("  ✓ pyproject.toml configured correctly")
    
    # Check README has title
    with open(base_dir / "README.md", 'r') as f:
        content = f.read()
        assert "# stackelberg-opt" in content
        print("  ✓ README.md has correct title")
    
    print("✓ File content is correct")

def count_lines():
    """Count total lines of code."""
    print("\nCounting lines of code...")
    
    base_dir = Path(__file__).parent
    total_lines = 0
    file_count = 0
    
    for py_file in base_dir.glob("stackelberg_opt/**/*.py"):
        if "__pycache__" not in str(py_file):
            with open(py_file, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                file_count += 1
    
    print(f"  Total Python files: {file_count}")
    print(f"  Total lines of code: {total_lines}")
    
    return total_lines

def main():
    """Run all structure tests."""
    print("Testing stackelberg-opt library structure")
    print("="*50)
    
    tests = [
        test_directory_structure,
        test_core_files,
        test_component_files,
        test_utility_files,
        test_setup_files,
        test_test_files,
        test_example_files,
        test_file_content
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n✗ {test.__name__} FAILED:")
            print(f"  {str(e)}")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test.__name__} FAILED with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Count lines
    total_lines = count_lines()
    
    print("\n" + "="*50)
    print(f"Structure tests run: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All structure tests passed!")
        print(f"✓ Successfully created library with {total_lines} lines of Python code")
        return 0
    else:
        print(f"\n✗ {failed} tests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())