#!/usr/bin/env python3
"""Test stackelberg-opt without external dependencies."""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_module_imports():
    """Test importing core modules that don't require external dependencies."""
    print("\n1. Testing Core Module Imports")
    print("-" * 40)
    
    try:
        from stackelberg_opt.core.module import Module, ModuleType
        print("✅ Imported Module and ModuleType")
        
        # Test Module creation
        module = Module(
            name="test_module",
            prompt="Test prompt",
            module_type=ModuleType.LEADER,
            dependencies=[]
        )
        assert module.name == "test_module"
        assert module.prompt == "Test prompt"
        assert module.module_type == ModuleType.LEADER
        print("✅ Module creation works")
        
        # Test ModuleType enum
        assert ModuleType.LEADER.value == "leader"
        assert ModuleType.FOLLOWER.value == "follower"
        assert ModuleType.INDEPENDENT.value == "independent"
        print("✅ ModuleType enum works")
        
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False
    
    try:
        from stackelberg_opt.core.candidate import SystemCandidate, ExecutionTrace
        print("✅ Imported SystemCandidate and ExecutionTrace")
        
        # Test SystemCandidate
        modules = {"test": module}
        candidate = SystemCandidate(
            modules=modules,
            candidate_id=1,
            generation=0
        )
        assert candidate.candidate_id == 1
        assert candidate.generation == 0
        assert len(candidate.modules) == 1
        print("✅ SystemCandidate creation works")
        
        # Test methods
        assert candidate.get_average_score() == 0.0  # No scores yet
        candidate.scores = {0: 0.8, 1: 0.9}
        assert candidate.get_average_score() == 0.85
        print("✅ SystemCandidate methods work")
        
        # Test ExecutionTrace
        trace = ExecutionTrace()
        trace.execution_order = ["module1", "module2"]
        trace.success = True
        assert len(trace.execution_order) == 2
        print("✅ ExecutionTrace works")
        
    except Exception as e:
        print(f"❌ Candidate import failed: {e}")
        return False
    
    return True

def test_utility_imports():
    """Test importing utilities that don't require external dependencies."""
    print("\n2. Testing Utility Imports")
    print("-" * 40)
    
    try:
        from stackelberg_opt.utils.cache import ComputationCache
        print("✅ Imported ComputationCache")
        
        # Test ComputationCache
        cache = ComputationCache(max_size=3)
        key = cache.make_key("operation", param1=1, param2="test")
        cache.set(key, "result")
        assert cache.get(key) == "result"
        print("✅ ComputationCache works")
        
        # Test LRU eviction
        key1 = cache.make_key("op", id=1)
        key2 = cache.make_key("op", id=2)
        key3 = cache.make_key("op", id=3)
        
        cache.set(key1, "result1")
        cache.set(key2, "result2")
        cache.get(key1)  # Access key1
        cache.set(key3, "result3")  # Should evict key2
        
        assert cache.get(key1) == "result1"
        assert cache.get(key2) is None  # Evicted
        assert cache.get(key3) == "result3"
        print("✅ LRU eviction works correctly")
        
    except Exception as e:
        print(f"❌ Utility import failed: {e}")
        return False
    
    return True

def test_component_structure():
    """Test component module structure without imports."""
    print("\n3. Testing Component Structure")
    print("-" * 40)
    
    component_files = [
        "mutator.py",
        "evaluator.py", 
        "feedback.py",
        "equilibrium.py",
        "stability.py",
        "constraints.py",
        "dependencies.py",
        "population.py"
    ]
    
    components_dir = Path("stackelberg_opt/components")
    all_exist = True
    
    for file_name in component_files:
        file_path = components_dir / file_name
        if file_path.exists():
            # Check file has a class definition
            with open(file_path, 'r') as f:
                content = f.read()
                if "class" in content:
                    print(f"✅ {file_name} exists and contains class definitions")
                else:
                    print(f"⚠️  {file_name} exists but no class found")
                    all_exist = False
        else:
            print(f"❌ {file_name} missing")
            all_exist = False
    
    return all_exist

def test_example_structure():
    """Test example files structure."""
    print("\n4. Testing Example Files")
    print("-" * 40)
    
    example_files = {
        "multi_hop_qa.py": ["MultiHopQAExecutor", "create_multi_hop_qa_system"],
        "simple_optimization.py": ["simple_optimization_example", "simple_task_executor"]
    }
    
    examples_dir = Path("stackelberg_opt/examples")
    all_good = True
    
    for file_name, expected_items in example_files.items():
        file_path = examples_dir / file_name
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
                items_found = all(item in content for item in expected_items)
                if items_found:
                    print(f"✅ {file_name} contains expected functions/classes")
                else:
                    print(f"⚠️  {file_name} missing some expected items")
                    all_good = False
        else:
            print(f"❌ {file_name} missing")
            all_good = False
    
    return all_good

def test_integration():
    """Test basic integration between modules."""
    print("\n5. Testing Module Integration")
    print("-" * 40)
    
    try:
        from stackelberg_opt.core.module import Module, ModuleType
        from stackelberg_opt.core.candidate import SystemCandidate
        
        # Create a simple system
        leader = Module(
            name="leader",
            prompt="Lead the task",
            module_type=ModuleType.LEADER,
            dependencies=[]
        )
        
        follower = Module(
            name="follower",
            prompt="Follow the leader",
            module_type=ModuleType.FOLLOWER,
            dependencies=["leader"]
        )
        
        modules = {"leader": leader, "follower": follower}
        
        # Create candidates
        parent = SystemCandidate(modules=modules, candidate_id=1, generation=0)
        child = SystemCandidate(modules=modules, candidate_id=2, generation=1, parent_id=1)
        
        assert child.parent_id == parent.candidate_id
        assert child.generation == parent.generation + 1
        print("✅ Parent-child relationship works")
        
        # Test scores
        parent.scores = {0: 0.7, 1: 0.8, 2: 0.9}
        child.scores = {0: 0.8, 1: 0.85, 2: 0.9}
        
        assert child.get_average_score() > parent.get_average_score()
        print("✅ Score comparison works")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing stackelberg-opt WITHOUT External Dependencies")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Utility Imports", test_utility_imports),
        ("Component Structure", test_component_structure),
        ("Example Structure", test_example_structure),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✅ All tests passed without external dependencies!")
        return 0
    else:
        print(f"\n❌ {failed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())