#!/usr/bin/env python3
"""Basic tests for stackelberg-opt that can run without pytest."""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting imports...")
    
    # Test core imports
    from stackelberg_opt import (
        StackelbergOptimizer,
        Module,
        ModuleType,
        SystemCandidate,
        ExecutionTrace,
        OptimizerConfig
    )
    print("✓ Core imports successful")
    
    # Test component imports
    from stackelberg_opt.components import (
        PromptMutator,
        CompoundSystemEvaluator,
        StackelbergFeedbackExtractor,
        StackelbergEquilibriumCalculator,
        StabilityCalculator,
        SemanticConstraintExtractor,
        DependencyAnalyzer,
        PopulationManager
    )
    print("✓ Component imports successful")
    
    # Test utility imports
    from stackelberg_opt.utils import (
        ResponseCache,
        ComputationCache,
        CheckpointManager,
        AutoCheckpointer,
        OptimizationVisualizer
    )
    print("✓ Utility imports successful")

def test_module_creation():
    """Test basic module creation."""
    print("\nTesting module creation...")
    
    from stackelberg_opt import Module, ModuleType
    
    # Create a simple module
    module = Module(
        name="test_module",
        prompt="Test prompt",
        module_type=ModuleType.LEADER,
        dependencies=[]
    )
    
    assert module.name == "test_module"
    assert module.prompt == "Test prompt"
    assert module.module_type == ModuleType.LEADER
    assert module.dependencies == []
    
    print("✓ Module creation successful")

def test_system_candidate():
    """Test SystemCandidate creation."""
    print("\nTesting SystemCandidate...")
    
    from stackelberg_opt import Module, ModuleType, SystemCandidate
    
    modules = {
        "test": Module("test", "prompt", ModuleType.LEADER)
    }
    
    candidate = SystemCandidate(
        modules=modules,
        candidate_id=1,
        generation=0
    )
    
    assert candidate.candidate_id == 1
    assert candidate.generation == 0
    assert len(candidate.modules) == 1
    assert candidate.parent_id is None
    
    print("✓ SystemCandidate creation successful")

def test_optimizer_config():
    """Test OptimizerConfig defaults."""
    print("\nTesting OptimizerConfig...")
    
    from stackelberg_opt import OptimizerConfig
    
    config = OptimizerConfig()
    
    assert config.budget == 1000
    assert config.population_size == 20
    assert config.mutation_rate == 0.7
    assert hasattr(config, 'model')
    
    print("✓ OptimizerConfig defaults correct")

def test_execution_trace():
    """Test ExecutionTrace."""
    print("\nTesting ExecutionTrace...")
    
    from stackelberg_opt import ExecutionTrace
    
    trace = ExecutionTrace()
    trace.execution_order = ["module1", "module2"]
    trace.module_outputs = {"module1": "output1"}
    trace.success = True
    trace.final_score = 0.8
    
    assert len(trace.execution_order) == 2
    assert trace.module_outputs["module1"] == "output1"
    assert trace.success == True
    assert trace.final_score == 0.8
    
    print("✓ ExecutionTrace working correctly")

def test_population_manager():
    """Test basic PopulationManager functionality."""
    print("\nTesting PopulationManager...")
    
    from stackelberg_opt import SystemCandidate
    from stackelberg_opt.components import PopulationManager
    
    manager = PopulationManager(max_size=5)
    
    # Add a candidate
    candidate = SystemCandidate(modules={}, candidate_id=1)
    candidate.scores = {0: 0.9}
    
    added, reason = manager.add_candidate(candidate, generation=1)
    
    assert added == True
    assert len(manager.population) == 1
    assert manager.population[0] == candidate
    
    print("✓ PopulationManager basic operations working")

def test_response_cache():
    """Test ResponseCache basic operations."""
    print("\nTesting ResponseCache...")
    
    from stackelberg_opt.utils import ResponseCache
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ResponseCache(cache_dir=Path(tmpdir))
        
        # Test set and get
        cache.set("prompt1", "model1", 0.7, "response1")
        result = cache.get("prompt1", "model1", 0.7)
        
        assert result == "response1"
        
        # Test cache miss
        result = cache.get("prompt2", "model1", 0.7)
        assert result is None
        
        print("✓ ResponseCache operations working")

def test_computation_cache():
    """Test ComputationCache."""
    print("\nTesting ComputationCache...")
    
    from stackelberg_opt.utils import ComputationCache
    
    cache = ComputationCache(max_size=10)
    
    # Test key generation
    key = cache.make_key("operation", param1=1, param2="test")
    assert isinstance(key, str)
    
    # Test set and get
    cache.set(key, "result")
    result = cache.get(key)
    
    assert result == "result"
    
    print("✓ ComputationCache working correctly")

def test_dependency_analyzer():
    """Test DependencyAnalyzer."""
    print("\nTesting DependencyAnalyzer...")
    
    from stackelberg_opt import Module, ModuleType
    from stackelberg_opt.components import DependencyAnalyzer
    
    modules = {
        "a": Module("a", "prompt", ModuleType.LEADER, dependencies=[]),
        "b": Module("b", "prompt", ModuleType.FOLLOWER, dependencies=["a"]),
        "c": Module("c", "prompt", ModuleType.FOLLOWER, dependencies=["b"])
    }
    
    analyzer = DependencyAnalyzer()
    analysis = analyzer.analyze_dependencies(modules, prompts_only=True)
    
    assert analysis['dependencies']['a'] == []
    assert analysis['dependencies']['b'] == ['a']
    assert analysis['dependencies']['c'] == ['b']
    assert analysis['properties']['is_dag'] == True
    
    print("✓ DependencyAnalyzer working correctly")

def main():
    """Run all tests."""
    print("Running basic stackelberg-opt tests")
    print("="*50)
    
    tests = [
        test_imports,
        test_module_creation,
        test_system_candidate,
        test_optimizer_config,
        test_execution_trace,
        test_population_manager,
        test_response_cache,
        test_computation_cache,
        test_dependency_analyzer
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {test.__name__} FAILED:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print(f"Tests run: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())