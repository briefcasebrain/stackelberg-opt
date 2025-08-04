"""Tests for the StackelbergOptimizer."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from stackelberg_opt import (
    StackelbergOptimizer,
    Module,
    ModuleType,
    SystemCandidate,
    ExecutionTrace,
    OptimizerConfig
)


@pytest.fixture
def simple_modules():
    """Create simple test modules."""
    return {
        "leader": Module(
            name="leader",
            prompt="Lead the task",
            module_type=ModuleType.LEADER,
            dependencies=[]
        ),
        "follower": Module(
            name="follower",
            prompt="Follow the leader",
            module_type=ModuleType.FOLLOWER,
            dependencies=["leader"]
        )
    }


@pytest.fixture
def mock_task_executor():
    """Create mock task executor."""
    async def executor(modules, input_data):
        trace = ExecutionTrace()
        trace.execution_order = ["leader", "follower"]
        trace.module_outputs = {
            "leader": "leader output",
            "follower": "follower output"
        }
        trace.module_timings = {"leader": 0.1, "follower": 0.2}
        trace.intermediate_scores = {"leader": 0.8, "follower": 0.9}
        trace.final_score = 0.85
        trace.success = True
        return "final output", trace
    
    return executor


@pytest.fixture
def train_data():
    """Create simple training data."""
    return [
        ("input1", "expected1"),
        ("input2", "expected2"),
        ("input3", "expected3")
    ]


def test_optimizer_initialization(simple_modules, train_data, mock_task_executor):
    """Test basic optimizer initialization."""
    config = OptimizerConfig(
        budget=10,
        population_size=5,
        mutation_rate=0.7
    )
    
    optimizer = StackelbergOptimizer(
        system_modules=simple_modules,
        train_data=train_data,
        task_executor=mock_task_executor,
        config=config
    )
    
    assert optimizer.config.budget == 10
    assert optimizer.config.population_size == 5
    assert len(optimizer.system_modules) == 2
    assert len(optimizer.train_data) == 3


def test_optimizer_validation():
    """Test optimizer input validation."""
    # Test empty modules
    with pytest.raises(ValueError, match="No modules provided"):
        StackelbergOptimizer(
            system_modules={},
            train_data=[("test", "test")],
            task_executor=lambda x, y: None
        )
    
    # Test empty training data
    modules = {"test": Module("test", "prompt", ModuleType.LEADER)}
    with pytest.raises(ValueError, match="No training data provided"):
        StackelbergOptimizer(
            system_modules=modules,
            train_data=[],
            task_executor=lambda x, y: None
        )
    
    # Test invalid task executor
    with pytest.raises(ValueError, match="Task executor must be callable"):
        StackelbergOptimizer(
            system_modules=modules,
            train_data=[("test", "test")],
            task_executor="not callable"
        )


def test_module_dependency_validation():
    """Test module dependency validation."""
    modules = {
        "a": Module("a", "prompt", ModuleType.LEADER, dependencies=["b"]),
        "b": Module("b", "prompt", ModuleType.FOLLOWER, dependencies=["c"])
        # Missing module "c"
    }
    
    with pytest.raises(ValueError, match="depends on non-existent module"):
        StackelbergOptimizer(
            system_modules=modules,
            train_data=[("test", "test")],
            task_executor=lambda x, y: None
        )


def test_optimizer_config_defaults():
    """Test default configuration values."""
    config = OptimizerConfig()
    
    assert config.budget == 1000
    assert config.population_size == 20
    assert config.mutation_rate == 0.7
    assert config.crossover_rate == 0.3
    assert config.llm_model == "gpt-3.5-turbo"
    assert config.enable_visualization == True


@pytest.mark.asyncio
async def test_candidate_evaluation(simple_modules, train_data, mock_task_executor):
    """Test candidate evaluation."""
    config = OptimizerConfig(budget=10)
    
    optimizer = StackelbergOptimizer(
        system_modules=simple_modules,
        train_data=train_data,
        task_executor=mock_task_executor,
        config=config
    )
    
    # Create a candidate
    candidate = SystemCandidate(
        modules=simple_modules,
        candidate_id=1,
        generation=0
    )
    
    # Evaluate candidate
    optimizer._evaluate_candidates([candidate])
    
    # Check results
    assert len(candidate.scores) > 0
    assert len(candidate.traces) > 0
    assert all(score > 0 for score in candidate.scores.values())


def test_candidate_mutation(simple_modules, train_data, mock_task_executor):
    """Test candidate mutation."""
    config = OptimizerConfig(budget=10)
    
    optimizer = StackelbergOptimizer(
        system_modules=simple_modules,
        train_data=train_data,
        task_executor=mock_task_executor,
        config=config
    )
    
    # Create parent candidate
    parent = SystemCandidate(
        modules=simple_modules,
        candidate_id=1,
        generation=0
    )
    
    # Mutate
    child = optimizer._mutate_candidate(parent)
    
    assert child.parent_id == parent.candidate_id
    assert child.generation == 1
    assert len(child.modules) == len(parent.modules)
    assert child.candidate_id != parent.candidate_id


def test_crossover(simple_modules, train_data, mock_task_executor):
    """Test candidate crossover."""
    config = OptimizerConfig(budget=10)
    
    optimizer = StackelbergOptimizer(
        system_modules=simple_modules,
        train_data=train_data,
        task_executor=mock_task_executor,
        config=config
    )
    
    # Create parents with different prompts
    parent1_modules = simple_modules.copy()
    parent1_modules["leader"] = Module(
        "leader", "Lead version 1", ModuleType.LEADER
    )
    parent1 = SystemCandidate(modules=parent1_modules, candidate_id=1)
    
    parent2_modules = simple_modules.copy()
    parent2_modules["leader"] = Module(
        "leader", "Lead version 2", ModuleType.LEADER
    )
    parent2 = SystemCandidate(modules=parent2_modules, candidate_id=2)
    
    # Perform crossover
    child = optimizer._crossover_candidates(parent1, parent2)
    
    assert child.parent_id == parent1.candidate_id
    assert len(child.modules) == len(parent1.modules)
    # Child should have prompt from one of the parents
    leader_prompt = child.modules["leader"].prompt
    assert leader_prompt in ["Lead version 1", "Lead version 2"]


def test_population_initialization(simple_modules, train_data, mock_task_executor):
    """Test population initialization."""
    config = OptimizerConfig(budget=50, population_size=5)
    
    optimizer = StackelbergOptimizer(
        system_modules=simple_modules,
        train_data=train_data,
        task_executor=mock_task_executor,
        config=config
    )
    
    # Initialize population
    optimizer._initialize_population()
    
    assert len(optimizer.population) > 0
    assert all(isinstance(c, SystemCandidate) for c in optimizer.population)
    

def test_multi_objective_scoring(simple_modules, train_data, mock_task_executor):
    """Test multi-objective candidate scoring."""
    config = OptimizerConfig(
        budget=10,
        performance_weight=0.5,
        equilibrium_weight=0.3,
        stability_weight=0.2
    )
    
    optimizer = StackelbergOptimizer(
        system_modules=simple_modules,
        train_data=train_data,
        task_executor=mock_task_executor,
        config=config
    )
    
    # Create candidates with known scores
    candidate = SystemCandidate(modules=simple_modules, candidate_id=1)
    candidate.scores = {0: 0.8, 1: 0.9, 2: 0.85}
    candidate.equilibrium_value = 0.7
    candidate.stability_score = 0.6
    
    optimizer.population_manager.population = [candidate]
    optimizer._update_best_candidate()
    
    assert optimizer.best_candidate == candidate