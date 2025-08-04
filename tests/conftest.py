"""
pytest configuration for stackelberg-opt tests.

This file provides shared fixtures and configuration for all tests.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict
import asyncio

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stackelberg_opt import Module, ModuleType, SystemCandidate, ExecutionTrace


@pytest.fixture
def simple_modules() -> Dict[str, Module]:
    """Create simple test modules."""
    return {
        "leader": Module(
            name="leader",
            prompt="Lead the task with clear instructions",
            module_type=ModuleType.LEADER,
            dependencies=[]
        ),
        "follower": Module(
            name="follower",
            prompt="Follow the leader's guidance carefully",
            module_type=ModuleType.FOLLOWER,
            dependencies=["leader"]
        )
    }


@pytest.fixture
def complex_modules() -> Dict[str, Module]:
    """Create complex test modules with multiple dependencies."""
    return {
        "orchestrator": Module(
            name="orchestrator",
            prompt="Orchestrate the entire workflow systematically",
            module_type=ModuleType.LEADER,
            dependencies=[]
        ),
        "analyzer": Module(
            name="analyzer",
            prompt="Analyze inputs based on orchestrator guidance",
            module_type=ModuleType.FOLLOWER,
            dependencies=["orchestrator"]
        ),
        "processor": Module(
            name="processor",
            prompt="Process the analyzed results efficiently",
            module_type=ModuleType.FOLLOWER,
            dependencies=["analyzer"]
        ),
        "validator": Module(
            name="validator",
            prompt="Validate all outputs for correctness",
            module_type=ModuleType.INDEPENDENT,
            dependencies=["processor"]
        )
    }


@pytest.fixture
def sample_candidate(simple_modules) -> SystemCandidate:
    """Create a sample system candidate."""
    candidate = SystemCandidate(
        modules=simple_modules,
        candidate_id=1,
        generation=0
    )
    candidate.scores = {0: 0.8, 1: 0.85, 2: 0.9}
    return candidate


@pytest.fixture
def sample_trace() -> ExecutionTrace:
    """Create a sample execution trace."""
    trace = ExecutionTrace()
    trace.execution_order = ["leader", "follower"]
    trace.module_outputs = {
        "leader": "Leader output",
        "follower": "Follower output"
    }
    trace.module_timings = {
        "leader": 0.1,
        "follower": 0.2
    }
    trace.intermediate_scores = {
        "leader": 0.8,
        "follower": 0.9
    }
    trace.success = True
    trace.final_score = 0.85
    return trace


@pytest.fixture
async def mock_task_executor():
    """Create a mock task executor for testing."""
    async def executor(modules, input_data):
        trace = ExecutionTrace()
        trace.execution_order = list(modules.keys())
        trace.module_outputs = {name: f"{name} output" for name in modules}
        trace.module_timings = {name: 0.1 for name in modules}
        trace.intermediate_scores = {name: 0.8 for name in modules}
        trace.success = True
        trace.final_score = 0.8
        return "final output", trace
    
    return executor


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_llm: marks tests that require LLM access"
    )