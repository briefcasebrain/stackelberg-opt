# stackelberg-opt Tests

This directory contains all tests for the stackelberg-opt library.

## Test Organization

### Core Test Files

1. **test_optimizer.py** - Tests for the main StackelbergOptimizer class
2. **test_components.py** - Tests for all component modules (mutator, evaluator, etc.)
3. **test_utils.py** - Tests for utility functions (caching, checkpointing, visualization)

### Special Test Files

4. **test_structure.py** - Verifies library structure without dependencies
5. **test_no_deps.py** - Tests core functionality without external dependencies
6. **test_with_mocks.py** - Tests with mocked dependencies
7. **test_basic.py** - Basic integration tests

### Supporting Files

- **conftest.py** - pytest configuration and shared fixtures
- **run_all_tests.py** - Main test runner with multiple modes
- **run_all_tests.sh** - Shell script for full test environment setup

## Running Tests

### Quick Tests (No Dependencies)

```bash
# Run structure tests only
python tests/run_all_tests.py --mode quick

# Run basic tests without dependencies
python tests/run_all_tests.py --mode basic
```

### Tests with Mocked Dependencies

```bash
# Run tests with mocked external libraries
python tests/run_all_tests.py --mode mocked
```

### Full Test Suite

```bash
# Run all tests (requires dependencies)
python tests/run_all_tests.py --mode full

# Or use pytest directly
pytest tests/ -v
```

### All Test Modes

```bash
# Run all available test modes
python tests/run_all_tests.py --mode all
```

## Test Categories

### Unit Tests

Test individual components in isolation:
- Module creation and validation
- Candidate operations
- Cache functionality
- Utility functions

### Integration Tests

Test component interactions:
- Optimizer workflow
- Mutation and evaluation pipeline
- Checkpoint save/load
- Visualization generation

### System Tests

Test complete optimization scenarios:
- Multi-hop QA optimization
- Complex module dependencies
- Population evolution
- Equilibrium calculation

## Writing New Tests

### Test Structure

```python
import pytest
from stackelberg_opt import Module, ModuleType

class TestNewFeature:
    """Tests for new feature."""
    
    def test_basic_functionality(self):
        """Test basic feature operations."""
        # Arrange
        module = Module("test", "prompt", ModuleType.LEADER)
        
        # Act
        result = some_operation(module)
        
        # Assert
        assert result is not None
    
    @pytest.mark.slow
    def test_complex_scenario(self, complex_modules):
        """Test complex feature scenario."""
        # Use fixtures from conftest.py
        pass
```

### Using Fixtures

Common fixtures available in `conftest.py`:
- `simple_modules` - Basic leader-follower modules
- `complex_modules` - Multi-module system
- `sample_candidate` - Pre-configured candidate
- `sample_trace` - Execution trace with data
- `mock_task_executor` - Async task executor

### Test Markers

Use markers to categorize tests:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_llm` - Tests requiring LLM access

## Coverage

Generate coverage reports:

```bash
# Run with coverage
pytest --cov=stackelberg_opt --cov-report=html

# View report
open htmlcov/index.html
```

## Continuous Integration

Tests are automatically run on:
- Every push to main branch
- Every pull request
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)

See `.github/workflows/tests.yml` for CI configuration.

## Troubleshooting

### Missing Dependencies

If tests fail due to missing dependencies:
```bash
pip install -e ..[dev]
python -m spacy download en_core_web_sm
```

### Import Errors

Ensure you're in the project root:
```bash
cd /path/to/stackelberg-opt
python -m pytest tests/
```

### Async Test Issues

For async test problems, ensure proper event loop:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```