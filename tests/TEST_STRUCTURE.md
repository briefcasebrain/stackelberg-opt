# Test Structure Overview

## Directory Organization

All tests are now organized under the `tests/` directory with the following structure:

```
tests/
├── README.md                    # Test documentation
├── TEST_RESULTS.md             # Test execution results
├── TEST_STRUCTURE.md           # This file
├── TEST_SUMMARY.md             # Test summary
├── conftest.py                 # pytest configuration and fixtures
├── run_all_tests.py            # Unified test runner (Python)
├── run_all_tests.sh            # Full test setup script (Shell)
│
├── test_optimizer.py           # Core optimizer tests
├── test_components.py          # Component module tests
├── test_utils.py              # Utility function tests
├── test_integration.py        # Integration tests
│
├── test_structure.py          # Library structure verification
├── test_no_deps.py           # Tests without dependencies
├── test_with_mocks.py        # Tests with mocked dependencies
├── test_basic.py             # Basic functionality tests
├── quick_test.py             # Quick verification script
└── run_tests.py              # Legacy test runner
```

## Test Categories

### 1. Core Test Suites
- **test_optimizer.py**: Tests for StackelbergOptimizer class
- **test_components.py**: Tests for all component modules
- **test_utils.py**: Tests for utility functions
- **test_integration.py**: Integration tests for component interactions

### 2. Special Test Suites
- **test_structure.py**: Verifies library structure without dependencies
- **test_no_deps.py**: Core functionality tests without external libraries
- **test_with_mocks.py**: Tests using mocked external dependencies
- **test_basic.py**: Basic integration tests

### 3. Test Infrastructure
- **conftest.py**: Shared fixtures and pytest configuration
- **run_all_tests.py**: Main test runner with multiple modes
- **run_all_tests.sh**: Complete test environment setup

## Running Tests

### Quick Command Reference

```bash
# Run all tests (from project root)
python tests/run_all_tests.py --mode all

# Run specific test modes
python tests/run_all_tests.py --mode quick    # Structure only
python tests/run_all_tests.py --mode basic    # No dependencies
python tests/run_all_tests.py --mode mocked   # With mocks
python tests/run_all_tests.py --mode full     # Full pytest suite

# Run with pytest directly
pytest tests/

# Run specific test files
pytest tests/test_optimizer.py
pytest tests/test_components.py -v

# Run by markers
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"       # Skip slow tests
```

### Full Test Setup

For a complete test environment with all dependencies:

```bash
cd /path/to/stackelberg-opt
./tests/run_all_tests.sh
```

This script will:
1. Create a virtual environment
2. Install all dependencies
3. Download required models
4. Run all test suites
5. Generate coverage reports

## Test Configuration

### pytest.ini
Located in project root, configures:
- Test discovery patterns
- Coverage settings
- Test markers
- Output options

### pyproject.toml
Contains additional test configuration:
- Test paths
- Coverage source
- Tool-specific settings

## Writing New Tests

### Test Template

```python
import pytest
from stackelberg_opt import Module, ModuleType

@pytest.mark.unit
class TestNewFeature:
    """Tests for new feature."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        # Arrange
        module = Module("test", "Test prompt here", ModuleType.LEADER)
        
        # Act
        result = some_operation(module)
        
        # Assert
        assert result is not None
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_complex_case(self, complex_modules, mock_task_executor):
        """Test complex scenario using fixtures."""
        # Test implementation
        pass
```

### Using Fixtures

Common fixtures from `conftest.py`:
- `simple_modules`: Basic leader-follower setup
- `complex_modules`: Multi-module system
- `sample_candidate`: Pre-configured candidate
- `sample_trace`: Execution trace with data
- `mock_task_executor`: Async task executor for testing

## Coverage Reports

After running tests, coverage reports are available:
- **Terminal**: Shown during test execution
- **HTML**: `htmlcov/index.html`
- **XML**: `coverage.xml` (for CI integration)

## Continuous Integration

Tests are automatically run on:
- Push to main branch
- Pull requests
- Multiple Python versions (3.8-3.11)

See `.github/workflows/tests.yml` for CI configuration.

## Benefits of New Structure

1. **Centralized**: All tests in one directory
2. **Organized**: Clear separation of test types
3. **Flexible**: Multiple ways to run tests
4. **Scalable**: Easy to add new test categories
5. **Maintainable**: Clear documentation and structure