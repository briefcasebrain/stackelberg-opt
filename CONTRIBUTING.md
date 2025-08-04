# Contributing to stackelberg-opt

We love your input! We want to make contributing to stackelberg-opt as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/aanshshah/stackelberg-opt/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/aanshshah/stackelberg-opt/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/aanshshah/stackelberg-opt.git
   cd stackelberg-opt
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e .[dev]
   python -m spacy download en_core_web_sm
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **pytest** for testing

### Running Code Quality Checks

```bash
# Format code
black stackelberg_opt tests

# Run linting
flake8 stackelberg_opt tests

# Run type checking
mypy stackelberg_opt

# Run all tests
pytest

# Run tests with coverage
pytest --cov=stackelberg_opt --cov-report=html
```

## Testing

- Write tests for any new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>80%)
- Tests should be in the `tests/` directory
- Follow existing test patterns and naming conventions

### Test Structure

```python
# tests/test_module.py
import pytest
from stackelberg_opt import Module, ModuleType

def test_module_creation():
    """Test basic module creation."""
    module = Module(
        name="test",
        prompt="Test prompt",
        module_type=ModuleType.LEADER
    )
    assert module.name == "test"
    assert module.prompt == "Test prompt"

@pytest.fixture
def sample_modules():
    """Fixture providing sample modules."""
    return {
        "leader": Module(...),
        "follower": Module(...)
    }

def test_complex_scenario(sample_modules):
    """Test with fixtures."""
    # Your test here
```

## Documentation

- Update docstrings for any changed functionality
- Follow Google-style docstrings
- Include examples in docstrings where helpful
- Update README.md if needed
- Add/update documentation in `docs/` for significant changes

### Docstring Example

```python
def optimize(self, modules: Dict[str, Module]) -> SystemCandidate:
    """
    Run optimization on the given modules.
    
    Args:
        modules: Dictionary mapping module names to Module instances
        
    Returns:
        Best SystemCandidate found during optimization
        
    Raises:
        OptimizationError: If optimization fails
        
    Examples:
        >>> optimizer = StackelbergOptimizer(...)
        >>> best = optimizer.optimize(modules)
        >>> print(best.get_average_score())
    """
```

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the CHANGELOG.md with your changes
3. The PR will be merged once you have the sign-off of at least one maintainer

## Community Guidelines

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Questions?

Feel free to open an issue with the "question" label or start a discussion in the [GitHub Discussions](https://github.com/aanshshah/stackelberg-opt/discussions).

Thank you for contributing to stackelberg-opt!