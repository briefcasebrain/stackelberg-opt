# Test Summary for stackelberg-opt

## Structure Tests ✅

All structure tests pass without dependencies:
- ✅ Directory structure is correct
- ✅ All 21 Python files created (5,495 lines of code)
- ✅ All configuration files present
- ✅ Documentation files complete

## Unit Tests 

The library includes comprehensive unit tests in the `tests/` directory:

### test_optimizer.py
- Tests for `StackelbergOptimizer` initialization and validation
- Tests for module dependency validation
- Tests for candidate evaluation and mutation
- Tests for crossover operations
- Tests for population management
- Tests for multi-objective scoring

### test_components.py
- Tests for `LLMPromptMutator` (including fallback mutations)
- Tests for `CompoundSystemEvaluator` (score calculations)
- Tests for `StackelbergFeedbackExtractor` (pattern extraction)
- Tests for `StackelbergEquilibriumCalculator` (with/without hierarchy)
- Tests for `StabilityCalculator` (performance and robustness)
- Tests for `PopulationManager` (parent selection methods)
- Tests for `DependencyAnalyzer` (explicit dependencies)
- Tests for `SemanticConstraintExtractor` (constraint extraction)

### test_utils.py
- Tests for `LLMCache` (persistence and key generation)
- Tests for `ComputationCache` (LRU eviction)
- Tests for `CheckpointManager` (save/load/delete operations)
- Tests for `AutoCheckpointer` (time/iteration-based checkpointing)
- Tests for `OptimizationVisualizer` (plot creation)

## Running Tests

To run the full test suite with pytest:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Download spaCy model
python -m spacy download en_core_web_sm

# Run tests
pytest -v

# Run with coverage
pytest --cov=stackelberg_opt --cov-report=html
```

## Dependencies Required for Tests

The tests require the following main dependencies:
- litellm (for LLM interactions)
- numpy, scipy, cvxpy (for optimization)
- networkx (for graph analysis)
- spacy, sentence-transformers (for NLP)
- matplotlib, seaborn (for visualization)
- pytest and related testing tools

All dependencies are listed in `requirements.txt` and `requirements-dev.txt`.

## Test Coverage

The test suite covers:
- Core functionality (modules, candidates, optimizer)
- All component classes
- Utility functions
- Error handling and edge cases
- Integration between components

The tests use mocking where appropriate to avoid external API calls during testing.