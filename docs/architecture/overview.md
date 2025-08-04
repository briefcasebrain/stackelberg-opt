# Stackelberg-Opt Architecture Overview

## Introduction

Stackelberg-Opt is a Python library implementing game-theoretic optimization for compound systems using Stackelberg equilibrium concepts. The library is designed to optimize multi-module systems where modules exhibit leader-follower dynamics, enabling sophisticated prompt optimization and system configuration.

## Core Concepts

### Stackelberg Game Theory

In Stackelberg games, players move sequentially:
- **Leaders** move first and commit to their strategies
- **Followers** observe leader actions and respond optimally
- The solution concept is the **Stackelberg equilibrium**

This framework naturally models many system architectures where some components (e.g., query generators) influence others (e.g., answer extractors).

### Bilevel Optimization

The library implements true bilevel optimization:
- **Upper level**: Optimize leader module prompts/configurations
- **Lower level**: Followers adapt to leader choices
- **Equilibrium**: System reaches stable leader-follower dynamics

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stackelberg Optimizer                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Core      │  │  Components  │  │     Utils        │ │
│  │             │  │              │  │                  │ │
│  │ • Optimizer │  │ • Mutator    │  │ • Cache         │ │
│  │ • Candidate │  │ • Evaluator  │  │ • Checkpoint    │ │
│  │ • Module    │  │ • Feedback   │  │ • Visualization │ │
│  └─────────────┘  │ • Equilibrium│  └──────────────────┘ │
│                   │ • Stability  │                        │
│                   │ • Constraints│                        │
│                   │ • Population │                        │
│                   └──────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ External Systems │
                    │                  │
                    │ • Language APIs │
                    │ • Task Executors│
                    │ • Evaluators    │
                    └──────────────────┘
```

## Design Principles

### 1. Modularity
- Clear separation of concerns
- Each component has a single responsibility
- Easy to extend or replace individual components

### 2. Asynchronous First
- Built on `asyncio` for concurrent operations
- Efficient parallel evaluation of candidates
- Non-blocking I/O for external API calls

### 3. Type Safety
- Comprehensive type hints throughout
- Runtime validation with dataclasses
- Protocol-based interfaces for extensibility

### 4. Robustness
- Automatic retries with exponential backoff
- Checkpointing for long-running optimizations
- Graceful error handling and recovery

### 5. Observability
- Structured logging at multiple levels
- Real-time progress tracking
- Comprehensive metrics and analysis

## Key Components

### Core Module (`stackelberg_opt.core`)

The foundation of the library:

- **StackelbergOptimizer**: Main optimization orchestrator
- **SystemCandidate**: Represents a system configuration
- **Module**: Defines individual system components

### Components (`stackelberg_opt.components`)

Pluggable optimization components:

- **Mutator**: Generates new candidates (e.g., PromptMutator)
- **Evaluator**: Assesses candidate performance
- **Feedback Extractor**: Analyzes system behavior
- **Equilibrium Finder**: Computes Stackelberg equilibria
- **Stability Analyzer**: Measures system stability
- **Constraint Manager**: Handles optimization constraints
- **Population Manager**: Manages candidate populations

### Utilities (`stackelberg_opt.utils`)

Supporting functionality:

- **Cache**: Memoization for expensive operations
- **Checkpoint**: Save/restore optimization state
- **Visualization**: Plotting and analysis tools

## Data Flow

```
Input Data → Task Executor → System Modules → Output
     ↓                            ↑
Evaluator ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
     ↓
Feedback Extractor
     ↓
Equilibrium Finder → Stability Analyzer
     ↓                        ↓
Constraint Manager ← ─ ─ ─ ─ ┘
     ↓
Population Manager → Mutator
     ↓                   ↓
Optimizer ← ─ ─ ─ ─ ─ ─ ┘
```

## Optimization Loop

1. **Initialization**
   - Load system modules and configuration
   - Initialize population with random/provided candidates

2. **Evaluation Phase**
   - Execute tasks with current candidates
   - Collect performance metrics
   - Extract feedback from executions

3. **Analysis Phase**
   - Compute Stackelberg equilibria
   - Analyze stability metrics
   - Check constraint satisfaction

4. **Evolution Phase**
   - Select best performers
   - Generate mutations using feedback
   - Update population

5. **Termination**
   - Check budget/convergence criteria
   - Save checkpoints
   - Return best candidate

## Extension Points

The library is designed for extensibility:

### Custom Mutators
```python
class CustomMutator(BaseMutator):
    async def mutate(self, candidate, feedback):
        # Custom mutation logic
        pass
```

### Custom Evaluators
```python
class CustomEvaluator(BaseEvaluator):
    async def evaluate(self, candidate, data):
        # Custom evaluation logic
        pass
```

### Custom Equilibrium Solvers
```python
class CustomEquilibriumFinder(BaseEquilibriumFinder):
    def find_equilibrium(self, game_matrix):
        # Custom equilibrium computation
        pass
```

## Performance Considerations

### Caching
- Automatic memoization of expensive operations
- Configurable cache policies
- Persistent cache options

### Parallelization
- Concurrent candidate evaluation
- Async API calls
- Batch processing support

### Memory Management
- Streaming data processing
- Configurable population sizes
- Automatic garbage collection

## Security Considerations

### API Key Management
- Environment variable support
- Secure key storage recommendations
- No hardcoded credentials

### Input Validation
- Sanitization of user inputs
- Protection against prompt injection
- Rate limiting support

### Output Safety
- Configurable output filters
- Safety checks for output content
- Audit logging capabilities

## Integration Patterns

### Standalone Usage
```python
optimizer = StackelbergOptimizer(config)
best_system = await optimizer.optimize()
```

### Framework Integration
```python
# Django/FastAPI integration
@app.post("/optimize")
async def optimize_endpoint(request):
    result = await optimizer.optimize_async()
    return result
```

### Pipeline Integration
```python
# ML pipeline integration
pipeline = Pipeline([
    DataLoader(),
    StackelbergOptimizer(),
    ResultProcessor()
])
```

## Future Directions

1. **Multi-objective Optimization**: Support for Pareto-optimal solutions
2. **Distributed Computing**: Scale across multiple machines
3. **AutoML Integration**: Automatic hyperparameter tuning
4. **Real-time Adaptation**: Online learning capabilities
5. **Explainability**: Better insights into optimization decisions