# API Design Documentation

This document describes the public API design of the Stackelberg-Opt library, including design decisions, patterns, and usage examples.

## Design Philosophy

### 1. Progressive Disclosure
- Simple things should be simple
- Complex things should be possible
- Start with sensible defaults, allow customization

### 2. Consistency
- Uniform naming conventions
- Predictable behavior patterns
- Consistent error handling

### 3. Type Safety
- Comprehensive type hints
- Runtime validation where needed
- Clear contracts via protocols

### 4. Async-First
- All I/O operations are async
- Sync wrappers for convenience
- Efficient resource utilization

## Core API

### Main Entry Point

```python
from stackelberg_opt import StackelbergOptimizer, Module, ModuleType

# Simple usage with defaults
optimizer = StackelbergOptimizer(
    system_modules=modules,
    train_data=data,
    task_executor=executor
)

# Async usage
result = await optimizer.optimize_async()

# Sync wrapper
result = optimizer.optimize()
```

### Configuration API

```python
from stackelberg_opt import OptimizerConfig

# Using configuration object
config = OptimizerConfig(
    budget=2000,
    population_size=50,
    mutation_rate=0.8,
    elite_ratio=0.1,
    checkpoint_interval=100
    # Model configuration is automatically loaded from environment variables:
    # STACKELBERG_MODEL and STACKELBERG_TEMPERATURE
)

optimizer = StackelbergOptimizer(
    system_modules=modules,
    train_data=data,
    task_executor=executor,
    config=config
)

# Or pass individual parameters
optimizer = StackelbergOptimizer(
    system_modules=modules,
    train_data=data,
    task_executor=executor,
    budget=2000,
    population_size=50
)
```

#### Environment Variables

The library uses environment variables for sensitive configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `STACKELBERG_MODEL` | Language model to use for mutations | `gpt-3.5-turbo` |
| `STACKELBERG_TEMPERATURE` | Temperature for generation (0.0-1.0) | `0.7` |
| `STACKELBERG_CACHE_DIR` | Directory for caching API responses | `.cache` |
| `STACKELBERG_CACHE_ENABLED` | Enable/disable caching | `true` |
| `STACKELBERG_LOG_LEVEL` | Logging level | `INFO` |

Additional environment variables are passed through to the underlying model provider (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### Module Definition API

```python
from stackelberg_opt import Module, ModuleType, ModuleConstraints

# Basic module
module = Module(
    name="query_generator",
    prompt="Generate a search query for: {question}",
    module_type=ModuleType.LEADER
)

# Module with dependencies
module = Module(
    name="answer_extractor",
    prompt="Extract answer from context: {context}",
    module_type=ModuleType.FOLLOWER,
    dependencies=["query_generator", "retriever"]
)

# Module with constraints
constraints = ModuleConstraints(
    max_length=500,
    required_vars=["question", "context"],
    forbidden_phrases=["ignore previous", "system:"],
    format_regex=r"^Answer: .+"
)

module = Module(
    name="formatter",
    prompt="Format the answer",
    module_type=ModuleType.FOLLOWER,
    constraints=constraints
)
```

### Task Executor API

```python
from stackelberg_opt import TaskExecutor, ExecutionResult

# Function-based executor
async def my_executor(modules: Dict[str, Module], input_data: Any) -> ExecutionResult:
    # Execute the compound system
    result = await run_system(modules, input_data)
    return ExecutionResult(
        output=result.output,
        metadata=result.metadata,
        success=result.success
    )

# Class-based executor
class MyTaskExecutor(TaskExecutor):
    async def execute(self, modules: Dict[str, Module], input_data: Any) -> ExecutionResult:
        # Implementation
        pass
    
    async def batch_execute(self, modules: Dict[str, Module], inputs: List[Any]) -> List[ExecutionResult]:
        # Efficient batch processing
        pass
```

### Evaluation API

```python
from stackelberg_opt import Evaluator, EvaluationMetrics

# Simple evaluator
async def accuracy_evaluator(output: Any, expected: Any) -> float:
    return 1.0 if output == expected else 0.0

# Complex evaluator
class CustomEvaluator(Evaluator):
    async def evaluate(
        self, 
        candidate: SystemCandidate,
        data: List[Tuple[Any, Any]]
    ) -> EvaluationMetrics:
        scores = []
        for input_data, expected in data:
            result = await self.executor(candidate.modules, input_data)
            score = self.score_function(result.output, expected)
            scores.append(score)
        
        return EvaluationMetrics(
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            min_score=np.min(scores),
            max_score=np.max(scores),
            total_cost=sum(r.cost for r in results),
            total_latency=sum(r.latency for r in results)
        )
```

### Mutation API

```python
from stackelberg_opt import Mutator, MutationStrategy

# Using built-in mutator
mutator = PromptMutator(
    model="gpt-4",
    temperature=0.7,
    strategies=[
        MutationStrategy.PARAPHRASE,
        MutationStrategy.EXPAND,
        MutationStrategy.COMPRESS
    ]
)

# Custom mutator
class MyMutator(Mutator):
    async def mutate(
        self,
        candidate: SystemCandidate,
        feedback: OptimizationFeedback
    ) -> List[SystemCandidate]:
        # Generate variations
        variations = []
        for module in candidate.modules.values():
            if module.module_type == ModuleType.LEADER:
                new_prompt = await self.modify_prompt(module.prompt, feedback)
                variations.append(self.create_variant(candidate, module.name, new_prompt))
        return variations
```

### Feedback API

```python
from stackelberg_opt import FeedbackExtractor, Feedback

# Built-in feedback
extractor = StackelbergFeedbackExtractor(
    extract_performance_gaps=True,
    extract_failure_patterns=True,
    extract_improvement_hints=True
)

# Custom feedback
class MyFeedbackExtractor(FeedbackExtractor):
    async def extract(
        self,
        results: List[ExecutionResult],
        scores: List[float]
    ) -> Feedback:
        # Analyze results
        gaps = self.find_performance_gaps(results, scores)
        patterns = self.find_patterns(results)
        
        return Feedback(
            performance_gaps=gaps,
            failure_patterns=patterns,
            suggested_improvements=self.suggest_improvements(gaps, patterns),
            strategic_insights=self.analyze_interactions(results)
        )
```

### Equilibrium API

```python
from stackelberg_opt import EquilibriumFinder, GameMatrix

# Using equilibrium finder
finder = StackelbergEquilibriumFinder()

# Find equilibrium
equilibrium = finder.find_equilibrium(
    leader_strategies=leader_prompts,
    follower_strategies=follower_prompts,
    payoff_matrix=evaluation_results
)

# Custom equilibrium
class MyEquilibriumFinder(EquilibriumFinder):
    def find_equilibrium(self, game_matrix: GameMatrix) -> Equilibrium:
        # Custom algorithm
        leader_strategy = self.solve_leader_problem(game_matrix)
        follower_response = self.compute_best_response(leader_strategy, game_matrix)
        
        return Equilibrium(
            leader_strategy=leader_strategy,
            follower_strategy=follower_response,
            leader_payoff=self.compute_payoff(leader_strategy, follower_response),
            stability_index=self.compute_stability(leader_strategy, follower_response)
        )
```

### Constraint API

```python
from stackelberg_opt import Constraint, ConstraintType

# Hard constraints (must be satisfied)
length_constraint = Constraint(
    name="max_prompt_length",
    type=ConstraintType.HARD,
    check=lambda prompt: len(prompt) <= 1000,
    error_message="Prompt exceeds maximum length"
)

# Soft constraints (penalized if violated)
performance_constraint = Constraint(
    name="min_accuracy",
    type=ConstraintType.SOFT,
    check=lambda metrics: metrics.accuracy >= 0.8,
    penalty=lambda metrics: max(0, 0.8 - metrics.accuracy) * 100
)

# Dynamic constraints
class AdaptiveConstraint(Constraint):
    def __init__(self):
        self.threshold = 0.8
    
    def check(self, metrics):
        return metrics.accuracy >= self.threshold
    
    def update(self, population_stats):
        # Adjust threshold based on population performance
        self.threshold = population_stats.percentile(75)
```

### Checkpointing API

```python
from stackelberg_opt import CheckpointManager

# Automatic checkpointing
optimizer = StackelbergOptimizer(
    # ... other args ...
    checkpoint_dir="./checkpoints",
    checkpoint_interval=50
)

# Manual checkpointing
checkpoint_manager = CheckpointManager("./checkpoints")

# Save state
await checkpoint_manager.save(
    generation=100,
    population=current_population,
    metrics=optimization_metrics,
    cache=evaluation_cache
)

# Resume from checkpoint
state = await checkpoint_manager.load_latest()
optimizer = StackelbergOptimizer.from_checkpoint(state)
```

### Visualization API

```python
from stackelberg_opt.utils import OptimizationVisualizer

# Create visualizer
viz = OptimizationVisualizer()

# Plot fitness evolution
viz.plot_fitness_evolution(
    optimization_history,
    save_path="fitness_evolution.png"
)

# Interactive dashboard
viz.create_dashboard(
    optimization_history,
    population_snapshots,
    port=8080
)

# Custom visualizations
fig = viz.plot_module_interactions(
    modules=optimizer.modules,
    interaction_strengths=computed_strengths,
    layout="hierarchical"
)
```

## Advanced Usage Patterns

### Custom Components with Dependency Injection

```python
from stackelberg_opt import create_optimizer

# Register custom components
optimizer = create_optimizer(
    mutator_class=MyCustomMutator,
    evaluator_class=MyCustomEvaluator,
    feedback_class=MyCustomFeedback,
    components={
        "cache": RedisCache(),
        "logger": CustomLogger(),
        "metrics": PrometheusMetrics()
    }
)
```

### Pipeline Integration

```python
# Scikit-learn style API
from stackelberg_opt import StackelbergOptimizer

class PromptOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self, budget=1000):
        self.budget = budget
        self.optimizer = None
    
    def fit(self, X, y):
        self.optimizer = StackelbergOptimizer(
            system_modules=self.create_modules(),
            train_data=list(zip(X, y)),
            task_executor=self.executor,
            budget=self.budget
        )
        self.best_system = self.optimizer.optimize()
        return self
    
    def transform(self, X):
        return [self.best_system.execute(x) for x in X]
```

### Distributed Optimization

```python
from stackelberg_opt.distributed import DistributedOptimizer

# Create distributed optimizer
optimizer = DistributedOptimizer(
    system_modules=modules,
    train_data=data,
    task_executor=executor,
    worker_nodes=["worker1:8080", "worker2:8080"],
    coordination_backend="redis://localhost:6379"
)

# Optimization runs across multiple nodes
result = await optimizer.optimize_async()
```

### Experiment Tracking

```python
from stackelberg_opt import Experiment

# Create experiment
with Experiment("multi_hop_qa_v2") as exp:
    # Configure experiment
    exp.log_params({
        "budget": 2000,
        "population_size": 50,
        "mutation_rate": 0.8
    })
    
    # Run optimization
    optimizer = StackelbergOptimizer(
        system_modules=modules,
        train_data=data,
        task_executor=executor,
        experiment=exp
    )
    
    result = optimizer.optimize()
    
    # Results automatically logged
    exp.log_metrics({
        "final_fitness": result.fitness,
        "convergence_generation": result.generation
    })
```

## Error Handling

### Exception Hierarchy

```python
StackelbergOptError
├── ConfigurationError
│   ├── InvalidModuleError
│   ├── MissingDependencyError
│   └── ConstraintDefinitionError
├── OptimizationError
│   ├── ConvergenceError
│   ├── BudgetExceededError
│   └── PopulationCollapseError
├── ExecutionError
│   ├── TaskExecutorError
│   ├── TimeoutError
│   └── ResourceLimitError
└── ComponentError
    ├── MutatorError
    ├── EvaluatorError
    └── EquilibriumError
```

### Error Handling Patterns

```python
from stackelberg_opt import StackelbergOptError, retry_with_backoff

# Graceful error handling
try:
    result = optimizer.optimize()
except ConvergenceError:
    # Fallback to best so far
    result = optimizer.get_best_candidate()
except BudgetExceededError as e:
    logger.warning(f"Budget exceeded: {e}")
    result = optimizer.get_best_candidate()
except StackelbergOptError as e:
    logger.error(f"Optimization failed: {e}")
    raise

# Automatic retries
@retry_with_backoff(max_attempts=3, exceptions=(ExecutionError,))
async def robust_execution(modules, data):
    return await task_executor(modules, data)
```

## Best Practices

### 1. Module Design
- Keep prompts focused and single-purpose
- Clearly define dependencies
- Use appropriate module types

### 2. Data Preparation
- Ensure diverse training data
- Include edge cases
- Balance dataset if needed

### 3. Performance Optimization
- Use caching for expensive operations
- Enable checkpointing for long runs
- Monitor resource usage

### 4. Testing
- Test individual modules first
- Validate task executor independently
- Use small budgets for initial testing

### 5. Production Deployment
- Set appropriate timeouts
- Implement proper logging
- Monitor optimization metrics
- Use gradual rollout for changes

## API Versioning

The library follows semantic versioning:

- **Major version**: Breaking API changes
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes

### Deprecation Policy

1. Features marked deprecated in version X.Y
2. Warning issued when used
3. Removed in version (X+1).0
4. Migration guide provided

Example:
```python
# Deprecated in 1.2, removed in 2.0
@deprecated("Use StackelbergOptimizer instead")
class Optimizer:
    pass
```