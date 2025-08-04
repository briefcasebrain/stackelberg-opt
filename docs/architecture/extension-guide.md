# Extension and Customization Guide

This guide explains how to extend and customize the Stackelberg-Opt library for your specific use cases.

## Overview

Stackelberg-Opt is designed with extensibility in mind. The library uses:
- **Protocol-based interfaces** for maximum flexibility
- **Dependency injection** for component substitution
- **Plugin architecture** for adding new capabilities
- **Hook system** for intercepting operations

## Core Extension Points

### 1. Custom Mutators

Create custom mutation strategies for your domain:

```python
from stackelberg_opt.components import BaseMutator
from stackelberg_opt.core import SystemCandidate, OptimizationFeedback
from typing import List, Protocol

class DomainSpecificMutator(BaseMutator):
    """Example: A mutator for scientific prompt optimization"""
    
    def __init__(self, domain_knowledge: Dict[str, Any]):
        super().__init__()
        self.domain_knowledge = domain_knowledge
        self.technical_terms = self._load_technical_vocabulary()
    
    async def mutate(
        self, 
        candidate: SystemCandidate,
        feedback: OptimizationFeedback
    ) -> List[SystemCandidate]:
        """Generate domain-specific mutations"""
        mutations = []
        
        # Strategy 1: Technical term substitution
        if feedback.suggests_clarity_issues():
            mutations.extend(
                await self._clarify_technical_terms(candidate)
            )
        
        # Strategy 2: Add domain constraints
        if feedback.suggests_accuracy_issues():
            mutations.extend(
                await self._add_domain_constraints(candidate)
            )
        
        # Strategy 3: Incorporate domain patterns
        mutations.extend(
            await self._apply_domain_patterns(candidate)
        )
        
        return mutations
    
    async def _clarify_technical_terms(
        self, 
        candidate: SystemCandidate
    ) -> List[SystemCandidate]:
        # Implementation
        pass
```

### 2. Custom Evaluators

Implement domain-specific evaluation logic:

```python
from stackelberg_opt.components import BaseEvaluator
from stackelberg_opt.core import EvaluationMetrics
import numpy as np

class MultiObjectiveEvaluator(BaseEvaluator):
    """Evaluate multiple objectives with Pareto optimization"""
    
    def __init__(
        self,
        objectives: List[Callable],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
    
    async def evaluate(
        self,
        candidate: SystemCandidate,
        data: List[Tuple[Any, Any]]
    ) -> EvaluationMetrics:
        # Evaluate each objective
        objective_scores = []
        for objective in self.objectives:
            scores = await self._evaluate_objective(
                candidate, data, objective
            )
            objective_scores.append(scores)
        
        # Compute Pareto front
        pareto_rank = self._compute_pareto_rank(objective_scores)
        
        # Weighted aggregation
        aggregate_score = self._aggregate_scores(
            objective_scores, self.weights
        )
        
        return EvaluationMetrics(
            primary_score=aggregate_score,
            objective_scores=objective_scores,
            pareto_rank=pareto_rank,
            metadata={
                "dominated_by": self._find_dominators(candidate),
                "dominates": self._find_dominated(candidate)
            }
        )
```

### 3. Custom Task Executors

Integrate with your existing systems:

```python
from stackelberg_opt.core import TaskExecutor, ExecutionResult
import aiohttp

class APITaskExecutor(TaskExecutor):
    """Execute tasks through external API endpoints"""
    
    def __init__(self, api_config: Dict[str, str]):
        self.api_endpoints = api_config
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def execute(
        self,
        modules: Dict[str, Module],
        input_data: Any
    ) -> ExecutionResult:
        # Build execution pipeline from modules
        pipeline = self._build_pipeline(modules)
        
        # Execute through API
        result = input_data
        for step in pipeline:
            endpoint = self.api_endpoints[step.name]
            async with self.session.post(
                endpoint,
                json={"input": result, "config": step.to_dict()}
            ) as response:
                result = await response.json()
        
        return ExecutionResult(
            output=result["output"],
            metadata=result.get("metadata", {}),
            latency=result.get("latency"),
            cost=result.get("cost", 0)
        )
```

### 4. Custom Equilibrium Solvers

Implement specialized equilibrium concepts:

```python
from stackelberg_opt.components import BaseEquilibriumFinder
import cvxpy as cp

class CorrelatedEquilibriumFinder(BaseEquilibriumFinder):
    """Find correlated equilibria using linear programming"""
    
    def find_equilibrium(self, game_matrix: GameMatrix) -> Equilibrium:
        # Set up linear program
        n_leader_actions = game_matrix.leader_actions
        n_follower_actions = game_matrix.follower_actions
        
        # Decision variables (probability distribution)
        p = cp.Variable((n_leader_actions, n_follower_actions))
        
        # Constraints
        constraints = []
        
        # Probability constraints
        constraints.append(p >= 0)
        constraints.append(cp.sum(p) == 1)
        
        # Incentive compatibility constraints
        for i in range(n_leader_actions):
            for j in range(n_follower_actions):
                # Leader IC
                for i_prime in range(n_leader_actions):
                    if i != i_prime:
                        constraints.append(
                            game_matrix.leader_payoff[i, j] >= 
                            game_matrix.leader_payoff[i_prime, j]
                        )
                
                # Follower IC (similar)
        
        # Objective (maximize social welfare)
        objective = cp.Maximize(
            cp.sum(cp.multiply(
                p,
                game_matrix.leader_payoff + game_matrix.follower_payoff
            ))
        )
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return CorrelatedEquilibrium(
            distribution=p.value,
            expected_payoffs=self._compute_expected_payoffs(p.value)
        )
```

### 5. Custom Constraints

Add domain-specific constraints:

```python
from stackelberg_opt.components import BaseConstraint
import re

class PromptSafetyConstraint(BaseConstraint):
    """Ensure prompts are safe and appropriate"""
    
    def __init__(self, safety_rules: List[Dict]):
        self.rules = safety_rules
        self.compiled_patterns = self._compile_patterns()
    
    def check(self, candidate: SystemCandidate) -> ConstraintResult:
        violations = []
        
        for module_name, module in candidate.modules.items():
            # Check each safety rule
            for rule in self.rules:
                if self._violates_rule(module.prompt, rule):
                    violations.append(
                        ConstraintViolation(
                            module=module_name,
                            rule=rule["name"],
                            severity=rule["severity"]
                        )
                    )
        
        return ConstraintResult(
            satisfied=len(violations) == 0,
            violations=violations,
            penalty=self._calculate_penalty(violations)
        )
    
    def _violates_rule(self, prompt: str, rule: Dict) -> bool:
        if rule["type"] == "regex":
            return bool(re.search(rule["pattern"], prompt))
        elif rule["type"] == "semantic":
            return self._semantic_check(prompt, rule)
        return False
```

## Plugin System

### Creating a Plugin

```python
from stackelberg_opt.plugins import Plugin, PluginRegistry

class TelemetryPlugin(Plugin):
    """Add telemetry and monitoring capabilities"""
    
    name = "telemetry"
    version = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.metrics_backend = config.get("backend", "prometheus")
        self.export_interval = config.get("interval", 60)
        self._init_metrics()
    
    def on_optimization_start(self, optimizer):
        """Hook: Called when optimization begins"""
        self.start_time = time.time()
        self.metrics.optimization_started.inc()
    
    def on_generation_complete(self, optimizer, generation: int):
        """Hook: Called after each generation"""
        self.metrics.generation_completed.inc()
        self.metrics.fitness_gauge.set(
            optimizer.best_candidate.fitness
        )
    
    def on_optimization_end(self, optimizer, result):
        """Hook: Called when optimization completes"""
        duration = time.time() - self.start_time
        self.metrics.optimization_duration.observe(duration)
        self.metrics.final_fitness.set(result.fitness)

# Register the plugin
PluginRegistry.register(TelemetryPlugin)

# Use in optimizer
optimizer = StackelbergOptimizer(
    # ... other args ...
    plugins=[
        TelemetryPlugin({"backend": "prometheus"})
    ]
)
```

### Available Hooks

```python
# Optimization lifecycle hooks
on_optimization_start(optimizer)
on_optimization_end(optimizer, result)
on_generation_start(optimizer, generation)
on_generation_complete(optimizer, generation)

# Component hooks
on_mutation_start(mutator, candidate)
on_mutation_complete(mutator, mutations)
on_evaluation_start(evaluator, candidate)
on_evaluation_complete(evaluator, metrics)

# State management hooks
on_checkpoint_save(checkpoint_manager, state)
on_checkpoint_load(checkpoint_manager, state)

# Error hooks
on_error(component, error)
on_retry(component, attempt, error)
```

## Advanced Customization

### Custom Population Strategies

```python
from stackelberg_opt.components import PopulationStrategy

class IslandPopulationStrategy(PopulationStrategy):
    """Island model for distributed evolution"""
    
    def __init__(
        self,
        n_islands: int,
        migration_rate: float,
        migration_interval: int
    ):
        self.islands = [[] for _ in range(n_islands)]
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.generation = 0
    
    def evolve(
        self,
        populations: List[List[SystemCandidate]],
        fitness_scores: List[List[float]]
    ) -> List[List[SystemCandidate]]:
        # Evolve each island independently
        evolved_islands = []
        for island, scores in zip(populations, fitness_scores):
            evolved = self._evolve_island(island, scores)
            evolved_islands.append(evolved)
        
        # Periodic migration
        self.generation += 1
        if self.generation % self.migration_interval == 0:
            evolved_islands = self._migrate(evolved_islands)
        
        return evolved_islands
```

### Custom Feedback Analyzers

```python
from stackelberg_opt.components import FeedbackAnalyzer

class CausalFeedbackAnalyzer(FeedbackAnalyzer):
    """Extract causal relationships from feedback"""
    
    def __init__(self, causal_model: Any):
        self.causal_model = causal_model
    
    async def analyze(
        self,
        execution_traces: List[ExecutionTrace]
    ) -> CausalFeedback:
        # Build causal graph from traces
        graph = self._build_causal_graph(execution_traces)
        
        # Identify causal paths
        critical_paths = self._find_critical_paths(graph)
        
        # Compute interventions
        suggested_interventions = self._compute_interventions(
            graph,
            critical_paths
        )
        
        return CausalFeedback(
            causal_graph=graph,
            critical_paths=critical_paths,
            interventions=suggested_interventions,
            confidence_scores=self._compute_confidence(graph)
        )
```

## Integration Examples

### 1. LangChain Integration

```python
from langchain.chains import Chain
from stackelberg_opt import Module, TaskExecutor

class LangChainExecutor(TaskExecutor):
    """Execute Stackelberg modules as LangChain chains"""
    
    def __init__(self, chain_factory: Callable):
        self.chain_factory = chain_factory
    
    async def execute(
        self,
        modules: Dict[str, Module],
        input_data: Any
    ) -> ExecutionResult:
        # Convert modules to LangChain format
        chains = {}
        for name, module in modules.items():
            chains[name] = self.chain_factory(
                prompt_template=module.prompt,
                dependencies=module.dependencies
            )
        
        # Execute chain
        result = await chains["root"].arun(input_data)
        
        return ExecutionResult(output=result)
```

### 2. Ray Integration

```python
import ray
from stackelberg_opt.distributed import DistributedOptimizer

@ray.remote
class RayWorker:
    """Ray actor for distributed evaluation"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    async def evaluate_batch(
        self,
        candidates: List[SystemCandidate],
        data: List[Any]
    ) -> List[EvaluationMetrics]:
        results = []
        for candidate in candidates:
            metric = await self.evaluator.evaluate(candidate, data)
            results.append(metric)
        return results

class RayDistributedOptimizer(DistributedOptimizer):
    """Distribute optimization using Ray"""
    
    def __init__(self, n_workers: int = 4, **kwargs):
        super().__init__(**kwargs)
        ray.init()
        self.workers = [
            RayWorker.remote(self.evaluator)
            for _ in range(n_workers)
        ]
    
    async def _parallel_evaluate(
        self,
        population: List[SystemCandidate]
    ) -> List[EvaluationMetrics]:
        # Distribute candidates across workers
        batch_size = len(population) // len(self.workers)
        futures = []
        
        for i, worker in enumerate(self.workers):
            start = i * batch_size
            end = start + batch_size if i < len(self.workers) - 1 else len(population)
            batch = population[start:end]
            futures.append(
                worker.evaluate_batch.remote(batch, self.train_data)
            )
        
        # Gather results
        results = await ray.get(futures)
        return [item for sublist in results for item in sublist]
```

### 3. MLflow Integration

```python
import mlflow
from stackelberg_opt import Experiment

class MLflowExperiment(Experiment):
    """Track experiments with MLflow"""
    
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        self.run = None
    
    def __enter__(self):
        self.run = mlflow.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = 0):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, artifact_path: str):
        mlflow.log_artifact(artifact_path)
```

## Best Practices

### 1. Component Design
- Keep components focused and single-purpose
- Use dependency injection for flexibility
- Implement proper error handling
- Add comprehensive logging

### 2. Performance
- Use async/await for I/O operations
- Implement caching where appropriate
- Batch operations when possible
- Profile before optimizing

### 3. Testing
- Unit test each component separately
- Integration test component interactions
- Use mocks for external dependencies
- Test edge cases and error conditions

### 4. Documentation
- Document all public interfaces
- Provide usage examples
- Explain design decisions
- Keep documentation up to date

## Common Patterns

### Factory Pattern for Components

```python
class ComponentFactory:
    """Factory for creating components with configuration"""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, component_class: Type):
        cls._registry[name] = component_class
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> Any:
        if name not in cls._registry:
            raise ValueError(f"Unknown component: {name}")
        
        component_class = cls._registry[name]
        return component_class(**config)

# Register components
ComponentFactory.register("prompt_mutator", PromptMutator)
ComponentFactory.register("genetic_mutator", GeneticMutator)

# Use factory
mutator = ComponentFactory.create(
    "llm_mutator",
    {"model": "gpt-4", "temperature": 0.7}
)
```

### Observer Pattern for Events

```python
class OptimizationObserver:
    """Observer for optimization events"""
    
    def __init__(self):
        self._observers = defaultdict(list)
    
    def subscribe(self, event: str, callback: Callable):
        self._observers[event].append(callback)
    
    def notify(self, event: str, **kwargs):
        for callback in self._observers[event]:
            callback(**kwargs)

# Usage
observer = OptimizationObserver()
observer.subscribe(
    "generation_complete",
    lambda gen, fitness: print(f"Gen {gen}: {fitness}")
)
```

### Strategy Pattern for Algorithms

```python
class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies"""
    
    def select_parents(
        self,
        population: List[SystemCandidate]
    ) -> List[SystemCandidate]:
        ...
    
    def generate_offspring(
        self,
        parents: List[SystemCandidate]
    ) -> List[SystemCandidate]:
        ...

class ElitistStrategy:
    """Elitist selection strategy"""
    
    def select_parents(self, population):
        # Select top performers
        return sorted(
            population,
            key=lambda x: x.fitness,
            reverse=True
        )[:self.elite_size]
```

## Troubleshooting Extensions

### Common Issues

1. **Component Not Found**
   ```python
   # Ensure component is registered
   from stackelberg_opt import registry
   registry.register_component("my_mutator", MyMutator)
   ```

2. **Async Compatibility**
   ```python
   # Use async context managers
   async with MyComponent() as component:
       result = await component.process(data)
   ```

3. **Type Compatibility**
   ```python
   # Implement required protocols
   class MyEvaluator(BaseEvaluator):
       # Must implement all abstract methods
       pass
   ```

## Conclusion

The Stackelberg-Opt library provides extensive customization options through:
- Clear extension points
- Protocol-based interfaces
- Plugin architecture
- Integration patterns

By following the patterns and examples in this guide, you can adapt the library to your specific optimization needs while maintaining compatibility with the core framework.