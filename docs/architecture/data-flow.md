# Data Flow and Interaction Diagrams

This document illustrates the data flow and component interactions within the Stackelberg-Opt library through detailed diagrams and explanations.

## High-Level Data Flow

### Main Optimization Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Optimization Loop                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐     ┌────────────┐     ┌──────────────┐            │
│  │  Start   │────▶│ Initialize │────▶│  Population  │            │
│  │          │     │ Population │     │              │            │
│  └──────────┘     └────────────┘     └──────┬───────┘            │
│                                              │                     │
│                                              ▼                     │
│  ┌──────────┐     ┌────────────┐     ┌──────────────┐            │
│  │ Converged│◀────│   Check    │◀────│  Evaluation  │            │
│  │    or    │ NO  │Convergence │     │              │            │
│  │  Budget  │     └────────────┘     └──────▲───────┘            │
│  │ Exceeded │            │                   │                     │
│  └────┬─────┘            │YES                │                     │
│       │                  ▼                   │                     │
│       │           ┌────────────┐     ┌──────┴───────┐            │
│       │           │   Return   │     │   Mutation   │            │
│       │           │    Best    │     │              │            │
│       │           └────────────┘     └──────▲───────┘            │
│       │                                      │                     │
│       │           ┌────────────┐     ┌──────┴───────┐            │
│       └──────────▶│  Feedback  │────▶│  Selection   │            │
│                   │ Extraction │     │              │            │
│                   └────────────┘     └──────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Interactions

### 1. Initialization Phase

```
User Input                  StackelbergOptimizer
    │                              │
    ├─modules─────────────────────▶│
    ├─train_data─────────────────▶│
    ├─task_executor──────────────▶│
    └─config─────────────────────▶│
                                   │
                                   ├──validate_modules()
                                   ├──build_dependency_graph()
                                   ├──initialize_components()
                                   └──create_initial_population()
                                             │
                                             ▼
                                    PopulationManager
                                             │
                                             ├──random_initialization()
                                             ├──seed_with_defaults()
                                             └──ensure_diversity()
```

### 2. Evaluation Phase

```
SystemCandidate                    Evaluator
      │                                │
      └──────────candidate────────────▶│
                                       │
                                       ├──prepare_modules()
                                       │        │
                                       │        ▼
                                       │   TaskExecutor
                                       │        │
                                       │        ├──execute_pipeline()
                                       │        ├──collect_outputs()
                                       │        └──measure_metrics()
                                       │                │
                                       │                ▼
                                       │         ExecutionResult
                                       │                │
                                       ├──score_outputs()◀─┘
                                       ├──aggregate_scores()
                                       └──compute_fitness()
                                                │
                                                ▼
                                         EvaluationMetrics
```

### 3. Feedback Extraction

```
EvaluationResults            FeedbackExtractor
       │                            │
       └──────results──────────────▶│
                                    │
                                    ├──analyze_failures()
                                    │       │
                                    │       ├──pattern_detection()
                                    │       ├──error_classification()
                                    │       └──root_cause_analysis()
                                    │
                                    ├──extract_improvements()
                                    │       │
                                    │       ├──performance_gaps()
                                    │       ├──module_bottlenecks()
                                    │       └──interaction_issues()
                                    │
                                    └──generate_feedback()
                                            │
                                            ▼
                                    OptimizationFeedback
```

### 4. Equilibrium Computation

```
Population                  EquilibriumFinder
    │                              │
    └────candidates───────────────▶│
                                   │
                                   ├──extract_strategies()
                                   │         │
                                   │         ├──leader_strategies[]
                                   │         └──follower_strategies[]
                                   │
                                   ├──build_game_matrix()
                                   │         │
                                   │         └──PayoffMatrix
                                   │
                                   ├──solve_stackelberg()
                                   │         │
                                   │         ├──enumerate_leader_actions()
                                   │         ├──compute_follower_responses()
                                   │         └──find_optimal_commitment()
                                   │
                                   └──return_equilibrium()
                                             │
                                             ▼
                                      StackelbergEquilibrium
```

### 5. Mutation Process

```
SystemCandidate + Feedback          Mutator
         │                            │
         └────────inputs─────────────▶│
                                      │
                                      ├──select_strategy()
                                      │       │
                                      │       ├──PARAPHRASE
                                      │       ├──EXPAND
                                      │       ├──COMPRESS
                                      │       └──CROSSOVER
                                      │
                                      ├──generate_variations()
                                      │       │
                                      │       └──Language_API
                                      │            │
                                      │            └──prompt_variations[]
                                      │
                                      ├──filter_candidates()
                                      │       │
                                      │       ├──semantic_similarity()
                                      │       ├──constraint_check()
                                      │       └──diversity_filter()
                                      │
                                      └──return_mutations()
                                                │
                                                ▼
                                         List[SystemCandidate]
```

## Data Structures Flow

### Module Dependency Resolution

```
Modules Dictionary              DependencyGraph
      │                               │
      └──────modules─────────────────▶│
                                      │
                                      ├──build_graph()
                                      │      │
                                      │      └─▶ NetworkX DiGraph
                                      │
                                      ├──detect_cycles()
                                      │      │
                                      │      └─▶ Error if cyclic
                                      │
                                      └──topological_sort()
                                             │
                                             ▼
                                      Execution Order
                                          [mod1, mod2, mod3, ...]
```

### Population Evolution

```
Generation N                    Generation N+1
     │                               │
     ├─Elite────────────────────────▶├─Elite (preserved)
     │                               │
     ├─Selected─────┐                ├─Offspring
     │              │                │     ▲
     │              ▼                │     │
     │          Mutation             │     │
     │              │                │     │
     │              └────────────────┼─────┘
     │                               │
     └─Discarded                     └─New Random
```

## Async Execution Flow

### Parallel Candidate Evaluation

```
                    ┌─────────────────┐
                    │ Evaluation Queue │
                    └────────┬─────────┘
                             │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Worker 1 │     │ Worker 2 │     │ Worker 3 │
    │          │     │          │     │          │
    │ Candidate│     │ Candidate│     │ Candidate│
    │    A     │     │    B     │     │    C     │
    └────┬─────┘     └────┬─────┘     └────┬─────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │ Results Collector│
                    └─────────────────┘
```

### Async Task Execution Pipeline

```
async def execute_pipeline():
    ┌─────────────────────────────────────┐
    │         Prepare Input Data          │
    └─────────────────┬───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │    Module 1 (Leader) Execution      │
    │  ┌─────────────────────────────┐   │
    │  │ await llm_api_call()        │   │
    │  └─────────────────────────────┘   │
    └─────────────────┬───────────────────┘
                      │
                      ├──────────┬────────────┐
                      ▼          ▼            ▼
    ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
    │Module 2 (Follower)││Module 3      ││Module 4      │
    │                  ││(Follower)     ││(Follower)     │
    │Depends on Mod 1  ││Depends on 1,2 ││Depends on 3   │
    └──────────┬───────┘└───────┬───────┘└───────┬───────┘
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Aggregate Results     │
                    └─────────────────────────┘
```

## State Management

### Optimization State Transitions

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐
│  INIT   │────▶│  EVALUATING  │────▶│  ANALYZING   │
└─────────┘     └──────────────┘     └──────┬───────┘
                                             │
                                             ▼
┌─────────┐     ┌──────────────┐     ┌──────────────┐
│COMPLETE │◀────│  SELECTING   │◀────│   MUTATING   │
└─────────┘     └──────────────┘     └──────────────┘
     ▲                                       │
     │                                       │
     └───────────────────────────────────────┘
                  (convergence/budget)
```

### Cache Interactions

```
                    ┌─────────────────┐
                    │  Cache Manager  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Memory Cache   │ │   Disk Cache    │ │Distributed Cache│
│                 │ │                 │ │                 │
│ • LRU Eviction  │ │ • Persistent    │ │ • Redis/Memcache│
│ • TTL Support   │ │ • Compressed    │ │ • Shared State  │
│ • Fast Access   │ │ • Large Capacity│ │ • Multi-process │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Error Propagation

### Error Handling Flow

```
                    Exception Raised
                          │
                          ▼
                ┌─────────────────────┐
                │  Component Handler  │
                └─────────┬───────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │   Retry?     │ │  Log &   │ │  Propagate   │
    │              │ │  Continue│ │  to Caller   │
    └──────┬───────┘ └──────────┘ └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Retry Logic  │
    │              │
    │ • Backoff    │
    │ • Max Attempts│
    └──────────────┘
```

## Metrics Collection

### Metrics Flow Through System

```
┌─────────────────────────────────────────────────────┐
│                  Metrics Collector                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Optimization Metrics          Component Metrics    │
│  ├─ fitness_history           ├─ mutation_count    │
│  ├─ convergence_rate          ├─ evaluation_time   │
│  ├─ population_diversity      ├─ cache_hit_rate    │
│  └─ best_candidate_trace      └─ api_call_count    │
│                                                     │
│  System Metrics               Resource Metrics      │
│  ├─ total_runtime             ├─ memory_usage      │
│  ├─ generation_count          ├─ cpu_utilization   │
│  ├─ success_rate              ├─ network_bandwidth │
│  └─ error_count               └─ api_costs         │
│                                                     │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   Metrics Store     │
                │                     │
                │ • Time Series DB    │
                │ • Prometheus Export │
                │ • JSON Logs        │
                └─────────────────────┘
```

## Configuration Propagation

### Config Flow Through Components

```
OptimizerConfig
      │
      ├──budget──────────────────▶ Optimizer.budget
      ├──population_size─────────▶ PopulationManager.size
      ├──mutation_rate───────────▶ Mutator.rate
      ├──elite_ratio─────────────▶ Selection.elite_ratio
      ├──checkpoint_interval─────▶ CheckpointManager.interval
      └──convergence_threshold───▶ ConvergenceChecker.threshold

Component-Specific Configs
      │
      ├──MutatorConfig──────────▶ Mutator
      ├──EvaluatorConfig────────▶ Evaluator
      ├──CacheConfig────────────▶ CacheManager
      └──VisualizationConfig───▶ Visualizer
```

## Summary

The data flow in Stackelberg-Opt follows these key principles:

1. **Unidirectional Flow**: Data flows in clear directions through the optimization loop
2. **Async Operations**: Parallel execution where possible for performance
3. **Clear Boundaries**: Well-defined interfaces between components
4. **Error Isolation**: Errors handled at appropriate levels
5. **State Management**: Explicit state transitions and persistence
6. **Metrics Collection**: Comprehensive monitoring at all levels

This architecture ensures:
- **Scalability**: Can handle large populations and datasets
- **Reliability**: Robust error handling and recovery
- **Observability**: Rich metrics and logging
- **Extensibility**: Clear extension points for customization
- **Performance**: Efficient resource utilization