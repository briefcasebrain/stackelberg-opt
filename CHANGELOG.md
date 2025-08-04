# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-XX

### Added
- Initial release of stackelberg-opt
- Core optimization framework with Stackelberg game-theoretic approach
- LLM-based prompt mutation using LiteLLM
- Comprehensive evaluation system with async support
- Multi-faceted stability analysis
- True bilevel optimization with multiple solver methods
- Population management with elite, diversity, and innovation archives
- Semantic constraint extraction for leader-follower relationships
- Dependency analysis with graph algorithms
- Checkpointing and recovery functionality
- Visualization tools for optimization progress
- Caching system for LLM responses
- Example implementations for multi-hop QA
- Comprehensive documentation
- Test suite with pytest
- Type hints throughout the codebase

### Features
- **Core Components**
  - `StackelbergOptimizer`: Main optimization orchestrator
  - `Module` and `SystemCandidate`: Core data structures
  - `OptimizerConfig`: Configuration management
  
- **Optimization Components**
  - `LLMPromptMutator`: Intelligent prompt mutations
  - `CompoundSystemEvaluator`: System evaluation with tracing
  - `StackelbergFeedbackExtractor`: Pattern-based feedback analysis
  - `StackelbergEquilibriumCalculator`: Bilevel equilibrium computation
  - `StabilityCalculator`: Multi-faceted stability metrics
  - `SemanticConstraintExtractor`: NLP-based constraint extraction
  - `DependencyAnalyzer`: Graph-based dependency analysis
  - `PopulationManager`: Evolutionary population management
  
- **Utilities**
  - `LLMCache`: Response caching for cost efficiency
  - `CheckpointManager`: State persistence and recovery
  - `OptimizationVisualizer`: Comprehensive plotting tools

[Unreleased]: https://github.com/aanshshah/stackelberg-opt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aanshshah/stackelberg-opt/releases/tag/v0.1.0