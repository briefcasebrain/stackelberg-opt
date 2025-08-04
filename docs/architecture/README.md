# Stackelberg-Opt Architecture Documentation

Welcome to the architecture documentation for Stackelberg-Opt, a Python library implementing game-theoretic optimization for compound systems.

## Documentation Structure

This directory contains comprehensive documentation about the library's architecture:

### 📋 [Overview](./overview.md)
High-level introduction to the library's architecture, core concepts, and design principles. Start here to understand the overall system design and Stackelberg game theory foundations.

### 🔧 [Component Architecture](./components.md)
Detailed documentation of all library components including:
- Core components (Optimizer, Module, SystemCandidate)
- Component modules (Mutators, Evaluators, Feedback extractors)
- Utility components (Cache, Checkpoint, Visualization)
- Component interactions and communication patterns

### 🔌 [API Design](./api-design.md)
Complete API reference and design patterns including:
- Public API documentation
- Configuration options
- Usage examples
- Error handling patterns
- Best practices

### 📊 [Data Flow](./data-flow.md)
Visual diagrams and explanations of:
- Main optimization loop
- Component interactions
- Async execution flows
- State management
- Error propagation
- Metrics collection

### 🚀 [Extension Guide](./extension-guide.md)
Guide for extending and customizing the library:
- Creating custom components
- Plugin system
- Integration patterns
- Advanced customization options
- Troubleshooting

## Quick Navigation

- **New to Stackelberg-Opt?** Start with the [Overview](./overview.md)
- **Looking for API details?** Check the [API Design](./api-design.md)
- **Want to extend the library?** Read the [Extension Guide](./extension-guide.md)
- **Understanding internals?** Explore [Component Architecture](./components.md) and [Data Flow](./data-flow.md)

## Architecture Highlights

### 🎯 Core Design Principles
1. **Modularity**: Clear separation of concerns with pluggable components
2. **Asynchronous First**: Built on asyncio for efficient concurrent operations
3. **Type Safety**: Comprehensive type hints and runtime validation
4. **Extensibility**: Protocol-based interfaces and plugin architecture
5. **Robustness**: Automatic retries, checkpointing, and error recovery

### 🏗️ Key Architectural Decisions

- **Game-Theoretic Foundation**: True bilevel optimization using Stackelberg equilibrium
- **Component-Based Design**: Each optimization aspect is a separate, replaceable component
- **Async/Await Pattern**: Non-blocking I/O for API calls and parallel evaluation
- **Caching Strategy**: Multi-level caching for expensive operations
- **Plugin System**: Hook-based extension mechanism for custom functionality

### 🔄 Optimization Flow

```
Initialize → Evaluate → Extract Feedback → Find Equilibrium → 
Mutate → Select → Check Convergence → Return Best
```

## Getting Started with Development

1. **Understand the Core**: Read the [Overview](./overview.md) to grasp fundamental concepts
2. **Explore Components**: Study [Component Architecture](./components.md) for implementation details
3. **Learn the API**: Review [API Design](./api-design.md) for usage patterns
4. **Trace the Flow**: Follow [Data Flow](./data-flow.md) to understand execution
5. **Extend as Needed**: Use [Extension Guide](./extension-guide.md) for customization

## Contributing to Architecture

When contributing architectural changes:

1. **Document First**: Update relevant architecture docs before implementing
2. **Maintain Consistency**: Follow established patterns and conventions
3. **Consider Extensibility**: Ensure changes don't break extension points
4. **Test Thoroughly**: Include unit and integration tests
5. **Update Examples**: Provide usage examples for new features

## Additional Resources

- **API Reference**: See `/docs/api/` for API documentation
- **Examples**: Check `/stackelberg_opt/examples/` for practical implementations
- **Tests**: Review `/tests/` for usage patterns and edge cases
- **Issues**: Report architecture concerns on GitHub Issues

## Version Compatibility

This documentation corresponds to Stackelberg-Opt version 0.1.0. Architecture may evolve in future versions while maintaining backward compatibility where possible.

---

For questions or clarifications about the architecture, please open an issue on the GitHub repository.