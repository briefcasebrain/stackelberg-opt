# stackelberg-opt Jupyter Notebooks

This directory contains interactive Jupyter notebooks demonstrating the usage and features of the stackelberg-opt library.

## Notebooks Overview

### 1. [Getting Started](01_getting_started.ipynb)
- **Purpose**: Introduction to basic concepts and usage
- **Topics Covered**:
  - Installation and setup
  - Creating modules with leader-follower-independent structure
  - Running basic optimization
  - Analyzing results
  - Simple visualization
- **Best For**: New users, quick start guide

### 2. [Multi-Hop QA Example](02_multi_hop_qa_example.ipynb)
- **Purpose**: Comprehensive example of a multi-hop question answering system
- **Topics Covered**:
  - Complex module dependencies
  - Strategic interactions between modules
  - Custom task executors
  - Performance analysis
  - Feedback extraction
- **Best For**: Understanding real-world applications

### 3. [Advanced Features](03_advanced_features.ipynb)
- **Purpose**: Deep dive into advanced capabilities
- **Topics Covered**:
  - Semantic constraint extraction
  - Population management strategies
  - Custom evaluation metrics
  - Checkpointing and recovery
  - Advanced visualizations
  - Performance optimization with caching
- **Best For**: Power users, production deployments

## Running the Notebooks

### Prerequisites

1. Install stackelberg-opt:
   ```bash
   pip install -e ..  # From notebooks directory
   ```

2. Install Jupyter:
   ```bash
   pip install jupyter notebook
   ```

3. Install additional dependencies for visualization:
   ```bash
   pip install matplotlib seaborn
   ```

### Starting Jupyter

From the notebooks directory:
```bash
jupyter notebook
```

Then open any notebook to begin exploring.

## Key Concepts

### Module Types
- **Leader**: Makes decisions first, influences the system
- **Follower**: Reacts to leader decisions
- **Independent**: Operates without strategic dependencies

### Optimization Process
1. **Population Initialization**: Create diverse candidates
2. **Evaluation**: Test candidates on training data
3. **Selection**: Choose parents for next generation
4. **Mutation/Crossover**: Create new candidates
5. **Equilibrium Calculation**: Find Stackelberg equilibrium
6. **Iteration**: Repeat until budget exhausted

### Key Metrics
- **Performance Score**: How well the system performs on tasks
- **Equilibrium Value**: Stackelberg game-theoretic equilibrium
- **Stability Score**: Consistency across different inputs

## Tips for Using the Notebooks

1. **Start Simple**: Begin with notebook 01 to understand basics
2. **Experiment**: Modify prompts and parameters to see effects
3. **Monitor Progress**: Use visualizations to track optimization
4. **Save Results**: Export optimized prompts for production use
5. **Use Caching**: Enable caching for faster experimentation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure stackelberg-opt is installed:
   ```bash
   pip install -e ..
   ```

2. **Async Errors**: Some cells use `await`. Ensure you're running in an async-compatible environment or use:
   ```python
   import asyncio
   asyncio.run(your_async_function())
   ```

3. **Visualization Issues**: Install matplotlib and seaborn:
   ```bash
   pip install matplotlib seaborn
   ```

### Getting Help

- Check the [main documentation](https://github.com/youraanshshah/stackelberg-opt)
- Review the example implementations in `stackelberg_opt/examples/`
- Open an issue on GitHub for bugs or questions

## Contributing

We welcome contributions! If you create useful notebooks:
1. Follow the existing naming convention
2. Include comprehensive markdown documentation
3. Test all code cells
4. Submit a pull request

## License

These notebooks are part of the stackelberg-opt project and are released under the MIT License.