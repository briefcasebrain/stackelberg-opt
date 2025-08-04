# stackelberg-opt

[![PyPI version](https://badge.fury.io/py/stackelberg-opt.svg)](https://badge.fury.io/py/stackelberg-opt)
[![Tests](https://github.com/aanshshah/stackelberg-opt/actions/workflows/tests.yml/badge.svg)](https://github.com/aanshshah/stackelberg-opt/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/stackelberg-opt/badge/?version=latest)](https://stackelberg-opt.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Stackelberg game-theoretic optimization for compound AI systems.

## 🎯 Who is this for?

stackelberg-opt is designed for:

- **AI/ML Engineers** building multi-agent or compound AI systems
- **Researchers** exploring game-theoretic approaches to AI coordination
- **Prompt Engineers** optimizing complex prompt chains and hierarchies
- **System Architects** designing AI systems with strategic component interactions
- **Data Scientists** working on multi-objective optimization problems

## 🚀 Use Cases

### 1. **Multi-Agent AI Systems**
Optimize AI agents that need to coordinate strategically, where some agents (leaders) make decisions that influence others (followers).

```python
# Example: Customer service system with routing agent (leader) and specialist agents (followers)
modules = {
    "router": Module(name="router", module_type=ModuleType.LEADER),
    "technical_support": Module(name="technical_support", module_type=ModuleType.FOLLOWER),
    "billing_support": Module(name="billing_support", module_type=ModuleType.FOLLOWER)
}
```

### 2. **Hierarchical Prompt Chains**
Build and optimize LLM systems where prompts depend on outputs from other prompts, creating strategic dependencies.

```python
# Example: Research assistant with query planning and execution
modules = {
    "query_planner": Module(name="query_planner", module_type=ModuleType.LEADER),
    "search_executor": Module(name="search_executor", module_type=ModuleType.FOLLOWER),
    "summarizer": Module(name="summarizer", module_type=ModuleType.FOLLOWER)
}
```

### 3. **Multi-Step Reasoning Systems**
Create AI systems that perform complex reasoning through multiple coordinated steps.

```python
# Example: Code generation with planning, implementation, and review
modules = {
    "architect": Module(name="architect", module_type=ModuleType.LEADER),
    "implementer": Module(name="implementer", module_type=ModuleType.FOLLOWER),
    "code_reviewer": Module(name="code_reviewer", module_type=ModuleType.INDEPENDENT)
}
```

### 4. **Retrieval-Augmented Generation (RAG) Systems**
Optimize RAG pipelines where retrieval strategies influence generation quality.

```python
# Example: Advanced RAG with strategic retrieval
modules = {
    "query_reformulator": Module(name="query_reformulator", module_type=ModuleType.LEADER),
    "retriever": Module(name="retriever", module_type=ModuleType.FOLLOWER),
    "generator": Module(name="generator", module_type=ModuleType.FOLLOWER)
}
```

### 5. **Automated Decision-Making Systems**
Build systems where high-level decisions guide lower-level actions.

```python
# Example: Trading system with strategy and execution layers
modules = {
    "strategy_selector": Module(name="strategy_selector", module_type=ModuleType.LEADER),
    "risk_analyzer": Module(name="risk_analyzer", module_type=ModuleType.FOLLOWER),
    "trade_executor": Module(name="trade_executor", module_type=ModuleType.FOLLOWER)
}
```

### 6. **Content Generation Pipelines**
Optimize creative AI systems with hierarchical content generation.

```python
# Example: Blog writing system
modules = {
    "topic_planner": Module(name="topic_planner", module_type=ModuleType.LEADER),
    "outline_generator": Module(name="outline_generator", module_type=ModuleType.FOLLOWER),
    "content_writer": Module(name="content_writer", module_type=ModuleType.FOLLOWER),
    "editor": Module(name="editor", module_type=ModuleType.INDEPENDENT)
}
```

## 💡 Why Stackelberg Optimization?

Traditional optimization treats all components equally. Stackelberg optimization recognizes that in many AI systems:

1. **Some components naturally lead** - They make decisions that constrain or influence others
2. **Some components naturally follow** - They react optimally to leader decisions
3. **Strategic interaction matters** - The best system isn't just individually optimized components

This game-theoretic approach finds **equilibrium solutions** where:
- Leaders anticipate follower responses
- Followers respond optimally to leader decisions
- The entire system reaches a stable, optimal configuration

## Features

- 🎯 **True Bilevel Optimization**: Implements Stackelberg equilibrium with leader-follower dynamics
- 🧠 **LLM Integration**: Built-in support for prompt optimization using LiteLLM
- 📊 **Comprehensive Analysis**: Multi-faceted stability and equilibrium metrics
- ⚡ **Async Support**: Fully async implementation for improved performance
- 📈 **Visualization**: Built-in plotting and analysis tools
- 💾 **Checkpointing**: Save and resume optimization runs
- 🔧 **Extensible**: Modular design for easy customization

## 🔍 When to Use stackelberg-opt

### ✅ Good Fit

- Your AI system has **hierarchical decision-making**
- Some components' outputs **influence or constrain** others
- You need **strategic coordination** between AI agents
- You want to optimize **multi-objective** systems
- You're building **compound AI systems** with complex interactions

### ❌ Not a Good Fit

- Simple, single-component systems
- Components with no interdependencies
- Systems where all components are equal peers
- One-shot optimization without strategic interaction

## Installation

```bash
pip install stackelberg-opt
```

For development:
```bash
pip install stackelberg-opt[dev]
```

### Additional Setup

1. **Download the required spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

2. **Configure environment variables:**

Create a `.env` file in your project root (see `.env.example` for template):

```bash
# Required: Specify your language model
STACKELBERG_MODEL=gpt-3.5-turbo

# Optional: Adjust temperature for generation
STACKELBERG_TEMPERATURE=0.7

# Required: Set your API key for the model provider
OPENAI_API_KEY=your-api-key-here  # For OpenAI models
# or
ANTHROPIC_API_KEY=your-api-key-here  # For Anthropic models
# or configure for your provider
```

The library uses environment variables for model configuration to keep credentials secure and make deployment easier. All model-related settings can be configured through environment variables.

## Quick Start

```python
from stackelberg_opt import StackelbergOptimizer, Module, ModuleType, OptimizerConfig

# Define your system modules
modules = {
    "query_generator": Module(
        name="query_generator",
        prompt="Generate a search query for the given question: {question}",
        module_type=ModuleType.LEADER,
        dependencies=[]
    ),
    "answer_extractor": Module(
        name="answer_extractor", 
        prompt="Extract the answer from the context.\nQuery: {query}\nContext: {context}",
        module_type=ModuleType.FOLLOWER,
        dependencies=["query_generator"]
    )
}

# Define your training data
train_data = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    # ... more examples
]

# Define your task executor
async def task_executor(modules, input_data):
    # Implement your system execution logic here
    # This should run the modules and return output + execution trace
    pass

# Configure and run optimization
config = OptimizerConfig(
    budget=1000,
    population_size=20,
    mutation_rate=0.7,
    llm_model="gpt-3.5-turbo"
)

optimizer = StackelbergOptimizer(
    system_modules=modules,
    train_data=train_data,
    task_executor=task_executor,
    config=config
)

# Run optimization
best_candidate = optimizer.optimize()

# Access optimized prompts
for name, module in best_candidate.modules.items():
    print(f"{name}: {module.prompt}")
```

## Advanced Usage

### Custom Components

You can extend the optimizer with custom components:

```python
from stackelberg_opt.components import LLMPromptMutator

class CustomMutator(LLMPromptMutator):
    def mutate_prompt(self, module, parent_candidate, feedback):
        # Your custom mutation logic
        return super().mutate_prompt(module, parent_candidate, feedback)
```

### Checkpointing

Save and resume optimization runs:

```python
from stackelberg_opt.utils import CheckpointManager

# Enable checkpointing
checkpoint_manager = CheckpointManager()
optimizer.checkpoint_manager = checkpoint_manager

# Optimization will auto-save checkpoints
best_candidate = optimizer.optimize()

# Later, restore from checkpoint
state = checkpoint_manager.load_checkpoint("checkpoint_name")
```

### Visualization

Visualize optimization progress:

```python
from stackelberg_opt.utils import OptimizationVisualizer

visualizer = OptimizationVisualizer()
visualizer.plot_optimization_progress(optimizer)
visualizer.plot_evolution_tree(optimizer.population)
visualizer.plot_module_dependency_graph(optimizer.dependency_analysis)
```

## Architecture

The library is organized into three main modules:

- **Core**: Contains the main optimizer, module definitions, and candidate representations
- **Components**: Specialized components for mutation, evaluation, feedback extraction, etc.
- **Utils**: Utility functions for caching, checkpointing, and visualization

### Key Concepts

1. **Modules**: Individual components of your AI system with specific prompts
2. **Leaders & Followers**: Modules with hierarchical relationships based on Stackelberg game theory
3. **System Candidates**: Complete configurations of all modules
4. **Equilibrium**: Optimal strategic balance between leader and follower modules
5. **Stability**: Robustness of configurations to perturbations and variations

## Examples

See the `examples/` directory for complete working examples:

- `simple_optimization.py`: Basic prompt optimization
- `multi_hop_qa.py`: Multi-hop question answering system

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/aanshshah/stackelberg-opt.git
cd stackelberg-opt

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run linting
flake8 stackelberg_opt tests
mypy stackelberg_opt

# Format code
black stackelberg_opt tests
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{stackelberg-opt,
  title = {stackelberg-opt: Stackelberg Game-Theoretic Optimization for Compound AI Systems},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/aanshshah/stackelberg-opt}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is inspired by research in:
- Bilevel optimization
- Game theory
- Prompt engineering
- Evolutionary algorithms

## Real-World Applications

stackelberg-opt has potential applications in:

### 🏢 Enterprise
- **Customer Service Automation**: Hierarchical routing and response systems
- **Supply Chain Optimization**: Strategic planning with cascading decisions
- **Resource Allocation**: Leader divisions allocating to follower teams

### 🔬 Research
- **Scientific Discovery**: Hypothesis generation leading experimental design
- **Literature Review**: Query planning guiding systematic searches
- **Data Analysis Pipelines**: Strategic data exploration workflows

### 🏥 Healthcare
- **Diagnostic Systems**: Symptom analysis leading specialized tests
- **Treatment Planning**: High-level strategies guiding specific interventions
- **Medical Imaging**: Region selection directing detailed analysis

### 💰 Finance
- **Risk Management**: Portfolio strategy influencing individual positions
- **Fraud Detection**: Pattern identification guiding investigation
- **Algorithmic Trading**: Market analysis leading execution strategies

### 🎓 Education
- **Personalized Learning**: Learning path planning with adaptive content
- **Assessment Systems**: Question selection based on student responses
- **Curriculum Design**: High-level objectives guiding specific lessons

## Support

- Documentation: [https://stackelberg-opt.readthedocs.io](https://stackelberg-opt.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/aanshshah/stackelberg-opt/issues)
- Discussions: [GitHub Discussions](https://github.com/aanshshah/stackelberg-opt/discussions)
