"""
Simple Optimization Example

This example demonstrates basic usage of stackelberg-opt for
optimizing a simple text transformation system.
"""

import asyncio
import logging
import os
from typing import Dict, Tuple
import re

from stackelberg_opt import (
    StackelbergOptimizer,
    Module,
    ModuleType,
    SystemCandidate,
    ExecutionTrace,
    OptimizerConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_text_transformation_system() -> Dict[str, Module]:
    """
    Create a simple text transformation system.
    
    This system demonstrates:
    - Leader module that extracts key information
    - Follower module that formats based on leader output
    - Independent module that adds polish
    """
    modules = {
        "key_extractor": Module(
            name="key_extractor",
            prompt="""Extract the key points from the following text.
Output a bullet-point list of the main ideas.

Text: {input_text}

Key points:""",
            module_type=ModuleType.LEADER,
            dependencies=[]
        ),
        
        "formatter": Module(
            name="formatter",
            prompt="""Given the key points, create a well-structured summary.

Key points:
{key_points}

Create a coherent paragraph that includes all points:""",
            module_type=ModuleType.FOLLOWER,
            dependencies=["key_extractor"]
        ),
        
        "polisher": Module(
            name="polisher",
            prompt="""Polish the following summary to be clear and engaging.

Summary: {summary}

Polished version:""",
            module_type=ModuleType.INDEPENDENT,
            dependencies=["formatter"]
        )
    }
    
    return modules


async def simple_task_executor(
    modules: Dict[str, Module], 
    input_text: str
) -> Tuple[str, ExecutionTrace]:
    """
    Simple task executor that transforms text through the pipeline.
    
    This executor simulates the transformation process without
    actually calling an LLM, making it suitable for testing.
    """
    trace = ExecutionTrace()
    trace.execution_order = []
    trace.module_outputs = {}
    trace.module_timings = {}
    trace.intermediate_scores = {}
    
    try:
        import time
        import random
        
        # Simulate key extraction (Leader)
        start_time = time.time()
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Extract simple key points
        sentences = input_text.split('.')
        key_points = "\n".join([
            f"• {s.strip()}" 
            for s in sentences[:3] 
            if s.strip()
        ])
        
        trace.execution_order.append("key_extractor")
        trace.module_outputs["key_extractor"] = key_points
        trace.module_timings["key_extractor"] = time.time() - start_time
        
        # Format summary (Follower)
        start_time = time.time()
        await asyncio.sleep(0.1)
        
        # Create summary from key points
        points = [p.strip('• ') for p in key_points.split('\n') if p.strip()]
        summary = " Furthermore, ".join(points) + "."
        
        trace.execution_order.append("formatter")
        trace.module_outputs["formatter"] = summary
        trace.module_timings["formatter"] = time.time() - start_time
        
        # Polish (Independent)
        start_time = time.time()
        await asyncio.sleep(0.1)
        
        # Simple polishing - capitalize and clean
        polished = summary.replace("Furthermore,", "Additionally,")
        polished = re.sub(r'\s+', ' ', polished).strip()
        polished = polished[0].upper() + polished[1:] if polished else polished
        
        trace.execution_order.append("polisher")
        trace.module_outputs["polisher"] = polished
        trace.module_timings["polisher"] = time.time() - start_time
        
        # Calculate scores based on output quality
        trace.intermediate_scores["key_extractor"] = 0.7 + random.random() * 0.3
        trace.intermediate_scores["formatter"] = 0.6 + random.random() * 0.3
        trace.intermediate_scores["polisher"] = 0.8 + random.random() * 0.2
        
        trace.success = True
        trace.final_score = sum(trace.intermediate_scores.values()) / len(trace.intermediate_scores)
        
        return polished, trace
        
    except Exception as e:
        logger.error(f"Error in execution: {e}")
        trace.success = False
        trace.error = str(e)
        trace.final_score = 0.0
        return "", trace


def simple_optimization_example():
    """
    Run a simple optimization example.
    
    This example shows:
    - Basic system setup
    - Configuration options
    - Running optimization
    - Examining results
    """
    # Create the system
    modules = create_text_transformation_system()
    
    # Create training data
    train_data = [
        (
            "Machine learning is a subset of artificial intelligence. It focuses on training algorithms. These algorithms learn from data patterns.",
            "Machine learning, a subset of artificial intelligence, focuses on training algorithms that learn from data patterns."
        ),
        (
            "Climate change affects global weather patterns. Rising temperatures cause extreme events. Immediate action is needed.",
            "Climate change affects global weather patterns, with rising temperatures causing extreme events that require immediate action."
        ),
        (
            "Regular exercise improves physical health. It also benefits mental wellbeing. Consistency is key to seeing results.",
            "Regular exercise improves both physical health and mental wellbeing, with consistency being key to seeing results."
        )
    ]
    
    # Configure the optimizer
    config = OptimizerConfig(
        budget=50,  # Small budget for demo
        population_size=5,
        mutation_rate=0.8,
        enable_caching=True,
        enable_checkpointing=True,
        checkpoint_interval=10,
        verbose=True
    )
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = StackelbergOptimizer(
        system_modules=modules,
        train_data=train_data,
        task_executor=simple_task_executor,
        config=config
    )
    
    # Run optimization
    logger.info("Starting optimization...")
    best_candidate = optimizer.optimize()
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*50)
    
    logger.info(f"\nBest candidate found:")
    logger.info(f"  ID: {best_candidate.candidate_id}")
    logger.info(f"  Generation: {best_candidate.generation}")
    logger.info(f"  Average score: {best_candidate.get_average_score():.3f}")
    logger.info(f"  Equilibrium value: {best_candidate.equilibrium_value:.3f}")
    logger.info(f"  Stability score: {best_candidate.stability_score:.3f}")
    
    # Show optimized prompts
    logger.info("\nOptimized prompts:")
    for name, module in best_candidate.modules.items():
        logger.info(f"\n{name} ({module.module_type.value}):")
        logger.info(f"  Original: {modules[name].prompt[:50]}...")
        logger.info(f"  Optimized: {module.prompt[:50]}...")
    
    # Test the optimized system
    logger.info("\n" + "="*50)
    logger.info("TESTING OPTIMIZED SYSTEM")
    logger.info("="*50)
    
    test_input = "Python is a versatile programming language. It has simple syntax. Many developers choose Python for various applications."
    logger.info(f"\nTest input: {test_input}")
    
    # Run through optimized system
    output, trace = asyncio.run(
        simple_task_executor(best_candidate.modules, test_input)
    )
    
    logger.info(f"\nOutput: {output}")
    logger.info(f"\nModule scores:")
    for module, score in trace.intermediate_scores.items():
        logger.info(f"  {module}: {score:.3f}")
    
    # Show population statistics
    if hasattr(optimizer, 'population_manager'):
        logger.info("\n" + "="*50)
        logger.info("POPULATION STATISTICS")
        logger.info("="*50)
        
        stats = optimizer.population_manager.get_statistics()
        logger.info(f"\nFinal population size: {stats['population_size']}")
        logger.info(f"Elite archive size: {stats['elite_size']}")
        logger.info(f"Diversity archive size: {stats['diversity_size']}")
        logger.info(f"Innovation archive size: {stats['innovation_size']}")
        
        if optimizer.population_manager.generation_stats:
            last_gen = max(optimizer.population_manager.generation_stats.keys())
            last_stats = optimizer.population_manager.generation_stats[last_gen]
            logger.info(f"\nFinal generation stats:")
            logger.info(f"  Average fitness: {last_stats.get('avg_fitness', 0):.3f}")
            logger.info(f"  Best fitness: {last_stats.get('best_fitness', 0):.3f}")
            logger.info(f"  Population diversity: {last_stats.get('diversity', 0):.3f}")
    
    return optimizer, best_candidate


def run_comparison_example():
    """
    Run a comparison between original and optimized systems.
    """
    logger.info("Running system comparison example...")
    
    # Get original and optimized systems
    original_modules = create_text_transformation_system()
    optimizer, best_candidate = simple_optimization_example()
    
    # Test inputs
    test_inputs = [
        "Data science combines statistics and programming. It extracts insights from data. Businesses use it for decision making.",
        "Renewable energy sources are sustainable. They reduce carbon emissions. Solar and wind power are growing rapidly.",
        "Artificial neural networks mimic brain structure. They process information in layers. Deep learning uses many layers."
    ]
    
    logger.info("\n" + "="*50)
    logger.info("SYSTEM COMPARISON")
    logger.info("="*50)
    
    for i, test_input in enumerate(test_inputs):
        logger.info(f"\nTest {i+1}: {test_input[:50]}...")
        
        # Run through original system
        orig_output, orig_trace = asyncio.run(
            simple_task_executor(original_modules, test_input)
        )
        
        # Run through optimized system
        opt_output, opt_trace = asyncio.run(
            simple_task_executor(best_candidate.modules, test_input)
        )
        
        logger.info(f"\nOriginal score: {orig_trace.final_score:.3f}")
        logger.info(f"Optimized score: {opt_trace.final_score:.3f}")
        logger.info(f"Improvement: {(opt_trace.final_score - orig_trace.final_score):.3f}")


if __name__ == "__main__":
    # Run the simple example
    simple_optimization_example()
    
    # Optionally run comparison
    # run_comparison_example()