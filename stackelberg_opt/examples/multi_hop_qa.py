"""
Multi-hop Question Answering Example

This example demonstrates how to use stackelberg-opt for optimizing
a multi-hop QA system with leader-follower dynamics.
"""

import asyncio
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
import litellm  # Language model library

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


def create_multi_hop_qa_system() -> Dict[str, Module]:
    """
    Create a multi-hop QA system configuration with leader-follower structure.
    
    Returns:
        Dictionary of modules for the QA system
    """
    modules = {
        # Leader module: Query Generator
        "query_generator": Module(
            name="query_generator",
            prompt="""Given a complex question that requires multiple steps to answer,
generate the first search query to find relevant information.

Question: {question}

Think about what information you need first and generate a focused search query.
Output only the search query.""",
            module_type=ModuleType.LEADER,
            dependencies=[]
        ),
        
        # Follower module: Context Retriever
        "context_retriever": Module(
            name="context_retriever",
            prompt="""Given a search query, retrieve and summarize relevant context.

Query: {query}

Simulate retrieving information for this query and provide a concise summary
of what you would find. Focus on facts that help answer the original question.""",
            module_type=ModuleType.FOLLOWER,
            dependencies=["query_generator"]
        ),
        
        # Follower module: Follow-up Query Generator
        "followup_generator": Module(
            name="followup_generator",
            prompt="""Based on the initial context, determine if more information is needed.
If so, generate a follow-up search query.

Question: {question}
Initial Query: {initial_query}
Context Found: {context}

If the context is sufficient, output "NONE". Otherwise, output a follow-up query.""",
            module_type=ModuleType.FOLLOWER,
            dependencies=["query_generator", "context_retriever"]
        ),
        
        # Independent module: Answer Synthesizer
        "answer_synthesizer": Module(
            name="answer_synthesizer",
            prompt="""Synthesize a comprehensive answer from all gathered information.

Question: {question}
Information Gathered: {all_context}

Provide a clear, accurate answer that addresses all aspects of the question.""",
            module_type=ModuleType.INDEPENDENT,
            dependencies=["context_retriever", "followup_generator"]
        )
    }
    
    return modules


class MultiHopQAExecutor:
    """Task executor for multi-hop question answering."""
    
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None):
        self.model = model or os.getenv('STACKELBERG_MODEL', 'gpt-3.5-turbo')
        self.temperature = temperature if temperature is not None else float(os.getenv('STACKELBERG_TEMPERATURE', '0.3'))
    
    async def __call__(
        self, 
        modules: Dict[str, Module], 
        question: str
    ) -> Tuple[str, ExecutionTrace]:
        """
        Execute the multi-hop QA pipeline.
        
        Args:
            modules: Dictionary of modules with their prompts
            question: The question to answer
            
        Returns:
            Tuple of (final answer, execution trace)
        """
        trace = ExecutionTrace()
        trace.execution_order = []
        trace.module_outputs = {}
        trace.module_timings = {}
        trace.intermediate_scores = {}
        
        try:
            # Step 1: Generate initial query (Leader)
            import time
            start_time = time.time()
            
            query_prompt = modules["query_generator"].prompt.format(question=question)
            initial_query = await self._call_model(query_prompt)
            
            trace.execution_order.append("query_generator")
            trace.module_outputs["query_generator"] = initial_query
            trace.module_timings["query_generator"] = time.time() - start_time
            
            # Step 2: Retrieve context (Follower)
            start_time = time.time()
            
            context_prompt = modules["context_retriever"].prompt.format(
                query=initial_query
            )
            context = await self._call_model(context_prompt)
            
            trace.execution_order.append("context_retriever")
            trace.module_outputs["context_retriever"] = context
            trace.module_timings["context_retriever"] = time.time() - start_time
            
            # Step 3: Generate follow-up if needed (Follower)
            start_time = time.time()
            
            followup_prompt = modules["followup_generator"].prompt.format(
                question=question,
                initial_query=initial_query,
                context=context
            )
            followup = await self._call_model(followup_prompt)
            
            trace.execution_order.append("followup_generator")
            trace.module_outputs["followup_generator"] = followup
            trace.module_timings["followup_generator"] = time.time() - start_time
            
            # Gather all context
            all_context = f"Initial search: {initial_query}\nContext: {context}"
            if followup != "NONE":
                # Simulate follow-up retrieval
                followup_context = await self._call_model(
                    f"Retrieve information for: {followup}"
                )
                all_context += f"\nFollow-up search: {followup}\nAdditional context: {followup_context}"
            
            # Step 4: Synthesize answer (Independent)
            start_time = time.time()
            
            answer_prompt = modules["answer_synthesizer"].prompt.format(
                question=question,
                all_context=all_context
            )
            final_answer = await self._call_model(answer_prompt)
            
            trace.execution_order.append("answer_synthesizer")
            trace.module_outputs["answer_synthesizer"] = final_answer
            trace.module_timings["answer_synthesizer"] = time.time() - start_time
            
            # Calculate intermediate scores based on output quality
            self._calculate_intermediate_scores(trace, modules)
            
            trace.success = True
            trace.final_score = sum(trace.intermediate_scores.values()) / len(trace.intermediate_scores)
            
            return final_answer, trace
            
        except Exception as e:
            logger.error(f"Error in QA execution: {e}")
            trace.success = False
            trace.error = str(e)
            trace.final_score = 0.0
            return "", trace
    
    async def _call_model(self, prompt: str) -> str:
        """Call the language model with the given prompt."""
        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return f"Error: {str(e)}"
    
    def _calculate_intermediate_scores(
        self, 
        trace: ExecutionTrace, 
        modules: Dict[str, Module]
    ):
        """Calculate quality scores for each module's output."""
        for module_name in trace.execution_order:
            output = trace.module_outputs[module_name]
            
            # Simple heuristic scoring
            score = 0.5  # Base score
            
            # Penalize empty or error outputs
            if not output or "Error" in output:
                score = 0.1
            # Reward substantial outputs
            elif len(output) > 50:
                score = 0.7
            # Bonus for structured/detailed outputs
            if len(output) > 100 and "\n" in output:
                score = 0.9
            
            # Leader modules get slightly higher weight
            if modules[module_name].module_type == ModuleType.LEADER:
                score *= 1.1
                score = min(1.0, score)
            
            trace.intermediate_scores[module_name] = score


async def run_qa_optimization_example():
    """Run the complete multi-hop QA optimization example."""
    
    # Create the QA system
    qa_modules = create_multi_hop_qa_system()
    
    # Create training data
    train_data = [
        (
            "What are the environmental impacts of electric vehicles compared to gasoline cars?",
            "Electric vehicles have lower operational emissions but higher manufacturing emissions..."
        ),
        (
            "How did the Renaissance influence modern scientific thought?",
            "The Renaissance promoted empirical observation and mathematical reasoning..."
        ),
        (
            "What are the main differences between quantum and classical computing?",
            "Quantum computers use qubits and superposition while classical computers use bits..."
        ),
        (
            "How do vaccines work and why are boosters sometimes needed?",
            "Vaccines train the immune system to recognize pathogens..."
        ),
        (
            "What factors contributed to the fall of the Roman Empire?",
            "Multiple factors including economic troubles, military problems, and political instability..."
        )
    ]
    
    # Create task executor
    task_executor = MultiHopQAExecutor()  # Uses environment variables
    
    # Configure optimizer
    config = OptimizerConfig(
        budget=100,  # Limit for example
        population_size=10,
        mutation_rate=0.7,
        # Model configuration from environment variables
        performance_weight=0.5,
        equilibrium_weight=0.3,
        stability_weight=0.2,
        enable_visualization=True,
        checkpoint_interval=25
    )
    
    # Create and run optimizer
    logger.info("Starting multi-hop QA optimization...")
    optimizer = StackelbergOptimizer(
        system_modules=qa_modules,
        train_data=train_data,
        task_executor=task_executor,
        config=config
    )
    
    # Run optimization
    best_candidate = await optimizer.optimize_async()
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*50)
    
    logger.info(f"\nBest candidate ID: {best_candidate.candidate_id}")
    logger.info(f"Average score: {best_candidate.get_average_score():.3f}")
    logger.info(f"Equilibrium value: {best_candidate.equilibrium_value:.3f}")
    logger.info(f"Stability score: {best_candidate.stability_score:.3f}")
    
    logger.info("\nOptimized prompts:")
    for module_name, module in best_candidate.modules.items():
        logger.info(f"\n{module_name} ({module.module_type.value}):")
        logger.info(f"{module.prompt[:200]}...")
    
    # Test the optimized system
    logger.info("\n" + "="*50)
    logger.info("TESTING OPTIMIZED SYSTEM")
    logger.info("="*50)
    
    test_question = "What are the causes and effects of climate change?"
    logger.info(f"\nTest question: {test_question}")
    
    answer, trace = await task_executor(best_candidate.modules, test_question)
    
    logger.info(f"\nAnswer: {answer}")
    logger.info(f"\nExecution trace:")
    for module in trace.execution_order:
        logger.info(f"  - {module}: {trace.intermediate_scores.get(module, 0):.3f}")
    
    return optimizer, best_candidate


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_qa_optimization_example())