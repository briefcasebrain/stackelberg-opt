"""
Integration tests for stackelberg-opt.

These tests verify that components work together correctly.
"""

import pytest
import asyncio
from typing import Dict, Tuple

from stackelberg_opt import (
    StackelbergOptimizer,
    Module,
    ModuleType,
    SystemCandidate,
    ExecutionTrace,
    OptimizerConfig
)
from stackelberg_opt.components import (
    PromptMutator,
    CompoundSystemEvaluator,
    StackelbergFeedbackExtractor,
    PopulationManager
)


@pytest.mark.integration
class TestOptimizerIntegration:
    """Integration tests for the optimizer workflow."""
    
    @pytest.mark.asyncio
    async def test_basic_optimization_flow(self, simple_modules, mock_task_executor):
        """Test basic optimization flow with minimal configuration."""
        config = OptimizerConfig(
            budget=5,
            population_size=3,
            mutation_rate=0.5
        )
        
        train_data = [("test input", "expected output")]
        
        optimizer = StackelbergOptimizer(
            system_modules=simple_modules,
            train_data=train_data,
            task_executor=mock_task_executor,
            config=config
        )
        
        # Run a few optimization steps
        optimizer._initialize_population()
        assert len(optimizer.population) > 0
        
        # Evaluate population
        await optimizer._evaluate_population()
        
        # Check that candidates have scores
        for candidate in optimizer.population:
            assert len(candidate.scores) > 0
            assert candidate.get_average_score() > 0
    
    def test_mutation_pipeline(self, simple_modules):
        """Test the mutation pipeline integration."""
        # Create components
        mutator = PromptMutator(cache_enabled=False)
        population_manager = PopulationManager(max_size=10)
        
        # Create parent candidate
        parent = SystemCandidate(modules=simple_modules, candidate_id=1)
        parent.scores = {0: 0.7}
        
        # Add to population
        population_manager.add_candidate(parent, generation=0)
        
        # Extract feedback
        feedback_extractor = StackelbergFeedbackExtractor()
        feedback = {
            'avg_score': 0.7,
            'module_type': ModuleType.LEADER,
            'failure_patterns': []
        }
        
        # Mutate module
        for module_name, module in simple_modules.items():
            new_prompt = mutator._fallback_mutation(module, feedback)
            assert new_prompt != module.prompt
            assert len(new_prompt) > len(module.prompt)


@pytest.mark.integration
class TestComponentIntegration:
    """Test interactions between different components."""
    
    def test_evaluator_feedback_integration(self, complex_modules):
        """Test evaluator and feedback extractor integration."""
        # Create execution traces
        traces = []
        for i in range(3):
            trace = ExecutionTrace()
            trace.execution_order = list(complex_modules.keys())
            trace.module_outputs = {
                name: f"Output {i} from {name}"
                for name in complex_modules
            }
            trace.intermediate_scores = {
                name: 0.6 + i * 0.1
                for name in complex_modules
            }
            trace.success = True
            trace.final_score = 0.7 + i * 0.1
            traces.append(trace)
        
        # Extract feedback
        feedback_extractor = StackelbergFeedbackExtractor()
        
        for module_name, module in complex_modules.items():
            feedback = feedback_extractor.extract_feedback(
                module_name,
                traces,
                module
            )
            
            assert 'avg_score' in feedback
            assert 'success_rate' in feedback
            assert 'stability' in feedback
            assert feedback['module_name'] == module_name
    
    def test_population_selection_strategies(self):
        """Test different selection strategies in population manager."""
        manager = PopulationManager(max_size=20)
        
        # Create diverse population
        for i in range(15):
            candidate = SystemCandidate(
                modules={"test": Module("test", f"prompt {i}", ModuleType.LEADER)},
                candidate_id=i
            )
            candidate.scores = {0: 0.5 + (i / 30)}  # Increasing scores
            manager.add_candidate(candidate, generation=0)
        
        # Test different selection methods
        methods = ['tournament', 'roulette', 'diverse', 'elite']
        
        for method in methods:
            parents = manager.select_parents(n=3, method=method)
            assert len(parents) == 3
            
            # Verify selection worked
            for parent in parents:
                assert parent in manager.population


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndOptimization:
    """End-to-end optimization tests."""
    
    @pytest.mark.asyncio
    async def test_complete_optimization_cycle(self, complex_modules):
        """Test a complete optimization cycle."""
        # Create task executor
        async def task_executor(modules: Dict[str, Module], input_data: str) -> Tuple[str, ExecutionTrace]:
            trace = ExecutionTrace()
            trace.execution_order = ["orchestrator", "analyzer", "processor", "validator"]
            trace.module_outputs = {name: f"{name} processed: {input_data}" for name in modules}
            trace.module_timings = {name: 0.1 for name in modules}
            trace.intermediate_scores = {name: 0.7 + len(input_data) * 0.01 for name in modules}
            trace.success = True
            trace.final_score = 0.75
            return "final result", trace
        
        # Configure optimizer
        config = OptimizerConfig(
            budget=10,
            population_size=5,
            mutation_rate=0.7,
            enable_caching=True,
            enable_checkpointing=False,
            verbose=False
        )
        
        # Training data
        train_data = [
            ("short", "expected short"),
            ("medium length input", "expected medium"),
            ("this is a longer input for testing", "expected long")
        ]
        
        # Create and run optimizer
        optimizer = StackelbergOptimizer(
            system_modules=complex_modules,
            train_data=train_data,
            task_executor=task_executor,
            config=config
        )
        
        # Run optimization
        best_candidate = await optimizer.optimize_async()
        
        # Verify results
        assert best_candidate is not None
        assert best_candidate.get_average_score() > 0
        assert best_candidate.generation > 0
        
        # Check that modules were mutated
        mutated = False
        for name, module in best_candidate.modules.items():
            if module.prompt != complex_modules[name].prompt:
                mutated = True
                break
        
        assert mutated, "At least one module should have been mutated"