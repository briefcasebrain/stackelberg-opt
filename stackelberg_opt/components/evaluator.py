"""
System evaluation component for Stackelberg optimization.

This module provides comprehensive evaluation of compound AI systems,
including execution tracing and performance scoring.
"""

import asyncio
import logging
import traceback
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np

from ..core.module import Module
from ..core.candidate import ExecutionTrace

logger = logging.getLogger(__name__)


class CompoundSystemEvaluator:
    """
    Production-ready evaluator with async execution and comprehensive tracing.
    
    Evaluates compound AI systems by executing them on test inputs and scoring
    their outputs. Supports parallel evaluation and detailed execution tracing.
    
    Attributes:
        task_executor: Function to execute system tasks
        timeout: Maximum execution time per evaluation
        max_parallel: Maximum parallel evaluations
        
    Examples:
        >>> evaluator = CompoundSystemEvaluator(task_executor, timeout=30.0)
        >>> score, trace = evaluator.evaluate(modules, input_data, expected_output)
    """
    
    def __init__(
        self, 
        task_executor: Callable, 
        timeout: float = 30.0,
        max_parallel: int = 4
    ):
        self.task_executor = task_executor
        self.timeout = timeout
        self.max_parallel = max_parallel
        self.evaluation_cache = {}
    
    async def evaluate_async(
        self, 
        modules: Dict[str, Module], 
        input_data: Any, 
        expected: Any
    ) -> Tuple[float, ExecutionTrace]:
        """
        Async evaluation with timeout.
        
        Args:
            modules: System modules to evaluate
            input_data: Input for the system
            expected: Expected output
            
        Returns:
            Tuple of (score, execution_trace)
        """
        try:
            # Run with timeout
            output, trace = await asyncio.wait_for(
                self._execute_system(modules, input_data),
                timeout=self.timeout
            )
            
            # Calculate score
            score = self._calculate_score(output, expected, trace)
            trace.final_score = score
            
            # Attribute scores to modules
            self._attribute_module_scores(trace, modules, score)
            
            # Analyze causal relationships
            self._analyze_causality(trace)
            
            return score, trace
            
        except asyncio.TimeoutError:
            trace = ExecutionTrace(success=False)
            trace.error_messages["system"] = f"Evaluation timeout after {self.timeout}s"
            return 0.0, trace
        except Exception as e:
            trace = ExecutionTrace(success=False)
            trace.error_messages["system"] = f"Evaluation error: {str(e)}"
            trace.metadata["traceback"] = traceback.format_exc()
            return 0.0, trace
    
    def evaluate(
        self, 
        modules: Dict[str, Module], 
        input_data: Any, 
        expected: Any
    ) -> Tuple[float, ExecutionTrace]:
        """
        Synchronous evaluation wrapper.
        
        Args:
            modules: System modules to evaluate
            input_data: Input for the system
            expected: Expected output
            
        Returns:
            Tuple of (score, execution_trace)
        """
        return asyncio.run(self.evaluate_async(modules, input_data, expected))
    
    async def evaluate_batch(
        self, 
        modules: Dict[str, Module], 
        batch: List[Tuple[Any, Any]]
    ) -> List[Tuple[float, ExecutionTrace]]:
        """
        Evaluate multiple inputs in parallel.
        
        Args:
            modules: System modules to evaluate
            batch: List of (input, expected) tuples
            
        Returns:
            List of (score, trace) tuples
        """
        tasks = []
        for input_data, expected in batch:
            task = self.evaluate_async(modules, input_data, expected)
            tasks.append(task)
        
        # Run with concurrency limit
        results = []
        for i in range(0, len(tasks), self.max_parallel):
            batch_tasks = tasks[i:i + self.max_parallel]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    trace = ExecutionTrace(success=False)
                    trace.error_messages["system"] = str(result)
                    results.append((0.0, trace))
                else:
                    results.append(result)
        
        return results
    
    async def _execute_system(
        self, 
        modules: Dict[str, Module], 
        input_data: Any
    ) -> Tuple[Any, ExecutionTrace]:
        """Execute the system with comprehensive tracing."""
        # This is where the actual task executor is called
        # In practice, this would be implemented based on the specific system
        return await asyncio.to_thread(self.task_executor, modules, input_data)
    
    def _calculate_score(
        self, 
        output: Any, 
        expected: Any, 
        trace: ExecutionTrace
    ) -> float:
        """
        Calculate score with multiple metrics.
        
        Args:
            output: System output
            expected: Expected output
            trace: Execution trace
            
        Returns:
            Score between 0 and 1
        """
        if isinstance(expected, str) and isinstance(output, str):
            # String comparison metrics
            exact_match = 1.0 if output.strip().lower() == expected.strip().lower() else 0.0
            
            # Calculate additional metrics
            similarity = SequenceMatcher(None, output.lower(), expected.lower()).ratio()
            
            # Token-level F1 (simplified)
            output_tokens = set(output.lower().split())
            expected_tokens = set(expected.lower().split())
            
            if expected_tokens:
                precision = len(output_tokens & expected_tokens) / len(output_tokens) \
                           if output_tokens else 0
                recall = len(output_tokens & expected_tokens) / len(expected_tokens)
                f1 = 2 * precision * recall / (precision + recall) \
                     if (precision + recall) > 0 else 0
            else:
                f1 = 0.0
            
            # Store detailed metrics
            trace.metadata['exact_match'] = exact_match
            trace.metadata['similarity'] = similarity
            trace.metadata['token_f1'] = f1
            
            # Weighted combination
            return 0.5 * exact_match + 0.3 * f1 + 0.2 * similarity
        
        elif isinstance(expected, dict) and isinstance(output, dict):
            # Structured comparison
            correct_keys = sum(1 for k in expected 
                             if k in output and output[k] == expected[k])
            total_keys = len(expected)
            return correct_keys / total_keys if total_keys > 0 else 0.0
        
        else:
            # Binary comparison
            return 1.0 if output == expected else 0.0
    
    def _attribute_module_scores(
        self, 
        trace: ExecutionTrace, 
        modules: Dict[str, Module], 
        total_score: float
    ):
        """
        Attribute scores to individual modules based on their contribution.
        
        Args:
            trace: Execution trace
            modules: System modules
            total_score: Overall system score
        """
        num_modules = len(modules)
        
        if trace.success:
            # Analyze execution order and timing
            total_time = sum(trace.module_timings.values())
            
            for name, module in modules.items():
                # Base attribution
                base_score = total_score / num_modules
                
                # Adjust based on module type
                if module.module_type.value == "leader":
                    type_weight = 1.3  # Leaders have more influence
                elif module.module_type.value == "follower":
                    type_weight = 0.9
                else:
                    type_weight = 1.0
                
                # Adjust based on execution time
                time_weight = 1.0
                if name in trace.module_timings and total_time > 0:
                    time_ratio = trace.module_timings[name] / total_time
                    time_weight = 0.8 + 0.4 * time_ratio  # Range: 0.8 to 1.2
                
                # Final score
                trace.intermediate_scores[name] = base_score * type_weight * time_weight
        
        else:
            # For failures, penalize modules with errors more
            for name in modules:
                if name in trace.error_messages:
                    trace.intermediate_scores[name] = 0.0
                elif name in trace.module_outputs:
                    # Partial credit for successful execution
                    trace.intermediate_scores[name] = 0.3
                else:
                    # No execution
                    trace.intermediate_scores[name] = 0.1
    
    def _analyze_causality(self, trace: ExecutionTrace):
        """
        Analyze causal relationships between module executions.
        
        Args:
            trace: Execution trace to analyze
        """
        # Build causal graph based on execution order and data flow
        for i in range(len(trace.execution_order) - 1):
            current = trace.execution_order[i]
            next_module = trace.execution_order[i + 1]
            
            # Check if output of current is used in input of next
            if (current in trace.module_outputs and 
                next_module in trace.module_inputs):
                
                current_output = str(trace.module_outputs[current])
                next_input = str(trace.module_inputs[next_module])
                
                # Simple check - in practice would use more sophisticated analysis
                if len(current_output) > 10 and current_output[:50] in next_input:
                    if current not in trace.causal_links:
                        trace.causal_links[current] = []
                    trace.causal_links[current].append(next_module)