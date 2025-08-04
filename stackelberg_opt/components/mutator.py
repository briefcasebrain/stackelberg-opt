"""
Prompt mutation for Stackelberg optimization.

This module provides intelligent prompt mutation using language models
to improve module prompts based on performance feedback.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.module import Module, ModuleType
from ..core.candidate import SystemCandidate
from ..utils.cache import ResponseCache

logger = logging.getLogger(__name__)


class PromptMutator:
    """
    Production-ready prompt mutator with caching and error handling.
    
    Uses language models to intelligently mutate prompts based on performance
    feedback, maintaining leader-follower constraints in the Stackelberg game.
    
    Attributes:
        model: Language model to use for mutations
        temperature: Temperature for generation
        max_retries: Maximum retry attempts for API calls
        cache: Optional cache for API responses
        
    Examples:
        >>> mutator = PromptMutator()  # Uses STACKELBERG_MODEL env var
        >>> new_prompt = mutator.mutate_prompt(module, candidate, feedback)
    """
    
    def __init__(
        self, 
        model: Optional[str] = None, 
        temperature: Optional[float] = None,
        max_retries: int = 3, 
        cache_enabled: bool = True
    ):
        self.model = model or os.getenv('STACKELBERG_MODEL', 'gpt-3.5-turbo')
        self.temperature = temperature if temperature is not None else float(os.getenv('STACKELBERG_TEMPERATURE', '0.7'))
        self.max_retries = max_retries
        self.cache = ResponseCache() if cache_enabled else None
        self.token_usage = defaultdict(int)
        self.api_costs = defaultdict(float)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """Make model API call with retries."""
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=500
        )
        
        # Track usage
        if hasattr(response, 'usage'):
            self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
            self.token_usage['completion_tokens'] += response.usage.completion_tokens
            self.token_usage['total_tokens'] += response.usage.total_tokens
        
        return response.choices[0].message.content.strip()
    
    def mutate_prompt(
        self, 
        module: Module, 
        parent_candidate: SystemCandidate, 
        feedback: Dict[str, Any]
    ) -> str:
        """
        Mutate prompt with comprehensive error handling.
        
        Args:
            module: Module to mutate
            parent_candidate: Parent candidate containing the module
            feedback: Performance feedback for the module
            
        Returns:
            Mutated prompt string
        """
        # Check cache first
        cache_key = f"{module.name}:{module.prompt}:{json.dumps(feedback, sort_keys=True)}"
        if self.cache:
            cached = self.cache.get(cache_key, self.model, self.temperature)
            if cached:
                logger.info(f"Using cached mutation for {module.name}")
                return cached
        
        try:
            if module.module_type == ModuleType.LEADER:
                mutation_prompt = self._create_leader_mutation_prompt(module, feedback)
            elif module.module_type == ModuleType.FOLLOWER:
                leader_constraints = self._extract_leader_constraints(
                    module, parent_candidate, feedback
                )
                mutation_prompt = self._create_follower_mutation_prompt(
                    module, feedback, leader_constraints
                )
            else:
                mutation_prompt = self._create_independent_mutation_prompt(module, feedback)
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert at optimizing prompts for AI systems. "
                              "Respond only with the improved prompt, no explanations."
                },
                {"role": "user", "content": mutation_prompt}
            ]
            
            new_prompt = self._call_model(messages)
            
            # Validate response
            if len(new_prompt) < 10 or len(new_prompt) > 2000:
                logger.warning(f"Invalid mutation response length: {len(new_prompt)}")
                return module.prompt
            
            # Cache successful response
            if self.cache:
                self.cache.set(cache_key, self.model, self.temperature, new_prompt)
            
            return new_prompt
            
        except Exception as e:
            logger.error(f"Model mutation failed: {e}")
            return self._fallback_mutation(module, feedback)
    
    def _fallback_mutation(self, module: Module, feedback: Dict[str, Any]) -> str:
        """Fallback mutation when model API fails."""
        base_prompt = module.prompt
        
        # Simple rule-based improvements
        improvements = []
        
        if feedback.get('avg_score', 0) < 0.3:
            improvements.append("Be more specific and detailed in your response.")
        
        if feedback.get('stability', 0) < 0.5:
            improvements.append("Ensure consistent behavior across different inputs.")
        
        if 'error' in ' '.join(feedback.get('failure_patterns', [])).lower():
            improvements.append("Handle edge cases and errors gracefully.")
        
        if improvements:
            return f"{base_prompt}\n\nAdditional guidelines:\n" + \
                   "\n".join(f"- {imp}" for imp in improvements)
        
        return base_prompt
    
    def _create_leader_mutation_prompt(self, module: Module, feedback: Dict[str, Any]) -> str:
        """Enhanced leader mutation prompt."""
        return f"""Improve this leader module prompt for '{module.name}':

Current prompt:
{module.prompt}

Performance Analysis:
- Average score: {feedback.get('avg_score', 0):.2f}
- Stability: {feedback.get('stability', 0):.2f}
- Success rate: {feedback.get('success_rate', 0):.2f}
- Error rate: {feedback.get('error_rate', 0):.2f}

Failure Patterns:
{chr(10).join(f"- {pattern}" for pattern in feedback.get('failure_patterns', ['None identified'])[:5])}

Successful Patterns:
{chr(10).join(f"- {pattern}" for pattern in feedback.get('success_patterns', ['None identified'])[:3])}

Module-specific feedback:
{feedback.get('module_feedback', 'No specific feedback')}

Downstream Impact Analysis:
{feedback.get('downstream_impact', 'No downstream analysis available')}

Requirements for the improved prompt:
1. Address identified failure patterns
2. Provide clearer, unambiguous guidance for downstream modules
3. Include specific examples or templates where helpful
4. Ensure robustness to input variations
5. Maximize information quality passed to followers
6. Include error handling instructions

Generate an improved prompt that addresses these issues while maintaining the core functionality."""
    
    def _create_follower_mutation_prompt(
        self, 
        module: Module, 
        feedback: Dict[str, Any], 
        leader_constraints: Dict[str, Any]
    ) -> str:
        """Enhanced follower mutation prompt."""
        return f"""Improve this follower module prompt for '{module.name}':

Current prompt:
{module.prompt}

Leader Module Information:
{self._format_leader_constraints(leader_constraints)}

Performance Analysis:
- Average score: {feedback.get('avg_score', 0):.2f}
- Stability: {feedback.get('stability', 0):.2f}
- Adaptation success: {feedback.get('adaptation_score', 0):.2f}
- Constraint satisfaction: {feedback.get('constraint_satisfaction', 0):.2f}

Failure Patterns:
{chr(10).join(f"- {pattern}" for pattern in feedback.get('failure_patterns', ['None identified'])[:5])}

Module-specific feedback:
{feedback.get('module_feedback', 'No specific feedback')}

Leader Output Variations:
{feedback.get('leader_variation_analysis', 'No variation analysis available')}

Requirements for the improved prompt:
1. Better utilize information from leader modules
2. Handle variations in leader outputs gracefully
3. Maintain performance despite upstream changes
4. Include specific strategies for common leader output patterns
5. Add validation for leader inputs
6. Optimize for the constraints provided by leaders

Generate an improved prompt that maximizes performance given leader constraints."""
    
    def _create_independent_mutation_prompt(self, module: Module, feedback: Dict[str, Any]) -> str:
        """Mutation prompt for independent modules."""
        return f"""Improve this independent module prompt for '{module.name}':

Current prompt:
{module.prompt}

Performance Analysis:
- Average score: {feedback.get('avg_score', 0):.2f}
- Stability: {feedback.get('stability', 0):.2f}
- Consistency: {feedback.get('consistency', 0):.2f}

Failure Patterns:
{chr(10).join(f"- {pattern}" for pattern in feedback.get('failure_patterns', ['None identified'])[:5])}

Requirements for the improved prompt:
1. Improve standalone performance
2. Ensure consistent behavior
3. Add clarity and specificity
4. Include relevant examples

Generate an improved prompt that addresses these issues."""
    
    def _extract_leader_constraints(
        self, 
        module: Module, 
        candidate: SystemCandidate, 
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive leader constraints."""
        constraints = {
            'prompts': {},
            'typical_outputs': {},
            'output_patterns': {},
            'performance_impact': {},
            'semantic_constraints': []
        }
        
        for dep_name in module.dependencies:
            dep_module = candidate.modules.get(dep_name)
            if dep_module and dep_module.module_type == ModuleType.LEADER:
                # Add prompt
                constraints['prompts'][dep_name] = dep_module.prompt
                
                # Extract output patterns from traces
                if candidate.traces:
                    outputs = []
                    for trace in candidate.traces.values():
                        if dep_name in trace.module_outputs:
                            outputs.append(trace.module_outputs[dep_name])
                    
                    if outputs:
                        # Get representative samples
                        constraints['typical_outputs'][dep_name] = outputs[:3]
                        
                        # Analyze patterns
                        if all(isinstance(o, str) for o in outputs):
                            lengths = [len(o) for o in outputs]
                            constraints['output_patterns'][dep_name] = {
                                'type': 'string',
                                'avg_length': np.mean(lengths),
                                'length_range': (min(lengths), max(lengths))
                            }
                
                # Performance impact
                if feedback.get('module_interactions'):
                    interaction = feedback['module_interactions'].get(
                        f"{dep_name}->{module.name}", {}
                    )
                    constraints['performance_impact'][dep_name] = interaction.get(
                        'impact_score', 0.5
                    )
        
        return constraints
    
    def _format_leader_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format constraints for prompt."""
        parts = []
        
        if constraints.get('prompts'):
            parts.append("Leader Module Prompts:")
            for name, prompt in constraints['prompts'].items():
                parts.append(f"  {name}: {prompt[:150]}...")
        
        if constraints.get('typical_outputs'):
            parts.append("\nTypical Leader Outputs:")
            for name, outputs in constraints['typical_outputs'].items():
                parts.append(f"  {name}: {str(outputs[0])[:100]}...")
        
        if constraints.get('output_patterns'):
            parts.append("\nOutput Patterns:")
            for name, pattern in constraints['output_patterns'].items():
                parts.append(f"  {name}: {pattern}")
        
        return "\n".join(parts) if parts else "No leader constraints available"
    
    def get_usage_report(self) -> Dict[str, Any]:
        """
        Get token usage and cost report.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'token_usage': dict(self.token_usage),
            'estimated_cost': self._estimate_cost(),
            'cache_stats': self.cache.memory_cache if self.cache else {}
        }
    
    def _estimate_cost(self) -> float:
        """Estimate API costs based on token usage."""
        # Rough estimates - adjust based on actual pricing
        cost_per_1k_prompt = 0.0015
        cost_per_1k_completion = 0.002
        
        prompt_cost = (self.token_usage['prompt_tokens'] / 1000) * cost_per_1k_prompt
        completion_cost = (self.token_usage['completion_tokens'] / 1000) * cost_per_1k_completion
        
        return prompt_cost + completion_cost