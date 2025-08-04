"""
System candidate representation for Stackelberg optimization.

This module contains the SystemCandidate class which represents
a complete system configuration with its performance history.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .module import Module


@dataclass
class ExecutionTrace:
    """
    Comprehensive execution trace with causal information.
    
    Tracks the execution of a system configuration including inputs,
    outputs, timings, errors, and causal relationships between modules.
    """
    module_inputs: Dict[str, Any] = field(default_factory=dict)
    module_outputs: Dict[str, Any] = field(default_factory=dict)
    module_timings: Dict[str, float] = field(default_factory=dict)
    error_messages: Dict[str, str] = field(default_factory=dict)
    intermediate_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    success: bool = True
    execution_order: List[str] = field(default_factory=list)
    causal_links: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemCandidate:
    """
    Represents a candidate system configuration.
    
    A SystemCandidate encapsulates a complete system configuration including
    all modules, their performance history, and genealogy information for
    evolutionary optimization.
    
    Attributes:
        modules: Dictionary mapping module names to Module instances
        scores: Performance scores indexed by task ID
        traces: Execution traces indexed by task ID
        equilibrium_value: Calculated Stackelberg equilibrium value
        stability_score: System stability metric
        parent_id: ID of parent candidate (for genealogy)
        candidate_id: Unique identifier for this candidate
        diversity_hash: Hash for diversity calculations
        generation: Generation number in evolutionary process
        creation_time: Timestamp of creation
        mutation_history: List of mutations applied
        performance_metrics: Additional performance metrics
        
    Examples:
        >>> modules = {
        ...     "leader": Module(name="leader", prompt="Lead", module_type=ModuleType.LEADER),
        ...     "follower": Module(name="follower", prompt="Follow", module_type=ModuleType.FOLLOWER)
        ... }
        >>> candidate = SystemCandidate(modules=modules, candidate_id=1)
    """
    modules: Dict[str, Module]
    scores: Dict[int, float] = field(default_factory=dict)
    traces: Dict[int, ExecutionTrace] = field(default_factory=dict)
    equilibrium_value: float = 0.0
    stability_score: float = 0.0
    parent_id: Optional[int] = None
    candidate_id: int = 0
    diversity_hash: str = ""
    generation: int = 0
    creation_time: float = field(default_factory=time.time)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate diversity hash if not provided."""
        if not self.diversity_hash:
            prompt_str = "".join([m.prompt for m in sorted(self.modules.values(), key=lambda x: x.name)])
            self.diversity_hash = hashlib.md5(prompt_str.encode()).hexdigest()[:8]
    
    def get_genealogy(self) -> List[int]:
        """
        Get list of ancestor IDs.
        
        Returns:
            List of ancestor candidate IDs
        """
        genealogy = []
        current_id = self.parent_id
        seen = set()
        
        while current_id is not None and current_id not in seen:
            genealogy.append(current_id)
            seen.add(current_id)
            # Would need access to population to continue
            break
        
        return genealogy
    
    def get_average_score(self) -> float:
        """
        Calculate average performance score.
        
        Returns:
            Average of all task scores
        """
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0
    
    def get_success_rate(self) -> float:
        """
        Calculate success rate from traces.
        
        Returns:
            Fraction of successful executions
        """
        if not self.traces:
            return 0.0
        successful = sum(1 for trace in self.traces.values() if trace.success)
        return successful / len(self.traces)
    
    def add_mutation(self, module_name: str, old_prompt: str, new_prompt: str, reason: str = ""):
        """
        Record a mutation in the history.
        
        Args:
            module_name: Name of mutated module
            old_prompt: Original prompt
            new_prompt: New prompt after mutation
            reason: Reason for mutation
        """
        self.mutation_history.append({
            'timestamp': time.time(),
            'module': module_name,
            'old_prompt': old_prompt,
            'new_prompt': new_prompt,
            'reason': reason
        })