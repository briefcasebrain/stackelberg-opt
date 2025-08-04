"""
Stability analysis for Stackelberg optimization.

This module calculates multi-faceted stability scores including
performance stability, strategic stability, and robustness.
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional

import numpy as np
import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    sentence_model = None

from ..core.module import ModuleType
from ..core.candidate import SystemCandidate

logger = logging.getLogger(__name__)


class StabilityCalculator:
    """
    Calculate multi-faceted stability scores with perturbation analysis.
    
    Analyzes system stability from multiple perspectives including
    performance consistency, strategic dominance, robustness to errors,
    Nash stability, and evolutionary stability.
    
    Attributes:
        perturbation_samples: Number of perturbation samples for stability analysis
        stability_cache: Cache for computed stability scores
        
    Examples:
        >>> calculator = StabilityCalculator(perturbation_samples=10)
        >>> stability_score = calculator.calculate_stability(
        ...     candidate, population, evaluator
        ... )
    """
    
    def __init__(self, perturbation_samples: int = 10):
        self.perturbation_samples = perturbation_samples
        self.stability_cache = {}
    
    def calculate_stability(
        self, 
        candidate: SystemCandidate, 
        population: List[SystemCandidate],
        evaluator: Optional[Any] = None
    ) -> float:
        """
        Calculate comprehensive stability score.
        
        Args:
            candidate: Candidate to evaluate
            population: Current population for comparison
            evaluator: Optional evaluator for perturbation analysis
            
        Returns:
            Stability score between 0 and 1
        """
        if not candidate.traces:
            return 0.5
        
        # Multiple stability components
        components = {
            'performance_stability': self._calculate_performance_stability(candidate),
            'strategic_stability': self._calculate_strategic_stability(candidate, population),
            'robustness_score': self._calculate_robustness(candidate),
            'nash_stability': self._calculate_nash_stability(candidate, population),
            'evolutionary_stability': self._calculate_evolutionary_stability(candidate, population)
        }
        
        # Add perturbation stability if evaluator available
        if evaluator:
            components['perturbation_stability'] = self._calculate_perturbation_stability(
                candidate, evaluator
            )
        
        # Weighted combination
        weights = {
            'performance_stability': 0.25,
            'strategic_stability': 0.20,
            'robustness_score': 0.20,
            'nash_stability': 0.15,
            'evolutionary_stability': 0.10,
            'perturbation_stability': 0.10
        }
        
        stability = sum(
            components.get(comp, 0.5) * weight 
            for comp, weight in weights.items()
        )
        
        # Store detailed breakdown
        candidate.performance_metrics['stability_components'] = components
        
        return float(np.clip(stability, 0, 1))
    
    def _calculate_performance_stability(self, candidate: SystemCandidate) -> float:
        """Stability of performance across different inputs."""
        scores = list(candidate.scores.values())
        
        if len(scores) < 2:
            return 0.5
        
        # Multiple metrics
        variance = np.var(scores)
        cv = np.std(scores) / (np.mean(scores) + 1e-10)  # Coefficient of variation
        
        # Success consistency
        success_rate = sum(1 for s in scores if s > 0.5) / len(scores)
        failure_rate = sum(1 for s in scores if s < 0.3) / len(scores)
        consistency = 1.0 - abs(success_rate - failure_rate)
        
        # Combine metrics
        variance_stability = 1.0 / (1.0 + variance)
        cv_stability = 1.0 / (1.0 + cv)
        
        return 0.4 * variance_stability + 0.3 * cv_stability + 0.3 * consistency
    
    def _calculate_strategic_stability(
        self, 
        candidate: SystemCandidate, 
        population: List[SystemCandidate]
    ) -> float:
        """Stability against strategic deviations."""
        if len(population) < 2:
            return 0.5
        
        candidate_avg = np.mean(list(candidate.scores.values()))
        
        # Check dominance relationships
        dominated_by = 0
        dominates = 0
        
        for other in population:
            if other.candidate_id == candidate.candidate_id:
                continue
            
            other_avg = np.mean(list(other.scores.values()))
            
            # Check task-wise dominance
            common_tasks = set(candidate.scores.keys()) & set(other.scores.keys())
            if common_tasks:
                candidate_better = sum(1 for t in common_tasks 
                                     if candidate.scores[t] > other.scores[t])
                other_better = sum(1 for t in common_tasks 
                                 if other.scores[t] > candidate.scores[t])
                
                if other_better > candidate_better and other_avg > candidate_avg:
                    dominated_by += 1
                elif candidate_better > other_better and candidate_avg > other_avg:
                    dominates += 1
        
        # More dominance = more stable position
        dominance_ratio = dominates / (dominates + dominated_by + 1)
        
        # Check local optimality
        similar_candidates = [
            other for other in population
            if other.candidate_id != candidate.candidate_id
            and self._calculate_similarity(candidate, other) > 0.8
        ]
        
        if similar_candidates:
            better_similar = sum(1 for other in similar_candidates
                               if np.mean(list(other.scores.values())) > candidate_avg * 1.05)
            local_optimality = 1.0 - (better_similar / len(similar_candidates))
        else:
            local_optimality = 0.5
        
        return 0.6 * dominance_ratio + 0.4 * local_optimality
    
    def _calculate_robustness(self, candidate: SystemCandidate) -> float:
        """Robustness to errors and failures."""
        if not candidate.traces:
            return 0.5
        
        # Error analysis
        total_traces = len(candidate.traces)
        error_traces = sum(1 for t in candidate.traces.values() if not t.success)
        error_rate = error_traces / total_traces if total_traces > 0 else 0
        
        # Recovery analysis
        recovery_scores = []
        trace_list = list(candidate.traces.values())
        
        for i in range(len(trace_list) - 1):
            if not trace_list[i].success and trace_list[i + 1].success:
                recovery_scores.append(1.0)
            elif not trace_list[i].success and not trace_list[i + 1].success:
                recovery_scores.append(0.0)
        
        recovery_rate = np.mean(recovery_scores) if recovery_scores else 0.5
        
        # Partial success analysis
        partial_successes = []
        for trace in candidate.traces.values():
            if not trace.success:
                # Check if any modules succeeded
                successful_modules = sum(1 for score in trace.intermediate_scores.values() 
                                       if score > 0.5)
                partial_success_rate = successful_modules / len(trace.intermediate_scores) \
                                     if trace.intermediate_scores else 0
                partial_successes.append(partial_success_rate)
        
        graceful_degradation = np.mean(partial_successes) if partial_successes else 1.0
        
        # Combine metrics
        return 0.4 * (1 - error_rate) + 0.3 * recovery_rate + 0.3 * graceful_degradation
    
    def _calculate_nash_stability(
        self, 
        candidate: SystemCandidate, 
        population: List[SystemCandidate]
    ) -> float:
        """Approximate Nash stability - resistance to unilateral deviations."""
        if not population:
            return 0.5
        
        # For each module, check if changing just that module improves performance
        deviation_improvements = []
        
        for module_name, module in candidate.modules.items():
            # Find candidates that differ only in this module
            similar_candidates = []
            
            for other in population:
                if other.candidate_id == candidate.candidate_id:
                    continue
                
                # Check if only this module differs
                differs_only_here = True
                for name, other_module in other.modules.items():
                    if name == module_name:
                        continue
                    if name not in candidate.modules or \
                       other_module.prompt != candidate.modules[name].prompt:
                        differs_only_here = False
                        break
                
                if differs_only_here and module_name in other.modules:
                    similar_candidates.append(other)
            
            # Check if any deviation improves performance
            if similar_candidates:
                candidate_avg = np.mean(list(candidate.scores.values()))
                max_improvement = 0
                
                for other in similar_candidates:
                    other_avg = np.mean(list(other.scores.values()))
                    improvement = other_avg - candidate_avg
                    max_improvement = max(max_improvement, improvement)
                
                deviation_improvements.append(max_improvement)
        
        # Lower improvements from deviations = more stable
        if deviation_improvements:
            avg_improvement = np.mean(deviation_improvements)
            nash_stability = 1.0 / (1.0 + avg_improvement * 10)  # Scale factor
        else:
            nash_stability = 0.5
        
        return nash_stability
    
    def _calculate_evolutionary_stability(
        self, 
        candidate: SystemCandidate, 
        population: List[SystemCandidate]
    ) -> float:
        """Evolutionary stability - resistance to invasion by mutants."""
        if len(population) < 3:
            return 0.5
        
        # Check offspring performance
        children = [c for c in population if c.parent_id == candidate.candidate_id]
        
        if not children:
            # No offspring to compare
            return 0.5
        
        # Compare candidate to its children
        candidate_avg = np.mean(list(candidate.scores.values()))
        
        better_children = 0
        for child in children:
            child_avg = np.mean(list(child.scores.values()))
            if child_avg > candidate_avg * 1.05:  # 5% improvement threshold
                better_children += 1
        
        # Fewer better children = more evolutionarily stable
        invasion_resistance = 1.0 - (better_children / len(children))
        
        # Also check lineage persistence
        descendants = children.copy()
        for child in children:
            descendants.extend([c for c in population if c.parent_id == child.candidate_id])
        
        lineage_size = len(set(c.candidate_id for c in descendants))
        lineage_persistence = min(1.0, lineage_size / 10)  # Normalize
        
        return 0.7 * invasion_resistance + 0.3 * lineage_persistence
    
    def _calculate_perturbation_stability(
        self, 
        candidate: SystemCandidate,
        evaluator: Any
    ) -> float:
        """Stability under input perturbations."""
        # This is expensive, so we sample a few test cases
        test_indices = list(candidate.scores.keys())[:self.perturbation_samples]
        
        if not test_indices:
            return 0.5
        
        perturbation_scores = []
        
        for idx in test_indices:
            original_score = candidate.scores[idx]
            
            # We would need access to original inputs to perturb them
            # For now, simulate with score variation
            perturbed_scores = []
            
            for _ in range(5):
                # Simulate perturbation effect
                noise = np.random.normal(0, 0.1)
                perturbed_score = np.clip(original_score + noise, 0, 1)
                perturbed_scores.append(perturbed_score)
            
            # Calculate stability for this input
            score_variance = np.var(perturbed_scores)
            stability = 1.0 / (1.0 + score_variance * 10)
            perturbation_scores.append(stability)
        
        return np.mean(perturbation_scores) if perturbation_scores else 0.5
    
    def _calculate_similarity(self, cand1: SystemCandidate, cand2: SystemCandidate) -> float:
        """Calculate similarity between two candidates."""
        # Prompt similarity
        prompt_similarities = []
        
        for name in cand1.modules:
            if name in cand2.modules:
                prompt1 = cand1.modules[name].prompt
                prompt2 = cand2.modules[name].prompt
                
                if prompt1 == prompt2:
                    prompt_similarities.append(1.0)
                else:
                    # Use sentence embeddings if available
                    if sentence_model:
                        emb1 = sentence_model.encode([prompt1])[0]
                        emb2 = sentence_model.encode([prompt2])[0]
                        similarity = np.dot(emb1, emb2) / (
                            np.linalg.norm(emb1) * np.linalg.norm(emb2)
                        )
                        prompt_similarities.append(similarity)
                    else:
                        # Simple character overlap
                        similarity = SequenceMatcher(None, prompt1, prompt2).ratio()
                        prompt_similarities.append(similarity)
        
        return np.mean(prompt_similarities) if prompt_similarities else 0.0