"""
Population management for evolutionary optimization.

This module manages the population of candidates with archives
for elite, diversity, and innovation preservation.
"""

import logging
import random
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    sentence_model = None

from ..core.candidate import SystemCandidate

logger = logging.getLogger(__name__)


class PopulationManager:
    """
    Advanced population management with archives and diversity preservation.
    
    Manages the main population and specialized archives for elite performers,
    diverse strategies, and innovative approaches. Implements sophisticated
    selection and replacement strategies.
    
    Attributes:
        max_size: Maximum population size
        archive_size: Size of each archive
        diversity_weight: Weight for diversity in selection
        
    Examples:
        >>> manager = PopulationManager(max_size=20, diversity_weight=0.3)
        >>> added, reason = manager.add_candidate(candidate, generation)
        >>> parents = manager.select_parents(n=2, method='tournament')
    """
    
    def __init__(
        self, 
        max_size: int = 20, 
        archive_size: int = 100,
        diversity_weight: float = 0.3
    ):
        self.max_size = max_size
        self.archive_size = archive_size
        self.diversity_weight = diversity_weight
        
        # Main population
        self.population: List[SystemCandidate] = []
        
        # Archives
        self.elite_archive: List[SystemCandidate] = []  # Best performers
        self.diversity_archive: List[SystemCandidate] = []  # Diverse strategies
        self.innovation_archive: List[SystemCandidate] = []  # Novel approaches
        
        # Tracking
        self.generation_stats = defaultdict(lambda: {
            'avg_fitness': 0.0,
            'best_fitness': 0.0,
            'diversity': 0.0,
            'innovations': 0
        })
    
    def add_candidate(
        self, 
        candidate: SystemCandidate, 
        current_generation: int
    ) -> Tuple[bool, str]:
        """
        Add candidate with detailed decision reasoning.
        
        Args:
            candidate: Candidate to add
            current_generation: Current generation number
            
        Returns:
            Tuple of (added_to_population, reason)
        """
        # Always add to appropriate archives
        self._update_archives(candidate)
        
        # Decision process
        decision, reason = self._should_add_to_population(candidate)
        
        if decision:
            self.population.append(candidate)
            self._maintain_population_limit()
            
            # Update generation stats
            self._update_generation_stats(current_generation)
            
            return True, reason
        
        return False, reason
    
    def _should_add_to_population(self, candidate: SystemCandidate) -> Tuple[bool, str]:
        """Detailed decision process for adding candidates."""
        # Always add if population is small
        if len(self.population) < self.max_size // 2:
            return True, "Population below minimum size"
        
        # Quality check
        quality_score, quality_reason = self._assess_quality(candidate)
        if quality_score > 0.8:
            return True, f"High quality: {quality_reason}"
        
        # Diversity check
        diversity_score, diversity_reason = self._assess_diversity(candidate)
        if diversity_score > 0.7:
            return True, f"High diversity: {diversity_reason}"
        
        # Innovation check
        innovation_score, innovation_reason = self._assess_innovation(candidate)
        if innovation_score > 0.6:
            return True, f"Innovation: {innovation_reason}"
        
        # Strategic value check
        strategic_score, strategic_reason = self._assess_strategic_value(candidate)
        if strategic_score > 0.7:
            return True, f"Strategic value: {strategic_reason}"
        
        # Combined score
        combined_score = (
            0.4 * quality_score +
            0.3 * diversity_score +
            0.2 * innovation_score +
            0.1 * strategic_score
        )
        
        if combined_score > 0.6:
            return True, f"Combined score: {combined_score:.2f}"
        
        return False, f"Below threshold (combined: {combined_score:.2f})"
    
    def _assess_quality(self, candidate: SystemCandidate) -> Tuple[float, str]:
        """Assess candidate quality."""
        if not candidate.scores:
            return 0.0, "No scores available"
        
        avg_score = np.mean(list(candidate.scores.values()))
        
        # Compare to population
        if self.population:
            pop_scores = [np.mean(list(p.scores.values())) for p in self.population 
                         if p.scores]
            if pop_scores:
                percentile = np.sum(avg_score > np.array(pop_scores)) / len(pop_scores)
                
                if percentile > 0.9:
                    return 0.95, f"Top 10% performer ({avg_score:.2f})"
                elif percentile > 0.75:
                    return 0.8, f"Top 25% performer ({avg_score:.2f})"
                else:
                    return percentile, f"{int(percentile*100)}th percentile"
        
        # Absolute thresholds
        if avg_score > 0.9:
            return 0.9, f"Excellent performance ({avg_score:.2f})"
        elif avg_score > 0.7:
            return 0.7, f"Good performance ({avg_score:.2f})"
        else:
            return avg_score, f"Average performance ({avg_score:.2f})"
    
    def _assess_diversity(self, candidate: SystemCandidate) -> Tuple[float, str]:
        """Assess how diverse the candidate is."""
        if not self.population:
            return 1.0, "First candidate"
        
        # Calculate distances to existing candidates
        distances = []
        
        for other in self.population:
            # Prompt diversity
            prompt_dist = self._calculate_prompt_distance(candidate, other)
            
            # Performance diversity
            if candidate.scores and other.scores:
                common_tasks = set(candidate.scores.keys()) & set(other.scores.keys())
                if common_tasks:
                    perf_dist = np.mean([abs(candidate.scores[t] - other.scores[t]) 
                                       for t in common_tasks])
                else:
                    perf_dist = 0.5
            else:
                perf_dist = 0.5
            
            # Strategic diversity
            strat_dist = abs(candidate.equilibrium_value - other.equilibrium_value)
            
            # Combined distance
            distance = 0.5 * prompt_dist + 0.3 * perf_dist + 0.2 * strat_dist
            distances.append(distance)
        
        # Average distance to population
        avg_distance = np.mean(distances)
        
        if avg_distance > 0.7:
            return 0.9, f"Very diverse (avg dist: {avg_distance:.2f})"
        elif avg_distance > 0.5:
            return 0.7, f"Moderately diverse (avg dist: {avg_distance:.2f})"
        else:
            return avg_distance, f"Low diversity (avg dist: {avg_distance:.2f})"
    
    def _assess_innovation(self, candidate: SystemCandidate) -> Tuple[float, str]:
        """Assess innovation level."""
        innovation_score = 0.0
        reasons = []
        
        # Check for novel prompt structures
        all_prompts = []
        for p in self.population:
            all_prompts.extend([m.prompt for m in p.modules.values()])
        
        for module in candidate.modules.values():
            if not any(self._is_similar_prompt(module.prompt, p) for p in all_prompts):
                innovation_score += 0.3
                reasons.append(f"Novel prompt for {module.name}")
        
        # Check for unusual performance patterns
        if candidate.scores:
            scores = list(candidate.scores.values())
            if len(scores) > 2:
                # High variance might indicate specialized strategy
                variance = np.var(scores)
                if variance > 0.2:
                    innovation_score += 0.2
                    reasons.append("Specialized strategy")
                
                # Unusual success pattern
                success_pattern = [1 if s > 0.7 else 0 for s in scores]
                if sum(success_pattern) > 0 and sum(success_pattern) < len(success_pattern):
                    # Partial success - might be innovative approach
                    innovation_score += 0.1
                    reasons.append("Unique success pattern")
        
        # Check for new module combinations
        if hasattr(candidate, 'mutation_history') and candidate.mutation_history:
            recent_mutations = candidate.mutation_history[-3:]
            if len(set(m.get('module') for m in recent_mutations)) > 2:
                innovation_score += 0.2
                reasons.append("Diverse mutation history")
        
        innovation_score = min(1.0, innovation_score)
        reason = "; ".join(reasons) if reasons else "No innovations detected"
        
        return innovation_score, reason
    
    def _assess_strategic_value(self, candidate: SystemCandidate) -> Tuple[float, str]:
        """Assess strategic value in game-theoretic sense."""
        strategic_score = 0.0
        reasons = []
        
        # High equilibrium value
        if candidate.equilibrium_value > 0.8:
            strategic_score += 0.4
            reasons.append(f"High equilibrium ({candidate.equilibrium_value:.2f})")
        
        # High stability
        if candidate.stability_score > 0.8:
            strategic_score += 0.3
            reasons.append(f"High stability ({candidate.stability_score:.2f})")
        
        # Balanced performance
        if candidate.scores:
            scores = list(candidate.scores.values())
            if len(scores) > 1 and np.std(scores) < 0.1 and np.mean(scores) > 0.6:
                strategic_score += 0.3
                reasons.append("Consistent performance")
        
        # Good lineage
        if self._has_successful_lineage(candidate):
            strategic_score += 0.2
            reasons.append("Successful lineage")
        
        strategic_score = min(1.0, strategic_score)
        reason = "; ".join(reasons) if reasons else "No strategic advantages"
        
        return strategic_score, reason
    
    def _calculate_prompt_distance(
        self, 
        cand1: SystemCandidate, 
        cand2: SystemCandidate
    ) -> float:
        """Calculate distance between candidates based on prompts."""
        distances = []
        
        for name in cand1.modules:
            if name in cand2.modules:
                prompt1 = cand1.modules[name].prompt
                prompt2 = cand2.modules[name].prompt
                
                if prompt1 == prompt2:
                    distances.append(0.0)
                elif sentence_model:
                    # Semantic distance
                    emb1 = sentence_model.encode([prompt1])[0]
                    emb2 = sentence_model.encode([prompt2])[0]
                    
                    # Cosine distance
                    similarity = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )
                    distance = 1.0 - similarity
                    distances.append(distance)
                else:
                    # Fallback to string distance
                    similarity = SequenceMatcher(None, prompt1, prompt2).ratio()
                    distances.append(1.0 - similarity)
        
        return np.mean(distances) if distances else 0.5
    
    def _is_similar_prompt(self, prompt1: str, prompt2: str, threshold: float = 0.9) -> bool:
        """Check if two prompts are similar."""
        if prompt1 == prompt2:
            return True
        
        if sentence_model:
            emb1 = sentence_model.encode([prompt1])[0]
            emb2 = sentence_model.encode([prompt2])[0]
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            return similarity > threshold
        else:
            return SequenceMatcher(None, prompt1, prompt2).ratio() > threshold
    
    def _has_successful_lineage(self, candidate: SystemCandidate) -> bool:
        """Check if candidate comes from successful lineage."""
        if not hasattr(candidate, 'parent_id') or candidate.parent_id is None:
            return False
        
        # Check parent performance
        parent = next((c for c in self.elite_archive 
                      if c.candidate_id == candidate.parent_id), None)
        
        return parent is not None
    
    def _update_archives(self, candidate: SystemCandidate):
        """Update various archives."""
        # Elite archive - best performers
        if candidate.scores:
            avg_score = np.mean(list(candidate.scores.values()))
            
            if avg_score > 0.8:
                self.elite_archive.append(candidate)
                # Keep only best
                if len(self.elite_archive) > self.archive_size:
                    self.elite_archive.sort(
                        key=lambda c: np.mean(list(c.scores.values())), 
                        reverse=True
                    )
                    self.elite_archive = self.elite_archive[:self.archive_size]
        
        # Diversity archive - unique strategies
        if self._is_diverse_strategy(candidate):
            self.diversity_archive.append(candidate)
            if len(self.diversity_archive) > self.archive_size:
                # Keep most diverse
                self._prune_diversity_archive()
        
        # Innovation archive - novel approaches
        innovation_score, _ = self._assess_innovation(candidate)
        if innovation_score > 0.5:
            self.innovation_archive.append(candidate)
            if len(self.innovation_archive) > self.archive_size // 2:
                # Keep most recent innovations
                self.innovation_archive = self.innovation_archive[-(self.archive_size // 2):]
    
    def _is_diverse_strategy(self, candidate: SystemCandidate) -> bool:
        """Check if candidate represents diverse strategy."""
        if not self.diversity_archive:
            return True
        
        # Check distance to existing diverse strategies
        min_distance = float('inf')
        
        for other in self.diversity_archive[-10:]:  # Check recent additions
            distance = self._calculate_prompt_distance(candidate, other)
            min_distance = min(min_distance, distance)
        
        return min_distance > 0.3
    
    def _prune_diversity_archive(self):
        """Prune diversity archive to maintain diversity."""
        if len(self.diversity_archive) <= self.archive_size:
            return
        
        # Use farthest point sampling
        remaining = self.diversity_archive.copy()
        selected = [remaining.pop(0)]  # Start with first
        
        while len(selected) < self.archive_size and remaining:
            # Find candidate farthest from selected
            max_min_dist = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                min_dist = min(
                    self._calculate_prompt_distance(candidate, sel)
                    for sel in selected
                )
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
        
        self.diversity_archive = selected
    
    def _maintain_population_limit(self):
        """Maintain population within size limits using sophisticated selection."""
        if len(self.population) <= self.max_size:
            return
        
        # Multi-objective scoring
        scores = []
        
        for candidate in self.population:
            # Performance component
            perf_score = np.mean(list(candidate.scores.values())) if candidate.scores else 0
            
            # Game-theoretic components
            game_score = 0.5 * candidate.equilibrium_value + 0.5 * candidate.stability_score
            
            # Diversity component
            diversity_score, _ = self._assess_diversity(candidate)
            
            # Age component (prefer some newer candidates)
            age_score = 1.0 / (1.0 + candidate.generation)
            
            # Combined score
            combined = (
                (1 - self.diversity_weight) * perf_score +
                self.diversity_weight * diversity_score +
                0.2 * game_score +
                0.1 * age_score
            )
            
            scores.append(combined)
        
        # Select top candidates
        sorted_indices = np.argsort(scores)[::-1]
        self.population = [self.population[i] for i in sorted_indices[:self.max_size]]
    
    def _update_generation_stats(self, generation: int):
        """Update statistics for current generation."""
        if not self.population:
            return
        
        # Calculate stats
        fitness_values = [np.mean(list(c.scores.values())) for c in self.population 
                         if c.scores]
        
        if fitness_values:
            self.generation_stats[generation]['avg_fitness'] = np.mean(fitness_values)
            self.generation_stats[generation]['best_fitness'] = np.max(fitness_values)
        
        # Calculate diversity
        if len(self.population) > 1:
            diversity = self._calculate_population_diversity()
            self.generation_stats[generation]['diversity'] = diversity
        
        # Count innovations
        innovations = sum(1 for c in self.population 
                         if hasattr(c, 'generation') and c.generation == generation)
        self.generation_stats[generation]['innovations'] = innovations
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of current population."""
        if len(self.population) < 2:
            return 0.0
        
        # Pairwise distances
        distances = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = self._calculate_prompt_distance(
                    self.population[i], 
                    self.population[j]
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def select_parents(self, n: int = 2, method: str = 'tournament') -> List[SystemCandidate]:
        """
        Select parents for reproduction.
        
        Args:
            n: Number of parents to select
            method: Selection method ('tournament', 'roulette', 'diverse')
            
        Returns:
            List of selected parent candidates
        """
        if len(self.population) < n:
            return self.population.copy()
        
        if method == 'tournament':
            return self._tournament_selection(n)
        elif method == 'roulette':
            return self._roulette_selection(n)
        elif method == 'diverse':
            return self._diverse_selection(n)
        else:
            return random.sample(self.population, n)
    
    def _tournament_selection(self, n: int, tournament_size: int = 3) -> List[SystemCandidate]:
        """Tournament selection."""
        selected = []
        
        for _ in range(n):
            tournament = random.sample(self.population, 
                                     min(tournament_size, len(self.population)))
            
            # Select best from tournament
            best = max(tournament, 
                      key=lambda c: np.mean(list(c.scores.values())) if c.scores else 0)
            selected.append(best)
        
        return selected
    
    def _roulette_selection(self, n: int) -> List[SystemCandidate]:
        """Roulette wheel selection."""
        # Calculate fitness values
        fitness_values = []
        for c in self.population:
            if c.scores:
                fitness = np.mean(list(c.scores.values()))
            else:
                fitness = 0.1
            fitness_values.append(fitness)
        
        # Normalize to probabilities
        fitness_array = np.array(fitness_values)
        fitness_array = fitness_array - np.min(fitness_array) + 0.1
        probs = fitness_array / np.sum(fitness_array)
        
        # Select
        indices = np.random.choice(len(self.population), size=n, p=probs, replace=False)
        return [self.population[i] for i in indices]
    
    def _diverse_selection(self, n: int) -> List[SystemCandidate]:
        """Select diverse parents."""
        if n <= 0:
            return []
        
        # Start with best performer
        selected = [max(self.population, 
                       key=lambda c: np.mean(list(c.scores.values())) if c.scores else 0)]
        
        # Add diverse candidates
        while len(selected) < n:
            # Find candidate most different from selected
            best_candidate = None
            best_min_dist = -1
            
            for candidate in self.population:
                if candidate in selected:
                    continue
                
                min_dist = min(
                    self._calculate_prompt_distance(candidate, sel)
                    for sel in selected
                )
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
            else:
                break
        
        return selected
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive population statistics.
        
        Returns:
            Dictionary of population statistics
        """
        stats = {
            'population_size': len(self.population),
            'elite_archive_size': len(self.elite_archive),
            'diversity_archive_size': len(self.diversity_archive),
            'innovation_archive_size': len(self.innovation_archive),
            'generation_history': dict(self.generation_stats)
        }
        
        if self.population:
            # Current population stats
            fitness_values = [np.mean(list(c.scores.values())) for c in self.population 
                            if c.scores]
            
            if fitness_values:
                stats['current_stats'] = {
                    'avg_fitness': np.mean(fitness_values),
                    'best_fitness': np.max(fitness_values),
                    'worst_fitness': np.min(fitness_values),
                    'fitness_std': np.std(fitness_values)
                }
            
            # Diversity metrics
            stats['diversity'] = self._calculate_population_diversity()
            
            # Module type distribution
            module_types = defaultdict(int)
            for candidate in self.population:
                for module in candidate.modules.values():
                    module_types[module.module_type.value] += 1
            stats['module_type_distribution'] = dict(module_types)
        
        return stats