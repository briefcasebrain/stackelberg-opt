"""
Semantic constraint extraction for Stackelberg optimization.

This module extracts constraints from leader modules that follower
modules must satisfy.
"""

import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Any, Set

import numpy as np

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

try:
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except:
    sentence_model = None

from ..core.module import Module
from ..core.candidate import SystemCandidate, ExecutionTrace

logger = logging.getLogger(__name__)


class SemanticConstraintExtractor:
    """
    Extract semantic constraints using NLP and logic.
    
    Analyzes leader module prompts and outputs to extract constraints
    that follower modules should satisfy for optimal performance.
    
    Examples:
        >>> extractor = SemanticConstraintExtractor()
        >>> constraints = extractor.extract_constraints(
        ...     follower_module, candidate
        ... )
    """
    
    def __init__(self):
        self.constraint_patterns = {
            'hard': [
                r'must\s+(\w+)',
                r'always\s+(\w+)',
                r'require[sd]?\s+(\w+)',
                r'mandatory\s+(\w+)'
            ],
            'soft': [
                r'should\s+(\w+)',
                r'prefer\s+(\w+)',
                r'recommend\s+(\w+)',
                r'suggest\s+(\w+)'
            ]
        }
    
    def extract_constraints(
        self, 
        follower_module: Module, 
        candidate: SystemCandidate
    ) -> Dict[str, Any]:
        """
        Extract comprehensive constraints from leader modules.
        
        Args:
            follower_module: Follower module to extract constraints for
            candidate: System candidate containing all modules
            
        Returns:
            Dictionary of extracted constraints
        """
        constraints = {
            'hard_constraints': [],
            'soft_constraints': [],
            'typical_inputs': [],
            'value_ranges': {},
            'semantic_constraints': [],
            'data_flow_constraints': [],
            'performance_constraints': {}
        }
        
        for dep_name in follower_module.dependencies:
            dep_module = candidate.modules.get(dep_name)
            if not dep_module:
                continue
            
            # Extract from prompts
            prompt_constraints = self._extract_from_prompt(dep_module.prompt)
            constraints['hard_constraints'].extend(prompt_constraints['hard'])
            constraints['soft_constraints'].extend(prompt_constraints['soft'])
            
            # Extract from execution traces
            if candidate.traces:
                trace_constraints = self._extract_from_traces(dep_name, candidate.traces)
                constraints['typical_inputs'].extend(trace_constraints['examples'])
                constraints['value_ranges'].update(trace_constraints['ranges'])
                constraints['data_flow_constraints'].extend(trace_constraints['data_flow'])
            
            # Extract semantic constraints
            semantic = self._extract_semantic_constraints(dep_module, follower_module)
            constraints['semantic_constraints'].extend(semantic)
            
            # Performance constraints
            perf = self._extract_performance_constraints(dep_name, candidate)
            constraints['performance_constraints'][dep_name] = perf
        
        return constraints
    
    def _extract_from_prompt(self, prompt: str) -> Dict[str, List[str]]:
        """Extract constraints from prompt using NLP and patterns."""
        constraints = {'hard': [], 'soft': []}
        
        # Pattern matching
        for constraint_type, patterns in self.constraint_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, prompt, re.IGNORECASE)
                for match in matches:
                    constraint_text = f"Must {match}" if constraint_type == 'hard' else f"Should {match}"
                    constraints[constraint_type].append(constraint_text)
        
        # NLP-based extraction if available
        if nlp:
            doc = nlp(prompt[:1000])  # Limit length
            
            # Extract modal verb constructions
            for token in doc:
                if token.dep_ == "aux" and token.text.lower() in ["must", "should", "shall"]:
                    # Get the verb phrase
                    verb = token.head
                    constraint_text = f"{token.text.capitalize()} {verb.text}"
                    
                    # Add objects/modifiers
                    for child in verb.children:
                        if child.dep_ in ["dobj", "attr", "prep"]:
                            constraint_text += f" {child.text}"
                    
                    if token.text.lower() in ["must", "shall"]:
                        constraints['hard'].append(constraint_text)
                    else:
                        constraints['soft'].append(constraint_text)
        
        return constraints
    
    def _extract_from_traces(
        self, 
        module_name: str, 
        traces: Dict[int, ExecutionTrace]
    ) -> Dict[str, Any]:
        """Extract constraints from execution patterns."""
        outputs = []
        timings = []
        data_flows = []
        
        for trace in traces.values():
            if module_name in trace.module_outputs:
                outputs.append(trace.module_outputs[module_name])
                
                if module_name in trace.module_timings:
                    timings.append(trace.module_timings[module_name])
                
                # Track data flow
                if module_name in trace.causal_links:
                    for target in trace.causal_links[module_name]:
                        data_flows.append({
                            'from': module_name,
                            'to': target,
                            'success': trace.intermediate_scores.get(target, 0) > 0.5
                        })
        
        result = {
            'examples': [],
            'ranges': {},
            'data_flow': []
        }
        
        if outputs:
            # Take diverse examples
            result['examples'] = self._select_diverse_examples(outputs, n=5)
            
            # Analyze output characteristics
            if all(isinstance(o, str) for o in outputs):
                lengths = [len(o) for o in outputs]
                result['ranges']['output_type'] = 'string'
                result['ranges']['length_range'] = (min(lengths), max(lengths))
                result['ranges']['avg_length'] = np.mean(lengths)
                
                # Common patterns
                if len(outputs) > 5:
                    # Extract common prefixes/suffixes
                    common_prefix = self._find_common_prefix(outputs[:10])
                    if len(common_prefix) > 3:
                        result['ranges']['common_prefix'] = common_prefix
            
            elif all(isinstance(o, (int, float)) for o in outputs):
                result['ranges']['output_type'] = 'numeric'
                result['ranges']['value_range'] = (min(outputs), max(outputs))
                result['ranges']['mean'] = np.mean(outputs)
                result['ranges']['std'] = np.std(outputs)
            
            elif all(isinstance(o, list) for o in outputs):
                result['ranges']['output_type'] = 'list'
                result['ranges']['length_range'] = (
                    min(len(o) for o in outputs), 
                    max(len(o) for o in outputs)
                )
        
        # Timing constraints
        if timings:
            result['ranges']['timing_range'] = (min(timings), max(timings))
            result['ranges']['avg_timing'] = np.mean(timings)
        
        # Data flow patterns
        if data_flows:
            success_rate = sum(1 for f in data_flows if f['success']) / len(data_flows)
            result['data_flow'] = [{
                'pattern': 'typical_flow',
                'success_rate': success_rate,
                'sample_flows': data_flows[:3]
            }]
        
        return result
    
    def _extract_semantic_constraints(
        self, 
        leader_module: Module, 
        follower_module: Module
    ) -> List[Dict[str, Any]]:
        """Extract semantic relationships between modules."""
        constraints = []
        
        # Check for explicit references
        if leader_module.name.lower() in follower_module.prompt.lower():
            constraints.append({
                'type': 'explicit_dependency',
                'description': f"Follower explicitly references {leader_module.name}",
                'strength': 'strong'
            })
        
        # Check for implicit semantic relationships
        if sentence_model:
            leader_emb = sentence_model.encode([leader_module.prompt])[0]
            follower_emb = sentence_model.encode([follower_module.prompt])[0]
            
            similarity = np.dot(leader_emb, follower_emb) / (
                np.linalg.norm(leader_emb) * np.linalg.norm(follower_emb)
            )
            
            if similarity > 0.7:
                constraints.append({
                    'type': 'semantic_alignment',
                    'description': f"High semantic similarity ({similarity:.2f})",
                    'strength': 'medium'
                })
        
        # Check for format specifications
        format_keywords = ['json', 'xml', 'csv', 'list', 'dict', 'array']
        for keyword in format_keywords:
            if keyword in leader_module.prompt.lower() and keyword in follower_module.prompt.lower():
                constraints.append({
                    'type': 'format_constraint',
                    'description': f"Both modules reference {keyword} format",
                    'strength': 'strong'
                })
        
        return constraints
    
    def _extract_performance_constraints(
        self, 
        module_name: str, 
        candidate: SystemCandidate
    ) -> Dict[str, float]:
        """Extract performance-related constraints."""
        constraints = {
            'min_latency': float('inf'),
            'max_latency': 0.0,
            'avg_latency': 0.0,
            'success_threshold': 0.0
        }
        
        latencies = []
        successes = []
        
        for trace in candidate.traces.values():
            if module_name in trace.module_timings:
                latency = trace.module_timings[module_name]
                latencies.append(latency)
                constraints['min_latency'] = min(constraints['min_latency'], latency)
                constraints['max_latency'] = max(constraints['max_latency'], latency)
            
            if module_name in trace.intermediate_scores:
                successes.append(trace.intermediate_scores[module_name])
        
        if latencies:
            constraints['avg_latency'] = np.mean(latencies)
        
        if successes:
            constraints['success_threshold'] = np.percentile(successes, 25)  # 25th percentile
        
        return constraints
    
    def _select_diverse_examples(self, outputs: List[Any], n: int = 5) -> List[Any]:
        """Select diverse examples from outputs."""
        if len(outputs) <= n:
            return outputs
        
        # For strings, use edit distance
        if all(isinstance(o, str) for o in outputs):
            selected = [outputs[0]]
            
            for _ in range(n - 1):
                max_min_dist = -1
                best_candidate = None
                
                for candidate in outputs:
                    if candidate in selected:
                        continue
                    
                    # Find minimum distance to selected
                    min_dist = float('inf')
                    for sel in selected:
                        dist = 1 - SequenceMatcher(None, candidate, sel).ratio()
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
            
            return selected
        
        # For other types, use random sampling
        import random
        return random.sample(outputs, n)
    
    def _find_common_prefix(self, strings: List[str]) -> str:
        """Find common prefix of strings."""
        if not strings:
            return ""
        
        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        
        return prefix