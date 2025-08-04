"""
Feedback extraction component for Stackelberg optimization.

This module analyzes execution traces to extract actionable feedback
for prompt improvement.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional

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

from ..core.module import Module, ModuleType
from ..core.candidate import ExecutionTrace, SystemCandidate

logger = logging.getLogger(__name__)


class StackelbergFeedbackExtractor:
    """
    Comprehensive feedback extraction with NLP and pattern mining.
    
    Analyzes execution traces to identify failure patterns, success patterns,
    and module-specific insights for improving prompts.
    
    Examples:
        >>> extractor = StackelbergFeedbackExtractor()
        >>> feedback = extractor.extract_feedback(
        ...     module_name="query_generator",
        ...     traces=execution_traces,
        ...     module=module,
        ...     population=population
        ... )
    """
    
    def __init__(self):
        self.pattern_cache = defaultdict(list)
        self.failure_patterns = defaultdict(lambda: defaultdict(int))
        self.success_patterns = defaultdict(lambda: defaultdict(int))
    
    def extract_feedback(
        self, 
        module_name: str, 
        traces: List[ExecutionTrace], 
        module: Module, 
        population: Optional[List[SystemCandidate]] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive feedback with pattern analysis.
        
        Args:
            module_name: Name of the module to analyze
            traces: List of execution traces
            module: The module instance
            population: Current population for comparative analysis
            
        Returns:
            Dictionary containing comprehensive feedback
        """
        feedback = {
            'module_name': module_name,
            'module_type': module.module_type.value,
            'iteration': len(traces),
            'avg_score': 0.0,
            'stability': 0.0,
            'success_rate': 0.0,
            'error_rate': 0.0,
            'failure_patterns': [],
            'success_patterns': [],
            'module_feedback': "",
            'adaptation_score': 0.0,
            'constraint_satisfaction': 0.0,
            'downstream_impact': {},
            'module_interactions': {},
            'improvement_suggestions': []
        }
        
        if not traces:
            return feedback
        
        # Basic metrics
        module_scores = []
        successes = 0
        errors = 0
        
        for trace in traces:
            if module_name in trace.intermediate_scores:
                module_scores.append(trace.intermediate_scores[module_name])
            
            if trace.success and trace.final_score > 0.5:
                successes += 1
            
            if module_name in trace.error_messages:
                errors += 1
        
        feedback['avg_score'] = np.mean(module_scores) if module_scores else 0.0
        feedback['success_rate'] = successes / len(traces) if traces else 0.0
        feedback['error_rate'] = errors / len(traces) if traces else 0.0
        
        # Stability analysis
        if len(module_scores) > 1:
            feedback['stability'] = 1.0 / (1.0 + np.std(module_scores))
        
        # Pattern extraction
        feedback['failure_patterns'] = self._extract_failure_patterns(module_name, traces)
        feedback['success_patterns'] = self._extract_success_patterns(module_name, traces)
        
        # Module-specific analysis
        if module.module_type == ModuleType.LEADER:
            feedback.update(self._analyze_leader_performance(module_name, traces))
        elif module.module_type == ModuleType.FOLLOWER:
            feedback.update(self._analyze_follower_performance(module_name, traces, module))
        
        # Downstream impact
        feedback['downstream_impact'] = self._analyze_downstream_impact(module_name, traces)
        
        # Module interactions
        feedback['module_interactions'] = self._analyze_module_interactions(module_name, traces)
        
        # Generate improvement suggestions
        feedback['improvement_suggestions'] = self._generate_suggestions(feedback)
        
        # Create narrative feedback
        feedback['module_feedback'] = self._generate_narrative_feedback(feedback)
        
        return feedback
    
    def _extract_failure_patterns(self, module_name: str, traces: List[ExecutionTrace]) -> List[str]:
        """Extract and rank failure patterns."""
        patterns = defaultdict(int)
        
        for trace in traces:
            if trace.final_score < 0.5 or module_name in trace.error_messages:
                # Analyze error messages
                if module_name in trace.error_messages:
                    error = trace.error_messages[module_name]
                    # Extract key phrases
                    key_phrases = self._extract_key_phrases(error)
                    for phrase in key_phrases:
                        patterns[phrase] += 1
                
                # Analyze problematic outputs
                if module_name in trace.module_outputs and trace.final_score < 0.3:
                    output = str(trace.module_outputs[module_name])
                    if len(output) < 10:
                        patterns["Empty or very short output"] += 1
                    elif len(output) > 1000:
                        patterns["Excessively long output"] += 1
                    
                    # Check for common issues
                    if "error" in output.lower():
                        patterns["Output contains error indicators"] += 1
                    if output.count("\n") > 20:
                        patterns["Excessive line breaks in output"] += 1
        
        # Return top patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return [f"{pattern} (occurred {count} times)" for pattern, count in sorted_patterns[:5]]
    
    def _extract_success_patterns(self, module_name: str, traces: List[ExecutionTrace]) -> List[str]:
        """Extract patterns from successful executions."""
        patterns = defaultdict(int)
        
        for trace in traces:
            if trace.final_score > 0.8 and module_name in trace.module_outputs:
                output = str(trace.module_outputs[module_name])
                
                # Analyze successful outputs
                if 50 < len(output) < 500:
                    patterns["Moderate length output"] += 1
                
                # Check for structure
                if output.count("\n") > 0 and output.count("\n") < 10:
                    patterns["Well-structured output with line breaks"] += 1
                
                # Module-specific patterns
                if module_name in trace.module_timings:
                    exec_time = trace.module_timings[module_name]
                    if exec_time < 1.0:
                        patterns["Fast execution (<1s)"] += 1
        
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        return [f"{pattern} (in {count} successful runs)" for pattern, count in sorted_patterns[:3]]
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text using NLP."""
        if not nlp:
            # Fallback to simple extraction
            words = text.lower().split()
            return [' '.join(words[i:i+3]) for i in range(0, len(words)-2, 2)][:5]
        
        doc = nlp(text[:1000])  # Limit length
        phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            phrases.append(chunk.text.lower())
        
        # Extract verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                phrase = token.text
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        phrase += " " + child.text
                phrases.append(phrase.lower())
        
        return phrases[:5]
    
    def _analyze_leader_performance(self, module_name: str, traces: List[ExecutionTrace]) -> Dict[str, Any]:
        """Specific analysis for leader modules."""
        analysis = {
            'consistency_score': 0.0,
            'guidance_quality': 0.0,
            'downstream_success_correlation': 0.0
        }
        
        # Analyze output consistency
        outputs = []
        for trace in traces:
            if module_name in trace.module_outputs:
                outputs.append(str(trace.module_outputs[module_name]))
        
        if len(outputs) > 1:
            # Calculate semantic similarity if sentence model available
            if sentence_model:
                embeddings = sentence_model.encode(outputs[:10])  # Limit for performance
                similarities = []
                for i in range(len(embeddings)-1):
                    sim = np.dot(embeddings[i], embeddings[i+1]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
                    )
                    similarities.append(sim)
                analysis['consistency_score'] = np.mean(similarities) if similarities else 0.5
            else:
                # Fallback to simple string similarity
                unique_outputs = len(set(outputs))
                analysis['consistency_score'] = 1.0 - (unique_outputs / len(outputs))
        
        # Analyze downstream impact
        downstream_scores = []
        for trace in traces:
            if module_name in trace.causal_links:
                affected_modules = trace.causal_links[module_name]
                affected_scores = [trace.intermediate_scores.get(m, 0) for m in affected_modules]
                if affected_scores:
                    downstream_scores.append(np.mean(affected_scores))
        
        analysis['guidance_quality'] = np.mean(downstream_scores) if downstream_scores else 0.5
        
        # Correlation with final success
        if len(outputs) > 2:
            final_scores = [trace.final_score for trace in traces if module_name in trace.module_outputs]
            output_lengths = [len(o) for o in outputs]
            if len(final_scores) == len(output_lengths):
                correlation = np.corrcoef(output_lengths, final_scores)[0, 1]
                analysis['downstream_success_correlation'] = abs(correlation)
        
        return analysis
    
    def _analyze_follower_performance(
        self, 
        module_name: str, 
        traces: List[ExecutionTrace], 
        module: Module
    ) -> Dict[str, Any]:
        """Specific analysis for follower modules."""
        analysis = {
            'adaptation_score': 0.0,
            'constraint_satisfaction': 0.0,
            'leader_dependency_strength': {}
        }
        
        # Analyze adaptation to different leader outputs
        leader_output_performance = defaultdict(list)
        
        for trace in traces:
            # Get leader outputs for this trace
            leader_outputs = {}
            for dep in module.dependencies:
                if dep in trace.module_outputs:
                    leader_outputs[dep] = str(trace.module_outputs[dep])[:100]  # Truncate for grouping
            
            if leader_outputs and module_name in trace.intermediate_scores:
                key = json.dumps(leader_outputs, sort_keys=True)
                score = trace.intermediate_scores[module_name]
                leader_output_performance[key].append(score)
        
        # Calculate adaptation score (low variance across different leader outputs)
        if leader_output_performance:
            variances = []
            for scores in leader_output_performance.values():
                if len(scores) > 1:
                    variances.append(np.var(scores))
            
            if variances:
                avg_variance = np.mean(variances)
                analysis['adaptation_score'] = 1.0 / (1.0 + avg_variance)
            else:
                analysis['adaptation_score'] = 0.5
        
        # Analyze dependency strength
        for dep in module.dependencies:
            dep_present_scores = []
            dep_absent_scores = []
            
            for trace in traces:
                if module_name in trace.intermediate_scores:
                    score = trace.intermediate_scores[module_name]
                    if dep in trace.module_outputs:
                        dep_present_scores.append(score)
                    else:
                        dep_absent_scores.append(score)
            
            if dep_present_scores and dep_absent_scores:
                strength = np.mean(dep_present_scores) - np.mean(dep_absent_scores)
                analysis['leader_dependency_strength'][dep] = strength
        
        return analysis
    
    def _analyze_downstream_impact(self, module_name: str, traces: List[ExecutionTrace]) -> Dict[str, Any]:
        """Analyze impact on downstream modules."""
        impact = {
            'affected_modules': [],
            'average_impact': 0.0,
            'critical_paths': []
        }
        
        affected_scores = defaultdict(list)
        
        for trace in traces:
            if module_name in trace.causal_links:
                for affected in trace.causal_links[module_name]:
                    if affected in trace.intermediate_scores:
                        affected_scores[affected].append(trace.intermediate_scores[affected])
        
        if affected_scores:
            impact['affected_modules'] = list(affected_scores.keys())
            all_impacts = []
            for module, scores in affected_scores.items():
                all_impacts.extend(scores)
            impact['average_impact'] = np.mean(all_impacts)
        
        return impact
    
    def _analyze_module_interactions(self, module_name: str, traces: List[ExecutionTrace]) -> Dict[str, Any]:
        """Analyze interactions with other modules."""
        interactions = {}
        
        # Find modules that frequently execute together
        co_execution = defaultdict(int)
        
        for trace in traces:
            if module_name in trace.execution_order:
                idx = trace.execution_order.index(module_name)
                
                # Check previous module
                if idx > 0:
                    prev_module = trace.execution_order[idx - 1]
                    co_execution[f"{prev_module}->{module_name}"] += 1
                
                # Check next module
                if idx < len(trace.execution_order) - 1:
                    next_module = trace.execution_order[idx + 1]
                    co_execution[f"{module_name}->{next_module}"] += 1
        
        # Calculate interaction scores
        for interaction, count in co_execution.items():
            interactions[interaction] = {
                'frequency': count / len(traces),
                'impact_score': 0.5  # Would need more analysis for actual impact
            }
        
        return interactions
    
    def _generate_suggestions(self, feedback: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Performance-based suggestions
        if feedback['avg_score'] < 0.3:
            suggestions.append("Major revision needed - consider restructuring the prompt entirely")
        elif feedback['avg_score'] < 0.6:
            suggestions.append("Add more specific instructions and examples to the prompt")
        
        # Stability-based suggestions
        if feedback['stability'] < 0.5:
            suggestions.append("Improve consistency by adding explicit handling for edge cases")
        
        # Error-based suggestions
        if feedback['error_rate'] > 0.2:
            suggestions.append("Add error handling instructions to the prompt")
        
        # Module-type specific suggestions
        if feedback['module_type'] == 'leader':
            if feedback.get('consistency_score', 0) < 0.5:
                suggestions.append("Standardize output format for better downstream processing")
            if feedback.get('guidance_quality', 0) < 0.5:
                suggestions.append("Provide more explicit guidance for follower modules")
        
        elif feedback['module_type'] == 'follower':
            if feedback.get('adaptation_score', 0) < 0.5:
                suggestions.append("Add strategies to handle variations in leader outputs")
            if feedback.get('constraint_satisfaction', 0) < 0.5:
                suggestions.append("Better incorporate leader constraints into processing")
        
        # Pattern-based suggestions
        if "Empty or very short output" in ' '.join(feedback.get('failure_patterns', [])):
            suggestions.append("Ensure the prompt encourages complete, detailed responses")
        
        if "Excessively long output" in ' '.join(feedback.get('failure_patterns', [])):
            suggestions.append("Add length constraints or summarization instructions")
        
        return suggestions
    
    def _generate_narrative_feedback(self, feedback: Dict[str, Any]) -> str:
        """Generate human-readable narrative feedback."""
        parts = []
        
        # Overall performance
        score_desc = (
            "excellent" if feedback['avg_score'] > 0.8 else
            "good" if feedback['avg_score'] > 0.6 else
            "poor"
        )
        parts.append(f"Module shows {score_desc} performance (avg: {feedback['avg_score']:.2f})")
        
        # Stability
        if feedback['stability'] < 0.5:
            parts.append("High variability in performance suggests need for more robust prompt")
        
        # Module-specific insights
        if feedback['module_type'] == 'leader':
            if feedback.get('downstream_impact', {}).get('average_impact', 0) < 0.5:
                parts.append("Leader module negatively impacts downstream performance")
        elif feedback['module_type'] == 'follower':
            if feedback.get('adaptation_score', 0) < 0.5:
                parts.append("Follower struggles to adapt to leader variations")
        
        # Key issues
        if feedback['failure_patterns']:
            parts.append(f"Main issue: {feedback['failure_patterns'][0]}")
        
        # Positive aspects
        if feedback['success_patterns']:
            parts.append(f"Strength: {feedback['success_patterns'][0]}")
        
        return "; ".join(parts)