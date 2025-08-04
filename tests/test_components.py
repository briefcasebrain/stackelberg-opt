"""Tests for optimization components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from stackelberg_opt import Module, ModuleType, SystemCandidate, ExecutionTrace
from stackelberg_opt.components import (
    PromptMutator,
    CompoundSystemEvaluator,
    StackelbergFeedbackExtractor,
    StackelbergEquilibriumCalculator,
    StabilityCalculator,
    SemanticConstraintExtractor,
    DependencyAnalyzer,
    PopulationManager
)


class TestPromptMutator:
    """Tests for PromptMutator."""
    
    def test_initialization(self):
        """Test mutator initialization."""
        mutator = PromptMutator(model="gpt-3.5-turbo", temperature=0.7)
        assert mutator.model == "gpt-3.5-turbo"
        assert mutator.temperature == 0.7
        assert mutator.cache is not None
    
    def test_fallback_mutation(self):
        """Test fallback mutation when LLM fails."""
        mutator = PromptMutator()
        
        module = Module("test", "Original prompt", ModuleType.LEADER)
        feedback = {
            'avg_score': 0.2,
            'stability': 0.3,
            'failure_patterns': ['error in output']
        }
        
        new_prompt = mutator._fallback_mutation(module, feedback)
        
        assert "Original prompt" in new_prompt
        assert "Additional guidelines" in new_prompt
        assert len(new_prompt) > len(module.prompt)
    
    @patch('litellm.completion')
    def test_successful_mutation(self, mock_completion):
        """Test successful LLM mutation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Improved prompt"))]
        mock_completion.return_value = mock_response
        
        mutator = PromptMutator(cache_enabled=False)
        module = Module("test", "Original prompt", ModuleType.LEADER)
        parent = SystemCandidate(modules={"test": module}, candidate_id=1)
        feedback = {'avg_score': 0.5}
        
        new_prompt = mutator.mutate_prompt(module, parent, feedback)
        
        assert new_prompt == "Improved prompt"
        mock_completion.assert_called_once()


class TestCompoundSystemEvaluator:
    """Tests for CompoundSystemEvaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        task_executor = Mock()
        evaluator = CompoundSystemEvaluator(task_executor, timeout=30.0)
        
        assert evaluator.task_executor == task_executor
        assert evaluator.timeout == 30.0
        assert evaluator.max_parallel == 4
    
    def test_score_calculation_strings(self):
        """Test score calculation for string outputs."""
        evaluator = CompoundSystemEvaluator(Mock())
        trace = ExecutionTrace()
        
        score = evaluator._calculate_score("hello world", "hello world", trace)
        assert score == 1.0  # Exact match
        
        score = evaluator._calculate_score("hello", "hello world", trace)
        assert 0 < score < 1  # Partial match
        
        score = evaluator._calculate_score("goodbye", "hello world", trace)
        assert score < 0.5  # Poor match
    
    def test_score_calculation_dicts(self):
        """Test score calculation for dictionary outputs."""
        evaluator = CompoundSystemEvaluator(Mock())
        trace = ExecutionTrace()
        
        output = {"a": 1, "b": 2}
        expected = {"a": 1, "b": 2}
        score = evaluator._calculate_score(output, expected, trace)
        assert score == 1.0
        
        output = {"a": 1, "b": 3}
        score = evaluator._calculate_score(output, expected, trace)
        assert score == 0.5  # 1 of 2 keys correct
    
    def test_module_score_attribution(self):
        """Test attribution of scores to individual modules."""
        evaluator = CompoundSystemEvaluator(Mock())
        
        modules = {
            "leader": Module("leader", "prompt", ModuleType.LEADER),
            "follower": Module("follower", "prompt", ModuleType.FOLLOWER)
        }
        
        trace = ExecutionTrace(success=True)
        trace.module_timings = {"leader": 0.5, "follower": 0.5}
        
        evaluator._attribute_module_scores(trace, modules, 1.0)
        
        assert "leader" in trace.intermediate_scores
        assert "follower" in trace.intermediate_scores
        assert trace.intermediate_scores["leader"] > trace.intermediate_scores["follower"]


class TestStackelbergFeedbackExtractor:
    """Tests for StackelbergFeedbackExtractor."""
    
    def test_basic_feedback_extraction(self):
        """Test basic feedback extraction."""
        extractor = StackelbergFeedbackExtractor()
        
        module = Module("test", "prompt", ModuleType.LEADER)
        traces = [
            ExecutionTrace(
                intermediate_scores={"test": 0.8},
                final_score=0.9,
                success=True
            ),
            ExecutionTrace(
                intermediate_scores={"test": 0.7},
                final_score=0.8,
                success=True
            )
        ]
        
        feedback = extractor.extract_feedback("test", traces, module)
        
        assert feedback['module_name'] == "test"
        assert feedback['avg_score'] == 0.75
        assert feedback['success_rate'] == 1.0
        assert feedback['error_rate'] == 0.0
    
    def test_failure_pattern_extraction(self):
        """Test extraction of failure patterns."""
        extractor = StackelbergFeedbackExtractor()
        
        module = Module("test", "prompt", ModuleType.LEADER)
        traces = [
            ExecutionTrace(
                final_score=0.2,
                module_outputs={"test": ""},
                intermediate_scores={"test": 0.1}
            ),
            ExecutionTrace(
                final_score=0.1,
                module_outputs={"test": "very short"},
                intermediate_scores={"test": 0.2}
            )
        ]
        
        feedback = extractor.extract_feedback("test", traces, module)
        
        assert len(feedback['failure_patterns']) > 0
        assert any("short output" in p for p in feedback['failure_patterns'])


class TestStackelbergEquilibriumCalculator:
    """Tests for StackelbergEquilibriumCalculator."""
    
    def test_no_hierarchy_case(self):
        """Test equilibrium calculation with no leader-follower structure."""
        calculator = StackelbergEquilibriumCalculator()
        
        modules = {
            "a": Module("a", "prompt", ModuleType.INDEPENDENT),
            "b": Module("b", "prompt", ModuleType.INDEPENDENT)
        }
        candidate = SystemCandidate(modules=modules, candidate_id=1)
        candidate.scores = {0: 0.8, 1: 0.9}
        candidate.traces = {0: ExecutionTrace()}
        
        equilibrium, info = calculator.calculate_equilibrium(candidate)
        
        assert equilibrium == 0.85  # Average of scores
        assert info['method'] == 'no_hierarchy'
        assert info['converged'] == True
    
    def test_with_hierarchy(self):
        """Test equilibrium calculation with leader-follower structure."""
        calculator = StackelbergEquilibriumCalculator()
        
        modules = {
            "leader": Module("leader", "prompt", ModuleType.LEADER),
            "follower": Module("follower", "prompt", ModuleType.FOLLOWER)
        }
        candidate = SystemCandidate(modules=modules, candidate_id=1)
        
        # Create traces with scores
        traces = {}
        for i in range(5):
            trace = ExecutionTrace()
            trace.intermediate_scores = {
                "leader": np.random.rand(),
                "follower": np.random.rand()
            }
            traces[i] = trace
        
        candidate.traces = traces
        
        equilibrium, info = calculator.calculate_equilibrium(candidate)
        
        assert isinstance(equilibrium, float)
        assert 0 <= equilibrium <= 1
        assert 'method' in info
        assert 'converged' in info


class TestStabilityCalculator:
    """Tests for StabilityCalculator."""
    
    def test_performance_stability(self):
        """Test performance stability calculation."""
        calculator = StabilityCalculator()
        
        candidate = SystemCandidate(modules={}, candidate_id=1)
        candidate.scores = {0: 0.8, 1: 0.8, 2: 0.8}  # Consistent scores
        candidate.traces = {0: ExecutionTrace()}
        
        stability = calculator._calculate_performance_stability(candidate)
        assert stability > 0.8  # High stability for consistent scores
        
        # Test with variable scores
        candidate.scores = {0: 0.2, 1: 0.9, 2: 0.5}
        stability = calculator._calculate_performance_stability(candidate)
        assert stability < 0.5  # Low stability for variable scores
    
    def test_robustness_calculation(self):
        """Test robustness calculation."""
        calculator = StabilityCalculator()
        
        # All successful traces
        traces = {
            0: ExecutionTrace(success=True),
            1: ExecutionTrace(success=True),
            2: ExecutionTrace(success=True)
        }
        candidate = SystemCandidate(modules={}, candidate_id=1)
        candidate.traces = traces
        
        robustness = calculator._calculate_robustness(candidate)
        assert robustness > 0.9  # High robustness for all successes
        
        # Mix of success and failure
        traces[1].success = False
        robustness = calculator._calculate_robustness(candidate)
        assert 0.4 < robustness < 0.8  # Medium robustness


class TestPopulationManager:
    """Tests for PopulationManager."""
    
    def test_initialization(self):
        """Test population manager initialization."""
        manager = PopulationManager(max_size=20, diversity_weight=0.3)
        
        assert manager.max_size == 20
        assert manager.diversity_weight == 0.3
        assert len(manager.population) == 0
        assert len(manager.elite_archive) == 0
    
    def test_add_candidate(self):
        """Test adding candidates to population."""
        manager = PopulationManager(max_size=5)
        
        candidate = SystemCandidate(modules={}, candidate_id=1)
        candidate.scores = {0: 0.9}
        
        added, reason = manager.add_candidate(candidate, generation=1)
        
        assert added == True
        assert len(manager.population) == 1
        assert "below minimum size" in reason.lower()
    
    def test_parent_selection(self):
        """Test parent selection methods."""
        manager = PopulationManager()
        
        # Add some candidates
        for i in range(5):
            candidate = SystemCandidate(modules={}, candidate_id=i)
            candidate.scores = {0: np.random.rand()}
            manager.population.append(candidate)
        
        # Test tournament selection
        parents = manager.select_parents(n=2, method='tournament')
        assert len(parents) == 2
        assert all(p in manager.population for p in parents)
        
        # Test roulette selection
        parents = manager.select_parents(n=2, method='roulette')
        assert len(parents) == 2
        
        # Test diverse selection
        parents = manager.select_parents(n=2, method='diverse')
        assert len(parents) == 2


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""
    
    def test_explicit_dependencies(self):
        """Test analysis of explicit dependencies."""
        analyzer = DependencyAnalyzer()
        
        modules = {
            "a": Module("a", "prompt", ModuleType.LEADER, dependencies=[]),
            "b": Module("b", "prompt", ModuleType.FOLLOWER, dependencies=["a"]),
            "c": Module("c", "prompt", ModuleType.FOLLOWER, dependencies=["b"])
        }
        
        analysis = analyzer.analyze_dependencies(modules, prompts_only=True)
        
        assert analysis['dependencies']['a'] == []
        assert analysis['dependencies']['b'] == ['a']
        assert analysis['dependencies']['c'] == ['b']
        assert analysis['properties']['is_dag'] == True
        assert analysis['module_roles']['a'] == ModuleType.LEADER
        assert analysis['module_roles']['b'] == ModuleType.FOLLOWER


class TestSemanticConstraintExtractor:
    """Tests for SemanticConstraintExtractor."""
    
    def test_prompt_constraint_extraction(self):
        """Test extraction of constraints from prompts."""
        extractor = SemanticConstraintExtractor()
        
        prompt = "You must provide accurate results. Always validate input."
        constraints = extractor._extract_from_prompt(prompt)
        
        assert len(constraints['hard']) > 0
        assert any("must" in c.lower() for c in constraints['hard'])
    
    def test_trace_constraint_extraction(self):
        """Test extraction of constraints from traces."""
        extractor = SemanticConstraintExtractor()
        
        traces = {
            0: ExecutionTrace(
                module_outputs={"leader": "output1"},
                module_timings={"leader": 0.5}
            ),
            1: ExecutionTrace(
                module_outputs={"leader": "output2"},
                module_timings={"leader": 0.6}
            )
        }
        
        constraints = extractor._extract_from_traces("leader", traces)
        
        assert len(constraints['examples']) > 0
        assert 'output_type' in constraints['ranges']
        assert 'timing_range' in constraints['ranges']