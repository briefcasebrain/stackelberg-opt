"""Tests for utility modules."""

import pytest
import numpy as np
import tempfile
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch

from stackelberg_opt import SystemCandidate
from stackelberg_opt.utils import (
    ResponseCache,
    ComputationCache,
    CheckpointManager,
    AutoCheckpointer,
    OptimizationVisualizer
)


class TestResponseCache:
    """Tests for ResponseCache."""
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))
            
            # Test set and get
            cache.set("prompt1", "model1", 0.7, "response1")
            assert cache.get("prompt1", "model1", 0.7) == "response1"
            assert cache.get("prompt1", "model1", 0.8) is None  # Different temperature
            assert cache.get("prompt2", "model1", 0.7) is None  # Different prompt
            
            # Test persistence
            cache2 = ResponseCache(cache_dir=Path(tmpdir))
            assert cache2.get("prompt1", "model1", 0.7) == "response1"
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = ResponseCache()
        
        key1 = cache._get_cache_key("prompt", "model", 0.7)
        key2 = cache._get_cache_key("prompt", "model", 0.7)
        key3 = cache._get_cache_key("different", "model", 0.7)
        
        assert key1 == key2  # Same inputs
        assert key1 != key3  # Different prompt
    
    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))
            
            cache.set("prompt1", "model", 0.7, "response1")
            cache.set("prompt2", "model", 0.7, "response2")
            
            stats = cache.get_stats()
            assert stats['total_entries'] == 2
            assert stats['cache_size_bytes'] > 0


class TestComputationCache:
    """Tests for ComputationCache."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = ComputationCache(max_size=3)
        
        # Test make_key
        key1 = cache.make_key("operation", param1=1, param2="test")
        key2 = cache.make_key("operation", param2="test", param1=1)  # Order shouldn't matter
        assert key1 == key2
        
        # Test set and get
        cache.set(key1, "result1")
        assert cache.get(key1) == "result1"
        assert cache.get("nonexistent") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = ComputationCache(max_size=2)
        
        key1 = cache.make_key("op", id=1)
        key2 = cache.make_key("op", id=2)
        key3 = cache.make_key("op", id=3)
        
        cache.set(key1, "result1")
        cache.set(key2, "result2")
        
        # Access key1 to make it more recent
        cache.get(key1)
        
        # Add key3, should evict key2 (least recently used)
        cache.set(key3, "result3")
        
        assert cache.get(key1) == "result1"  # Still present
        assert cache.get(key2) is None  # Evicted
        assert cache.get(key3) == "result3"  # New entry
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ComputationCache()
        
        key = cache.make_key("test", id=1)
        cache.set(key, "result")
        cache.get(key)
        cache.get(key)
        
        stats = cache.get_stats()
        assert stats['total_entries'] == 1
        assert stats['total_accesses'] == 3  # 1 set + 2 gets


class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))
            
            # Create test state
            state = {
                'generation': 10,
                'evaluations_used': 100,
                'best_score': 0.95
            }
            
            # Save checkpoint
            manager.save_checkpoint(state, "test_checkpoint")
            
            # Load checkpoint
            loaded_state = manager.load_checkpoint("test_checkpoint")
            
            assert loaded_state['generation'] == 10
            assert loaded_state['evaluations_used'] == 100
            assert loaded_state['best_score'] == 0.95
    
    def test_checkpoint_summary(self):
        """Test checkpoint summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))
            
            # Create state with candidate
            candidate = SystemCandidate(modules={}, candidate_id=1)
            candidate.scores = {0: 0.9, 1: 0.8}
            candidate.equilibrium_value = 0.85
            candidate.stability_score = 0.7
            
            state = {
                'generation': 5,
                'evaluations_used': 50,
                'population': [candidate],
                'best_candidate': candidate
            }
            
            manager.save_checkpoint(state, "test")
            
            # Check summary file
            summary_file = Path(tmpdir) / "test_summary.json"
            assert summary_file.exists()
            
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            assert summary['generation'] == 5
            assert summary['best_candidate']['equilibrium'] == 0.85
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))
            
            # Create multiple checkpoints
            manager.save_checkpoint({'data': 1}, "checkpoint1")
            manager.save_checkpoint({'data': 2}, "checkpoint2")
            
            checkpoints = manager.list_checkpoints()
            
            assert len(checkpoints) == 2
            assert any(c['name'] == 'checkpoint1' for c in checkpoints)
            assert any(c['name'] == 'checkpoint2' for c in checkpoints)
    
    def test_delete_checkpoint(self):
        """Test checkpoint deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))
            
            manager.save_checkpoint({'data': 1}, "test")
            assert len(manager.list_checkpoints()) == 1
            
            manager.delete_checkpoint("test")
            assert len(manager.list_checkpoints()) == 0


class TestAutoCheckpointer:
    """Tests for AutoCheckpointer."""
    
    def test_time_based_checkpointing(self):
        """Test time-based checkpoint triggering."""
        manager = Mock()
        auto_cp = AutoCheckpointer(manager, time_interval=1.0)  # 1 second
        
        # Should checkpoint immediately (first time)
        assert auto_cp.should_checkpoint(iteration=1) == True
        
        # Should not checkpoint immediately after
        assert auto_cp.should_checkpoint(iteration=2) == False
        
        # Wait and check again
        import time
        time.sleep(1.1)
        assert auto_cp.should_checkpoint(iteration=3) == True
    
    def test_iteration_based_checkpointing(self):
        """Test iteration-based checkpoint triggering."""
        manager = Mock()
        auto_cp = AutoCheckpointer(manager, iteration_interval=10)
        
        assert auto_cp.should_checkpoint(iteration=10) == True
        assert auto_cp.should_checkpoint(iteration=15) == False
        assert auto_cp.should_checkpoint(iteration=20) == True
    
    def test_checkpoint_saving(self):
        """Test automatic checkpoint saving."""
        manager = Mock()
        auto_cp = AutoCheckpointer(manager, iteration_interval=10)
        
        state = {'test': 'data'}
        auto_cp.checkpoint(state, iteration=10)
        
        manager.save_checkpoint.assert_called_once()
        args = manager.save_checkpoint.call_args[0]
        assert args[0] == state
        assert 'auto_10_' in args[1]


class TestOptimizationVisualizer:
    """Tests for OptimizationVisualizer."""
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_creation(self, mock_savefig):
        """Test that plots are created without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = OptimizationVisualizer(save_dir=Path(tmpdir))
            
            # Create mock optimizer with population manager
            optimizer = Mock()
            optimizer.population_manager = Mock()
            optimizer.population_manager.generation_stats = {
                1: {'avg_fitness': 0.5, 'best_fitness': 0.6, 'diversity': 0.3, 'innovations': 2},
                2: {'avg_fitness': 0.6, 'best_fitness': 0.7, 'diversity': 0.35, 'innovations': 1}
            }
            optimizer.population_manager.elite_archive = []
            optimizer.population_manager.diversity_archive = []
            optimizer.population_manager.innovation_archive = []
            optimizer.population = []
            
            # Test optimization progress plot
            visualizer.plot_optimization_progress(optimizer, save=True)
            
            # Verify plot was saved
            assert mock_savefig.called
    
    @patch('matplotlib.pyplot.savefig')
    def test_evolution_tree_plot(self, mock_savefig):
        """Test evolution tree plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = OptimizationVisualizer(save_dir=Path(tmpdir))
            
            # Create population with parent-child relationships
            parent = SystemCandidate(modules={}, candidate_id=1, generation=0)
            parent.scores = {0: 0.7}
            
            child = SystemCandidate(modules={}, candidate_id=2, generation=1, parent_id=1)
            child.scores = {0: 0.8}
            
            population = [parent, child]
            
            visualizer.plot_evolution_tree(population, save=True)
            
            assert mock_savefig.called