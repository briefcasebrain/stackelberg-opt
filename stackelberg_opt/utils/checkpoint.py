"""
Checkpoint management for Stackelberg optimization.

This module provides functionality to save and restore optimization
state for long-running optimization processes.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Handle saving and loading optimization state.
    
    Saves complete optimizer state including population, archives,
    and optimization history. Supports both pickle format for
    complete state and JSON format for human-readable summaries.
    
    Attributes:
        checkpoint_dir: Directory for storing checkpoints
        
    Examples:
        >>> manager = CheckpointManager(Path("checkpoints"))
        >>> manager.save_checkpoint(optimizer_state, "generation_100")
        >>> restored_state = manager.load_checkpoint("generation_100")
    """
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, state: Dict[str, Any], name: Optional[str] = None):
        """
        Save complete optimizer state.
        
        Args:
            state: Dictionary containing optimizer state
            name: Checkpoint name (defaults to timestamp)
        """
        if name is None:
            name = f"checkpoint_{int(time.time())}"
        
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        
        # Save pickle for complete state
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
        
        # Also save human-readable summary
        self._save_summary(state, name)
    
    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """
        Load optimizer state from checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Dictionary containing restored state
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _save_summary(self, state: Dict[str, Any], name: str):
        """Save human-readable summary."""
        summary_path = self.checkpoint_dir / f"{name}_summary.json"
        
        summary = {
            'timestamp': time.time(),
            'checkpoint_name': name
        }
        
        # Extract key information
        if 'generation' in state:
            summary['generation'] = state['generation']
        
        if 'evaluations_used' in state:
            summary['evaluations_used'] = state['evaluations_used']
        
        if 'population' in state and state['population']:
            population = state['population']
            summary['population_size'] = len(population)
            
            # Best candidate info
            if hasattr(population[0], 'scores'):
                scores = []
                for candidate in population:
                    if hasattr(candidate, 'scores') and candidate.scores:
                        avg_score = float(np.mean(list(candidate.scores.values())))
                        scores.append(avg_score)
                
                if scores:
                    summary['best_score'] = max(scores)
                    summary['avg_score'] = np.mean(scores)
                    summary['score_std'] = np.std(scores)
        
        if 'best_candidate' in state and state['best_candidate']:
            candidate = state['best_candidate']
            summary['best_candidate'] = {
                'id': getattr(candidate, 'candidate_id', 'unknown'),
                'generation': getattr(candidate, 'generation', 0)
            }
            
            if hasattr(candidate, 'equilibrium_value'):
                summary['best_candidate']['equilibrium'] = float(candidate.equilibrium_value)
            
            if hasattr(candidate, 'stability_score'):
                summary['best_candidate']['stability'] = float(candidate.stability_score)
        
        # Save summary
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save summary: {e}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for pkl_file in self.checkpoint_dir.glob("*.pkl"):
            name = pkl_file.stem
            summary_file = self.checkpoint_dir / f"{name}_summary.json"
            
            checkpoint_info = {
                'name': name,
                'path': str(pkl_file),
                'size_bytes': pkl_file.stat().st_size,
                'modified_time': pkl_file.stat().st_mtime
            }
            
            # Load summary if available
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    checkpoint_info['summary'] = summary
                except:
                    pass
            
            checkpoints.append(checkpoint_info)
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return checkpoints
    
    def delete_checkpoint(self, name: str):
        """
        Delete a checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        summary_path = self.checkpoint_dir / f"{name}_summary.json"
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint {name}")
        
        if summary_path.exists():
            summary_path.unlink()
    
    def cleanup_old_checkpoints(self, keep_recent: int = 10):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_recent: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_recent:
            return
        
        # Delete old checkpoints
        for checkpoint in checkpoints[keep_recent:]:
            self.delete_checkpoint(checkpoint['name'])
        
        logger.info(f"Cleaned up {len(checkpoints) - keep_recent} old checkpoints")


class AutoCheckpointer:
    """
    Automatic checkpointing based on time or iteration intervals.
    
    Examples:
        >>> auto_cp = AutoCheckpointer(
        ...     checkpoint_manager, 
        ...     time_interval=3600,  # Every hour
        ...     iteration_interval=100  # Every 100 iterations
        ... )
        >>> if auto_cp.should_checkpoint(iteration=150):
        ...     auto_cp.checkpoint(optimizer_state)
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        time_interval: Optional[float] = None,
        iteration_interval: Optional[int] = None
    ):
        self.checkpoint_manager = checkpoint_manager
        self.time_interval = time_interval
        self.iteration_interval = iteration_interval
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_iteration = 0
    
    def should_checkpoint(self, iteration: int) -> bool:
        """
        Check if checkpoint should be saved.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            True if checkpoint should be saved
        """
        current_time = time.time()
        
        # Check time interval
        if self.time_interval and (current_time - self.last_checkpoint_time) >= self.time_interval:
            return True
        
        # Check iteration interval
        if self.iteration_interval and (iteration - self.last_checkpoint_iteration) >= self.iteration_interval:
            return True
        
        return False
    
    def checkpoint(self, state: Dict[str, Any], iteration: int):
        """
        Save checkpoint and update tracking.
        
        Args:
            state: Optimizer state to save
            iteration: Current iteration number
        """
        name = f"auto_{iteration}_{int(time.time())}"
        self.checkpoint_manager.save_checkpoint(state, name)
        
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_iteration = iteration