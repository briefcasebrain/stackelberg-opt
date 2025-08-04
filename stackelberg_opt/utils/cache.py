"""
Caching utilities for Stackelberg optimization.

This module provides caching functionality for API responses
and other expensive computations.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Cache for API responses to avoid redundant calls.
    
    Stores API responses both in memory and on disk for persistence
    across runs.
    
    Attributes:
        cache_dir: Directory for persistent cache storage
        memory_cache: In-memory cache for fast access
        
    Examples:
        >>> cache = ResponseCache(cache_dir=Path(".cache"))
        >>> cached_response = cache.get(prompt, model, temperature)
        >>> if not cached_response:
        ...     response = call_api(prompt)
        ...     cache.set(prompt, model, temperature, response)
    """
    
    def __init__(self, cache_dir: Path = Path(".response_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self._load_cache()
    
    def _get_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature setting
            
        Returns:
            SHA256 hash as cache key
        """
        content = f"{model}:{temperature}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.memory_cache = json.load(f)
                logger.info(f"Loaded {len(self.memory_cache)} cached responses")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.memory_cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.memory_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """
        Get cached response.
        
        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature setting
            
        Returns:
            Cached response or None if not found
        """
        key = self._get_cache_key(prompt, model, temperature)
        return self.memory_cache.get(key)
    
    def set(self, prompt: str, model: str, temperature: float, response: str):
        """
        Cache response.
        
        Args:
            prompt: The prompt text
            model: Model name
            temperature: Temperature setting
            response: Response to cache
        """
        key = self._get_cache_key(prompt, model, temperature)
        self.memory_cache[key] = response
        self._save_cache()
    
    def clear(self):
        """Clear all cached responses."""
        self.memory_cache = {}
        self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'total_entries': len(self.memory_cache),
            'cache_size_bytes': len(json.dumps(self.memory_cache).encode()),
            'cache_dir': str(self.cache_dir)
        }


class ComputationCache:
    """
    Generic cache for expensive computations.
    
    Can be used to cache equilibrium calculations, stability scores,
    or other expensive operations.
    
    Examples:
        >>> cache = ComputationCache()
        >>> key = cache.make_key("equilibrium", candidate_id=123)
        >>> result = cache.get(key)
        >>> if result is None:
        ...     result = expensive_computation()
        ...     cache.set(key, result)
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
        self.access_order = []
    
    def make_key(self, operation: str, **kwargs) -> str:
        """
        Create cache key from operation and parameters.
        
        Args:
            operation: Operation name
            **kwargs: Operation parameters
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent keys
        sorted_kwargs = sorted(kwargs.items())
        key_str = f"{operation}:" + ":".join(f"{k}={v}" for k, v in sorted_kwargs)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self.cache:
            # Update access tracking
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Check size limit
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict least recently used
            self._evict_lru()
        
        self.cache[key] = value
        self.access_count[key] = 1
        self.access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        # Find least recently used key
        key_last_access = {}
        for i, key in enumerate(self.access_order):
            key_last_access[key] = i
        
        if key_last_access:
            lru_key = min(key_last_access.keys(), key=lambda k: key_last_access[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
            # Clean up access order
            self.access_order = [k for k in self.access_order if k != lru_key]
    
    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.access_count.clear()
        self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'total_entries': len(self.cache),
            'total_accesses': sum(self.access_count.values()),
            'max_size': self.max_size,
            'most_accessed': max(self.access_count.items(), key=lambda x: x[1])
                            if self.access_count else None
        }