"""
Module definitions for Stackelberg optimization.

This module contains the core data structures for defining modules
in a compound AI system with leader-follower dynamics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any


class ModuleType(Enum):
    """Types of modules in the Stackelberg hierarchy."""
    LEADER = "leader"
    FOLLOWER = "follower"
    INDEPENDENT = "independent"


@dataclass
class Module:
    """
    Represents a module in the compound AI system.
    
    Attributes:
        name: Unique identifier for the module
        prompt: The prompt template for this module
        module_type: Whether this is a leader, follower, or independent module
        dependencies: List of module names this module depends on
        version: Version number for tracking changes
        metadata: Additional module metadata
        
    Examples:
        >>> leader = Module(
        ...     name="query_generator",
        ...     prompt="Generate a search query for: {input}",
        ...     module_type=ModuleType.LEADER,
        ...     dependencies=[]
        ... )
        >>> follower = Module(
        ...     name="answer_extractor",
        ...     prompt="Extract answer from context: {context}",
        ...     module_type=ModuleType.FOLLOWER,
        ...     dependencies=["query_generator"]
        ... )
    """
    name: str
    prompt: str
    module_type: ModuleType
    dependencies: List[str] = field(default_factory=list)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate module after initialization."""
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"Invalid module name: {self.name}")
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValueError(f"Invalid prompt for module {self.name}")
        if len(self.prompt) < 10:
            raise ValueError(f"Prompt too short for module {self.name}")
    
    def __hash__(self):
        """Make module hashable for use in sets/dicts."""
        return hash((self.name, self.prompt, self.version))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize module to dictionary.
        
        Returns:
            Dictionary representation of the module
        """
        return {
            'name': self.name,
            'prompt': self.prompt,
            'module_type': self.module_type.value,
            'dependencies': self.dependencies,
            'version': self.version,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Module':
        """
        Deserialize module from dictionary.
        
        Args:
            data: Dictionary containing module data
            
        Returns:
            Module instance
        """
        return cls(
            name=data['name'],
            prompt=data['prompt'],
            module_type=ModuleType(data['module_type']),
            dependencies=data.get('dependencies', []),
            version=data.get('version', 1),
            metadata=data.get('metadata', {})
        )