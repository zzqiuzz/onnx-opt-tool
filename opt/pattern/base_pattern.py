import logging

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, TypeVar, Set, Dict, Any
from .constraints import Constraints
from ..onnx_helper import ONNXNode

logger = logging.getLogger(__name__)
# 定义类型变量，约束注册的是Pattern子类/实例
PatternType = TypeVar("PatternType", bound="Pattern")

class Pattern(ABC):
    
    REGISTER_PATTERNS = dict()
    
    def __init__(self, name : str, priority: int = 0):
        self._name = name
        self._priority = priority
        self.constraints = None
    
    @property
    def name(self):
        return self._name
    
    @property
    def priority(self):
        return self._priority
    
    @classmethod
    def register_pattern(cls, pattern : PatternType):    
        if not isinstance(pattern, cls):
            raise TypeError(f"The registerd pattern must be {cls.__name__}, input pattern type:{type(pattern)}")
        
        if pattern.name in dir(cls) and not pattern.name.startswith("_"):
            logger.warning(
                 "Registered pattern: {:} is hidden by a Pattern attribute or pattern with the same name. "
                "This pattern will never be called!".format(pattern.__name__)
            )
        if pattern.name in cls.REGISTER_PATTERNS:
            logger.warning(
                "This pattern has been registerd, action will be the newer pattern wiil override the older one."
            )
            
        cls.REGISTER_PATTERNS[pattern.name] = pattern
        
    @classmethod
    def register(cls):  
        def register_func(pattern_cls: PatternType) -> PatternType:
            if issubclass(pattern_cls, cls):
                isinstance = pattern_cls()
                cls.register_pattern(isinstance)
                logger.debug(f"Pattern {pattern_cls.name} has been registered.")
            return pattern_cls
        return register_func
        
    @abstractmethod
    def match(self, node, graph) -> List:
        NotImplemented
        
    def add_constraint(self, constraint : Constraints | None):
        if self.constraints is None:
            self.constraints = []
        self.constraints.append(constraint)
    
    def __repr__(self): 
        return f"Pattern(name={self.name}, priority={self.priority}, constraints={len(self.constraints)})"
    
    
@dataclass
class MatchResult:
    """Data class for pattern matching results, encapsulates pattern matching-related information"""
    
    # Mandatory initialization parameters (core params from original __init__)
    pattern: Pattern
    matched_nodes: List[ONNXNode]
    
    # Derived attribute (not part of initialization, calculated from matched_nodes)
    node_ids: Set[str] = field(init=False)
    
    node_names: Set[str] = field(init=False)
    
    # Optional parameters (empty defaults using default_factory to avoid mutable default value trap)
    inputs: List[Any] = field(default_factory=list)  # Inputs of new subgraph or single node
    outputs: List[Any] = field(default_factory=list) # Outputs of new subgraph or single node
    attrs: Dict[str, Any] = field(default_factory=dict) # Attributes of new subgraph or single node

    def __post_init__(self) -> None:
        """Post-initialization processing: Calculate derived attribute `node_ids`"""
        self.node_ids = {node.id for node in self.matched_nodes}
        self.node_names = {node.name for node in self.matched_nodes}

    def __repr__(self) -> str:
        """Custom string representation (preserves original format)"""
        return f"MatchResult(pattern={self.pattern.name}, nodes={[n.id for n in self.matched_nodes]})"
    