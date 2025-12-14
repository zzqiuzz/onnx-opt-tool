import logging

from abc import ABC, abstractmethod
from typing import List, TypeVar
from .constraints import Constraints
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