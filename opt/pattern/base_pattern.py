from abc import ABC, abstractmethod
from typing import List
from .constraints import Constraints


class Pattern(ABC):
    def __init__(self, name : str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.constraints = None
         
    @abstractmethod
    def match(self, node, graph) -> List:
        NotImplemented
        
    def add_constraint(self, constraint : Constraints | None):
        if self.constraints is None:
            self.constraints = []
        self.constraints.append(constraint)
    
    def __repr__(self): 
        return f"Pattern(name={self.name}, priority={self.priority}, constraints={len(self.constraints)})"