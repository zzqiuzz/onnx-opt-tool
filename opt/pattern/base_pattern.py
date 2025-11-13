from abc import ABC, abstractmethod
from typing import List


class Pattern(ABC):
    def __init__(self, name : str, priority: int = 0):
        self.name = name
        self.priority = priority
        self.constraints = None
         
    @abstractmethod
    def match(self, node, graph) -> List:
        NotImplemented
        
    def __repr__(self): 
        return f"Pattern(name={self.name}, priority={self.priority}, constraints={len(self.constraints)})"