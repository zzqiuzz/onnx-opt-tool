from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..onnx_helper.onnx_graph import ONNXGraph
from ..onnx_helper.onnx_node import ONNXNode

class Constraints(ABC):
    
    @abstractmethod
    def check(self, node : ONNXNode, graph : ONNXGraph) ->bool:
        NotImplemented
        
        
class OpTypeConstraint(Constraints):
    def __init__(self, op_type: str):
        self.op_type = op_type
        
    def check(self, node, graph) -> bool:
        return node.is_op(self.op_type)
    
class AttrConstraint(Constraints):
    def __init__(self, attr_name: str, value: Any, comparator: str = "=="):
        self.attr_name = attr_name
        self.value = value
        self.comparator = comparator

    def check(self, node: ONNXNode, graph: ONNXGraph) -> bool:
        attr_value = node.get_attr(self.attr_name)
        if attr_value is None:
            return False
        
        if self.comparator == "==":
            return attr_value == self.value
        elif self.comparator == "!=":
            return attr_value != self.value
        elif self.comparator == ">":
            return attr_value > self.value
        elif self.comparator == "<":
            return attr_value < self.value
        elif self.comparator == ">=":
            return attr_value >= self.value
        elif self.comparator == "<=":
            return attr_value <= self.value
        else:
            raise ValueError(f"Unsupported comparator: {self.comparator}")
        
__all__ = [OpTypeConstraint, AttrConstraint]