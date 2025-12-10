from .base_pattern import Pattern
from .constraints import OpTypeConstraint
from ..onnx_helper import ONNXNode, ONNXGraph
from typing import List, Optional

class ConvReLUPattern(Pattern):
    def __init__(self):
        super().__init__(name="ConvReLU", priority=10)
        self.add_constraint(OpTypeConstraint("Conv"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
        if not all(ct.check(node, graph) for ct in self.constraints):
            return None

        conv_outputs = node.outputs
        if len(conv_outputs) != 1:
            return None

        # relu_candidates = graph.name_to_nodes.get(conv_outputs[0], [])
        relu_candidates = graph.get_successors(node)
        relu_node = None
        for candidate in relu_candidates:
            if candidate.is_op("Relu"):
                relu_node = candidate
                break

        if not relu_node:
            return None

        if len(relu_node.inputs) < 1 or relu_node.inputs[0] != conv_outputs[0]:
            return None

        return [node, relu_node]
    
__all__ = ["ConvReLUPattern"]