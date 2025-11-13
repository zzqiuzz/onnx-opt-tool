from base_pattern import Pattern
from constraints import OpTypeConstraint
from ..onnx_helper import ONNXNode, ONNXGraph
from typing import List, Dict, Any, Optional

class ConvBNPattern(Pattern):
    def __init__(self):
        super().__init__(name="ConvBN", priority=10)
        self.add_constraint(OpTypeConstraint("Conv"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
        # 1. 检查当前节点是否为Conv
        if not all(ct.check(node, graph) for ct in self.constraints):
            return None

        # 2. 获取Conv的所有输出节点（应该只有一个BN）
        conv_outputs = node.outputs
        if len(conv_outputs) != 1:
            return None

        bn_candidates = graph.name_to_nodes.get(conv_outputs[0], [])
        bn_node = None
        for candidate in bn_candidates:
            if candidate.is_op("BatchNormalization"):
                bn_node = candidate
                break

        if not bn_node:
            return None

        # 3. 检查BN的输入是否只有Conv的输出（简化版，不考虑其他输入如scale/bias）
        if len(bn_node.inputs) < 1 or bn_node.inputs[0] != conv_outputs[0]:
            return None

        return [node, bn_node]
    
__all__ = [ConvBNPattern]