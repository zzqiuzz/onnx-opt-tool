import logging
import numpy as np

from .base_pattern import Pattern, MatchResult
from .constraints import OpTypeConstraint
from ..onnx_helper import ONNXNode, ONNXGraph
from typing import List, Optional

logger = logging.getLogger(__name__)


@Pattern.register()
class MatMulAddPattern(Pattern):
    """

    input -> MatMul -> Add -> (optional) Relu -> output
        -->  input -> MatMulPlugin -> output

    """

    def __init__(self):
        super().__init__(name="MatMulAddPattern", priority=10)
        self.add_constraint(OpTypeConstraint("MatMul"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
        """
        Match the MatMulAddPattern subgraph described in the class docstring.
        Returns MatchResult with:
          - matched_nodes: the nodes in the matched subgraph (ordered)
          - inputs: list with the main input tensor name
          - outputs: list with the final output tensor name
          - attrs: extra info like withoutRelu
        """
        if not all(ct.check(node, graph) for ct in self.constraints):
            return None

        matmul_node = node
        matmul_inputs = matmul_node.inputs
        matmul_input0, matmul_input1 = graph.get_initializer_by_name(
            matmul_inputs[0]
        ), graph.get_initializer_by_name(matmul_inputs[1])

        if (matmul_input0 is not None and matmul_input1 is not None) or (
            matmul_input0 is None and matmul_input1 is None
        ):
            return None
        matmul_input_constant, matmul_input_variable_name = (
            (matmul_input0, matmul_inputs[1])
            if matmul_input0 is not None
            else (matmul_input1, matmul_inputs[0])
        )

        # successor should be Add
        add_node = graph.get_successors(matmul_node)
        if len(add_node) != 1 or not add_node[0].is_op("Add"):
            return None
        add_node = add_node[0]
        matmul_output = matmul_node.outputs[0]
        for inp in add_node.inputs:
            if inp != matmul_output:
                add_input_constant = graph.get_initializer_by_name(inp)
        if add_input_constant is None:
            return None
        outputs = add_node.outputs
        matched_nodes = [
            matmul_node,
            add_node,
        ]
        attrs = {"withoutRelu": 1}
        relu_node = graph.get_successors(add_node)
        if len(relu_node) == 1 and relu_node[0].is_op("Relu"):
            relu_node = relu_node[0]
            outputs = relu_node.outputs
            attrs["withoutRelu"] = 0
            matched_nodes.append(relu_node)

        return MatchResult(
            pattern=self,
            matched_nodes=matched_nodes,
            inputs=[matmul_input_variable_name, matmul_input_constant, add_input_constant],
            outputs=outputs,
            attrs=attrs,
        )


__all__ = ["MatMulAddPattern"]
