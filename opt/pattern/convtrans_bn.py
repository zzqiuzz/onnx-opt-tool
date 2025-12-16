from .base_pattern import Pattern, MatchResult
from .constraints import OpTypeConstraint
from ..onnx_helper import ONNXNode, ONNXGraph   

@Pattern.register()
class ConvTransBNPattern(Pattern):
    def __init__(self):
        super().__init__(name="ConvTransBNPattern", priority=10)
        self.add_constraint(OpTypeConstraint("ConvTranspose"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> MatchResult | None:
        # check if current node is ConvTranspose
        if not all(ct.check(node, graph) for ct in self.constraints):
            return None
        
        inputs = node.inputs
        conv_outputs = node.outputs
        if len(conv_outputs) != 1:
            return None

        bn_candidates = graph.get_successors(node)
        bn_node = None
        for candidate in bn_candidates:
            if candidate.is_op("BatchNormalization"):
                bn_node = candidate
                break

        if not bn_node:
            return None
  
        outputs = bn_node.outputs
        if len(bn_node.inputs) < 1 or bn_node.inputs[0] != conv_outputs[0]:
            return None 

        return MatchResult(
            pattern=self,
            matched_nodes=[node, bn_node],
            inputs=inputs,
            outputs=outputs,
            attrs={}
        )
    
__all__ = ["ConvTransBNPattern"]