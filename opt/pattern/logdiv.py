from .base_pattern import Pattern, MatchResult
from .constraints import OpTypeConstraint
from ..onnx_helper import ONNXNode, ONNXGraph   

@Pattern.register()
class LogDivPattern(Pattern):
    '''
        To be configured with config file for each Project
        This pattern is expecially matched for Bevod model optimization, 
        to transform 
        
        input1 ---                                 input1 --- Log ---
                  \                                                  \
                  Div ---- Log --- output    ===>                    Div --- output
                  /                                                  /
        input2 ---                                 input2 --- Log ---
    '''
    def __init__(self):
        super().__init__(name="LogDivPattern", priority=10)
        self.add_constraint(OpTypeConstraint("Log"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> MatchResult | None: 
        if not all(ct.check(node, graph) for ct in self.constraints):
            return None
 
        if len(node.inputs) < 1 or len(node.outputs) != 1:
            return None

        # only one node precedes the Log node
        div_node = None
        preds = graph.get_predecessors(node) 
        if preds[0].is_op("Div"):
            div_node = preds[0] 

        if not div_node:
            return None

        for div_input in div_node.inputs:
            if graph.is_constant_input(div_input):
                return None

        # 返回匹配结果：matched_nodes 中先放 Div 再放 Log，
        # inputs 使用 Div 的 inputs（两个输入），outputs 使用 Log 的 outputs
        return MatchResult(
            pattern=self,
            matched_nodes=[div_node, node],
            inputs=div_node.inputs,
            outputs=node.outputs,
            attrs={}
        )
    
__all__ = ["LogDivPattern"]