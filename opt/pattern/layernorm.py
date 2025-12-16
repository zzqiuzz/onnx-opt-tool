import logging

from .base_pattern import Pattern, MatchResult
from .constraints import OpTypeConstraint
from ..onnx_helper import ONNXNode, ONNXGraph
from typing import List, Optional 

logger = logging.getLogger(__name__)
 
@Pattern.register()
class LayerNormPattern(Pattern):
    '''  
                ---ReduceMean --     Pow - ReduceMean - Add - Sqrt                            |
            /                 \  /                               \                            |
        Input                   Sub                                Div - (Mul - Add) - Output |  --> Input -> LayerNorm -> Output
            \                 /  \                               /                            |
                ----------------     -----------------------------                            |
    '''
    def __init__(self):
        super().__init__(name="LayerNormPattern", priority=10)
        self.add_constraint(OpTypeConstraint("ReduceMean"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
        """
            Match the LayerNorm subgraph described in the class docstring.
            Returns MatchResult with:
              - matched_nodes: the nodes in the matched subgraph (ordered)
              - inputs: list with the main input tensor name
              - outputs: list with the final output tensor name
              - attrs: extra info like epsilon, scale_name, bias_name (if found)
        """
        if not all(ct.check(node, graph) for ct in self.constraints):
            return None

        # start from first ReduceMean
        reduce_mean1 = node
        # successor should be Sub
        subs = graph.get_successors(reduce_mean1)
        if len(subs) != 1 or not subs[0].is_op("Sub"):
            return None
        sub = subs[0]

        # determine the main input (the one that is not the ReduceMean output)
        # ONNXNode.inputs expected to be list of tensor names
        try:
            input_names = [n for n in sub.inputs if n not in reduce_mean1.outputs]
            if not input_names:
                return None
            main_input = input_names[0]
        except Exception:
            return None

        # Sub -> Pow
        pow_and_Div = graph.get_successors(sub)
        if len(pow_and_Div) != 2 or not set(nd.op_type for nd in pow_and_Div) == {"Pow", "Div"}:
            return None
        pown = pow_and_Div[0]

        # Pow -> ReduceMean (variance)
        reduce_mean2_list = graph.get_successors(pown)
        if len(reduce_mean2_list) != 1 or not reduce_mean2_list[0].is_op("ReduceMean"):
            return None
        reduce_mean2 = reduce_mean2_list[0]

        # ReduceMean -> Add (epsilon)
        adds = graph.get_successors(reduce_mean2)
        if len(adds) != 1 or not adds[0].is_op("Add"):
            return None
        add_eps = adds[0]

        # Add -> Sqrt
        sqrts = graph.get_successors(add_eps)
        if len(sqrts) != 1 or not sqrts[0].is_op("Sqrt"):
            return None
        sqrt = sqrts[0]

        # Sqrt -> Div
        divs = graph.get_successors(sqrt)
        if len(divs) != 1 or not divs[0].is_op("Div"):
            return None
        div = divs[0]
        node_output_shape = graph.get_output_shape_by_name(div.outputs[0])
        # ensure Div consumes the Sub result and sqrt result (order-agnostic)
        if pow_and_Div[1] is not div:
            return None
        
        # if Div connected with Mul and Add nodes: Div -> Mul -> Add (scale and bias)
        muls = graph.get_successors(div)
        if len(muls) != 1 or not muls[0].is_op("Mul"):
            return None
        mul = muls[0]

        adds2 = graph.get_successors(mul)
        if len(adds2) != 1 or not adds2[0].is_op("Add"):
            return None
        add_bias = adds2[0]

        # final output
        outputs = list(add_bias.outputs)

        # try to find scale and bias tensor names (constants)
        scale_name = None
        bias_name = None
        # Mul inputs: one should be Div output, other is scale const
        for inp in mul.inputs:
            if inp not in div.outputs:
                scale_name = inp
                break
        
        scale_array = graph.get_initializer_by_name(scale_name)
        for inp in add_bias.inputs:
            if inp not in mul.outputs:
                bias_name = inp
                break
        bias_array = graph.get_initializer_by_name(bias_name)
        # get scale array
        
        
        # get bias array

        # attempt to read epsilon from Add (one input is a scalar constant)
        eps = None
        for inp in add_eps.inputs:
            if inp not in reduce_mean2.outputs: 
                eps = graph.get_initializer_by_name(inp)
                          
        # parse axis from ReduceMean node
        axis = reduce_mean1.attrs.get("axes", None)
        if not axis:
            logger.warning("Parse axis for LayerNorm failed.")
            return None
        
        matched_nodes = [
            reduce_mean1, 
            sub, 
            pown, 
            reduce_mean2, 
            add_eps, 
            sqrt, 
            div, 
            mul, 
            add_bias
        ]
        attrs = {
            "epsilon": eps.item(), 
            "axis" : len(node_output_shape) - len(axis)
        }
        
        return MatchResult(pattern=self, 
                           matched_nodes=matched_nodes, 
                           inputs=[main_input, scale_array, bias_array], 
                           outputs=outputs, 
                           attrs=attrs)

__all__ = ["LayerNormPattern"]