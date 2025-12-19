import logging
import numpy as np

from .base_pattern import Pattern, MatchResult
from .constraints import OpTypeConstraint
from ..onnx_helper import ONNXNode, ONNXGraph
from typing import List, Optional 

logger = logging.getLogger(__name__)
 
@Pattern.register()
class CustomAttnPattern(Pattern):
    '''  
        input_q -- Reshape -- Transpose -- Div --
                                                 \        
                                                Matmul -- Softmax -- Matmul ---Transpose -- Reshape -- Output
                                                 /                    /
        input_k -- Reshape -- Transpose --------                     /
                                                                    /        
                                                                   /
        input_v -- Reshape ---Transpose ---------------------------
        
                                       ||
                                       ||
                                      \||/
                                       \/
                   
        input_q -- Reshape ---
                              \
        input_v -- Reshape --- Attention -- Reshape -- output               
                              /
        input_k -- Reshape ---
        
    '''
    def __init__(self):
        super().__init__(name="CustomAttnPattern", priority=10)
        self.add_constraint(OpTypeConstraint("Softmax"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
        """
            Match the CustomAttention subgraph described in the class docstring.
            Returns MatchResult with:
              - matched_nodes: the nodes in the matched subgraph (ordered)
              - inputs: list with the main input tensor name
              - outputs: list with the final output tensor name
              - attrs: extra info like epsilon, scale_name, bias_name (if found)
        """
        if not all(ct.check(node, graph) for ct in self.constraints):
            return None

        # start from Softmax
        softmax_node = node
        # predecessor should be Matmul
        matmuls = graph.get_predecessors(softmax_node)
        if len(matmuls) != 1 or not matmuls[0].is_op("MatMul"):
            return None
        pre_matmul = matmuls[0]  
        
        if pre_matmul_shape := graph.get_output_shape_by_name(pre_matmul.outputs[0]): 
            if len(pre_matmul_shape) != 3:
                return None
            q_seq_values = np.array([0, pre_matmul_shape[1]], dtype=np.int32)
            k_seq_values = np.array([0, pre_matmul_shape[2]], dtype=np.int32)
        else:
            raise ValueError("Cannot infer pre_matmul output shape")
        
        q_k_nodes = graph.get_predecessors(pre_matmul)
        if len(q_k_nodes) !=2:
            return None
        div_node = None
        k_trans_node = None
        for nd in q_k_nodes:
            if nd.is_op("Div"):
                div_node = nd 
            if nd.is_op("Transpose"):
                k_trans_node = nd
                
        if not (div_node and k_trans_node):
            return None
        
        # trace q branch
        # Div predecessor should be Transpose
        if q_trans_node := graph.get_predecessors(div_node):
            if len(q_trans_node) !=1 or not q_trans_node[0].is_op("Transpose"):
                return None
            q_trans_node = q_trans_node[0]
        else:
            return None
            
        if q_reshape_node := graph.get_predecessors(q_trans_node):
            if len(q_reshape_node) !=1 or not q_reshape_node[0].is_op("Reshape"):
                return None 
        q_input_name = q_trans_node.inputs[0] 
        
        # trace k branch
        if k_reshape_node := graph.get_predecessors(k_trans_node):
            if len(k_reshape_node) !=1 or not k_reshape_node[0].is_op("Reshape"):
                return None
        k_input_name = k_trans_node.inputs[0]
        
        # trace forward v branch
        if post_matmul := graph.get_successors(softmax_node):
            if len(post_matmul) !=1 or not post_matmul[0].is_op("MatMul"):
                return None
            post_matmul = post_matmul[0]

        v_trans_node = None
        for nd in (nodes:= graph.get_predecessors(post_matmul)):
            if  nd.is_op("Softmax"):
                continue
            if nd.is_op("Transpose"):
                v_trans_node = nd
        if len(nodes) !=2 or not v_trans_node:
            return None
            
        if v_reshape_node := graph.get_predecessors(v_trans_node):
            if len(v_reshape_node) !=1 or not v_reshape_node[0].is_op("Reshape"):
                return None
        v_input_name = v_trans_node.inputs[0]
        
        # trace output branch
        if post_transpose := graph.get_successors(post_matmul):
            if len(post_transpose) !=1 or not post_transpose[0].is_op("Transpose"):
                return None 
            post_transpose = post_transpose[0]
        output_reshape = graph.get_successors(post_transpose)
        if len(output_reshape) !=1 or not output_reshape[0].is_op("Reshape"):
            return None
        outputs = post_transpose.outputs 
        matched_nodes = [
            softmax_node,
            pre_matmul,
            div_node,
            q_trans_node,
            k_trans_node,
            post_matmul,
            v_trans_node,
            post_transpose
        ]
        
        return MatchResult(pattern=self, 
                           matched_nodes=matched_nodes, 
                           inputs=[
                                q_input_name,
                                k_input_name,
                                v_input_name,
                                q_seq_values,
                                k_seq_values
                            ], 
                           outputs=outputs)

__all__ = ["CustomAttnPattern"]