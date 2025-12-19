import numpy as np
import onnx_graphsurgeon as gs

from ..pattern import MatchResult

@gs.Graph.register()
def fuse_customattn(self, match_result : MatchResult):
    """
    Args: match_result
    
    Returns:
        返回融合后的新 LayerNorm 节点
    """
   # names from match
    q_name = match_result.inputs[0]
    k_name = match_result.inputs[1]
    v_name = match_result.inputs[2]
    seq_q_tensor = match_result.inputs[3]
    seq_k_tensor = match_result.inputs[4]
    output_name = match_result.outputs[0]

    # fetch tensors from gs graph (gs_graph.tensors() maps names -> tensors)
    tensors = self.tensors()
    q = tensors.get(q_name)
    k = tensors.get(k_name)
    v = tensors.get(v_name)
    
    outputs = tensors.get(output_name)
    
    # tensor's output is node.
    for outp in q.outputs[::]:
        if outp.name in match_result.node_names:
            q.outputs.remove(outp)
            
    for outp in k.outputs[::]:
        if outp.name in match_result.node_names:
            k.outputs.remove(outp)
            
    for outp in v.outputs[::]:
        if outp.name in match_result.node_names:
            v.outputs.remove(outp)
    
    for inp in outputs.inputs[::]:
        if inp.name in match_result.node_names:
            outputs.inputs.remove(inp)

        
    seq_q_tensor = gs.Constant(name= output_name + "_seq_q_tensor", values=seq_q_tensor)
    seq_k_tensor = gs.Constant(name= output_name + "_seq_k_tensor", values=seq_k_tensor)
     
    customattn_node = self.layer(op="CustomFFAttn",
                name=output_name + "_customattn",
                inputs=[
                    q, 
                    k, 
                    v, 
                    seq_q_tensor, 
                    seq_k_tensor
                ],
                outputs=[outputs],
                )
    return customattn_node