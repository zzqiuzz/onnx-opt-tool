import numpy as np
import onnx_graphsurgeon as gs

from ..pattern import MatchResult

@gs.Graph.register()
def fuse_matmuladd(self, match_result : MatchResult):
    """
    Args: match_result
    
    Returns:
        返回融合后的新 matmul 节点
    """
   # names from match
    input_name = match_result.inputs[0]
    scale = match_result.inputs[1]
    bias = match_result.inputs[2]
    output_name = match_result.outputs[0]
    attrs = match_result.attrs

    # fetch tensors from gs graph (gs_graph.tensors() maps names -> tensors)
    tensors = self.tensors()
    inputs = tensors.get(input_name)
    outputs = tensors.get(output_name)
    
    # tensor's output is node.
    for outp in inputs.outputs[::]:
        if outp.name in match_result.node_names:
            inputs.outputs.remove(outp)
    
    for inp in outputs.inputs[::]:
        if inp.name in match_result.node_names:
            outputs.inputs.remove(inp)
     
    
    # create LayerNormalization node. If your target runtime doesn't have "LayerNormalization",
    # you can instead create the classic subgraph. Here we show the single op case:
  
    ln_node = self.layer(op="MatMulPlugin",
                name=output_name + "_matmul_plugin",
                inputs=[inputs, scale, bias],
                outputs=[outputs],
                attrs= attrs
                )
    return ln_node
