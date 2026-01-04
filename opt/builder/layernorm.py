import numpy as np
import onnx_graphsurgeon as gs

from ..pattern import MatchResult

@gs.Graph.register()
def fuse_layernorm(self, match_result : MatchResult):
    """
    Args: match_result
    
    Returns:
        返回融合后的新 LayerNorm 节点
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
    
    # If scale/bias are not present, create neutral scale=1 and bias=0 of appropriate shape
    # We attempt to infer last dimension from an existing initializer or from input shape if available.
    if scale is None:
        # a safe fallback: use scalar 1.0 (best to replace with correct shape)
        scale = np.array(1.0, dtype=np.float32)
    if bias is None:
        bias = np.array(0.0, dtype=np.float32)
        
    scale = gs.Constant(name= input_name + "_ln_scale", values=scale)
    bias = gs.Constant(name= input_name + "_ln_bias", values=bias)
    
    # create LayerNormalization node. If your target runtime doesn't have "LayerNormalization",
    # you can instead create the classic subgraph. Here we show the single op case:
    if False:
        ln_node = self.layer(op="LayerNormalization",
                            name=output_name + "_LayerNorm",
                            inputs=[inputs, scale, bias],
                            outputs=[outputs],
                            attrs=attrs)
    else:
        eps = gs.Constant(name= input_name + "_ln_eps", values=np.array(attrs["epsilon"], dtype=np.float32))
        ln_node = self.layer(op="NvLayerNormPlugin",
                    name=output_name + "_LayerNorm",
                    inputs=[inputs, scale, bias, eps],
                    outputs=[outputs],
                    )
    return ln_node
