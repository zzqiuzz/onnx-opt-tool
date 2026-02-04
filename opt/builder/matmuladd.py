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
    matmul_node, add_node = match_result.matched_nodes[0], match_result.matched_nodes[1]
    input_name = match_result.inputs[0]
    scale = match_result.inputs[1]
    bias = match_result.inputs[2]
    output_name = match_result.outputs[0]
    attrs = match_result.attrs
    if len(scale.shape) != 2:
        return 
    tensors = self.tensors()
    inputs = tensors.get(input_name)
    outputs = tensors.get(output_name) 
    for outp in inputs.outputs[::]:
        if outp.name in match_result.node_names:
            inputs.outputs.remove(outp) 
    for inp in outputs.inputs[::]:
        if inp.name in match_result.node_names:
            outputs.inputs.remove(inp) 
    scale = gs.Constant(name=matmul_node.name + "_scale", values=scale)
    bias = gs.Constant(name=add_node.name + "_bias", values=bias)
    input_all = [inputs, scale, bias]
    if scale.shape[-1] %8 != 0:
        patch_tensor_fixNot8x = gs.Constant(name=matmul_node.name + "_patch_tensor_fixNot8x", values=np.array([0.], dtype=np.float32))
        input_all.append(patch_tensor_fixNot8x)
        
    ln_node = self.layer(op="MatMulPlugin",
                name=output_name + "_matmul_plugin",
                inputs=input_all,
                outputs=[outputs],
                attrs= attrs
                )
    return ln_node
