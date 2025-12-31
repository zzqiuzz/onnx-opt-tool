import numpy as np
import onnx_graphsurgeon as gs

from ..pattern import MatchResult

@gs.Graph.register()
def replace_log_div(self, match_result : MatchResult): 
    tensors = self.tensors()
    
    div_node, log_node = match_result.matched_nodes  
    div_input_0, div_input_1 = match_result.inputs
    
    div_input_0_tensor, div_input_1_tensor = tensors.get(div_input_0), tensors.get(div_input_1) 
    for outp in div_input_0_tensor.outputs[::]:
        if outp.name in match_result.node_names:
            div_input_0_tensor.outputs.remove(outp)
    
    for outp in div_input_1_tensor.outputs[::]:
        if outp.name in match_result.node_names:
            div_input_1_tensor.outputs.remove(outp) 
    log_output = match_result.outputs[0]
    log_output = tensors.get(log_output) 
    for inp in log_output.inputs[::]:
        if inp.name in match_result.node_names:
            log_output.inputs.remove(inp)
     
    
    
 
    log_0_output = gs.Variable(name=f"{log_node.name}_0_out", dtype=log_output.dtype, shape=log_output.shape)
    log_1_output = gs.Variable(name=f"{log_node.name}_1_out", dtype=log_output.dtype, shape=log_output.shape)
    log_node_0 = self.layer(
        op="Log",
        inputs=[div_input_0_tensor],
        outputs=[log_0_output], 
        name=f"{log_node.name}_0"
    )
    
    log_node_1 = self.layer(
        op="Log",
        inputs=[div_input_1_tensor],
        outputs=[log_1_output], 
        name=f"{log_node.name}_1"
    )
    
    div_node = self.layer(
        op="Sub",
        inputs=[log_0_output, log_1_output],
        outputs=[log_output], 
        name=f"{div_node.name}"
    )

    return log_node_0, log_node_1, div_node