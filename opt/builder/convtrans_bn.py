import numpy as np
import onnx_graphsurgeon as gs

from ..pattern import MatchResult

@gs.Graph.register()
def fuse_convtrans_bn(self, match_result: MatchResult):
    """
    Args: 
        match_result: 包含ConvTranspose和BatchNormalization节点的匹配结果
    Returns:
        返回融合后的新 ConvTranspose 节点
    """
         
    convtrans_node, bn_node = match_result.matched_nodes
    
    convtrans_input_name  = match_result.inputs[0]
    convtrans_weight_name = match_result.inputs[1]
    convtrans_bias_name   = match_result.inputs[2] if len(match_result.inputs) > 2 else None 
    convtrans_input = self.tensors().get(convtrans_input_name)
    convtrans_weight = self.tensors().get(convtrans_weight_name)
    convtrans_bias = self.tensors().get(convtrans_bias_name)
    
    bn_output_name = match_result.outputs[0]
    bn_output = self.tensors().get(bn_output_name) 
    
    for outp in convtrans_input.outputs[::]:
        if outp.name in match_result.node_names:
            convtrans_input.outputs.remove(outp)
    
    for inp in bn_output.inputs[::]:
        if inp.name in match_result.node_names:
            bn_output.inputs.remove(inp)

    # ConvTranspose权重维度：ONNX标准 [C_in, C_out/groups, kH, kW]
    weight = convtrans_weight.values
    groups = convtrans_node.attrs.get("group", 1)  # 默认分组为1
    C_out_per_group = weight.shape[1]
    C_out = C_out_per_group * groups  # 输出通道数 = 每组输出通道 * 分组数

    # 校验BN参数维度与ConvTranspose输出通道匹配
    scale_name = bn_node.inputs[1] 
    bias_name = bn_node.inputs[2]
    mean_name = bn_node.inputs[3]
    var_name = bn_node.inputs[4]
    scale = self.tensors().get(scale_name)
    bias = self.tensors().get(bias_name)
    mean = self.tensors().get(mean_name)
    var = self.tensors().get(var_name)
    
    scale = scale.values
    bias = bias.values
    mean = mean.values
    var = var.values
    
    
    
    if scale.shape[0] != C_out:
        raise ValueError(f"BN scale维度({scale.shape[0]})与ConvTranspose输出通道({C_out})不匹配！")
   
    epsilon = bn_node.attrs.get("epsilon", 1e-5)

    # 处理ConvTranspose无bias的场景（初始bias为C_out维度的0）
    if convtrans_bias is None:
        bias = np.zeros(C_out, dtype=weight.dtype)
    else:
        bias = convtrans_bias.values
        # 校验原bias维度
        if bias.shape[0] != C_out:
            raise ValueError(f"ConvTranspose bias维度({bias.shape[0]})与输出通道({C_out})不匹配！")

    # BN核心公式：y = (x - mean)/sqrt(var + eps) * scale + bias
    denom = np.sqrt(var + epsilon)
    bn_scale_fused = scale / denom
    bn_bias_fused = bias - mean * bn_scale_fused

    # 适配分组卷积的广播形状：[1, C_out/groups, 1, 1]（匹配weight的[C_in, C_out/groups, kH, kW]）
    # 分组场景下：bn_scale_fused先reshape为[groups, C_out_per_group]，再展平为[1, C_out_per_group*groups, 1, 1]会广播错误
    # 正确方式：按分组拆分scale后广播到每个分组的输出通道
    bn_scale_broadcast = bn_scale_fused.reshape(groups, C_out_per_group).reshape(1, -1, 1, 1)
    weight_fused = weight * bn_scale_broadcast  # 广播维度匹配：[C_in, C_out/groups, kH, kW]

    # Bias融合：直接按C_out维度计算（无需广播）
    bias_fused = bias * bn_scale_fused + bn_bias_fused

    weight_dtype = weight.dtype
    weight_fused = weight_fused.astype(weight_dtype)
    bias_fused = bias_fused.astype(weight_dtype)
    # 融合权重
    fused_weight = gs.Constant(
        name=f"{convtrans_weight.name}_fused",
        values=weight_fused
    )
    # 融合偏置
    fused_bias_name = f"{convtrans_bias.name}_fused" if convtrans_bias else f"{convtrans_node.name}_bias_fused"
    fused_bias = gs.Constant(
        name=fused_bias_name,
        values=bias_fused
    )

    # 创建融合后的ConvTranspose节点
    fused_convtrans_node = self.layer(
        op="ConvTranspose",
        inputs=[convtrans_input, fused_weight, fused_bias],
        outputs=[bn_output],
        attrs=convtrans_node.attrs,
        name=f"{convtrans_node.name}_fused"
    )

    return fused_convtrans_node
