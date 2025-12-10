import numpy as np
import onnx_graphsurgeon as gs

@gs.Graph.register()
def fuse_convtrans_bn(self, convtrans_node: gs.Node, bn_node: gs.Node):
    """
    Args:
        convtrans_node: ConvTranspose 节点
        bn_node: 紧随其后的 BatchNorm 节点
    
    Returns:
        返回融合后的新 ConvTranspose 节点
    """
        
    # 获取原始节点输入/输出
    convtrans_input  = convtrans_node.inputs[0]
    convtrans_weight = convtrans_node.inputs[1]
    convtrans_bias   = convtrans_node.inputs[2] if len(convtrans_node.inputs) > 2 else None
    # disconnect input's output link to convtrans_node
    convtrans_input.outputs.remove(convtrans_node) 
    bn_output = bn_node.outputs[0]
    # disconnect bn_output's input link to bn_node
    bn_output.inputs.remove(bn_node)  

    # ConvTranspose权重维度：ONNX标准 [C_in, C_out/groups, kH, kW]
    weight = convtrans_weight.values
    groups = convtrans_node.attrs.get("group", 1)  # 默认分组为1
    C_out_per_group = weight.shape[1]
    C_out = C_out_per_group * groups  # 输出通道数 = 每组输出通道 * 分组数

    # 校验BN参数维度与ConvTranspose输出通道匹配
    bn_scale = bn_node.inputs[1].values
    if bn_scale.shape[0] != C_out:
        raise ValueError(f"BN scale维度({bn_scale.shape[0]})与ConvTranspose输出通道({C_out})不匹配！")
    bn_bias = bn_node.inputs[2].values
    bn_mean = bn_node.inputs[3].values
    bn_var = bn_node.inputs[4].values
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
    denom = np.sqrt(bn_var + epsilon)
    bn_scale_fused = bn_scale / denom
    bn_bias_fused = bn_bias - bn_mean * bn_scale_fused

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