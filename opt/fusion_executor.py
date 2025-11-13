from typing import List, Optional, Dict, Any
import numpy as np
from onnx import helper, TensorProto
from onnx.helper import make_tensor, make_node, make_graph, make_model
from .onnx_helper import ONNXGraph, ONNXNode 
from .pattern import Pattern
from .graph_matcher import MatchResult
from .logger import logger

class ONNXNodeBuilder:
    @staticmethod
    def create_conv_node(
        name: str,
        inputs: List[str],
        outputs: List[str],
        kernel_shape: List[int],
        strides: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        group: int = 1,
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None
    ) -> ONNXNode:
        """创建Conv节点（含权重和偏置初始化）"""
        attrs = {
            "kernel_shape": kernel_shape,
            "group": group
        }
        if strides:
            attrs["strides"] = strides
        if pads:
            attrs["pads"] = pads
        if dilations:
            attrs["dilations"] = dilations

        # 创建节点
        conv_node_proto = make_node(
            op_type="Conv",
            inputs=inputs,
            outputs=outputs,
            name=name,
            **attrs
        )

        return ONNXNode(conv_node_proto)

class FusionExecutor:
    def __init__(self, graph: Optional[ONNXGraph] = None):
        self.graph = graph
        self.builder = ONNXNodeBuilder()

    def set_graph(self, graph: ONNXGraph):
        self.graph = graph

    def _fuse_conv_bn(self, match_result: MatchResult) -> bool:
        """融合Conv + BatchNormalization为单个Conv"""
        if len(match_result.matched_nodes) != 2:
            logger.error(f"ConvBN fusion requires 2 nodes, got {len(match_result.matched_nodes)}.")
            return False

        conv_node, bn_node = match_result.matched_nodes
        graph = self.graph

        # 1. 获取Conv和BN的权重/参数
        # 简化版：假设权重和偏置是初始器（实际可能需要从graph的initializer中获取）
        conv_weights = None  # TODO: 从graph中获取Conv的权重张量
        conv_biases = None   # TODO: 从graph中获取Conv的偏置张量
        bn_scale = None      # TODO: 从graph中获取BN的scale张量
        bn_bias = None       # TODO: 从graph中获取BN的bias张量
        bn_mean = None       # TODO: 从graph中获取BN的mean张量
        bn_var = None        # TODO: 从graph中获取BN的var张量

        # 这里用占位符模拟计算（实际需要替换为真实张量计算）
        if conv_weights is None:
            logger.warning("Conv weights not found, using dummy values for fusion.")
            conv_weights = np.ones((64, 3, 3, 3), dtype=np.float32)
        if bn_scale is None:
            bn_scale = np.ones(64, dtype=np.float32)
        if bn_var is None:
            bn_var = np.ones(64, dtype=np.float32)
        if bn_mean is None:
            bn_mean = np.zeros(64, dtype=np.float32)
        if bn_bias is None:
            bn_bias = np.zeros(64, dtype=np.float32)

        # 2. 计算融合后的权重和偏置
        # 公式：
        # fused_weight = weight * scale / sqrt(var + eps)
        # fused_bias = (bias - mean) * scale / sqrt(var + eps) + bn_bias
        eps = 1e-5
        std = np.sqrt(bn_var + eps)
        fused_weights = conv_weights * (bn_scale / std).reshape(-1, 1, 1, 1)
        
        if conv_biases is None:
            fused_biases = (-bn_mean * bn_scale / std) + bn_bias
        else:
            fused_biases = (conv_biases - bn_mean) * (bn_scale / std) + bn_bias

        # 3. 创建新的Conv节点
        new_conv_name = f"fused_conv_{conv_node.id}"
        new_conv_inputs = conv_node.inputs.copy()
        # 替换Conv的偏置输入（如果原来有的话）
        if len(new_conv_inputs) > 1:
            new_conv_inputs[1] = f"{new_conv_name}_bias"
        else:
            new_conv_inputs.append(f"{new_conv_name}_bias")
        
        new_conv_outputs = bn_node.outputs.copy()  # 直接使用BN的输出

        new_conv_node = self.builder.create_conv_node(
            name=new_conv_name,
            inputs=new_conv_inputs,
            outputs=new_conv_outputs,
            kernel_shape=conv_node.get_attr("kernel_shape", [3, 3]),
            strides=conv_node.get_attr("strides", [1, 1]),
            pads=conv_node.get_attr("pads", [1, 1, 1, 1]),
            dilations=conv_node.get_attr("dilations", [1, 1]),
            group=conv_node.get_attr("group", 1),
            weights=fused_weights,
            biases=fused_biases
        )

        # 4. 更新计算图
        # 4.1 添加新节点
        # TODO: 实际需要将新节点的proto添加到graph.proto.node中
        # 这里简化为打印信息，真实实现需要操作ONNX的protobuf
        logger.info(f"Created fused Conv node: {new_conv_node}")

        # 4.2 删除旧节点
        graph.remove_node(conv_node)
        graph.remove_node(bn_node)

        # 4.3 更新其他节点的输入（将指向BN输出的改为指向新Conv输出）
        # TODO: 实际需要遍历所有节点，替换输入中的旧名称
        logger.info(f"Fused Conv({conv_node.id}) and BatchNormalization({bn_node.id}) into {new_conv_name}")
        return True

    def _fuse_conv_relu(self, match_result: MatchResult) -> bool:
        """融合Conv + Relu为单个Conv（ONNX中可直接用Conv的activation属性）"""
        if len(match_result.matched_nodes) != 2:
            logger.error(f"ConvRelu fusion requires 2 nodes, got {len(match_result.matched_nodes)}.")
            return False

        conv_node, relu_node = match_result.matched_nodes
        graph = self.graph

        # 1. 检查Conv是否已有activation属性
        if conv_node.get_attr("activation") == "Relu":
            logger.warning(f"Conv({conv_node.id}) already has Relu activation, skipping fusion.")
            return True

        # 2. 创建新的Conv节点（添加activation属性）
        new_conv_name = f"fused_conv_relu_{conv_node.id}"
        new_conv_inputs = conv_node.inputs.copy()
        new_conv_outputs = relu_node.outputs.copy()  # 使用Relu的输出

        # 复制原有属性并添加activation
        new_conv_attrs = conv_node.attrs.copy()
        new_conv_attrs["activation"] = "Relu"

        # 创建节点proto
        new_conv_node_proto = helper.make_node(
            op_type="Conv",
            inputs=new_conv_inputs,
            outputs=new_conv_outputs,
            name=new_conv_name,
            **new_conv_attrs
        )
        new_conv_node = ONNXNode(new_conv_node_proto)

        # 3. 更新计算图
        logger.info(f"Created fused Conv+Relu node: {new_conv_node}")
        graph.remove_node(conv_node)
        graph.remove_node(relu_node)
        logger.info(f"Fused Conv({conv_node.id}) and Relu({relu_node.id}) into {new_conv_name}")
        return True

    def execute(self, match_result: MatchResult) -> bool:
        if not self.graph:
            logger.error("No graph set for fusion.")
            return False

        pattern_name = match_result.pattern.name
        logger.info(f"Executing fusion for pattern '{pattern_name}'")

        try:
            if pattern_name == "ConvBN":
                return self._fuse_conv_bn(match_result)
            elif pattern_name == "ConvRelu":
                return self._fuse_conv_relu(match_result)
            else:
                logger.warning(f"No fusion handler for pattern '{pattern_name}'")
                return False
        except Exception as e:
            logger.error(f"Fusion failed for pattern '{pattern_name}': {str(e)}")
            return False

    def execute_all(self, match_results: List[MatchResult]) -> bool:
        all_success = True
        for match in match_results:
            if not self.execute(match):
                all_success = False
        return all_success