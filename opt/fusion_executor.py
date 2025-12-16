import onnx
import logging
import numpy as np
import onnx_graphsurgeon as gs

from typing import List, Optional
from onnx import helper
from onnx.helper import make_node
from .onnx_helper import ONNXGraph, ONNXNode 
from .graph_matcher import MatchResult
from .builder import fuse_convtrans_bn

logger = logging.getLogger(__name__)
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
    def __init__(self, graph: gs.Graph = None):
        self.graph = graph 
        self.gs_nodes = None  # Placeholder for gs nodes
        self.gs_fusion = False

    def set_graph(self, graph: gs.Graph):
        self.graph = graph
        
    def get_graph(self) -> gs.Graph:
        return self.graph
        
    def set_gs_nodes(self, gs_nodes: dict):
        self.gs_nodes = gs_nodes

    # TODO factory pattern
    def _fuse_convtrans_bn(self, match_result: MatchResult) -> bool:
        # gs_graph = gs.import_onnx(self.onnx_model_proto)
        if len(match_result.matched_nodes) != 2:
            logger.error(f"ConvTransBN fusion requires 2 nodes, got {len(match_result.matched_nodes)}.")
            return False
        
        convtrans_node, bn_node = match_result.matched_nodes
        gs_convtrasn_node       = self.gs_nodes[convtrans_node.name]
        gs_bn_node              = self.gs_nodes[bn_node.name]
        self.graph.fuse_convtrans_bn(gs_convtrasn_node, gs_bn_node)
        # self.graph.cleanup().toposort()
        
    def _fuse_layernorm(self, match_result: MatchResult) -> bool:
        self.graph.fuse_layernorm(match_result)
        # self.graph.cleanup().toposort()

    def execute(self, match_result: MatchResult) -> bool:
        if not self.graph:
            logger.error("No graph set for fusion.")
            return False

        pattern_name = match_result.pattern.name
        logger.info(f"Executing fusion for pattern '{pattern_name}'")

        try:
            if pattern_name == "ConvTransBNPattern":
                self._fuse_convtrans_bn(match_result) 
            elif pattern_name == "LayerNormPattern":
                self._fuse_layernorm(match_result)
            else:
                logger.warning(f"No fusion handler for pattern '{pattern_name}'")
                return False
            self.graph.cleanup().toposort()
            self.gs_fusion = True
            return True
        except Exception as e:
            logger.error(f"Fusion failed for pattern '{pattern_name}': {str(e)}")
            return False
        

    def execute_all(self, match_results: List[MatchResult]) -> bool:
        all_success = True
        for match in match_results:
            if not self.execute(match):
                all_success = False
        return all_success
    
    def get_gs_model_proto(self) -> onnx.ModelProto:
        '''
            only return when gs fusion finished.
        '''
        self.onnx_model_proto = gs.export_onnx(self.graph) if self.gs_fusion else None
        return self.onnx_model_proto