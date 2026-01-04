import onnx
import logging
import numpy as np
import onnx_graphsurgeon as gs

from typing import List, Optional
from onnx import helper
from onnx.helper import make_node
from .onnx_helper import ONNXGraph, ONNXNode 
from .graph_matcher import MatchResult
from .builder import *

logger = logging.getLogger(__name__)

class FusionExecutor:
    def __init__(self, graph: gs.Graph = None):
        self.graph = graph  
        self.gs_fusion = False

    def set_graph(self, graph: gs.Graph):
        self.graph = graph
        
    def get_graph(self) -> gs.Graph:
        return self.graph 

    def execute(self, match_result: MatchResult) -> bool:
        if not self.graph:
            logger.error("No graph set for fusion.")
            return False 
        pattern_name = match_result.pattern.name
        logger.debug(f"Executing fusion for pattern '{pattern_name}'") 
        if pattern_name == "ConvTransBNPattern":
            self.graph.fuse_convtrans_bn(match_result) 
        elif pattern_name == "LayerNormPattern":
            self.graph.fuse_layernorm(match_result)
        elif pattern_name == "CustomAttnPattern":
            self.graph.fuse_customattn(match_result)
        elif pattern_name == "LogDivPattern":
            self.graph.replace_log_div(match_result)
        else:
            logger.warning(f"No fusion handler for pattern '{pattern_name}'")
            return False
        self.graph.cleanup().toposort()
        self.gs_fusion = True
        
        return True  

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
    