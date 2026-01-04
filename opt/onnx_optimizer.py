import logging

from typing import Optional
from .onnx_helper import ONNXModel 
from .graph_matcher import GraphMatcher
from .fusion_executor import FusionExecutor
from .config import Config, default_config

logger = logging.getLogger(__name__)

class ONNXOptimizer:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.model: Optional[ONNXModel] = None
        self.matcher = GraphMatcher()
        self.executor = FusionExecutor()

    def load_model(self, onnx_path: str) -> bool: 
        self.model = ONNXModel.load(onnx_path)
        digraph  = self.model.get_digraph()
        gs_graph = self.model.get_gs_graph() 
        if digraph and gs_graph:
            self.matcher.set_graph(digraph)
            self.executor.set_graph(gs_graph) 
            logger.info(f"Model loaded successfully: {onnx_path}")
            return True
        else:
            logger.error("Failed to load graph from model.")
            return False 

    def optimize(self) -> bool:
        if not self.model or not self.model.get_digraph():
            logger.error("No model loaded.")
            return False
 
        all_success = True 
        match_results = self.matcher.match_all(allow_overlap=self.config.allow_overlap)
        
        if not match_results:
            logger.info("No matches found, optimization complete.")
            all_success = False
        success = self.executor.execute_all(match_results)
        
        if not success: 
            all_success = False 
        
        self.model.update_onnx_model_proto(self.executor.get_gs_model_proto())
            
        logger.info(f"Optimization finished. Success: {all_success}")
        return all_success

    def save_model(self, path: str):
        if self.model:
            return self.model.save(path)
        else:
            logger.error("No model to save.")
        return False

    def get_optimized_model(self) -> Optional[ONNXModel]:
        return self.model