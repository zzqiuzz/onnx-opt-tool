import logging

from typing import List, Optional
from .onnx_helper import ONNXModel
from .pattern import Pattern
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
        try:
            self.model = ONNXModel.load(onnx_path)
            digraph  = self.model.get_digraph()
            gs_graph = self.model.get_gs_graph() 
            if digraph and gs_graph:
                self.matcher.set_graph(digraph)
                self.executor.set_graph(gs_graph)
                self.executor.set_gs_nodes(self.model.gs_nodes)
                logger.info(f"Model loaded successfully: {onnx_path}")
                return True
            else:
                logger.error("Failed to load graph from model.")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def add_pattern(self, pattern: Pattern):
        self.matcher.add_pattern(pattern)
        logger.info(f"Added pattern: {pattern.name} (priority: {pattern.priority})")

    def add_patterns(self, patterns: List[Pattern]):
        for pattern in patterns:
            self.add_pattern(pattern)

    def optimize(self, iterations: int = 1) -> bool:
        if not self.model or not self.model.get_digraph():
            logger.error("No model loaded.")
            return False

        logger.info(f"Starting optimization with {iterations} iterations...")
        all_success = True
        for i in range(iterations):
            logger.info(f"\n--- Optimization Iteration {i+1}/{iterations} ---")
            match_results = self.matcher.match_all(allow_overlap=self.config.allow_overlap)
            if not match_results:
                logger.info("No matches found, optimization complete.")
                break
            success = self.executor.execute_all(match_results)
            if not success:
                logger.error(f"Iteration {i+1} failed.")
                all_success = False
                break
            
            self.model.update_onnx_model_proto(self.executor.get_gs_model_proto())
            
        logger.info(f"\nOptimization finished. Success: {all_success}")
        return all_success

    def save_model(self, path: str):
        if self.model:
            self.model.save(path)
        else:
            logger.error("No model to save.")

    def get_optimized_model(self) -> Optional[ONNXModel]:
        return self.model