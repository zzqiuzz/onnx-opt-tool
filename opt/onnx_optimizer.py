from typing import List, Optional
from .onnx_helper import ONNXModel
from .pattern import Pattern
from .graph_matcher import GraphMatcher
from .fusion_executor import FusionExecutor
from .config import Config, default_config
from .logger import logger

class ONNXOptimizer:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or default_config
        self.model: Optional[ONNXModel] = None
        self.matcher = GraphMatcher()
        self.executor = FusionExecutor()

        # 初始化日志级别
        logger.set_level(self.config.log_level)

    def load_model(self, path: str) -> bool:
        try:
            self.model = ONNXModel.load(path)
            graph = self.model.get_graph()
            if graph:
                self.matcher.set_graph(graph)
                self.executor.set_graph(graph)
                logger.info(f"Model loaded successfully: {path}")
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
        """执行优化（支持迭代优化）"""
        if not self.model or not self.model.get_graph():
            logger.error("No model loaded.")
            return False

        logger.info(f"Starting optimization with {iterations} iterations...")
        all_success = True

        for i in range(iterations):
            logger.info(f"\n--- Optimization Iteration {i+1}/{iterations} ---")
            
            # 1. 执行匹配
            match_results = self.matcher.match_all(allow_overlap=self.config.allow_overlap)
            if not match_results:
                logger.info("No matches found, optimization complete.")
                break

            # 2. 执行融合
            success = self.executor.execute_all(match_results)
            if not success:
                logger.error(f"Iteration {i+1} failed.")
                all_success = False
                break

            # 3. 可视化（如果开启）
            if self.config.visualize:
                # TODO: 实现匹配结果可视化（如使用networkx+matplotlib）
                logger.info("Visualization not implemented yet.")

        logger.info(f"\nOptimization finished. Success: {all_success}")
        return all_success

    def save_model(self, path: str):
        if self.model:
            self.model.save(path)
        else:
            logger.error("No model to save.")

    def get_optimized_model(self) -> Optional[ONNXModel]:
        return self.model