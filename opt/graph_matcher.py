from typing import List, Dict, Optional, Set
from .onnx_helper import ONNXGraph, ONNXNode 
from .pattern import Pattern
from .logger import logger

class MatchResult:
    def __init__(self, pattern: Pattern, matched_nodes: List[ONNXNode]):
        self.pattern = pattern
        self.matched_nodes = matched_nodes
        self.node_ids = {node.id for node in matched_nodes}

    def __repr__(self):
        return f"MatchResult(pattern={self.pattern.name}, nodes={[n.id for n in self.matched_nodes]})"

class GraphMatcher:
    def __init__(self, graph: Optional[ONNXGraph] = None):
        self.graph = graph
        self.patterns: List[Pattern] = []
        self.match_results: List[MatchResult] = []

    def set_graph(self, graph: ONNXGraph):
        self.graph = graph

    def add_pattern(self, pattern: Pattern):
        self.patterns.append(pattern)
        # 按优先级排序（高优先级在前）
        self.patterns.sort(key=lambda p: p.priority, reverse=True)

    def clear_patterns(self):
        self.patterns.clear()

    def match_all(self, allow_overlap: bool = False) -> List[MatchResult]:
        if not self.graph:
            logger.error("No graph set for matching.")
            return []

        self.match_results.clear()
        matched_node_ids: Set[int] = set()
        sorted_nodes = self.graph.topological_sort()

        logger.info(f"Starting pattern matching on {len(sorted_nodes)} nodes with {len(self.patterns)} patterns...")

        for node in sorted_nodes:
            # 如果不允许重叠，跳过已匹配的节点
            if not allow_overlap and node.id in matched_node_ids:
                continue

            for pattern in self.patterns:
                matched_nodes = pattern.match(node, self.graph)
                if matched_nodes:
                    # 检查是否有重叠节点（如果不允许）
                    if not allow_overlap:
                        new_node_ids = {n.id for n in matched_nodes}
                        if new_node_ids & matched_node_ids:
                            continue  # 有重叠，跳过

                    match_result = MatchResult(pattern, matched_nodes)
                    self.match_results.append(match_result)
                    matched_node_ids.update(new_node_ids if allow_overlap else {n.id for n in matched_nodes})
                    logger.debug(f"Matched pattern '{pattern.name}' at nodes {[n.id for n in matched_nodes]}")
                    break  # 一个节点只匹配一个最高优先级的pattern

        logger.info(f"Found {len(self.match_results)} matches.")
        return self.match_results

    def get_match_results(self) -> List[MatchResult]:
        return self.match_results