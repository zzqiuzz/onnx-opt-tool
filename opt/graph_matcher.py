import logging
from typing import List, Optional, Set, Dict, Any
from .onnx_helper import ONNXGraph, ONNXNode 
from .pattern import Pattern, MatchResult


logger = logging.getLogger(__name__)



class GraphMatcher:
    def __init__(self, graph: Optional[ONNXGraph] = None):
        self.graph = graph
        self.match_results: List[MatchResult] = []

    def set_graph(self, graph: ONNXGraph):
        self.graph = graph

    @property
    def patterns(self):
        """
        get all registered patterns and sort in priority-first order. 
        """
        patterns = list(Pattern.REGISTER_PATTERNS.values())
        patterns.sort(key=lambda p: p.priority, reverse=True)
        
        return patterns

    def match_all(self, allow_overlap: bool = False) -> List[MatchResult]:
        if not self.graph:
            logger.error("No graph set for matching.")
            return []

        self.match_results.clear()
        matched_node_ids: Set[int] = set()
        sorted_nodes = self.graph.topological_sort()

        # get all registered patterns 
        logger.info(f"Starting pattern matching on {len(sorted_nodes)} nodes with {len(self.patterns)} patterns...")

        for node in sorted_nodes:
            # 如果不允许重叠，跳过已匹配的节点
            if not allow_overlap and node.id in matched_node_ids:
                continue

            for pattern in self.patterns:
                match_result = pattern.match(node, self.graph)
                if match_result:
                    # 检查是否有重叠节点（如果不允许）
                    if not allow_overlap:
                        new_node_ids = match_result.node_ids
                        if new_node_ids & matched_node_ids:
                            continue  # 有重叠，跳过
 
                    self.match_results.append(match_result)
                    matched_node_ids.update({} if allow_overlap else new_node_ids)
                    logger.debug(f"Matched pattern '{pattern.name}' at nodes {match_result.node_names}")
                    break  # 一个节点只匹配一个最高优先级的pattern

        logger.info(f"Found {len(self.match_results)} matches.")
        return self.match_results

    def get_match_results(self) -> List[MatchResult]:
        return self.match_results