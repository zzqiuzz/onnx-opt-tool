from onnx import GraphProto
from typing import Dict, List, Optional, Iterable
from .onnx_node import ONNXNode
from ..logger import logger
import networkx as nx

class ONNXGraph:
    def __init__(self, graph_proto: GraphProto):
        self.proto = graph_proto
        self.nodes: Dict[int, ONNXNode] = {}  # node.id -> ONNXNode
        self.name_to_nodes: Dict[str, List[ONNXNode]] = {}  # output name -> nodes
        self.graph: nx.DiGraph = nx.DiGraph()

        self._build_graph()

    def _build_graph(self):
        # 1. 创建所有节点对象
        for node_proto in self.proto.node:
            node = ONNXNode(node_proto)
            self.nodes[node.id] = node
            self.graph.add_node(node.id, op_type=node.op_type)

            # 2. 建立输出名到节点的映射
            for output in node.outputs:
                if output not in self.name_to_nodes:
                    self.name_to_nodes[output] = []
                self.name_to_nodes[output].append(node)

        # 3. 建立边（基于张量依赖）
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp in self.name_to_nodes:
                    for prev_node in self.name_to_nodes[inp]:
                        self.graph.add_edge(prev_node.id, node.id, tensor=inp)

    def get_node_by_id(self, node_id: int) -> Optional[ONNXNode]:
        return self.nodes.get(node_id)

    def get_nodes_by_op_type(self, op_type: str) -> List[ONNXNode]:
        return [node for node in self.nodes.values() if node.is_op(op_type)]

    def get_predecessors(self, node: ONNXNode) -> List[ONNXNode]:
        pred_ids = self.graph.predecessors(node.id)
        return [self.get_node_by_id(pid) for pid in pred_ids if self.get_node_by_id(pid)]

    def get_successors(self, node: ONNXNode) -> List[ONNXNode]:
        succ_ids = self.graph.successors(node.id)
        return [self.get_node_by_id(sid) for sid in succ_ids if self.get_node_by_id(sid)]

    def topological_sort(self) -> List[ONNXNode]:
        """返回拓扑排序后的节点列表"""
        try:
            sorted_ids = list(nx.topological_sort(self.graph))
            return [self.get_node_by_id(nid) for nid in sorted_ids if self.get_node_by_id(nid)]
        except nx.NetworkXUnfeasible:
            logger.warning("Graph contains a cycle, cannot perform topological sort.")
            return list(self.nodes.values())

    def remove_node(self, node: ONNXNode):
        if node.id in self.nodes:
            del self.nodes[node.id]
            self.graph.remove_node(node.id)

            # 更新 name_to_nodes
            for output in node.outputs:
                if output in self.name_to_nodes:
                    self.name_to_nodes[output] = [n for n in self.name_to_nodes[output] if n.id != node.id]
                    if not self.name_to_nodes[output]:
                        del self.name_to_nodes[output]

    def __repr__(self):
        return f"ONNXGraph(nodes={len(self.nodes)}, edges={self.graph.number_of_edges()})"