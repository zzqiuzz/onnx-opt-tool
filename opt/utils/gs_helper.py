import onnx_graphsurgeon as gs

def get_node_dict(graph: gs.Graph):
    """
    创建节点名称到节点对象的映射字典
    
    Args:
        graph: ONNX GraphSurgeon 图对象
    
    Returns:
        dict: {node_name: node_object}
    """
    return {node.name: node for node in graph.nodes}

@gs.Graph.register()
def get_nodes_by_op(self, op_type: str) -> dict:
    """
    获取指定操作类型的所有节点，按名称组织
    
    Args:
        op_type: 操作类型，如 "Conv", "BatchNormalization"
    
    Returns:
        dict: {node_name: node_object}
    """
    return {
        node.name: node 
        for node in self.nodes 
        if node.op == op_type
    }

@gs.Graph.register()
def get_node_by_name(self, name: str) -> gs.Node:
    """
    根据名称精确查找节点
    
    Args:
        name: 节点名称
    
    Returns:
        gs.Node: 找到的节点，不存在则返回 None
    """
    for node in self.nodes:
        if node.name == name:
            return node
    return None

@gs.Graph.register()
def find_nodes_by_pattern(self, pattern: str) -> dict:
    """
    根据名称模式模糊查找节点
    
    Args:
        pattern: 名称包含的字符串，如 "resnet/block1"
    
    Returns:
        dict: {node_name: node_object}
    """
    return {
        node.name: node
        for node in self.nodes
        if pattern in node.name
    }