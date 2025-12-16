from onnx import NodeProto
from typing import Dict, Any, List, Optional

class ONNXNode:
    def __init__(self, node_proto: NodeProto):
        self.proto = node_proto
        self.name = node_proto.name
        self.id = id(node_proto)  # 使用内存地址作为唯一ID
        self.op_type = node_proto.op_type
        self.inputs = list(node_proto.input)
        self.outputs = list(node_proto.output)
        self.attrs = self._parse_attrs()

    def _parse_attrs(self) -> Dict[str, Any]:
        attrs = {}
        for attr in self.proto.attribute:
            # 简化版属性解析，仅处理常见类型
            if attr.HasField('f'):
                attrs[attr.name] = attr.f
            elif attr.HasField('i'):
                attrs[attr.name] = attr.i
            elif attr.HasField('s'):
                attrs[attr.name] = attr.s.decode('utf-8')
            elif attr.HasField('t'):
                attrs[attr.name] = attr.t
            elif attr.floats:
                attrs[attr.name] = list(attr.floats)
            elif attr.ints:
                attrs[attr.name] = list(attr.ints)
        return attrs

    def get_attr(self, name: str, default: Any = None) -> Any:
        return self.attrs.get(name, default)

    def is_op(self, op_type: str) -> bool:
        return self.op_type == op_type
    
    def has_intersection(self, list1: List) -> bool:
        return bool(set(self.inputs) & set(list1)) or False

    def __repr__(self):
        return f"ONNXNode(id={self.id}, op={self.op_type}, inputs={self.inputs}, outputs={self.outputs})"
    