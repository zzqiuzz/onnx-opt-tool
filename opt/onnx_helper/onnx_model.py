import onnx
from onnx import ModelProto
from typing import Optional
from .onnx_graph import ONNXGraph
from ..logger import logger

class ONNXModel:
    def __init__(self, model_proto: Optional[ModelProto] = None):
        self.proto = model_proto
        self.graph: Optional[ONNXGraph] = ONNXGraph(model_proto.graph) if model_proto else None

    @classmethod
    def load(cls, path: str) -> 'ONNXModel':
        logger.info(f"Loading ONNX model from {path}")
        model_proto = onnx.load(path)
        return cls(model_proto)

    def save(self, path: str):
        if not self.proto:
            logger.error("Cannot save empty model.")
            return
        logger.info(f"Saving optimized ONNX model to {path}")
        onnx.save(self.proto, path)

    def get_graph(self) -> Optional[ONNXGraph]:
        return self.graph

    def __repr__(self):
        return f"ONNXModel(ir_version={self.proto.ir_version if self.proto else None}, graph={self.graph})"