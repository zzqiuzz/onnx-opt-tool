import os
import logging
import onnx
import shutil
import numpy as np
import onnx_graphsurgeon as gs
import onnxruntime as ort

from typing import Dict, List, Optional 
from pathlib import Path

logger = logging.getLogger(__name__)

def infer_model_and_save_outputs(
    model_path: str,
    output_dir: str,
    input_data: Optional[Dict[str, np.ndarray]] = None,
    input_shape: Optional[Dict[str, List[int]]] = None,
    input_dtype: np.dtype = np.float32,
    seed: int = 42,
    dump_data = False
) -> Dict[str, np.ndarray]: 
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory prepared: {output_dir}")
 
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider']) 
 
    np.random.seed(seed)
    if input_data is None:
        input_data = {} 
        if input_shape is None:
            input_shape = {}
            for input_meta in sess.get_inputs(): 
                shape = []
                for dim in input_meta.shape:
                    if dim is None or isinstance(dim, str):
                        shape.append(1)
                    else:
                        shape.append(dim)
                input_shape[input_meta.name] = shape 
        for input_name, shape in input_shape.items():
            input_data[input_name] = np.random.randn(*shape).astype(input_dtype)
            logger.info(f"Generated random input: {input_name}, shape: {shape}, dtype: {input_dtype}")
    else: 
        for input_name, data in input_data.items():
            logger.info(f"Using custom input: {input_name}, shape: {data.shape}, dtype: {data.dtype}")
 
    logger.info("\nRunning model inference...")
    output_names = [out.name for out in sess.get_outputs()]
    outputs = sess.run(output_names, input_data)
    output_dict = dict(zip(output_names, outputs))
    if dump_data:
        logger.info(f"Saving {len(output_dict)} outputs to {output_dir}...")
        for name, tensor in output_dict.items():
            save_path = os.path.join(output_dir, f"{name}.npy")
            np.save(save_path, tensor)
            logger.info(f"  Saved {name}: shape={tensor.shape}, dtype={tensor.dtype} -> {save_path}")

    return output_dict


def calculate_mse(original: np.ndarray, dquant: np.ndarray) -> float:
    if original.shape != dquant.shape:
        raise ValueError(f"orignal tensor shape not equal to that of dquant.")

    squared_errors = np.square(original - dquant)
    mse = np.mean(squared_errors)

    return float(mse)


def cosine_similarity(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    if vec1.shape != vec2.shape:
        raise ValueError("两个向量的形状必须相同")
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 and norm2 == 0:
        return 1.0
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_dict_input_data(data_path: str) -> dict:
    data = {}
    for calib_file_name in os.listdir(data_path):
        calib_file_path = os.path.join(data_path, calib_file_name)
        np_data = np.load(calib_file_path)
        for key, value in np_data.items():
            data[key] = value

    return data

  
def insert_op_output(
    model_path: str, 
    dump_model_path: str,
    op_type = None,
    insert_node_names = None
) -> list:
    if op_type and insert_node_names:
        raise NotImplementedError(f"Currently can't support insert output of specified  operator concurrently according to op_type and insert_node_names.")
    
    onnx_model_proto = onnx.load(model_path)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model_proto) # 
    graph = gs.import_onnx(inferred_model)
    graph.cleanup().toposort()
    dump_output_names = []
    node_names = [] 
    for node in graph.nodes: 
        if insert_node_names:
            if node.name in insert_node_names:
                output = node.outputs[0] 
                dump_var_name = f"{node.name}" 
                output.name = dump_var_name
                if output not in graph.outputs:
                    dump_output_names.append(output) 
        else:
            assert op_type 
            for node_input in node.inputs:
                pre_nodes = list(node_input.inputs)
                if any(pre_node.op in op_type for pre_node in pre_nodes):
                    node_names.append(node.name)
                    output = node.outputs[0] 
                    dump_var_name = f"{node.name}" 
                    output.name = dump_var_name
                    if output not in graph.outputs:
                        dump_output_names.append(output)
                    break
        
    graph.outputs += dump_output_names  
    graph.cleanup().toposort()    
    saved_onnx = gs.export_onnx(graph)
    onnx.checker.check_model(saved_onnx)
    onnx.save(saved_onnx, dump_model_path)  
    
    return node_names
  
    
if __name__ == "__main__": 
    float_onnx = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/online.onnx"
    float_dump_onnx = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/online_dump.onnx"

    quant_onnx = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/normal_quant_80.onnx"
    quant_dump_onnx = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/normal_quant_80_dump.onnx"
    inserted_node_names = insert_op_output(quant_onnx, quant_dump_onnx, op_type=["DequantizeLinear"])
    insert_op_output(float_onnx, float_dump_onnx, insert_node_names=inserted_node_names)
    