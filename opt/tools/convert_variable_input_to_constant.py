import onnx_graphsurgeon as gs
import onnx
import numpy as np
import onnxoptimizer

def convert_inputs_to_constants(
    model_path: str,
    output_path: str,
    input_constants: dict,  # {input_name: numpy_array},
    unused_output : list
):
    """批量转换多个输入为常量"""
    
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    
    # 为每个输入创建常量节点
    for input_name, const_value in input_constants.items():
        const_node = gs.Constant(
            name=f"constant_{input_name}",
            values=const_value
        )
        
        # 替换使用该输入的地方
        for node in graph.nodes:
            for i, inp in enumerate(node.inputs):
                if inp.name == input_name:
                    node.inputs[i] = const_node
    
    # 从输入中移除已转换的
    graph.inputs = [
        inp for inp in graph.inputs 
        if inp.name not in input_constants.keys()
    ]
    
    for output in graph.outputs[::]:
        if output.name in unused_output:
            graph.outputs.remove(output)
            
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()
    optimized_model = gs.export_onnx(graph)
    # optimized_model = onnxoptimizer.optimize(optimized_model)
    onnx.save(optimized_model, output_path)


def verify_ogs_model(output_onnx_path: str):
    """验证修改后的模型"""
    import onnxruntime as ort
    sess = ort.InferenceSession(output_onnx_path)
    remaining_inputs = [(inp.name, inp.shape, inp.type) for inp in sess.get_inputs()]
    print(f"模型剩余输入（已移除常量输入）：")
    for name, shape, dtype in remaining_inputs:
        print(f"  - {name}: shape={shape}, dtype={dtype}")


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 配置路径和常量映射
    INPUT_ONNX = "/home/uto/workspace/my/onnx-opt-tool/onnx/online.onnx"    # 原始模型路径
    OUTPUT_ONNX = "/home/uto/workspace/my/onnx-opt-tool/onnx/online_const_ogs.onnx"       # 修改后模型路径
    
    infer_data_path = "/home/uto/workspace/demos/bevod/res50_8b_online_inputs_0_0208.npz"
    reserved_img_input_name = ["img", "intrinsic", "img2lidar"]
    np_data = np.load(infer_data_path)
    CONST_MAPPING = {}
    for key, value in np_data.items():
        if key in reserved_img_input_name:
            continue 
        CONST_MAPPING[key] = value
    
    unused_output = ["mem_embedding",
                     "mem_timestamp",
                     "mem_egopose",
                     "mem_ref_point",
                     "mem_velo",
                     "outs_dec"]

    convert_inputs_to_constants(
        INPUT_ONNX,
        OUTPUT_ONNX,
        CONST_MAPPING,
        unused_output
    )

 