import onnx
from onnx import helper
import copy
from typing import Dict, List

def make_initializers_unique(onnx_model_path: str, output_model_path: str):
    """
    将ONNX模型中每个算子的initializer替换为独有的副本
    
    Args:
        onnx_model_path: 输入ONNX模型路径
        output_model_path: 输出修改后模型的路径
    """
    # 1. 加载模型并检查有效性
    model = onnx.load(onnx_model_path) 
    graph = model.graph

    # 2. 提取原始initializer，建立名称到张量的映射
    original_initializers: Dict[str, onnx.TensorProto] = {}
    for init in graph.initializer:
        original_initializers[init.name] = init

    # 3. 存储新的initializer（每个算子独有的）
    new_initializers: List[onnx.TensorProto] = []

    # 4. 遍历每个节点，处理其使用的initializer输入
    for node_idx, node in enumerate(graph.node):
        # 遍历节点的每个输入
        for input_idx, input_name in enumerate(node.input):
            # 检查该输入是否是initializer
            if input_name in original_initializers:
                # 生成唯一的新名称：原名称 + _node_ + 节点索引 + _input_ + 输入索引
                # （也可以用节点名，避免索引重复，但节点名可能含特殊字符）
                new_init_name = f"{input_name}_node_{node_idx}_input_{input_idx}"
                
                # 复制原始initializer，修改名称为新名称（深拷贝避免引用问题）
                original_init = original_initializers[input_name]
                new_init = copy.deepcopy(original_init)
                new_init.name = new_init_name
                
                # 将新initializer加入列表
                new_initializers.append(new_init)
                
                # 修改节点的输入为新的initializer名称
                node.input[input_idx] = new_init_name

    # 5. 替换模型的initializer为新的独有版本
    del graph.initializer[:]  # 清空原有initializer
    graph.initializer.extend(new_initializers)  # 添加新的initializer

    # 6. 保存修改后的模型，并验证合法性
    onnx.save(model, output_model_path)
    # 验证修改后的模型 

if __name__ == "__main__": 
    INPUT_MODEL = "/home/uto/workspace/taijia_workspace/bev_od_orin_work/OneKeyPAT/workdir/aiv_mw_1224/model/online_opt_named.onnx"
    OUTPUT_MODEL = "/home/uto/workspace/taijia_workspace/bev_od_orin_work/OneKeyPAT/workdir/aiv_mw_1224/model/online_opt_named_initializer.onnx"
    
    make_initializers_unique(INPUT_MODEL, OUTPUT_MODEL)
    
     