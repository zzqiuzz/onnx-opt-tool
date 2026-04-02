import os
import gc
import logging
import onnx
import shutil
import numpy as np
import onnx_graphsurgeon as gs


from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def infer_generator(sess, output_name, infer_data_reader):
    per_output_dict = {output_name: []}
    for iter_data in infer_data_reader: 
        output = sess.run(output_name, iter_data)
        yield {output_name: output}
        del output  


def infer_model_and_save_outputs( 
    output_dir: str, 
    sess,
    infer_data_reader,  
    dump_data = False,
): 
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory prepared: {output_dir}")
 
 
    logger.info("\nRunning model inference...")  
    output_names = [out.name for out in sess.get_outputs()] 
    for iter_data in infer_data_reader: 
        iter_output = sess.run(output_names, iter_data) 
        yield iter_output
        del iter_output
    #     per_batch_output.append(output)
        
    # per_batch_output = np.concatenate(per_batch_output, axis=0).astype(per_batch_output[0].dtype)
    
    # if dump_data:
    #     logger.info(f"Saving {per_output_name} outputs to {output_dir}...") 
    #     save_path = os.path.join(output_dir, f"{per_output_name}.npy")
    #     np.save(save_path, per_batch_output)
    #     logger.info(f"  Saved {per_output_name}: shape={per_batch_output.shape}, dtype={per_batch_output.dtype} -> {save_path}")
    # yield per_output_name, per_batch_output
    # del per_batch_output 


def calculate_snr(y_pred: np.ndarray, y_real: np.ndarray, reduction: str='mean') -> np.ndarray:
    """
    用NumPy计算预测数组和真实数组之间的SNR误差（噪声功率/信号功率）
    功能与torch_snr_error完全一致，仅将后端从PyTorch替换为NumPy
    
    Args:
        y_pred (np.ndarray): 预测值数组（任意维度）
        y_real (np.ndarray): 真实值数组（需与y_pred形状一致）
        reduction (str, optional): 结果归约方式，支持'mean'/'sum'/'none'，默认'mean'
    
    Raises:
        ValueError: 数组形状不一致时抛出
        ValueError: 不支持的归约方式时抛出
    
    Returns:
        np.ndarray: 计算得到的SNR误差（标量或数组，取决于reduction）
    """
    # 1. 形状校验：确保预测值和真实值形状完全一致
    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for arrays with different shape. '
                         f'({y_pred.shape} and {y_real.shape})')
    
    # 2. 归一化归约参数：转为小写，避免大小写敏感
    reduction = str(reduction).lower()

    # 3. 维度适配：如果是1维数组，添加批次维度（变为[1, N]）
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=0)
        y_real = np.expand_dims(y_real, axis=0)

    # 4. 展平数组：保留第0维（批次维度），将后续所有维度展平为一维
    # 例如：形状[batch, H, W] → [batch, H*W]
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    y_real_flat = y_real.reshape(y_real.shape[0], -1)

    # 5. 计算功率（核心逻辑）
    # 噪声功率：每个样本所有元素的(pred-real)²之和（按最后一维求和）
    noise_power = np.sum(np.square(y_pred_flat - y_real_flat), axis=-1)
    # 信号功率：每个样本所有元素的real²之和，加1e-7避免除以0
    signal_power = np.sum(np.square(y_real_flat), axis=-1)
    # SNR误差 = 噪声功率 / (信号功率 + 极小值)
    snr = noise_power / (signal_power + 1e-7)

    # 6. 归约处理：根据指定方式返回结果
    if reduction == 'mean':
        return np.mean(snr)
    elif reduction == 'sum':
        return np.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method: {reduction}. Only "mean", "sum", "none" are supported.')


def calculate_snr(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算两个浮点向量/张量的信噪比（Signal-to-Noise Ratio, SNR），单位为分贝（dB）
    核心公式：SNR(dB) = 10 * log10( (信号能量 + ε) / (噪声能量 + ε) )
    其中：信号能量 = ||x||₂² = Σx_i²（原始向量能量），噪声能量 = ||x - y||₂² = Σ(x_i - y_i)²（两向量误差能量）
          ε为防除零微小偏置，避免分母/分子为0导致的数值异常

    参数：
        x: np.ndarray - 原始浮点向量/张量（如量化前的FP32向量），任意维度
        y: np.ndarray - 对比浮点向量/张量（如量化反量化后的FP32向量），需与x同形状
        eps: float - 防止除零的微小偏置，默认1e-8（可根据向量幅值调整为1e-6/1e-9）

    返回：
        float - SNR结果（dB），值越高表示两向量差异越小，量化损失越低
                量化任务中，SNR≥20dB表示噪声占比≤1%，SNR≥30dB为优秀（噪声占比≤0.1%）

    异常：
        ValueError - 若x和y形状不一致时抛出
    """
    # 严格校验输入形状，确保维度/尺寸完全一致，避免计算错误
    if x.shape != y.shape:
        raise ValueError(f"输入向量形状不匹配！原始向量x形状：{x.shape}，对比向量y形状：{y.shape}")
    
    # 多维张量自动展平为1D向量，消除维度干扰（适配1D/2D/3D/4D等任意维度输入）
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # 计算信号能量：原始向量的二范数平方
    signal_energy = np.sum(np.square(x_flat))
    # 计算噪声能量：两向量误差的二范数平方
    noise_energy = np.sum(np.square(x_flat - y_flat))
    
    # 计算SNR（dB），加入eps防止分子/分母为0（如x全0、两向量完全匹配的极端情况）
    snr_db = 10 * np.log10((signal_energy + eps) / (noise_energy + eps))
    
    return snr_db


def min_max_norm(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Min-Max归一化，基于外部传入的最值（保证X/Y归一化参数一致）"""
    if x_max - x_min < 1e-8:  # 避免除零
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

def calc_norm_mse(x: np.ndarray, y: np.ndarray) -> float:
    """
    对单个量化算子：先基于X独立Min-Max归一化，再计算X_norm和Y_norm的MSE
    :param x: 原始浮点张量（FP32），np.ndarray，任意维度
    :param y: 量化反量化张量（如INT8→FP32），np.ndarray，与x形状完全一致
    :return: 归一化后的MSE值，float，非负
    """
    # 步骤1：基于原始张量X计算归一化参数（仅用X，避免Y的量化误差干扰）
    x_min, x_max = x.min(), x.max()
    # 步骤2：用同一套参数归一化X和Y
    x_norm = min_max_norm(x, x_min, x_max)
    y_norm = min_max_norm(y, x_min, x_max)
    # 步骤3：计算归一化后的MSE
    mse = np.mean(np.square(x_norm - y_norm))
    return float(mse)


def calculate_mse(original: np.ndarray, dquant: np.ndarray) -> float:
    if original.shape != dquant.shape:
        raise ValueError(f"orignal tensor shape not equal to that of dquant.")

    squared_errors = np.square(original - dquant)
    mse = np.mean(squared_errors)

    return float(mse)


def calculate_nmse(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算两个浮点向量/张量的归一化均方误差（Normalized MSE, NMSE）
    核心公式：NMSE = ||x - y||₂² / (||x||₂² + ε)
    其中 ||·||₂² 表示二范数的平方，ε为防除零微小偏置

    参数：
        x: np.ndarray - 原始浮点向量/张量（如量化前的FP32向量），任意维度
        y: np.ndarray - 对比浮点向量/张量（如量化反量化后的FP32向量），需与x同形状
        eps: float - 防止分母为0的微小偏置，默认1e-8（可根据需求调整为1e-6/1e-9）

    返回：
        float - 归一化均方误差结果，值越接近0表示两个向量差异越小

    异常：
        ValueError - 若x和y形状不一致时抛出
    """
    # 输入形状校验，确保两个向量维度/尺寸完全一致
    if x.shape != y.shape:
        raise ValueError(f"输入向量形状不匹配！x形状：{x.shape}，y形状：{y.shape}")
    
    # 将多维张量展平为1D向量，消除维度干扰（适配任意维度输入）
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # 计算误差的二范数平方：||x - y||₂² = Σ(x_i - y_i)²
    error_squared_norm = np.sum((x_flat - y_flat) ** 2)
    # 计算原始向量的二范数平方：||x||₂² = Σx_i²
    x_squared_norm = np.sum(x_flat ** 2)
    
    # 计算NMSE，加入eps防止分母为0（如x全为0的极端情况）
    nmse = error_squared_norm / (x_squared_norm + eps)
    
    return nmse


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


def calculate_kl_divergence(float_data, quant_data, min_value, max_value, bins=50, epsilon=1e-10):
    """
    计算浮点向量与量化向量的KL散度（D_KL(P||Q)，P=浮点分布，Q=量化分布）
    基于直方图将向量转换为概率分布，统一[-20,20]边界和分箱数，处理零概率坑点
    :param float_data: 浮点向量（list/numpy.ndarray，一维），基准分布P
    :param quant_data: 量化向量（list/numpy.ndarray，一维），对比分布Q
    :param bins: 直方图分箱数，与绘制直方图的bins一致，默认50
    :param range_lim: 数值边界，固定为[-20,20]，与直方图一致
    :param epsilon: 极小平滑值，解决Q=0的除零/对数无意义问题，默认1e-10
    :return: kl_div: 计算得到的KL散度值（非负浮点数）
    """
    # 步骤1：将向量转换为numpy数组，兼容list输入
    float_arr = np.asarray(float_data)
    quant_arr = np.asarray(quant_data)
    
    # 步骤2：核心——基于相同分箱/边界，计算两个向量的直方图频次（与直方图绘制逻辑完全一致）
    # normed=True已废弃，改用density=True表示概率密度→归一化为总和为1的概率分布
    p_freq, _ = np.histogram(float_arr, bins=bins, range=(min_value, max_value), density=True)
    q_freq, _ = np.histogram(quant_arr, bins=bins, range=(min_value, max_value), density=True)
    
    # 步骤3：零概率平滑处理（KL散度计算的核心坑点修复）
    # 给P、Q各加极小值，再重新归一化，保证：1. Q无0值 2. 仍为合法概率分布（总和为1）
    p_smooth = p_freq + epsilon
    q_smooth = q_freq + epsilon
    p_prob = p_smooth / np.sum(p_smooth)  # 基准概率分布P
    q_prob = q_smooth / np.sum(q_smooth)  # 对比概率分布Q
    
    # 步骤4：计算KL散度（按离散公式，使用自然对数np.log，也可改用np.log2得到比特单位）
    kl_div = np.sum(p_prob * np.log(p_prob / q_prob))
    
    return kl_div


def get_dict_input_data(data_path: str) -> dict:
    data = {}
    for calib_file_name in os.listdir(data_path):
        calib_file_path = os.path.join(data_path, calib_file_name)
        np_data = np.load(calib_file_path)
        for key, value in np_data.items():
            data[key] = value

    return data



def get_onnx_nodes_names(model_path):
    model = onnx.load(model_path)
    graph_nodes = model.graph.node 
    initializer_names = {init.name for init in model.graph.initializer} 
    operator_names = list({
        node.name for node in graph_nodes
        if node.op_type != "Constant" and node.name not in initializer_names
    })
    return operator_names
  
def insert_op_output(
    model_path: str, 
    dump_model_path: str,
    op_type = None,
    insert_node_names = None
):
    if op_type and insert_node_names:
        raise NotImplementedError(
            f"Currently can't support insert output of specified  operator concurrently according to op_type and insert_node_names."
        )

    onnx_model_proto = onnx.load(model_path)
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model_proto)  #
    graph = gs.import_onnx(inferred_model)
    graph.cleanup().toposort()
    dump_output_names = {} 
    for node in graph.nodes: 
        if insert_node_names:
            if node.name in insert_node_names:
                output = node.outputs[0]  
                if output not in graph.outputs: 
                    output.name = node.name
                    dump_output_names[node.name] = output
        # else:
        #     assert op_type 
        #     for node_input in node.inputs:
        #         pre_nodes = list(node_input.inputs)
        #         if any(pre_node.op in op_type for pre_node in pre_nodes): 
        #             output = node.outputs[0] 
        #             dump_var_name = f"{node.name}" 
        #             output.name = dump_var_name
        #             if output not in graph.outputs:
        #                 dump_output_names.append(output)
        #             break 
    sorted_items = sorted(dump_output_names.items(), key=lambda x: insert_node_names.index(x[0]))
  
    graph.outputs += list(dict(sorted_items).values())  
    graph.cleanup().toposort()    
    saved_onnx = gs.export_onnx(graph)
    saved_onnx.ir_version = 10
    onnx.checker.check_model(saved_onnx)
    onnx.save(saved_onnx, dump_model_path)  
     
  
    
if __name__ == "__main__": 
    float_onnx = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/online.onnx"
    float_dump_onnx = (
        "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/online_dump.onnx"
    )

    quant_onnx = (
        "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/normal_quant_80.onnx"
    )
    quant_dump_onnx = (
        "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/normal_quant_80_dump.onnx"
    )
    inserted_node_names = insert_op_output(
        quant_onnx, quant_dump_onnx, op_type=["DequantizeLinear"]
    )
    insert_op_output(float_onnx, float_dump_onnx, insert_node_names=inserted_node_names)
