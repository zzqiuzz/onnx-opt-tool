import os
import math
import numpy as np
import pandas as pd
import onnxruntime as ort

from tabulate import tabulate  
from opt.tools.analy.util import (
    get_dict_input_data, 
    infer_model_and_save_outputs,
    calculate_mse, 
    cosine_similarity, 
    calculate_nmse,
    calculate_snr,
    calc_norm_mse
)


def analyze(
    float_onnx_path: str,
    qdq_onnx_path: str,
    data_path: str,
    float_output_dir,
    quant_output_dir,
    dump_data = False,
    csv_path = "",
    topk_csv_file_path = "",
    topk_mse = 10,
    show = True,
    quant_op_names=[]
) -> list[str]:   
    sess = ort.InferenceSession
    float_sess = sess(float_onnx_path, providers=['CUDAExecutionProvider']) 
    infer_data = get_dict_input_data(data_path) 
    input_names = [] 
    input_shape = {}
    for input_meta in float_sess.get_inputs(): 
        shape = []
        for dim in input_meta.shape:
            if dim is None or isinstance(dim, str):
                shape.append(1)
            else:
                shape.append(dim)
        input_shape[input_meta.name] = shape 
        input_names.append(input_meta.name)
        
    n_itr = int(infer_data[input_names[0]].shape[0] / input_shape[input_names[0]][0])  
    infer_data_list = [{} for _ in range(n_itr)] 
    for input_name in input_names:
        for idx, calib_data in enumerate(
            np.array_split(infer_data[input_name], n_itr, axis=0)
        ):  
            infer_data_list[idx][input_name] = calib_data 
             
    float_output_generator = infer_model_and_save_outputs(
        sess=float_sess,
        output_dir=quant_output_dir,  
        dump_data=dump_data,
        infer_data_reader=iter(infer_data_list)
    )
    
    infer_data_reader = iter(infer_data_list) 
    quant_sess = sess(qdq_onnx_path, providers=['CUDAExecutionProvider']) 
    quant_output_generator = infer_model_and_save_outputs(
        sess=quant_sess,
        output_dir=quant_output_dir,  
        dump_data=dump_data,
        infer_data_reader=infer_data_reader
    ) 
    output_names = [out.name for out in quant_sess.get_outputs()] 

    mse_dict = {output_name: [] for output_name in output_names} 
    nmse_dict = {output_name: [] for output_name in output_names} 
    cosine_dict = {output_name: [] for output_name in output_names} 
    snr_dict = {output_name: [] for output_name in output_names} 
    it = 1
    for float_output, quant_output in zip(float_output_generator, quant_output_generator):
        # print(output_name)
        for output_name , float_tensor, quant_tensor in zip(output_names, float_output, quant_output):
            # print(float_tensor.shape, quant_tensor.shape) 
            assert float_tensor.shape == quant_tensor.shape
            mse_error = calculate_mse(float_tensor.flatten(), quant_tensor.flatten()) # 增加下consine的计算 平均即可
            nmse_error = calculate_nmse(float_tensor.flatten(), quant_tensor.flatten()) # 增加下consine的计算 平均即可
            # error = calc_norm_mse(float_tensor.flatten(), quant_tensor.flatten()) # 增加下consine的计算 平均即可
            cosine_sim = cosine_similarity(float_tensor.flatten(), quant_tensor.flatten())
            snr = calculate_snr(float_tensor.flatten(), quant_tensor.flatten())
            # mse_dict[output_name] = (mse_dict[output_name] * (it - 1) + error ) / it
            mse_dict[output_name].append(mse_error)
            nmse_dict[output_name].append(nmse_error)
            cosine_dict[output_name].append(cosine_sim)
            snr_dict[output_name].append(snr)
        print(f"processing iter_{it} done...")
        it += 1
    for output_name, errors in mse_dict.items():
        mse_dict[output_name] = sum(errors) / len(errors)
    for output_name, errors in nmse_dict.items():
        nmse_dict[output_name] = sum(errors) / len(errors)
    for output_name, cosine_sim in cosine_dict.items():
        cosine_dict[output_name] = sum(cosine_sim) / len(cosine_sim)
    for output_name, snr in snr_dict.items():
        snr_dict[output_name] = sum(snr) / len(snr)
    rows = []
    for name, mse, nmse, cosin_sim, snr in zip(output_names, mse_dict.values(), nmse_dict.values(), cosine_dict.values(), snr_dict.values()):
        rows.append({
            "op_name": name,
            "mse": mse,
            "nmse": nmse,
            "cosine_sim": cosin_sim,
            "snr": snr
        })

    df = pd.DataFrame(rows, columns=["op_name", "mse", "nmse", "cosine_sim", "snr"])
 
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"\nSaved comparison CSV to: {csv_path}")

    if show:
        display_df = df.copy()
        display_df["mse"] = display_df["mse"].apply(lambda x: f"{x:.6e}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x))
        display_df["nmse"] = display_df["nmse"].apply(lambda x: f"{x:.6f}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x)) 
        display_df["cosine_sim"] = display_df["cosine_sim"].apply(lambda x: f"{x:.6f}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x)) 
        print(tabulate(display_df.values.tolist(), headers=display_df.columns.tolist(), tablefmt="github", showindex=False)) 
     
    df_filtered = df[df["op_name"].isin(quant_op_names)].copy() 
    
    # numeric_snr = pd.to_numeric(df_filtered["snr"], errors="coerce")
    # snr_desc = df_filtered.assign(snr_numeric=numeric_snr).dropna(subset=["snr_numeric"]).sort_values("snr_numeric", ascending=False)
    # topk_snr = snr_desc.tail(topk_mse) 
    # if not topk_snr.empty:
    #     topk_display = topk_snr.copy()
    #     topk_display["snr"] = topk_display["snr"].apply(lambda x: f"{x:.6e}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x))
    #     print(f"\nTop {len(topk_snr)} tensors with largest snr:")
    #     print(tabulate(topk_display.values.tolist(), headers=topk_display.columns.tolist(), tablefmt="github", showindex=False))
 
    #     topk_snr.to_csv(topk_csv_file_path.replace("mse", "snr"), index=False)
    #     print(f"\nSaved top-k largest snr CSV to: {topk_csv_file_path.replace("mse", "snr")}")
    #     topk_names = topk_snr["op_name"].tolist()
    #     print("-----------topk snr: ", topk_names)
        
    # else:
    #     print("\nNo numeric snr values available to compute top-20.")
    #     return []
    
    
    numeric_cosine_sim = pd.to_numeric(df_filtered["cosine_sim"], errors="coerce")
    cosine_sim_desc = df_filtered.assign(cosine_sim_numeric=numeric_cosine_sim).dropna(subset=["cosine_sim_numeric"]).sort_values("cosine_sim_numeric", ascending=False)
    topk_cosine_sim = cosine_sim_desc.tail(topk_mse) 
    if not topk_cosine_sim.empty:
        topk_display = topk_cosine_sim.copy()
        topk_display["cosine_sim"] = topk_display["cosine_sim"].apply(lambda x: f"{x:.6e}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x))
        print(f"\nTop {len(topk_cosine_sim)} tensors with largest cosine_sim:")
        print(tabulate(topk_display.values.tolist(), headers=topk_display.columns.tolist(), tablefmt="github", showindex=False))
 
        topk_cosine_sim.to_csv(topk_csv_file_path.replace("mse", "cosine_sim"), index=False)
        print(f"\nSaved top-k largest cosine_sim CSV to: {topk_csv_file_path.replace('mse', 'cosine_sim')}")
        topk_names = topk_cosine_sim["op_name"].tolist()
        print("-----------topk cosine_sim: ", topk_names)
        
    else:
        print("\nNo numeric cosine_sim values available to compute.")
        return []
     
     
     
    numeric_nmse = pd.to_numeric(df_filtered["nmse"], errors="coerce")
    nmse_desc = df_filtered.assign(nmse_numeric=numeric_nmse).dropna(subset=["nmse_numeric"]).sort_values("nmse_numeric", ascending=False)
 
    topk = nmse_desc.head(topk_mse) 
    if not topk.empty:
        topk_display = topk.copy()
        topk_display["nmse"] = topk_display["nmse"].apply(lambda x: f"{x:.6e}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x))
        print(f"\nTop {len(topk)} tensors with largest NMSE:")
        print(tabulate(topk_display.values.tolist(), headers=topk_display.columns.tolist(), tablefmt="github", showindex=False))
 
        topk.to_csv(topk_csv_file_path, index=False)
        print(f"\nSaved top-k largest NMSE CSV to: {topk_csv_file_path}")
        topk_names = topk["op_name"].tolist()
        
    else:
        print("\nNo numeric NMSE values available to compute.")
        return []
     
    
    numeric_mse = pd.to_numeric(df_filtered["mse"], errors="coerce")
    mse_desc = (
        df_filtered.assign(mse_numeric=numeric_mse)
        .dropna(subset=["mse_numeric"])
        .sort_values("mse_numeric", ascending=False)
    )

    topk = mse_desc.head(topk_mse)
    if not topk.empty:
        topk_display = topk.copy()
        topk_display["mse"] = topk_display["mse"].apply(
            lambda x: f"{x:.6e}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x)
        )
        print(f"\nTop {len(topk)} tensors with largest MSE:")
        print(
            tabulate(
                topk_display.values.tolist(),
                headers=topk_display.columns.tolist(),
                tablefmt="github",
                showindex=False,
            )
        )

        topk.to_csv(topk_csv_file_path, index=False)
        print(f"\nSaved top-k largest MSE CSV to: {topk_csv_file_path}")
        topk_names = topk["op_name"].tolist()

    else:
        print("\nNo numeric MSE values available to compute.")
        return []

    return topk_names


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    quant_output_dir = os.path.join(current_dir, "quant_operator_outputs_black")
    csv_file = os.path.join(current_dir, "result_black.csv")
    qdq_onnx_path = (
        "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/normal_quant_80_dump.onnx"
    )
    qdq_onnx_path = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/trial_blacklist_quant.onnx"
    float_output_dir = os.path.join(current_dir, "float_operator_outputs")
    float_onnx_path = (
        "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/online_dump.onnx"
    )

    data_path = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/batch1"

    blacklist_op = analyze(
        float_onnx_path,
        qdq_onnx_path,
        data_path,
        float_output_dir,
        quant_output_dir,
        csv_path=csv_file,
        topk_mse=50,
    )
    print("----", blacklist_op)
