import os
import math
import numpy as np
import pandas as pd

from tabulate import tabulate  
from opt.tools.analy.util import get_dict_input_data, infer_model_and_save_outputs, \
                    calculate_mse, cosine_similarity


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
    inserted_op_names=[]
) -> list[str]:   
    infer_data = get_dict_input_data(data_path) 
    quant_output_dict = infer_model_and_save_outputs(
        model_path=qdq_onnx_path,
        output_dir=quant_output_dir, 
        input_data=infer_data, 
        dump_data=dump_data
    )
    
    float_output_dict = infer_model_and_save_outputs(
        model_path=float_onnx_path,
        output_dir=float_output_dir, 
        input_data=infer_data, 
        dump_data=dump_data
    ) 
    if quant_output_dict.keys() != float_output_dict.keys():
        raise ValueError(f"Length Not Equal.")

    all_keys = float_output_dict.keys()

    rows = []
    for key in all_keys:
        q_arr = quant_output_dict[key]
        f_arr = float_output_dict[key]
        note = "" 
        q_shape = tuple(q_arr.shape)
        f_shape = tuple(f_arr.shape)
        if np.prod(q_shape) != np.prod(f_shape):
            note = "shape_mismatch_truncated"
        try:
            mse_val = calculate_mse(f_arr, q_arr)
        except Exception:
            mse_val = float("nan")
            note = note or "mse_error"
        try:
            cos_val = cosine_similarity(f_arr, q_arr)
        except Exception:
            cos_val = float("nan")
            note = note or "cos_error"
        rows.append({
            "op_name": key,
            "float_shape": f_shape,
            "quant_shape": q_shape,
            "mse": mse_val,
            "cosine": cos_val,
            "note": note
        })

    df = pd.DataFrame(rows, columns=["op_name", "float_shape", "quant_shape", "mse", "cosine", "note"])
 
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"\nSaved comparison CSV to: {csv_path}")
    
    if show:
        display_df = df.copy()
        display_df["mse"] = display_df["mse"].apply(lambda x: f"{x:.6e}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x))
        display_df["cosine"] = display_df["cosine"].apply(lambda x: f"{x:.6f}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x)) 
        print(tabulate(display_df.values.tolist(), headers=display_df.columns.tolist(), tablefmt="github", showindex=False)) 
     
    df_filtered = df[df["op_name"].isin(inserted_op_names)].copy() 
    numeric_mse = pd.to_numeric(df_filtered["mse"], errors="coerce")
    mse_desc = df_filtered.assign(mse_numeric=numeric_mse).dropna(subset=["mse_numeric"]).sort_values("mse_numeric", ascending=False)
 
    topk = mse_desc.head(topk_mse) 
    if not topk.empty:
        topk_display = topk.copy()
        topk_display["mse"] = topk_display["mse"].apply(lambda x: f"{x:.6e}" if (isinstance(x, (int, float)) and not math.isnan(x)) else str(x))
        print(f"\nTop {len(topk)} tensors with largest MSE:")
        print(tabulate(topk_display.values.tolist(), headers=topk_display.columns.tolist(), tablefmt="github", showindex=False))
 
        topk.to_csv(topk_csv_file_path, index=False)
        print(f"\nSaved top-k largest MSE CSV to: {topk_csv_file_path}")
        topk_names = topk["op_name"].tolist()
        
    else:
        print("\nNo numeric MSE values available to compute top-20.")
        return []
        
    return topk_names



if __name__ == "__main__":  
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    
    quant_output_dir = os.path.join(current_dir, "quant_operator_outputs_black")
    csv_file = os.path.join(current_dir, "result_black.csv")
    qdq_onnx_path = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/normal_quant_80_dump.onnx"  
    qdq_onnx_path = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/trial_blacklist_quant.onnx"  
    float_output_dir = os.path.join(current_dir, "float_operator_outputs"  )                 
    float_onnx_path = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/online_dump.onnx" 
     
    data_path = "/home/uto/workspace/my/Model-Optimizer-0.40.0/examples/bevod_hdt/batch1"
    
    blacklist_op = analyze(float_onnx_path, qdq_onnx_path, data_path, float_output_dir, quant_output_dir, csv_path=csv_file, topk_mse=50)
    print("----", blacklist_op)
    