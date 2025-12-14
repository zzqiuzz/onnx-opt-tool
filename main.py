from opt import ONNXOptimizer, Config
from opt.logger import setup_global_logging

logger = setup_global_logging()
logger.info("===== GO =====")

def main():
    config = Config(
        allow_overlap=False,
        log_level=10,  # DEBUG级别
        visualize=False
    )
    
    optimizer = ONNXOptimizer(config=config)

    # onnx_path = "/home/uto/workspace/my/onnx-opt-tool/conv_trans_bn_original.onnx"
    onnx_path = "/home/zhengzhe/workspace/uto/onnx-opt-tool/custom_model_with_bn.onnx"
    if not optimizer.load_model(onnx_path):
        logger.info("Failed to load model.")
        return
    if optimizer.optimize(iterations=1):
        output_path = "/home/zhengzhe/workspace/uto/onnx-opt-tool/custom_model_with_bn_optimized_model.onnx"
        # output_path = "optimized_model.onnx"
        optimizer.save_model(output_path)
        logger.info(f"Optimized model saved to: {output_path}")
    else:
        logger.info("Optimization failed.")
        
    # TODO
    # compare outputs of optimized onnx with previous one

if __name__ == "__main__":
    main()