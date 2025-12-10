from opt import ONNXOptimizer, ConvBNPattern, ConvReLUPattern, ConvTransBNPattern, Config
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
    onnx_path = "/home/uto/workspace/taijia_workspace/LG_HDT_nn_model_zoo/front_lane/0930_v71_hdt_iter_120000_fusion.onnx"
    if not optimizer.load_model(onnx_path):
        logger.info("Failed to load model.")
        return

    # 添加融合模式 后面改为全局注册
    patterns = [
        # ConvBNPattern(), 
        # ConvReLUPattern(),
        ConvTransBNPattern()
    ]
    optimizer.add_patterns(patterns)
    if optimizer.optimize(iterations=1):
        output_path = "/home/uto/workspace/taijia_workspace/LG_HDT_nn_model_zoo/front_lane/0930_v71_hdt_iter_120000_fusion_optimized_model.onnx"
        # output_path = "optimized_model.onnx"
        optimizer.save_model(output_path)
        logger.info(f"Optimized model saved to: {output_path}")
    else:
        logger.info("Optimization failed.")
        
    # TODO
    # compare outputs of optimized onnx with previous one

if __name__ == "__main__":
    main()