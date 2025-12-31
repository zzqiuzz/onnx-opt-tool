import argparse
from opt import ONNXOptimizer, Config
from opt.logger import setup_global_logging

logger = setup_global_logging()
logger.info("===== GO =====")

def main():
    parser = argparse.ArgumentParser(description="Optimize an ONNX model and save the result.")
    parser.add_argument("input_model", help="Path to input ONNX model to optimize")
    parser.add_argument("output_model", help="Path where the optimized ONNX model will be saved")
    args = parser.parse_args()

    config = Config(
        allow_overlap=False,
        log_level=10,  # DEBUG级别
        visualize=False
    )
    
    optimizer = ONNXOptimizer(config=config)
 
    if not optimizer.load_model(args.input_model):
        logger.error(f"Failed to load model: {args.input_model}")
        return

    if optimizer.optimize():
        if optimizer.save_model(args.output_model):
            logger.info(f"Optimized model saved to: {args.output_model}")
        else:
            logger.error(f"Failed to save optimized model to: {args.output_model}")
    else:
        logger.info("Optimization failed.")
        
    # TODO
    # compare outputs of optimized onnx with previous one
    # to be ingrated with onnx_infer

if __name__ == "__main__":
    main()