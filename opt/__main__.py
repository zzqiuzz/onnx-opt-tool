import argparse
import onnx
import onnxsim
from opt import ONNXOptimizer, Config
from opt.logger import setup_global_logging


def simplify_onnx(input_path: str, output_path: str) -> bool:
    try:
        model = onnx.load(input_path)
        model_simplified, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_simplified, output_path)
            return True
        else:
            return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Optimize an ONNX model and save the result.")
    parser.add_argument("input_model", help="Path to input ONNX model to optimize")
    parser.add_argument("output_model", help="Path where the optimized model will be saved")
    parser.add_argument("--excluded_pass", nargs='*', default=[], help="List of optimization passes to exclude", choices=["ConvTransBNPattern", "LayerNormPattern", "CustomAttnPattern", "LogDivPattern", "MatMulAddPattern"])
    parser.add_argument("-l", "--log-level", type=int, default=1, help="Log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)")
    parser.add_argument("--skip_simplify", action="store_true", help="Skip onnxsim simplification")
    args = parser.parse_args()

    logger = setup_global_logging(log_level=args.log_level)
    logger.info("===== GO =====")

    if not args.skip_simplify:
        temp_path = args.output_model + ".simplified.onnx"
        logger.info("Running onnxsim simplification...")
        if simplify_onnx(args.input_model, temp_path):
            input_path = temp_path
            logger.info("onnxsim simplification done.")
        else:
            input_path = args.input_model
            logger.warning("onnxsim simplification failed, using original input.")

    config = Config(
        allow_overlap=False,
        log_level=10,
        visualize=False,
        excluded_opt_pass=args.excluded_pass
    )

    optimizer = ONNXOptimizer(config=config)

    if not optimizer.load_model(input_path):
        logger.error(f"Failed to load model: {input_path}")
        return

    if optimizer.optimize():
        if optimizer.save_model(args.output_model):
            logger.info(f"Optimized model saved to: {args.output_model}")
        else:
            logger.error(f"Failed to save optimized model to: {args.output_model}")
    else:
        logger.info("Optimization failed.")

    if not args.skip_simplify:
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    main()
