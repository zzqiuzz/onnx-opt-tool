from opt import ONNXOptimizer, ConvBNPattern, Config

def main():
    # 1. 配置优化器
    config = Config(
        allow_overlap=False,
        log_level=10,  # DEBUG级别
        visualize=False
    )

    # 2. 创建优化器实例
    optimizer = ONNXOptimizer(config=config)

    # 3. 加载ONNX模型（替换为你的模型路径）
    model_path = "path/to/your/model.onnx"
    if not optimizer.load_model(model_path):
        print("Failed to load model.")
        return

    # 4. 添加融合模式
    patterns = [
        ConvBNPattern(), 
    ]
    optimizer.add_patterns(patterns)

    # 5. 执行优化（最多2次迭代）
    if optimizer.optimize(iterations=2):
        # 6. 保存优化后的模型
        output_path = "path/to/optimized_model.onnx"
        optimizer.save_model(output_path)
        print(f"Optimized model saved to: {output_path}")
    else:
        print("Optimization failed.")

if __name__ == "__main__":
    main()