# onnx-opt-tool

A small utility for optimizing ONNX graphs. It performs common subgraph fusions and operator rewrites to improve inference performance or integrate with custom runtimes.

## Features
- Fuse LayerNorm subgraphs composed of multiple small operators into `NvLayerNormPlugin`
- Fuse attention/FFN related subgraphs into `CustomFFAttn`
- Rewrite `log(A/B)` as `log(A) - log(B)`
- Other optimizations for common graph patterns

## Installation
1. Clone the repository:
```bash
git clone <gitlab-url>
cd onnx-opt-tool
```

2. Build a wheel:
```bash
python setup.py bdist_wheel
```

3. Install the generated wheel (example):
```bash
pip install dist/onnx_opt*.whl
```

## Usage
Run the optimizer from the command line to optimize an ONNX model:
```bash
python -m opt input_model.onnx output_model.onnx
```

Example:
```bash
python -m opt ./models/resnet.onnx ./models/resnet_opt.onnx
```

You can seemlessly call the api like:
```
from opt import ONNXOptimizer
optimizer = ONNXOptimizer()
optimizer.load_model(input_onnx_path)
optimizer.optimize()
optimizer.save_model(output_onnx_path)
```

## Notes
- The optimizer attempts to preserve numerical semantics, but you should run regression tests for critical scenarios.
- To add or adjust fusion rules, inspect the implementation files in the repository and submit a PR.

## License & Contributing
Issues and pull requests are welcome. Please follow the project's coding style and test conventions.
