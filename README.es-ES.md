# onnx-opt-tool

Una pequeña utilidad para optimizar gráficos ONNX. Realiza fusiones comunes de subgráficos y reescrituras de operadores para mejorar el rendimiento de la inferencia o integrar con runtimes personalizados.

## Características
- Fusar subgráficos LayerNorm compuestos por múltiples operadores pequeños en `NvLayerNormPlugin`
- Fusar subgráficos relacionados con atención/FFN en `CustomFFAttn`
- Reescribir `log(A/B)` como `log(A) - log(B)`
- Fusar MatMul+Add+(Relu) en `MatMulPlugin`
- Simplificación automática de onnxsim antes de optimizaciones personalizadas

## Instalación
1. Clonar el repositorio:
```bash
git clone <gitlab-url>
cd onnx-opt-tool
```

2. Construir una rueda:
```bash
python setup.py bdist_wheel
```

3. Instalar la rueda generada (ejemplo):
```bash
pip install dist/onnx_opt*.whl
```

## Uso
Ejecuta el optimizador desde la línea de comandos para optimizar un modelo ONNX:
```bash
python -m opt input_model.onnx output_model.onnx
```

Ejemplo:
```bash
python -m opt ./models/resnet.onnx ./models/resnet_opt.onnx
```

### Opciones
- `--exclude_pass`: Excluir pasadas de optimización específicas (p. ej., `--exclude_pass LayerNormPattern`)
- `-l, --log-level`: Establecer nivel de registro (0=DEBUG, 1=INFO, 2=ADVERTENCIA, 3=ERROR)
- `--skip_simplify`: Saltar la simplificación de onnxsim

Puedes llamar de manera fluida a la API:
```python
from opt import ONNXOptimizer
optimizer = ONNXOptimizer()
optimizer.load_model(input_onnx_path)
optimizer.optimize()
optimizer.save_model(output_onnx_path)
```

## Notas
- El optimizador intenta preservar la semántica numérica, pero debes ejecutar pruebas de regresión para escenarios críticos.
- Para agregar o ajustar reglas de fusión, inspecciona los archivos de implementación en el repositorio y envía un PR.

## Licencia & Contribución
Se aceptan problemas y solicitudes de extracción. Por favor, sigue el estilo de codificación y convenciones de prueba del proyecto.
