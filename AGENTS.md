# AGENTS.md - ONNX Optimization Tool

## Project Overview

This is an ONNX graph optimization tool that performs subgraph fusion and operator rewrites. It uses pattern matching with graph traversal and fuses matched subgraphs into custom plugin operators.

### Key Dependencies
- `onnx>=1.15.0` - ONNX model handling
- `onnx-graphsurgeon>=0.4.0` - Graph manipulation and fusion
- `numpy` - Array operations
- `networkx` - Graph data structure for pattern matching

### Core Architecture
```
opt/
├── onnx_optimizer.py      # Main optimizer class
├── config.py              # Config dataclass
├── graph_matcher.py       # Pattern matching engine
├── fusion_executor.py     # Executes fusions on matched patterns
├── onnx_helper/          # ONNX model/graph/node abstractions
├── pattern/               # Pattern definitions (LayerNorm, Attention, etc.)
├── builder/               # Fusion builders (create fused nodes)
├── logger/                # Logging setup
├── utils/                 # Helper utilities
└── tools/                 # Standalone utilities
```

---

## Build/Lint/Test Commands

### Installation
```bash
# Development install with all dependencies
pip install -e ".[dev]"

# Production install
python setup.py bdist_wheel
pip install dist/onnx_opt*.whl
```

### Running the Tool
```bash
# Via module
python -m opt input_model.onnx output_model.onnx

# Via main.py
python main.py input_model.onnx output_model.onnx -l 0  # -l sets log level (0=DEBUG)

# Python API
from opt import ONNXOptimizer, Config
optimizer = ONNXOptimizer(config=Config(allow_overlap=False))
optimizer.load_model("input.onnx")
optimizer.optimize()
optimizer.save_model("output.onnx")
```

### Linting (from setup.py dev dependencies)
```bash
# Flake8
flake8 opt/ --max-line-length=100

# Black formatting
black opt/ --line-length=100
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_layernorm.py

# Run a single test function
pytest tests/test_layernorm.py::test_layernorm_basic

# Run tests matching a pattern
pytest -k "layernorm"

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=opt --cov-report=html
```

---

## Code Style Guidelines

### Python Version
- **Minimum**: Python 3.8
- **Target**: Python 3.8-3.10

### Formatting
- **Line length**: 100 characters (enforced by flake8)
- **Indentation**: 4 spaces (avoid tabs)
- **Black** is used for code formatting

### Imports
Organize imports in three sections separated by blank lines:
1. Standard library (`import logging`, `from typing import`, `from abc import`)
2. Third-party packages (`import onnx`, `import numpy as np`, `import onnx_graphsurgeon as gs`)
3. Local/relative imports (`from .pattern import`, `from ..onnx_helper import`)

Example:
```python
import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import onnx_graphsurgeon as gs

from .base_pattern import Pattern, MatchResult
from ..onnx_helper import ONNXNode, ONNXGraph
```

### Type Hints
- Use `Optional[X]` instead of `X | None` for Python 3.8 compatibility
- Use `X | Y` syntax only if targeting Python 3.10+
- Always include return types for public methods
- Use `List`, `Dict`, `Set` from `typing` (not built-in generics)

```python
def load_model(self, onnx_path: str) -> bool:
    ...

def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
    ...
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `ONNXOptimizer`, `LayerNormPattern`, `MatchResult`)
- **Functions/methods**: `snake_case` (e.g., `load_model`, `fuse_layernorm`, `get_successors`)
- **Variables**: `snake_case` (e.g., `matched_nodes`, `output_shape`, `scale_array`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `LOG_LEVEL_MAP`)
- **Private members**: Prefix with `_` (e.g., `_name`, `_priority`)
- **Module-level variables**: `snake_case`

### Dataclasses
Use `@dataclass` for data containers with `field(default_factory=...)` for mutable defaults:

```python
@dataclass
class MatchResult:
    pattern: Pattern
    matched_nodes: List[ONNXNode]
    node_ids: Set[str] = field(init=False)  # Calculated in __post_init__
    node_names: Set[str] = field(init=False)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_ids = {node.id for node in self.matched_nodes}
```

### Abstract Base Classes
Use `ABC` and `@abstractmethod` for interface definitions:

```python
from abc import ABC, abstractmethod

class Pattern(ABC):
    @property
    def name(self) -> str:
        return self._name
    
    @abstractmethod
    def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
        NotImplemented
```

### Pattern Registration
Patterns use a decorator-based registration system:

```python
@Pattern.register()  # Registers automatically in Pattern.REGISTER_PATTERNS
class LayerNormPattern(Pattern):
    def __init__(self):
        super().__init__(name="LayerNormPattern", priority=10)
        self.add_constraint(OpTypeConstraint("ReduceMean"))

    def match(self, node: ONNXNode, graph: ONNXGraph) -> Optional[List[ONNXNode]]:
        # Return MatchResult on success, None on failure
        ...
```

### Error Handling
- Use logging for errors: `logger.error("message")`
- Return `False` or `None` on failure, `True` or valid object on success
- Use assertions sparingly: `assert self.output_shape, f"tensor shape is {self.output_shape}"`
- Raise `ValueError` or `TypeError` for programming errors

```python
def get_initializer_by_name(self, name: str) -> Optional[np.ndarray]:
    for initializer in self.graph_proto.initializer:
        if initializer.name == name:
            return self.initializer2array(initializer)
    return None
```

### Logging
Use module-level loggers with `__name__`:

```python
import logging

logger = logging.getLogger(__name__)

class SomeClass:
    def some_method(self):
        logger.debug("Detailed info for debugging")
        logger.info("Normal operational info")
        logger.warning("Something unexpected but handled")
        logger.error("Something failed")
```

### Docstrings
Use docstrings for:
- Module-level documentation
- Class docstrings (including ASCII diagrams for patterns)
- Public method docstrings with Args/Returns sections

```python
class LayerNormPattern(Pattern):
    '''  
        ---ReduceMean --     Pow - ReduceMean - Add - Sqrt                            |
    /                 \  /                               \                            |
    Input                   Sub                                Div - (Mul - Add) - Output |
    \                 /  \                               /                            |
        ----------------     -----------------------------                            |
    '''
```

### Comments
- Use inline comments sparingly, only when code is non-obvious
- Chinese comments are acceptable (project uses mixed English/Chinese)
- Avoid commented-out code; use version control instead

### ONNX-Specific Patterns

#### Node Matching
```python
# Check node type
if node.is_op("ReduceMean"):
    ...

# Get successors/predecessors
successors = graph.get_successors(node)
predecessors = graph.get_predecessors(node)

# Check tensor constants
if graph.is_constant_input(tensor_name):
    value = graph.get_initializer_by_name(tensor_name)
```

#### GraphSurgeon Fusion Builders
```python
@gs.Graph.register()  # Register as method on gs.Graph
def fuse_layernorm(self, match_result: MatchResult):
    tensors = self.tensors()
    inputs = tensors.get(input_name)
    
    # Remove old node connections
    for outp in inputs.outputs[::]:
        if outp.name in match_result.node_names:
            inputs.outputs.remove(outp)
    
    # Create new fused node
    ln_node = self.layer(op="NvLayerNormPlugin", ...)
    return ln_node
```

---

## File Organization

### Pattern Files (`opt/pattern/`)
Each pattern has:
1. `pattern/<name>.py` - Pattern class with `@Pattern.register()` decorator
2. `builder/<name>.py` - Fusion builder with `@gs.Graph.register()` decorator

### __init__.py Exports
```python
# opt/pattern/__init__.py
from .base_pattern import *
from .layernorm import *
# etc.

# opt/__init__.py
from .onnx_optimizer import ONNXOptimizer 
from .config import Config
from .pattern import *
```

### Adding a New Pattern
1. Create `opt/pattern/<name>.py` with Pattern class
2. Create `opt/builder/<name>.py` with fusion builder
3. Update `opt/pattern/__init__.py` with new import
4. Update `opt/builder/__init__.py` with new import
5. Add `elif` branch in `fusion_executor.py` `execute()` method
