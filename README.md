# Jones Digital Compiler

The `digital_compiler` package is the reusable, policy-driven AJL circuit
compiler used by the Jones-polynomial research notebooks in this repository.

## Installation

From this directory:

```bash
pip install -r requirements.txt
```

For a core-only installation:

```bash
pip install .
```

## Verification

```bash
ruff check digital_compiler tests
pytest
```

## Basic Usage

```python
from digital_compiler import AJLCompiler, AJLPathModel, CompilerConfig

model = AJLPathModel(strands=2, level=5)
compiler = AJLCompiler(model, CompilerConfig())
result = compiler.compile_hadamard_test("s1^2", "10", part="real")
```

The default Level 3 policy uses exact no-ancilla MCX lowering through recursive
phase decomposition and Gray-code multiplexed rotations. Use
`Level3Policy(mcx=CleanAncillaMCX())` to opt into the clean-ancilla Toffoli
ladder.

See `../notebooks/AJL-Compiler-Package-Demo.ipynb` for a complete walkthrough.
