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

The repository notebooks intentionally import the checkout's `compiler/`
source tree, even if the active environment contains an older non-editable
installation. For ordinary Python sessions, refresh that installation after a
source change from the repository root:

```bash
pip install --no-build-isolation --force-reinstall --no-deps ./compiler
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

## Parallel Generator Scheduling

Scheduling is independently hot-swappable. The default
`SerialGeneratorScheduling` preserves the existing low-width circuit. Use
`CommutingLayerScheduling` to execute pairwise-distant generators in coherent
layers:

```python
from digital_compiler import (
    AJLCompiler,
    AJLPathModel,
    CommutingLayerScheduling,
    CompilerConfig,
)

config = CompilerConfig(
    scheduling=CommutingLayerScheduling(max_lanes=None),
)
compiler = AJLCompiler(AJLPathModel(strands=4, level=5), config)
compilation = compiler.compile_hadamard_test("1 3", "1010")
```

`max_lanes=None` reserves up to `strands // 2` lanes; a positive integer caps
the width. Each lane has a separate prefix-height selector. Controlled circuits
use a logarithmic-depth clean-control fanout tree, and clean-ancilla MCX
workspace is partitioned by lane. Registers are reused between layers and are
uncomputed after use.

Only generators satisfying `abs(i - j) >= 2` can share a layer, and the
compiler validates that a custom scheduling policy neither loses generators
nor reorders a noncommuting pair. Prefix-height computation may still limit the
depth improvement when simultaneous generators have overlapping prefixes.

See `../notebooks/AJL-Compiler-Package-Demo.ipynb` for a complete walkthrough.
