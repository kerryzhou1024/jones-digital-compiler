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
from digital_compiler import JonesProblem, compare_circuits

problem = JonesProblem(
    "s1^2",
    closure="trace",
    strands=2,
    k=5,
)

exact = problem.evaluate()
real_level_3 = problem.circuit(part="real", circuit_level=3, measure=True)
all_levels = problem.compile("10", "real")

real_level_3.display(title="Hopf real — Level 3", max_lines=12)
real_level_3.info()  # rich table in a notebook

print(real_level_3.info())          # deterministic terminal table
resources = real_level_3.info().as_dict()
```

The façade exposes all valid AJL paths and the subset required by its closure:

```python
print(problem.valid_paths)
print(problem.evaluation_paths)

for compiled in problem.circuits(
    path="10",
    part="real",
    circuit_level="all",
):
    print(compiled.circuit_level, compiled.depth())
```

`circuit()` and `circuits()` return `CompiledCircuit` records. They print as
Qiskit circuit drawings, display themselves in notebooks with `.display()`,
forward common circuit attributes, and expose the raw Qiskit object as
`.circuit`.

`CompiledCircuit.info()` describes that one compiler-level logical circuit.
It selects the gate-family vocabulary from the recorded compiler level,
separates measurements from quantum gate totals, and retains exact Qiskit
operation counts and depths in `exact_gate_stats` and `as_dict()`.

When `path` is omitted from a singular `circuit()` call, the façade uses
`problem.valid_paths[0]`. Omitting `path` from plural `circuits()` still selects
the complete closure workload.

In a notebook, display the complete workload without retaining it:

```python
from digital_compiler.notebook import show_problem_circuits

show_problem_circuits(
    problem,
    title="Trace Hopf",
    path="10",
    part="real",
    circuit_level="all",
)
```

`AJLPathModel`, `AJLCompiler`, and `AJLJonesEvaluator` remain available as the
advanced reusable engines underneath this API.

Compare compatible or intentionally different circuits without inventing a
normalized cost model:

```python
trace_circuit = problem.circuit("10", "real", circuit_level=3)
plat_problem = JonesProblem(
    "s2^2",
    closure="plat",
    strands=4,
    k=5,
    writhe=2,
)
plat_circuit = plat_problem.circuit("1010", "real", circuit_level=3)

comparison = compare_circuits(
    {
        "Trace |10>": trace_circuit,
        "Plat |1010>": plat_circuit,
    }
)
comparison              # rich side-by-side notebook table
print(comparison)        # deterministic terminal table
comparison.as_dict()    # structured research data
```

These are pre-transpilation, single-circuit logical resources. They are not
physical-qubit, surface-code, or total closure-workload estimates. Level 3
retains its detailed gate-family view and still contains arbitrary rotations.

To see every compiler level of one Hadamard-test component side by side, use
`compilation_info`. It is the `AJLCompiler` counterpart to `compare_circuits`
and builds on the same `CircuitInfo` reports, so there is a single reporting
path:

```python
from digital_compiler import AJLCompiler, AJLPathModel, compilation_info

compilation = AJLCompiler(AJLPathModel(2, 5)).compile_hadamard_test("s1^2", "10")
report = compilation_info(compilation)

report                                  # side-by-side notebook table
print(report)                           # deterministic terminal table
report.as_dict()                        # structured research data

levels = dict(report.reports)
levels["Level 2"].quantum_gate_count
levels["Level 2"].compiler_policies["prefix_height_loads"]
```

A bare compilation carries no closure or AJL root — those are evaluation
concepts supplied by `JonesProblem` — so those identity fields read `n/a`.
Use `JonesProblem` when the report should describe a closure workload.

## Level 4 Clifford+T

Level 4 is opt-in because arbitrary rotations require an explicit approximation
budget:

```python
from digital_compiler import CliffordTConfig, CompilerConfig, JonesProblem

config = CompilerConfig(
    level4=CliffordTConfig(
        synthesis_error_budget=1e-3,
        optimization_level=2,
        seed_transpiler=0,
    )
)
problem = JonesProblem("s1^2", strands=2, k=5, config=config)

level_4 = problem.circuit("10", "real", circuit_level=4)
print(level_4.info())

approximate = problem.evaluate(circuit_level=4)
print(approximate.value)
print(approximate.value_synthesis_error_bound)
```

The compiler translates Level 3 through an exact Clifford+Rz boundary, replaces
exact pi/4 rotations, uniformly allocates the circuit synthesis budget across
the remaining Rz rotations, and uses Qiskit's Gridsynth-backed Rz synthesis.
Level 4 reports the `Clifford` and `T` families together with exact gate
statistics, T count and depth, T-layer widths, and synthesis provenance.
It remains a logical circuit: physical routing, surface-code cycles, magic-state
factories, and physical-qubit estimates are downstream concerns.

## Trace and Plat Evaluation

Statevector trace evaluation enumerates the complete valid path basis and
returns `path_estimates` and `path_amplitudes`. Shot-based trace evaluation uses
AJL endpoint-weighted path sampling instead:

```python
sampled = problem.evaluate(
    method="shots",
    shots=10_000,
    success_probability=0.99,
    seed=7,
)

print(sampled.value)
print(sampled.path_sampling)  # "ajl_weighted"
print(sampled.total_shots)    # 20_000: 10_000 per complex component
print(sampled.trace_samples)
print(sampled.value_additive_error_bound)
```

Alternatively, request an absolute additive error on the returned Jones value
and let the evaluator choose the shots:

```python
error_driven = problem.evaluate(
    method="shots",
    target_additive_error=0.1,
    success_probability=0.99,
    seed=7,
)

print(error_driven.shots_per_component)
print(error_driven.total_shots)  # twice shots_per_component
print(error_driven.value_additive_error_bound)  # <= 0.1
```

Duplicate sampled paths are grouped into variable-shot Qiskit jobs.
`circuit_count` is the number of unique path/component circuits, while
`total_shots` is always twice `shots_per_component` for a complex estimate. A
sampled trace does not build a complete amplitude table, so its `path_estimates`
and `path_amplitudes` are `None`. Pass `path_seed` to control path selection
independently when supplying a custom sampler. Target-error selection and
reported additive bounds also apply to shot-based plat evaluation.

For `M` shots per real/imaginary component and success probability `p`, the
normalized complex bound is
`sqrt(4 * log(4 / (1 - p)) / M)`. At the default `p=0.75`, this is
`sqrt(16 * log(2) / M)`. If the closure normalization is `C`, the Jones-value
bound is `abs(C)` times the normalized bound, and a requested Jones-value error
`E` selects
`ceil(4 * abs(C)**2 * log(4 / (1 - p)) / E**2)` shots per component.

The AJL paper states its trace guarantee as normalized epsilon times
`d**(strands - 1)`. `target_additive_error` is instead the absolute statistical
tolerance on the returned Jones value, so the evaluator performs that
normalization conversion. `shots` and `target_additive_error` are mutually
exclusive. `success_probability` may be used with explicit, default, or
error-derived shots.

These Hoeffding bounds assume ideal independent sampling. They exclude hardware
noise, correlations introduced by a custom sampler, floating-point roundoff,
and model error. Level-4 synthesis error remains separate in
`value_synthesis_error_bound`; it is not consumed from
`target_additive_error`.

Select a plat closure to evaluate only `|1010...10>`:

```python
from digital_compiler import JonesProblem

plat = JonesProblem(
    "s1",
    strands=2,
    k=5,
    closure="plat",
    writhe=-1,
)

print(plat.evaluate().value)
print(plat.reference_value())
```

Plat closure requires an even number of strands. Supply the writhe of the
chosen oriented plat diagram explicitly: unlike the standard trace orientation,
it need not equal the braid word's exponent sum. The normalization follows the
[AJL plat-closure formula](../references/literature/0511096v2.pdf).

The default Level 3 policy uses exact no-ancilla MCX lowering through recursive
phase decomposition and Gray-code multiplexed rotations. Use
`Level3Policy(mcx=CleanAncillaMCX())` to opt into the clean-ancilla Toffoli
ladder.

Prefix heights use rolling routing by default. A live height selector moves
between consecutive generator indices using only the intervening path steps.
The selector stores `height - 1`, requiring `ceil(log2(k - 1))` qubits for the
`k - 1` valid AJL vertices. Thus `k=5` uses two selector qubits per lane, and
the clean all-zero register already represents vertex 1. Use the former
compute/apply/uncompute behavior as an explicit resource baseline:

```python
from digital_compiler import CompilerConfig, RecomputePrefixHeight

baseline = CompilerConfig(prefix_height=RecomputePrefixHeight())
```

## Parallel Generator Scheduling

Scheduling is independently hot-swappable. The default
`SerialGeneratorScheduling` preserves the existing low-width circuit. Use
`CommutingLayerScheduling` to execute pairwise-distant generators in coherent
layers:

```python
from digital_compiler import (
    CommutingLayerScheduling,
    CompilerConfig,
    JonesProblem,
    TreeControlFanout,
)

config = CompilerConfig(
    scheduling=CommutingLayerScheduling(max_lanes=None),
)
problem = JonesProblem("1 3", strands=4, k=5, config=config)
compilation = problem.compile("1010", "real")

fanout_config = CompilerConfig(
    scheduling=CommutingLayerScheduling(max_lanes=None),
    control_distribution=TreeControlFanout(),
)
```

`max_lanes=None` reserves up to `strands // 2` lanes; a positive integer caps
the width. Each lane has a separate prefix-height selector. Controlled circuits
share one experiment control by default, so no control ancillas are required.
Use `TreeControlFanout` to trade one clean ancilla per additional lane for
disjoint controls prepared by a logarithmic-depth CNOT tree. Clean-ancilla MCX
workspace remains partitioned by lane. Registers are reused between layers and
are uncomputed after use. Rolling prefix-height routing matches live selectors
to the next layer by minimum movement distance, cleans unmatched lanes before
the next layer, and cleans every lane after the final layer.

Only generators satisfying `abs(i - j) >= 2` can share a layer, and the
compiler validates that a custom scheduling policy neither loses generators
nor reorders a noncommuting pair. When ready work exceeds `max_lanes`, the
scheduler prioritizes generators on the longest remaining dependency paths.
Prefix-height computation may still limit the depth improvement when
simultaneous generators have overlapping prefixes.

See `../notebooks/parallel-generators.ipynb` for the scheduling and
control-distribution experiments, and `../notebooks/compiler-demo.ipynb` for a
complete walkthrough.
