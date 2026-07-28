"""Audited Level-3 to Level-4 Clifford+T lowering."""

from __future__ import annotations

from dataclasses import dataclass

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    OptimizeCliffordT,
    SubstitutePi4Rotations,
    SynthesizeRZRotations,
)

from .lowering import assert_level_3_contract
from .policies import CliffordTConfig

CLIFFORD_GATE_NAMES = frozenset({"h", "s", "sdg", "x", "z", "cx"})
T_GATE_NAMES = frozenset({"t", "tdg"})
CLIFFORD_T_GATE_NAMES = CLIFFORD_GATE_NAMES | T_GATE_NAMES
CLIFFORD_T_BASIS = ("h", "s", "sdg", "x", "z", "cx", "t", "tdg")
CLIFFORD_RZ_BASIS = (*CLIFFORD_T_BASIS, "rz")


@dataclass(frozen=True)
class LogicalResourceReport:
    """Logical resources for one Level-4 Clifford+T circuit."""

    logical_qubits: int
    total_gate_count: int
    total_depth: int
    clifford_count: int
    clifford_depth: int
    cx_count: int
    cx_depth: int
    t_gate_count: int
    tdg_gate_count: int
    t_count: int
    t_depth: int
    t_layer_widths: tuple[int, ...]
    measurement_count: int
    original_rz_count: int
    arbitrary_rotation_count: int
    synthesis_error_budget: float
    per_rotation_error: float | None


@dataclass(frozen=True)
class CliffordTCompilation:
    """A Level-4 circuit and its synthesis audit trail."""

    circuit: QuantumCircuit
    config: CliffordTConfig
    resources: LogicalResourceReport
    qiskit_version: str


def assert_clifford_t_contract(circuit: QuantumCircuit) -> None:
    """Reject operations outside the fixed Level-4 contract."""

    allowed_non_gates = {"measure", "barrier"}
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, Gate):
            if operation.name not in CLIFFORD_T_GATE_NAMES:
                raise AssertionError(
                    f"operation {operation.name!r} violates Level 4"
                )
        elif operation.name not in allowed_non_gates:
            raise AssertionError(
                f"operation {operation.name!r} violates Level 4"
            )


def t_layer_widths(circuit: QuantumCircuit) -> tuple[int, ...]:
    """Return the dependency-aware number of T gates in each T layer."""

    level_by_qubit = {qubit: 0 for qubit in circuit.qubits}
    widths: dict[int, int] = {}
    for instruction in circuit.data:
        qubits = tuple(instruction.qubits)
        if not qubits:
            continue
        level = max(level_by_qubit[qubit] for qubit in qubits)
        if instruction.operation.name in T_GATE_NAMES:
            level += 1
            widths[level] = widths.get(level, 0) + 1
        for qubit in qubits:
            level_by_qubit[qubit] = level

    maximum = max(widths, default=0)
    return tuple(widths.get(level, 0) for level in range(1, maximum + 1))


def logical_resource_report(
    circuit: QuantumCircuit,
    *,
    original_rz_count: int,
    arbitrary_rotation_count: int,
    config: CliffordTConfig,
) -> LogicalResourceReport:
    """Measure one circuit after enforcing the Level-4 contract."""

    assert_clifford_t_contract(circuit)
    gate_names = [
        instruction.operation.name
        for instruction in circuit.data
        if isinstance(instruction.operation, Gate)
    ]
    layers = t_layer_widths(circuit)
    per_rotation_error = (
        None
        if arbitrary_rotation_count == 0
        else config.synthesis_error_budget / arbitrary_rotation_count
    )
    return LogicalResourceReport(
        logical_qubits=circuit.num_qubits,
        total_gate_count=len(gate_names),
        total_depth=circuit.depth(
            lambda instruction: isinstance(instruction.operation, Gate)
        ),
        clifford_count=sum(
            name in CLIFFORD_GATE_NAMES for name in gate_names
        ),
        clifford_depth=circuit.depth(
            lambda instruction: (
                isinstance(instruction.operation, Gate)
                and instruction.operation.name in CLIFFORD_GATE_NAMES
            )
        ),
        cx_count=sum(name == "cx" for name in gate_names),
        cx_depth=circuit.depth(
            lambda instruction: instruction.operation.name == "cx"
        ),
        t_gate_count=sum(name == "t" for name in gate_names),
        tdg_gate_count=sum(name == "tdg" for name in gate_names),
        t_count=sum(name in T_GATE_NAMES for name in gate_names),
        t_depth=len(layers),
        t_layer_widths=layers,
        measurement_count=sum(
            instruction.operation.name == "measure"
            for instruction in circuit.data
        ),
        original_rz_count=original_rz_count,
        arbitrary_rotation_count=arbitrary_rotation_count,
        synthesis_error_budget=config.synthesis_error_budget,
        per_rotation_error=per_rotation_error,
    )


class CliffordTCompiler:
    """Translate Level 3 through Clifford+Rz into Clifford+T."""

    def __init__(self, config: CliffordTConfig):
        self.config = config
        self._canonicalizer = generate_preset_pass_manager(
            optimization_level=config.optimization_level,
            basis_gates=list(CLIFFORD_RZ_BASIS),
            approximation_degree=1.0,
            seed_transpiler=config.seed_transpiler,
        )

    def compile(self, level_3_circuit: QuantumCircuit) -> CliffordTCompilation:
        assert_level_3_contract(level_3_circuit)
        source_metadata = dict(level_3_circuit.metadata or {})

        canonical = self._canonicalizer.run(level_3_circuit)
        original_rz_count = int(canonical.count_ops().get("rz", 0))
        exact_discrete = SubstitutePi4Rotations(
            approximation_degree=1.0
        )(canonical)
        arbitrary_rotation_count = int(
            exact_discrete.count_ops().get("rz", 0)
        )

        if arbitrary_rotation_count:
            per_rotation_error = (
                self.config.synthesis_error_budget
                / arbitrary_rotation_count
            )
            synthesized = SynthesizeRZRotations(
                synthesis_error=per_rotation_error,
                cache_error=0.0,
            )(exact_discrete)
        else:
            per_rotation_error = None
            synthesized = exact_discrete

        optimized = OptimizeCliffordT(
            basis_gates=list(CLIFFORD_T_BASIS)
        )(synthesized)
        optimized.name = f"level_4_clifford_t({level_3_circuit.name})"
        optimized.metadata = {
            **source_metadata,
            "compiler_level": 4,
            "source_level": 3,
            "gate_contract": "clifford_t",
            "clifford_t_synthesis": {
                **self.config.metadata(),
                "qiskit_version": qiskit.__version__,
                "basis_gates": CLIFFORD_T_BASIS,
                "original_rz_count": original_rz_count,
                "arbitrary_rotation_count": arbitrary_rotation_count,
                "per_rotation_error": per_rotation_error,
            },
        }
        resources = logical_resource_report(
            optimized,
            original_rz_count=original_rz_count,
            arbitrary_rotation_count=arbitrary_rotation_count,
            config=self.config,
        )
        return CliffordTCompilation(
            circuit=optimized,
            config=self.config,
            resources=resources,
            qiskit_version=qiskit.__version__,
        )
