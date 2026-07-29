from __future__ import annotations

import numpy as np
import pytest
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from digital_compiler import (
    AJLCompiler,
    AJLPathModel,
    CliffordTCompiler,
    CliffordTConfig,
    CompilerConfig,
    JonesProblem,
    assert_clifford_t_contract,
    compilation_info,
    register_signature,
    t_layer_widths,
)


def phase_insensitive_operator_error(
    left: QuantumCircuit,
    right: QuantumCircuit,
) -> float:
    left_operator = Operator(left).data
    right_operator = Operator(right).data
    overlap = np.vdot(left_operator, right_operator)
    phase = float(np.angle(overlap))
    return float(
        np.linalg.norm(
            left_operator - np.exp(-1j * phase) * right_operator,
            ord=2,
        )
    )


@pytest.mark.parametrize(
    "budget",
    [0.0, -1.0, 1.1, float("inf"), True, object()],
)
def test_clifford_t_config_rejects_invalid_error_budgets(
    budget: object,
) -> None:
    with pytest.raises(ValueError, match="synthesis_error_budget"):
        CliffordTConfig(budget)


@pytest.mark.parametrize("optimization_level", [-1, 4, 1.5, True])
def test_clifford_t_config_rejects_invalid_optimization_levels(
    optimization_level: object,
) -> None:
    with pytest.raises(ValueError, match="optimization_level"):
        CliffordTConfig(1e-3, optimization_level=optimization_level)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_clifford_t_config_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed_transpiler"):
        CliffordTConfig(1e-3, seed_transpiler=seed)


def test_level_4_requires_an_explicit_policy() -> None:
    problem = JonesProblem("s1", strands=2)

    with pytest.raises(ValueError, match="CliffordTConfig"):
        problem.circuit("10", "real", circuit_level=4)
    with pytest.raises(ValueError, match="CliffordTConfig"):
        problem.evaluate(circuit_level=4)


def test_level_4_preserves_registers_and_reports_resources() -> None:
    config = CompilerConfig(level4=CliffordTConfig(1e-3))
    compiler = AJLCompiler(AJLPathModel(2, 5), config)
    compilation = compiler.compile_hadamard_test(
        "s1",
        "10",
        measure=True,
    )

    assert compilation.level_4_clifford_t is not None
    assert compilation.level_4_resources is not None
    assert register_signature(
        compilation.level_4_clifford_t
    ) == register_signature(compilation.level_3_single_control)
    assert_clifford_t_contract(compilation.level_4_clifford_t)

    resources = compilation.level_4_resources
    assert resources.logical_qubits == compilation.logical_qubits
    assert resources.measurement_count == 1
    assert resources.original_rz_count >= resources.arbitrary_rotation_count
    assert resources.arbitrary_rotation_count > 0
    assert resources.per_rotation_error == pytest.approx(
        config.level4.synthesis_error_budget
        / resources.arbitrary_rotation_count
    )
    assert resources.t_count == (
        resources.t_gate_count + resources.tdg_gate_count
    )
    assert resources.t_count == sum(resources.t_layer_widths)
    assert resources.t_depth == len(resources.t_layer_widths)
    assert resources.cx_count > 0

    metadata = compilation.level_4_clifford_t.metadata
    assert metadata["compiler_level"] == 4
    assert metadata["source_level"] == 3
    assert metadata["gate_contract"] == "clifford_t"
    synthesis = metadata["clifford_t_synthesis"]
    assert synthesis["qiskit_version"] == qiskit.__version__
    assert synthesis["allocation"] == "uniform"
    assert synthesis["cache_error"] == 0.0

    level_4 = dict(compilation_info(compilation).reports)["Level 4"]
    assert level_4.compiler_level == 4
    assert set(level_4.gate_families) == {"Clifford", "T"}
    assert level_4.level_4_resources["t_count"] == resources.t_count

    repeated = compiler.compile_hadamard_test("s1", "10", measure=True)
    assert repeated.level_4_clifford_t == compilation.level_4_clifford_t
    assert repeated.level_4_resources == compilation.level_4_resources


def test_level_4_operator_error_respects_the_component_budget() -> None:
    budget = 1e-3
    compiler = AJLCompiler(
        AJLPathModel(2, 5),
        CompilerConfig(level4=CliffordTConfig(budget)),
    )
    level_3 = compiler.level_3_single_control_circuit(
        "s1",
        "10",
        measure=False,
    )
    level_4 = compiler.lower_to_level_4(level_3)

    assert level_4.resources.arbitrary_rotation_count > 0
    assert (
        phase_insensitive_operator_error(level_3, level_4.circuit)
        <= budget + 1e-10
    )


def test_exact_clifford_t_input_consumes_no_synthesis_budget() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.t(0)
    circuit.cx(0, 1)

    result = CliffordTCompiler(CliffordTConfig(1e-6)).compile(circuit)

    assert result.resources.arbitrary_rotation_count == 0
    assert result.resources.per_rotation_error is None
    assert Operator(circuit).equiv(Operator(result.circuit))
    assert_clifford_t_contract(result.circuit)


def test_dependency_aware_t_layers() -> None:
    circuit = QuantumCircuit(3)
    circuit.t(0)
    circuit.tdg(1)
    circuit.cx(0, 2)
    circuit.t(2)

    assert t_layer_widths(circuit) == (2, 1)


def test_level_4_reporting_groups_clifford_and_t() -> None:
    problem = JonesProblem(
        "s1",
        strands=2,
        config=CompilerConfig(level4=CliffordTConfig(1e-3)),
    )
    compiled = problem.circuit(
        "10",
        "real",
        circuit_level=4,
        measure=True,
    )
    report = compiled.info()

    assert report.compiler_level == 4
    assert set(report.gate_families) == {"Clifford", "T"}
    assert sum(report.gate_families.values()) == report.quantum_gate_count
    assert report.level_4_resources is not None
    assert (
        report.level_4_resources["t_count"]
        == report.gate_families["T"]
    )
    assert report.level_4_resources["t_depth"] > 0
    assert report.exact_gate_stats["t"]["count"] > 0
    assert report.measurement_count == 1
    assert "excludes physical mapping" in " ".join(report.scope_notes)


def test_all_levels_include_level_4_only_when_configured() -> None:
    baseline = JonesProblem("s1", strands=2)
    configured = JonesProblem(
        "s1",
        strands=2,
        config=CompilerConfig(level4=CliffordTConfig(1e-3)),
    )

    assert [
        circuit.circuit_level
        for circuit in baseline.circuits(
            path="10",
            part="real",
            circuit_level="all",
        )
    ] == [1, 2, 3]
    assert [
        circuit.circuit_level
        for circuit in configured.circuits(
            path="10",
            part="real",
            circuit_level="all",
        )
    ] == [1, 2, 3, 4]
