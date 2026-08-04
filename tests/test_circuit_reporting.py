from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from qiskit.circuit import Gate

from digital_compiler import (
    CircuitComparison,
    CircuitInfo,
    CleanAncillaMCX,
    CompilerConfig,
    JonesProblem,
    Level3Policy,
    compare_circuits,
)


def trace_circuit(*, measure: bool = True):
    return JonesProblem("s1^2", strands=2, k=5).circuit(
        "10",
        "real",
        circuit_level=3,
        measure=measure,
    )


def plat_circuit(*, measure: bool = True):
    return JonesProblem(
        "s2^2",
        closure="plat",
        writhe=2,
        strands=4,
        k=5,
    ).circuit(
        "1010",
        "real",
        circuit_level=3,
        measure=measure,
    )


def test_level_3_trace_and_plat_resource_regressions() -> None:
    trace = trace_circuit().info()
    plat = plat_circuit().info()

    assert isinstance(trace, CircuitInfo)
    assert (
        trace.closure,
        trace.path_label,
        trace.compiler_level,
        trace.logical_qubits,
        trace.quantum_gate_count,
        trace.quantum_gate_depth,
        trace.measurement_count,
    ) == ("trace", "10", 3, 5, 57, 49, 1)
    assert trace.gate_families == {
        "CNOT": 28,
        "CRz": 2,
        "H": 2,
        "Ry": 12,
        "Rz": 12,
        "X": 1,
    }
    assert trace.exact_gate_stats["rz"] == {"count": 12, "depth": 10}

    assert (
        plat.closure,
        plat.path_label,
        plat.compiler_level,
        plat.logical_qubits,
        plat.quantum_gate_count,
        plat.quantum_gate_depth,
        plat.measurement_count,
    ) == ("plat", "1010", 3, 7, 64, 50, 1)
    assert plat.gate_families["CNOT"] == 32
    assert plat.gate_families["X"] == 4
    assert "T/Tdg" not in plat.gate_families
    assert plat.exact_gate_stats["rz"] == {"count": 12, "depth": 10}

    for report in (trace, plat):
        assert sum(report.gate_families.values()) == report.quantum_gate_count
        assert (
            sum(row["count"] for row in report.exact_gate_stats.values())
            == report.quantum_gate_count
        )


def test_reports_select_gate_families_from_the_recorded_level() -> None:
    problem = JonesProblem("s1^2", strands=2)
    reports = [
        compiled.info()
        for compiled in problem.circuits(
            path="10",
            part="real",
            circuit_level="all",
            measure=False,
        )
    ]

    assert [report.compiler_level for report in reports] == [1, 2, 3]
    assert reports[0].gate_families["controlled varphi"] == 2
    assert not any(name.startswith("MCX[") for name in reports[1].gate_families)
    assert any(name.startswith("MCPhase[") for name in reports[1].gate_families)
    assert reports[2].gate_families["CNOT"] == 28
    assert "controlled varphi" not in reports[2].gate_families


def test_measurements_are_separate_from_quantum_resources() -> None:
    measured = trace_circuit(measure=True).info()
    unmeasured = trace_circuit(measure=False).info()

    assert measured.quantum_gate_count == unmeasured.quantum_gate_count
    assert measured.quantum_gate_depth == unmeasured.quantum_gate_depth
    assert measured.measurement_count == 1
    assert measured.non_gate_operations == {"measure": 1}
    assert measured.measured is True
    assert unmeasured.measurement_count == 0
    assert unmeasured.non_gate_operations == {}
    assert unmeasured.measured is False


def test_report_recomputes_and_unknown_gates_fall_back_to_exact_names() -> None:
    compiled = trace_circuit(measure=False)
    before = compiled.info()
    compiled.circuit.append(Gate("research_custom", 1, []), [0])
    after = compiled.info()

    assert after.quantum_gate_count == before.quantum_gate_count + 1
    assert after.gate_families["research_custom"] == 1
    assert after.exact_gate_stats["research_custom"]["count"] == 1


def test_report_is_immutable_structured_and_has_deterministic_rich_output() -> None:
    report = trace_circuit().info()
    first_dict = report.as_dict()
    second_dict = report.as_dict()

    assert first_dict == second_dict
    assert first_dict["identity"]["signed_generators"] == (1, 1)
    assert first_dict["resources"]["quantum_registers"] == (
        ("ctrl", 1),
        ("path", 2),
        ("height", 2),
        ("adder_work", 0),
    )
    assert first_dict["compiler"]["height_strategy"] == "multiplexed"
    assert first_dict["compiler"]["height_encoding"] == "vertex_minus_one"
    assert first_dict["compiler"]["control_distribution"] == "shared"
    assert first_dict["compiler"]["prefix_height_strategy"] == "rolling"
    assert first_dict["compiler"]["prefix_height_loads"] == 1
    assert first_dict["compiler"]["prefix_height_moves"] == 0
    assert first_dict["compiler"]["prefix_height_unloads"] == 1
    assert first_dict["compiler"]["prefix_height_path_steps"] == 0
    assert "not the complete closure workload" in str(report)
    assert "not a final Clifford+T estimate" in report._repr_html_()
    assert str(report) == str(report)

    first_dict["gates"]["families"]["CNOT"] = -1
    assert report.gate_families["CNOT"] == 28
    with pytest.raises(TypeError):
        report.gate_families["CNOT"] = -1
    with pytest.raises(TypeError):
        report.exact_gate_stats["cx"]["count"] = -1
    with pytest.raises(FrozenInstanceError):
        report.quantum_gate_count = 0


def test_compare_circuits_is_ordered_rich_and_does_not_invent_deltas() -> None:
    comparison = compare_circuits(
        {
            "Trace <10>": trace_circuit(),
            "Plat |1010>": plat_circuit(),
        }
    )

    assert isinstance(comparison, CircuitComparison)
    assert [label for label, _ in comparison.reports] == [
        "Trace <10>",
        "Plat |1010>",
    ]
    assert "Each column describes one compiled circuit" in comparison.warnings[0]
    assert "Compiler policies differ" not in comparison.warnings
    assert "Trace &lt;10&gt;" in comparison._repr_html_()
    assert "Trace <10>" not in comparison._repr_html_()
    assert "quantum gates" in str(comparison)
    assert "delta" not in str(comparison).lower()
    assert list(comparison.as_dict()["circuits"]) == [
        "Trace <10>",
        "Plat |1010>",
    ]


def test_comparison_reports_missing_families_and_mixed_context_warnings() -> None:
    problem = JonesProblem("s1^2", strands=2)
    level_1 = problem.circuit("10", "real", circuit_level=1)
    clean_config = CompilerConfig(
        level3=Level3Policy(mcx=CleanAncillaMCX()),
    )
    level_3 = JonesProblem(
        "s1^2",
        strands=2,
        config=clean_config,
    ).circuit("10", "real", circuit_level=3)

    comparison = compare_circuits({"Level 1": level_1, "Level 3": level_3})
    output = str(comparison)

    assert "gate: controlled varphi" in output
    assert any("Compiler levels differ" in warning for warning in comparison.warnings)
    assert any("Compiler policies differ" in warning for warning in comparison.warnings)


def test_compare_circuits_validates_its_mapping() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compare_circuits({})
    with pytest.raises(ValueError, match="non-empty"):
        compare_circuits({"": trace_circuit()})
    with pytest.raises(TypeError, match="CompiledCircuit"):
        compare_circuits({"raw": trace_circuit().circuit})
