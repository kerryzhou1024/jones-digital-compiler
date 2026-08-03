from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from digital_compiler import (
    AJLCompiler,
    AJLPathModel,
    CleanAncillaMCX,
    CompilerConfig,
    DenseAJLReference,
    Level3Policy,
    MultiplexedHeightSynthesis,
    RecomputePrefixHeight,
    SwitchCaseHeightSynthesis,
    circuit_gate_count_depth,
    compilation_info,
    level_1_varphi_names,
    register_signature,
)
from digital_compiler.lowering import assert_level_2_contract, assert_level_3_contract

TOL = 1e-9


def statevector_from_basis(circuit, basis_index: int) -> Statevector:
    return Statevector.from_int(basis_index, 2**circuit.num_qubits).evolve(circuit)


def braid_builder(compiler: AJLCompiler, level: int, controlled: bool):
    if level == 2:
        return (
            compiler.controlled_level_2_braid_circuit
            if controlled
            else compiler.level_2_braid_circuit
        )
    return (
        compiler.controlled_level_3_braid_circuit
        if controlled
        else compiler.level_3_braid_circuit
    )


def assert_braid_matches_dense(
    compiler: AJLCompiler,
    reference: DenseAJLReference,
    word: str,
    level: int,
    controlled: bool,
) -> None:
    circuit = braid_builder(compiler, level, controlled)(word)
    target = reference.full_braid_matrix(word)
    path_dimension = 2**compiler.strands
    scratch_shift = compiler.strands + (1 if controlled else 0)

    for control_bit in ((0, 1) if controlled else (0,)):
        for input_path in reference.paths:
            path_index = reference.little_endian_index(input_path)
            input_index = control_bit | (path_index << 1) if controlled else path_index
            state = statevector_from_basis(circuit, input_index)
            for output_path_index in range(path_dimension):
                output_index = (
                    control_bit | (output_path_index << 1)
                    if controlled
                    else output_path_index
                )
                expected = (
                    1.0 if output_path_index == path_index else 0.0
                ) if controlled and control_bit == 0 else target[output_path_index, path_index]
                assert abs(state.data[output_index] - expected) < TOL

            leakage = 0.0
            for output_index, amplitude in enumerate(state.data):
                same_control = not controlled or (output_index & 1) == control_bit
                clean_scratch = (output_index >> scratch_shift) < path_dimension
                if not (same_control and clean_scratch):
                    leakage += float(abs(amplitude) ** 2)
            assert math.sqrt(leakage) < TOL


@pytest.mark.parametrize("level", [3, 4, 5, 6, 9])
def test_height_encoding_uses_the_minimum_zero_based_selector(level: int) -> None:
    model = AJLPathModel(strands=2, level=level)
    compiler = AJLCompiler(model)
    expected_width = (level - 2).bit_length()

    assert compiler.height_qubits == expected_width
    assert compiler.height_selector_qubits == expected_width
    assert len(compiler.projector_basis_angles) == 1 << expected_width
    assert len(compiler.projector_alignment_angles) == 1 << expected_width

    for height in model.valid_heights:
        encoded_height = height - 1
        assert compiler.projector_basis_angles[encoded_height] == pytest.approx(
            model.projector_angle(height)
        )
        assert compiler.projector_alignment_angles[encoded_height] == pytest.approx(
            -2.0 * model.projector_angle(height)
        )

    for encoded_height in range(level - 1, 1 << expected_width):
        assert compiler.projector_basis_angles[encoded_height] == 0.0
        assert compiler.projector_alignment_angles[encoded_height] == 0.0

    circuit = compiler.level_2_braid_circuit("")
    assert circuit.metadata["height_encoding"] == "vertex_minus_one"
    assert circuit.metadata["compiler_config"]["height_encoding"] == "vertex_minus_one"


def test_height_one_load_and_unload_are_gate_free() -> None:
    compiler = AJLCompiler(AJLPathModel(strands=2, level=5))
    circuit = QuantumCircuit(compiler.height_qubits)
    height = list(circuit.qubits)

    compiler._compute_prefix_height(circuit, (), height, index=1)
    compiler._uncompute_prefix_height(circuit, (), height, index=1)

    assert circuit.data == []


def test_k9_minimal_height_width_needs_no_clean_adder_workspace() -> None:
    config = CompilerConfig(level3=Level3Policy(mcx=CleanAncillaMCX()))
    compiler = AJLCompiler(AJLPathModel(strands=3, level=9), config)

    assert compiler.height_qubits == 3
    assert compiler.work_qubits_per_lane == 0
    assert compiler.work_qubits == 0


@pytest.mark.parametrize(
    "height_policy",
    [MultiplexedHeightSynthesis(), SwitchCaseHeightSynthesis()],
)
@pytest.mark.parametrize("level", [3, 4, 5, 6, 9])
@pytest.mark.parametrize("compiler_level", [2, 3])
@pytest.mark.parametrize("controlled", [False, True])
def test_zero_based_height_encoding_matches_dense_across_levels(
    height_policy,
    level: int,
    compiler_level: int,
    controlled: bool,
) -> None:
    model = AJLPathModel(strands=3, level=level)
    compiler = AJLCompiler(model, CompilerConfig(height=height_policy))
    assert_braid_matches_dense(
        compiler,
        DenseAJLReference(model),
        "1 2 -1",
        level=compiler_level,
        controlled=controlled,
    )


@pytest.mark.parametrize(
    "height_policy",
    [MultiplexedHeightSynthesis(), SwitchCaseHeightSynthesis()],
)
@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("word", ["2", "-2", "1 2 -1"])
def test_braid_levels_match_dense_oracle(height_policy, level, controlled, word) -> None:
    model = AJLPathModel(strands=3, level=5)
    compiler = AJLCompiler(model, CompilerConfig(height=height_policy))
    assert_braid_matches_dense(
        compiler,
        DenseAJLReference(model),
        word,
        level,
        controlled,
    )


@pytest.mark.parametrize("controlled", [False, True])
def test_level_2_and_level_3_are_equivalent(controlled: bool) -> None:
    model = AJLPathModel(3, 5)
    compiler = AJLCompiler(model)
    builder = (
        compiler.controlled_level_2_braid_circuit
        if controlled
        else compiler.level_2_braid_circuit
    )
    level_2 = builder("1 2 -1")
    level_3 = compiler.lower_to_level_3(level_2)
    for control_bit in ((0, 1) if controlled else (0,)):
        for path in model.valid_paths():
            path_index = DenseAJLReference.little_endian_index(path)
            basis = control_bit | (path_index << 1) if controlled else path_index
            np.testing.assert_allclose(
                statevector_from_basis(level_2, basis).data,
                statevector_from_basis(level_3, basis).data,
                atol=TOL,
            )


@pytest.mark.parametrize("part", ["real", "imag"])
@pytest.mark.parametrize(
    "height_policy",
    [MultiplexedHeightSynthesis(), SwitchCaseHeightSynthesis()],
)
def test_hadamard_test_matches_dense_amplitude(part: str, height_policy) -> None:
    model = AJLPathModel(2, 5)
    compiler = AJLCompiler(model, CompilerConfig(height=height_policy))
    reference = DenseAJLReference(model)
    compilation = compiler.compile_hadamard_test("s1^2", "10", part)
    expected = reference.path_amplitude(compilation.word, compilation.initial_path)
    expected_component = expected.real if part == "real" else expected.imag

    for circuit in (
        compilation.level_2_multicontrolled,
        compilation.level_3_single_control,
    ):
        state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
        probabilities = state.probabilities(qargs=[0])
        observed = float(probabilities[0] - probabilities[1])
        assert abs(observed - expected_component) < TOL
        scratch_leakage = sum(
            float(abs(amplitude) ** 2)
            for index, amplitude in enumerate(state.data)
            if (index >> (1 + compiler.strands)) != 0
        )
        assert math.sqrt(scratch_leakage) < TOL

    assert register_signature(compilation.level_1_varphi) == register_signature(
        compilation.level_2_multicontrolled
    ) == register_signature(compilation.level_3_single_control)
    assert_level_2_contract(compilation.level_2_multicontrolled)
    assert_level_3_contract(compilation.level_3_single_control)


@dataclass(frozen=True)
class TwoWorkspaceMCX:
    """Test policy that changes layout while delegating exact synthesis."""

    name: str = "test_two_workspace_mcx"

    @staticmethod
    def clean_ancillas(control_count: int) -> int:
        del control_count
        return 2

    @staticmethod
    def append(circuit, controls, target, work=()) -> None:
        CleanAncillaMCX().append(circuit, controls, target, work)


def test_policy_injection_changes_workspace_dispatch_and_metadata() -> None:
    level_3 = Level3Policy(mcx=TwoWorkspaceMCX())
    config = CompilerConfig(level3=level_3)
    compiler = AJLCompiler(AJLPathModel(2, 5), config)
    compilation = compiler.compile_hadamard_test("s1^2", "10")

    assert compiler.work_qubits == 2
    assert compiler.logical_qubits == 7
    assert register_signature(compilation.level_1_varphi)[0][-1] == ("adder_work", 2)
    metadata = compilation.level_3_single_control.metadata
    assert metadata["lowering_policies"]["mcx"] == "test_two_workspace_mcx"
    assert metadata["compiler_config"]["workspace_qubits"] == 2


def test_default_policy_uses_no_workspace_and_preserves_semantics() -> None:
    model = AJLPathModel(2, 5)
    compiler = AJLCompiler(model)
    compilation = compiler.compile_hadamard_test("s1^2", "10")

    assert compiler.work_qubits == 0
    assert compiler.logical_qubits == 5
    expected_registers = (
        ("ctrl", 1),
        ("path", 2),
        ("height", 2),
        ("adder_work", 0),
    )
    assert register_signature(compilation.level_1_varphi)[0] == expected_registers
    assert register_signature(compilation.level_2_multicontrolled)[0] == expected_registers
    assert register_signature(compilation.level_3_single_control)[0] == expected_registers
    assert compilation.level_1_varphi.num_qubits == 5
    assert compilation.level_2_multicontrolled.num_qubits == 5
    assert compilation.level_3_single_control.num_qubits == 5

    metadata = compilation.level_3_single_control.metadata
    assert metadata["lowering_policies"]["mcx"] == "no_ancilla_recursive_phase"
    assert metadata["compiler_config"]["workspace_qubits"] == 0

    level_2_state = Statevector.from_instruction(
        compilation.level_2_multicontrolled.remove_final_measurements(inplace=False)
    )
    level_3_state = Statevector.from_instruction(
        compilation.level_3_single_control.remove_final_measurements(inplace=False)
    )
    np.testing.assert_allclose(level_2_state.data, level_3_state.data, atol=TOL)
    assert_level_3_contract(compilation.level_3_single_control)

    expected = DenseAJLReference(model).path_amplitude("s1^2", "10").real
    probabilities = level_3_state.probabilities(qargs=[0])
    observed = float(probabilities[0] - probabilities[1])
    assert abs(observed - expected) < TOL


@pytest.mark.parametrize("controlled", [False, True])
def test_default_level_3_braid_matches_dense_oracle(controlled: bool) -> None:
    model = AJLPathModel(3, 5)
    compiler = AJLCompiler(model)
    assert_braid_matches_dense(
        compiler,
        DenseAJLReference(model),
        "1 2 -1",
        level=3,
        controlled=controlled,
    )


def test_clean_ancilla_policy_remains_available_as_an_opt_in() -> None:
    config = CompilerConfig(level3=Level3Policy(mcx=CleanAncillaMCX()))
    compiler = AJLCompiler(AJLPathModel(2, 5), config)
    compilation = compiler.compile_hadamard_test("s1^2", "10")

    assert compiler.work_qubits == 0
    assert compiler.logical_qubits == 5
    assert register_signature(compilation.level_1_varphi)[0][-1] == (
        "adder_work",
        0,
    )
    metadata = compilation.level_3_single_control.metadata
    assert metadata["lowering_policies"]["mcx"] == "clean_ancilla_toffoli_ladder"
    assert metadata["compiler_config"]["workspace_qubits"] == 0


def test_default_no_ancilla_policy_reduces_t_count_for_k17() -> None:
    model = AJLPathModel(3, 17)
    default = AJLCompiler(model).compile_hadamard_test("s2", "110")
    clean_config = CompilerConfig(level3=Level3Policy(mcx=CleanAncillaMCX()))
    clean = AJLCompiler(model, clean_config).compile_hadamard_test("s2", "110")

    default_counts = default.level_3_single_control.count_ops()
    clean_counts = clean.level_3_single_control.count_ops()
    default_t_count = default_counts.get("t", 0) + default_counts.get("tdg", 0)
    clean_t_count = clean_counts.get("t", 0) + clean_counts.get("tdg", 0)

    assert default_t_count == 28
    assert clean_t_count == 112
    assert default_t_count < clean_t_count


def test_gate_count_depth_report_uses_exact_gate_names_and_parallel_depth() -> None:
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.x(1)
    circuit.cx(0, 1)
    circuit.measure(0, 0)

    report = circuit_gate_count_depth(circuit)
    assert report == {
        "total": {"count": 3, "depth": 2},
        "by_gate": {
            "cx": {"count": 1, "depth": 1},
            "h": {"count": 1, "depth": 1},
            "x": {"count": 1, "depth": 1},
        },
    }


def test_default_k5_resources_and_metadata_regression() -> None:
    compiler = AJLCompiler(AJLPathModel(2, 5))
    compilation = compiler.compile_hadamard_test("s1^2", "10")
    assert compiler.height_qubits == 2
    assert compiler.work_qubits == 0
    assert compiler.logical_qubits == 5
    assert dict(compilation.level_2_multicontrolled.count_ops()) == {
        "cx": 20,
        "ry": 12,
        "x": 5,
        "h": 2,
        "p": 2,
        "mcphase": 2,
        "measure": 1,
    }
    assert dict(compilation.level_3_single_control.count_ops()) == {
        "cx": 28,
        "ry": 12,
        "x": 5,
        "rz": 12,
        "h": 2,
        "crz": 2,
        "measure": 1,
    }
    assert level_1_varphi_names(compilation.level_1_varphi) == (
        "c_varphi_sigma_1_plus",
        "c_varphi_sigma_1_plus",
    )
    levels = dict(compilation_info(compilation).reports)
    assert levels["Level 3"].compiler_policies["lowering_policies"] == {
        "mcx": "no_ancilla_recursive_phase",
        "multi_controlled_rotations": "gray_code_sparse_multiplexor",
        "multi_controlled_phase": "recursive_corrected_rz",
    }
    assert {report.logical_qubits for report in levels.values()} == {5}
    assert levels["Level 2"].compiler_policies["prefix_height_strategy"] == "rolling"
    assert levels["Level 2"].compiler_policies["prefix_height_loads"] == 1
    assert levels["Level 2"].compiler_policies["prefix_height_moves"] == 0
    assert levels["Level 2"].compiler_policies["prefix_height_unloads"] == 1
    assert levels["Level 2"].compiler_policies["prefix_height_path_steps"] == 0
    assert levels["Level 2"].compiler_policies["height_encoding"] == "vertex_minus_one"
    assert levels["Level 1"].quantum_gate_count == 5
    assert levels["Level 2"].quantum_gate_count == 43
    assert levels["Level 3"].quantum_gate_count == 61

    baseline = AJLCompiler(
        AJLPathModel(2, 5),
        CompilerConfig(prefix_height=RecomputePrefixHeight()),
    ).compile_hadamard_test("s1^2", "10")
    baseline_levels = dict(compilation_info(baseline).reports)
    assert sum(baseline.level_2_multicontrolled.count_ops().values()) == 44
    assert sum(baseline.level_3_single_control.count_ops().values()) == 62
    assert (
        baseline_levels["Level 2"].compiler_policies["prefix_height_strategy"]
        == "recompute"
    )
    assert baseline_levels["Level 2"].quantum_gate_count == 43
    assert baseline_levels["Level 3"].quantum_gate_count == 61


def test_compilation_info_reports_one_column_per_compiler_level(capsys) -> None:
    compilation = AJLCompiler(AJLPathModel(2, 5)).compile_hadamard_test("s1^2", "10")
    report = compilation_info(compilation)

    assert [label for label, _ in report.reports] == [
        "Level 1",
        "Level 2",
        "Level 3",
    ]
    # A bare compilation has no closure or AJL root, but it does know its word.
    assert {info.closure for _, info in report.reports} == {"n/a"}
    assert {info.word for _, info in report.reports} == {"sigma_1 sigma_1"}

    print(report)
    output = capsys.readouterr().out

    assert "Circuit comparison" in output
    for label in ("Level 1", "Level 2", "Level 3"):
        assert label in output
    # Level 2 keeps its multi-controlled phase family, and measurements are
    # reported separately from the quantum gate families.
    assert "MCPhase" in output
    assert "gate: measure" not in output
    assert "measurements" in output
