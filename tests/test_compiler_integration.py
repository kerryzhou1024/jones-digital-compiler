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
    NoAncillaMCX,
    SwitchCaseHeightSynthesis,
    circuit_gate_count_depth,
    compilation_summary,
    level_1_varphi_names,
    print_compilation_summary,
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
    assert compiler.logical_qubits == 8
    assert register_signature(compilation.level_1_varphi)[0][-1] == ("adder_work", 2)
    metadata = compilation.level_3_single_control.metadata
    assert metadata["lowering_policies"]["mcx"] == "test_two_workspace_mcx"
    assert metadata["compiler_config"]["workspace_qubits"] == 2


def test_no_ancilla_policy_removes_physical_workspace_and_preserves_semantics() -> None:
    model = AJLPathModel(2, 5)
    config = CompilerConfig(level3=Level3Policy(mcx=NoAncillaMCX()))
    compiler = AJLCompiler(model, config)
    compilation = compiler.compile_hadamard_test("s1^2", "10")

    assert compiler.work_qubits == 0
    assert compiler.logical_qubits == 6
    expected_registers = (
        ("ctrl", 1),
        ("path", 2),
        ("height", 3),
        ("adder_work", 0),
    )
    assert register_signature(compilation.level_1_varphi)[0] == expected_registers
    assert register_signature(compilation.level_2_multicontrolled)[0] == expected_registers
    assert register_signature(compilation.level_3_single_control)[0] == expected_registers
    assert compilation.level_1_varphi.num_qubits == 6
    assert compilation.level_2_multicontrolled.num_qubits == 6
    assert compilation.level_3_single_control.num_qubits == 6

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
def test_no_ancilla_level_3_braid_matches_dense_oracle(controlled: bool) -> None:
    model = AJLPathModel(3, 5)
    compiler = AJLCompiler(
        model,
        CompilerConfig(level3=Level3Policy(mcx=NoAncillaMCX())),
    )
    assert_braid_matches_dense(
        compiler,
        DenseAJLReference(model),
        "1 2 -1",
        level=3,
        controlled=controlled,
    )


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
    assert compiler.height_qubits == 3
    assert compiler.work_qubits == 1
    assert compiler.logical_qubits == 7
    assert dict(compilation.level_2_multicontrolled.count_ops()) == {
        "cx": 40,
        "ry": 24,
        "x": 21,
        "ccx": 4,
        "h": 2,
        "p": 2,
        "mcphase": 2,
        "measure": 1,
    }
    assert dict(compilation.level_3_single_control.count_ops()) == {
        "cx": 72,
        "ry": 24,
        "x": 21,
        "t": 16,
        "tdg": 12,
        "rz": 12,
        "h": 10,
        "crz": 2,
        "measure": 1,
    }
    assert level_1_varphi_names(compilation.level_1_varphi) == (
        "c_varphi_sigma_1_plus",
        "c_varphi_sigma_1_plus",
    )
    summary = compilation_summary(compilation)
    assert summary["level_3_lowering_policies"] == {
        "mcx": "clean_ancilla_toffoli_ladder",
        "multi_controlled_rotations": "gray_code_sparse_multiplexor",
        "multi_controlled_phase": "recursive_corrected_rz",
    }
    assert summary["logical_qubits_each_level"] == 7
    assert summary["level_1_gate_count_depth"]["total"]["count"] == 5
    assert summary["level_2_gate_count_depth"]["total"]["count"] == 95
    assert summary["level_3_gate_count_depth"]["total"]["count"] == 169


def test_printed_compilation_summary_has_one_clear_table_per_level(capsys) -> None:
    compilation = AJLCompiler(AJLPathModel(2, 5)).compile_hadamard_test("s1^2", "10")
    print_compilation_summary(compilation)
    output = capsys.readouterr().out

    assert output.count("exact gate count | depth") == 3
    assert output.count("TOTAL quantum gates") == 3
    assert "gate                     count | depth" in output
    assert "mcphase" in output
    assert "measure" not in output
