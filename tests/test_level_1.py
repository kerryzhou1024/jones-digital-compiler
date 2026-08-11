from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from digital_compiler import (
    TOL,
    AJLCompiler,
    AJLPathModel,
    CleanAncillaMCX,
    CompilerConfig,
    DenseAJLReference,
    JonesProblem,
    Level3Policy,
    register_signature,
)
from digital_compiler.notebook import _level_1_serial_svg, show_scrollable_circuit
from digital_compiler.primitives import PrefixAdderGate


def hadamard_component(circuit) -> float:
    state = Statevector.from_instruction(circuit)
    probabilities = state.probabilities(qargs=[0])
    return float(probabilities[0] - probabilities[1])


def test_sigma_3_sigma_4_sigma_1_has_the_reference_level_1_structure() -> None:
    circuit = AJLCompiler(AJLPathModel(5, 5)).level_1_varphi_circuit(
        "3 4 1",
        "10101",
    )

    assert register_signature(circuit) == (
        (("ctrl", 1), ("path", 5), ("height", 2)),
        (("meas", 1),),
    )
    assert circuit.num_qubits == 8
    assert circuit.metadata["prefix_height_loads"] == 1
    assert circuit.metadata["prefix_height_moves"] == 2
    assert circuit.metadata["prefix_height_unloads"] == 0
    assert circuit.metadata["prefix_height_path_steps"] == 6

    semantic = [
        instruction
        for instruction in circuit.data
        if instruction.operation.name != "x"
    ]
    assert [instruction.operation.name for instruction in semantic] == [
        "h",
        "level_1_adder_plus_1_2",
        "c_varphi_sigma_3_plus",
        "level_1_adder_plus_3",
        "c_varphi_sigma_4_plus",
        "level_1_adder_minus_1_2_3",
        "c_varphi_sigma_1_plus",
        "h",
        "measure",
    ]
    assert [
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        for instruction in semantic[1:7]
    ] == [
        (1, 2, 6, 7),
        (0, 3, 4, 6, 7),
        (3, 6, 7),
        (0, 4, 5, 6, 7),
        (1, 2, 3, 6, 7),
        (0, 1, 2, 6, 7),
    ]


@pytest.mark.parametrize("height_qubits", range(1, 5))
@pytest.mark.parametrize("path_qubits", range(1, 4))
@pytest.mark.parametrize("inverse", [False, True])
def test_prefix_adder_gate_matches_signed_path_sum(
    height_qubits: int,
    path_qubits: int,
    inverse: bool,
) -> None:
    gate = PrefixAdderGate(
        tuple(range(2, 2 + path_qubits)),
        height_qubits,
        inverse=inverse,
    )
    operator = Operator(gate).data
    modulus = 1 << height_qubits

    for bits in product((0, 1), repeat=path_qubits):
        path_basis = sum(bit << index for index, bit in enumerate(bits))
        delta = sum(2 * bit - 1 for bit in bits)
        if inverse:
            delta = -delta
        for encoded_height in range(modulus):
            basis = path_basis | (encoded_height << path_qubits)
            observed = int(np.argmax(np.abs(operator[:, basis])))
            expected_height = (encoded_height + delta) % modulus
            assert observed == path_basis | (expected_height << path_qubits)
            assert abs(operator[observed, basis] - 1.0) < TOL


@pytest.mark.parametrize("path_qubits", range(1, 4))
def test_prefix_adder_and_inverse_compose_to_identity(path_qubits: int) -> None:
    indices = tuple(range(path_qubits))
    forward = Operator(PrefixAdderGate(indices, 3)).data
    inverse = Operator(PrefixAdderGate(indices, 3, inverse=True)).data
    forward_map = np.argmax(np.abs(forward), axis=0)
    inverse_map = np.argmax(np.abs(inverse), axis=0)

    np.testing.assert_array_equal(
        inverse_map[forward_map],
        np.arange(forward.shape[0]),
    )


def test_level_1_stays_lean_when_later_levels_need_workspace() -> None:
    config = CompilerConfig(level3=Level3Policy(mcx=CleanAncillaMCX()))
    compiler = AJLCompiler(AJLPathModel(3, 17), config)
    compilation = compiler.compile_hadamard_test("2", "110", measure=False)

    assert compiler.work_qubits > 0
    assert register_signature(compilation.level_1_varphi)[0] == (
        ("ctrl", 1),
        ("path", 3),
        ("height", 4),
    )
    assert any(
        register.name == "adder_work" and register.size == compiler.work_qubits
        for register in compilation.level_2_multicontrolled.qregs
    )
    assert compilation.logical_qubits == 8


def test_level_1_degenerate_words_and_readout_shape() -> None:
    compiler = AJLCompiler(AJLPathModel(2, 5))
    identity = compiler.level_1_varphi_circuit("", "10", measure=False)
    sigma_1 = compiler.level_1_varphi_circuit("-1", "10", "imag", measure=True)

    assert not any(
        instruction.operation.name.startswith("level_1_adder_")
        for instruction in identity.data
    )
    assert not any(
        instruction.operation.name.startswith("level_1_adder_")
        for instruction in sigma_1.data
    )
    assert [
        instruction.operation.name
        for instruction in sigma_1.data
        if instruction.operation.name in {"c_varphi_sigma_1_minus", "sdg", "measure"}
    ] == ["c_varphi_sigma_1_minus", "sdg", "measure"]
    assert identity.count_ops().get("measure", 0) == 0


@pytest.mark.parametrize("part", ["real", "imag"])
def test_sigma_3_sigma_4_sigma_1_lowered_circuits_match_dense_for_all_paths(
    part: str,
) -> None:
    model = AJLPathModel(5, 5)
    compiler = AJLCompiler(model)
    reference = DenseAJLReference(model)

    for path in model.valid_paths():
        compilation = compiler.compile_hadamard_test(
            "3 4 1",
            path,
            part,
            measure=False,
        )
        amplitude = reference.path_amplitude("3 4 1", path)
        expected = amplitude.real if part == "real" else amplitude.imag
        for circuit in (
            compilation.level_2_multicontrolled,
            compilation.level_3_single_control,
        ):
            assert abs(hadamard_component(circuit) - expected) < TOL
            state = Statevector.from_instruction(circuit)
            clean_dimension = 1 << (1 + model.strands)
            assert np.linalg.norm(state.data[clean_dimension:]) < TOL


def test_serial_svg_is_symbolic_exact_and_non_mutating(monkeypatch) -> None:
    compiled = JonesProblem("3 4 1", strands=5, k=5).circuit(
        "10101",
        "real",
        circuit_level=1,
    )
    circuit = compiled.circuit
    original_data = tuple(circuit.data)
    original_metadata = dict(circuit.metadata)
    original_counts = circuit.count_ops()
    original_depth = circuit.depth()

    svg = _level_1_serial_svg(circuit)

    assert svg is not None
    for text in (
        "|a₁⟩",
        "|a₅⟩",
        "|h₁⟩",
        "|h₂⟩",
        "Adder₁,₂",
        "Adder₃",
        "Adder†₁,₂,₃",
        "φ(σ₃)⁽ᶻ⁾",
        "φ(σ₄)⁽ᶻ⁾",
        "φ(σ₁)",
    ):
        assert text in svg
    assert 'class="operation controlled-varphi"' in svg
    assert 'class="operation prefix-adder"' in svg
    assert tuple(circuit.data) == original_data
    assert circuit.metadata == original_metadata
    assert circuit.count_ops() == original_counts
    assert circuit.depth() == original_depth
    assert compiled.info().gate_families["prefix Adder"] == 3
    assert compiled.info().gate_families["controlled varphi"] == 3

    displayed = []
    monkeypatch.setattr(
        "digital_compiler.notebook.display",
        lambda value: displayed.append(value),
    )
    show_scrollable_circuit(circuit, "Reference Level 1")
    assert len(displayed) == 1
    assert "level-1-ajl-circuit" in displayed[0].data
    assert "Reference Level 1" in displayed[0].data


def test_parallel_level_1_uses_the_qiskit_display_fallback() -> None:
    from digital_compiler import CommutingLayerScheduling

    config = CompilerConfig(scheduling=CommutingLayerScheduling(max_lanes=2))
    circuit = AJLCompiler(AJLPathModel(4, 5), config).level_1_varphi_circuit(
        "1 3",
        "1010",
    )

    assert _level_1_serial_svg(circuit) is None
