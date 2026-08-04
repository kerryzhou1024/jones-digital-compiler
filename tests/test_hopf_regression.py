from __future__ import annotations

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from digital_compiler import (
    AJLCompiler,
    AJLPathModel,
    BraidWord,
    HadamardTestCompilation,
    append_fixed_height_braid,
    append_hadamard_readout,
    prepare_basis_path,
)


def hadamard_component(circuit: QuantumCircuit) -> float:
    state = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    probabilities = state.probabilities(qargs=[0])
    return float(probabilities[0] - probabilities[1])


def test_hopf_jones_polynomial_from_compiled_path_amplitudes() -> None:
    model = AJLPathModel(2, 5)
    compiler = AJLCompiler(model)
    word = BraidWord.power(1, 2)
    amplitudes = {}
    for path in model.valid_paths():
        real = hadamard_component(
            compiler.compile_hadamard_test(word, path, "real").level_2_multicontrolled
        )
        imag = hadamard_component(
            compiler.compile_hadamard_test(word, path, "imag").level_2_multicontrolled
        )
        amplitudes[path] = complex(real, imag)

    observed = model.trace_closure_jones(word, amplitudes)
    expected = -model.A**10 - model.A**2
    assert abs(observed - expected) < 1e-9


def test_sigma3_squared_four_strand_closure_matches_analytic_jones_value() -> None:
    model = AJLPathModel(4, 5)
    compiler = AJLCompiler(model)
    word = BraidWord.power(3, 2)
    amplitudes = {}

    for path in model.valid_paths():
        real = hadamard_component(
            compiler.compile_hadamard_test(word, path, "real").level_2_multicontrolled
        )
        imag = hadamard_component(
            compiler.compile_hadamard_test(word, path, "imag").level_2_multicontrolled
        )
        amplitudes[path] = complex(real, imag)

    observed = model.trace_closure_jones(word, amplitudes)
    expected_split_union = model.d**2 * (-(model.A**10) - model.A**2)
    expected_radical = complex(
        -0.5,
        -0.5 * math.sqrt(5.0 + 2.0 * math.sqrt(5.0)),
    )
    assert abs(observed - expected_split_union) < 1e-9
    assert abs(expected_split_union - expected_radical) < 1e-9


def test_fixed_height_hopf_uses_only_public_primitives() -> None:
    model = AJLPathModel(2, 5)
    compiler = AJLCompiler(model)
    word = BraidWord.power(1, 2)
    path_bits = model.coerce_path("10")

    circuits = []
    for level in (1, 2):
        control = QuantumRegister(1, "ctrl")
        path = QuantumRegister(2, "path")
        measurement = ClassicalRegister(1, "meas")
        circuit = QuantumCircuit(control, path, measurement, name=f"fixed_height_l{level}")
        prepare_basis_path(circuit, path, path_bits)
        circuit.h(control[0])
        for generator in word.generators:
            if level == 1:
                circuit.append(
                    compiler.controlled_varphi_gate(generator, include_workspace=False),
                    [control[0], *path],
                )
            else:
                append_fixed_height_braid(
                    circuit,
                    model,
                    [control[0]],
                    path[0],
                    path[1],
                    1,
                    generator.sign,
                )
        append_hadamard_readout(circuit, control[0], "real", measurement[0])
        circuits.append(circuit)

    level_1, level_2 = circuits
    level_2.metadata = {"compiler_level": 2, "gate_contract": "ajl_multicontrolled"}
    level_3 = compiler.lower_to_level_3(level_2)
    result = HadamardTestCompilation(
        word=word,
        initial_path=path_bits,
        part="real",
        level_1_varphi=level_1,
        level_2_multicontrolled=level_2,
        level_3_single_control=level_3,
        config=compiler.config,
    )

    assert result.logical_qubits == 3
    assert dict(level_2.count_ops()) == {
        "x": 1,
        "h": 2,
        "p": 2,
        "cx": 4,
        "mcphase": 2,
        "measure": 1,
    }
    assert dict(level_3.count_ops()) == {
        "rz": 12,
        "cx": 12,
        "x": 1,
        "h": 2,
        "crz": 2,
        "measure": 1,
    }
