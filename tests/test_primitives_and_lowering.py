from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import MCPhaseGate, MCXGate, RXGate, RYGate, RZGate, UCRYGate
from qiskit.quantum_info import Operator, Statevector

from digital_compiler import (
    CleanAncillaMCX,
    GrayCodeMCR,
    Level3Policy,
    NoAncillaMCX,
    QuantumAdder,
    RecursiveMCPhase,
    append_standard_toffoli,
    append_uniformly_controlled_ry,
)
from digital_compiler.lowering import SingleControlLowerer, assert_level_3_contract


def unitary_error(left: QuantumCircuit, right: QuantumCircuit) -> float:
    return float(np.max(np.abs(Operator(left).data - Operator(right).data)))


@pytest.mark.parametrize("control_count", [1, 2, 3, 4])
def test_gray_code_uniform_rotation_matches_qiskit_oracle(control_count: int) -> None:
    angles = [
        math.sin(index + 0.37) + 0.07 * index
        for index in range(1 << control_count)
    ]
    custom = QuantumCircuit(control_count + 1)
    append_uniformly_controlled_ry(
        custom,
        range(1, control_count + 1),
        0,
        angles,
    )
    oracle = QuantumCircuit(control_count + 1)
    oracle.append(UCRYGate(angles), [0, *range(1, control_count + 1)])
    assert unitary_error(custom, oracle) < 1e-10
    assert custom.count_ops().get("cx", 0) == 1 << control_count


def test_standard_toffoli_inventory_and_unitary() -> None:
    custom = QuantumCircuit(3)
    append_standard_toffoli(custom, 0, 1, 2)
    oracle = QuantumCircuit(3)
    oracle.ccx(0, 1, 2)
    assert unitary_error(custom, oracle) < 1e-10
    assert dict(custom.count_ops()) == {"h": 2, "cx": 6, "t": 4, "tdg": 3}


@pytest.mark.parametrize(
    "method_name,direction",
    [("add_path_step", 1), ("subtract_path_step", -1)],
)
@pytest.mark.parametrize("width", range(1, 6))
def test_path_step_adder_matches_signed_modular_arithmetic(
    width: int,
    method_name: str,
    direction: int,
) -> None:
    circuit = QuantumCircuit(width + 1)
    getattr(QuantumAdder(width), method_name)(
        circuit,
        0,
        range(1, width + 1),
    )

    modulus = 1 << width
    for basis in range(1 << (width + 1)):
        path_bit = basis & 1
        height = basis >> 1
        signed_step = 2 * path_bit - 1
        expected_height = (height + direction * signed_step) % modulus
        expected_basis = path_bit | (expected_height << 1)

        actual = Statevector.from_int(basis, 1 << (width + 1)).evolve(circuit)
        expected = Statevector.from_int(expected_basis, 1 << (width + 1))
        np.testing.assert_allclose(actual.data, expected.data, atol=1e-9)


@pytest.mark.parametrize("width", range(1, 6))
def test_path_step_add_and_subtract_are_exact_inverses(width: int) -> None:
    circuit = QuantumCircuit(width + 1)
    adder = QuantumAdder(width)
    adder.add_path_step(circuit, 0, range(1, width + 1))
    adder.subtract_path_step(circuit, 0, range(1, width + 1))

    assert unitary_error(circuit, QuantumCircuit(width + 1)) < 1e-10


@pytest.mark.parametrize("method_name", ["add_path_step", "subtract_path_step"])
@pytest.mark.parametrize("width", range(1, 6))
def test_path_step_adder_uses_at_most_width_minus_one_controls(
    width: int,
    method_name: str,
) -> None:
    circuit = QuantumCircuit(width + 1)
    getattr(QuantumAdder(width), method_name)(
        circuit,
        0,
        range(1, width + 1),
    )

    x_control_counts = [
        instruction.operation.num_ctrl_qubits
        for instruction in circuit.data
        if isinstance(instruction.operation, ControlledGate)
        and instruction.operation.base_gate.name == "x"
    ]
    assert max(x_control_counts, default=0) == width - 1
    if width == 1:
        assert dict(circuit.count_ops()) == {"x": 1}


@pytest.mark.parametrize("axis,gate", [("x", RXGate), ("y", RYGate), ("z", RZGate)])
@pytest.mark.parametrize("control_count", [1, 2, 3, 4])
def test_mcr_lowering(axis: str, gate, control_count: int) -> None:
    angle = 0.31 + 0.17 * control_count + 0.03 * ord(axis)
    semantic = QuantumCircuit(control_count + 1)
    semantic.append(
        gate(angle).control(control_count, annotated=False),
        range(control_count + 1),
    )
    lowered = SingleControlLowerer(Level3Policy()).lower(semantic)
    assert unitary_error(semantic, lowered) < 1e-9
    assert_level_3_contract(lowered)


@pytest.mark.parametrize("control_count", [1, 2, 3, 4])
def test_mcphase_lowering_including_global_phase(control_count: int) -> None:
    angle = 0.43 + 0.11 * control_count
    semantic = QuantumCircuit(control_count + 1)
    semantic.append(MCPhaseGate(angle, control_count), range(control_count + 1))
    lowered = SingleControlLowerer(Level3Policy()).lower(semantic)
    assert unitary_error(semantic, lowered) < 1e-9


@pytest.mark.parametrize("control_count", [2, 3, 4])
def test_clean_ancilla_mcx_and_workspace_cleanup(control_count: int) -> None:
    data = QuantumRegister(control_count + 1, "data")
    work = QuantumRegister(control_count - 2, "adder_work")
    semantic = QuantumCircuit(data, work)
    semantic.append(MCXGate(control_count), data)
    lowered = SingleControlLowerer(Level3Policy(mcx=CleanAncillaMCX())).lower(semantic)

    control_mask = (1 << control_count) - 1
    for basis in range(1 << (control_count + 1)):
        state = Statevector.from_int(basis, 2**lowered.num_qubits).evolve(lowered)
        expected = basis
        if basis & control_mask == control_mask:
            expected ^= 1 << control_count
        oracle = Statevector.from_int(expected, 2**lowered.num_qubits)
        np.testing.assert_allclose(state.data, oracle.data, atol=1e-9)
        leakage = np.sum(np.abs(state.data[1 << (control_count + 1) :]) ** 2)
        assert math.sqrt(float(leakage)) < 1e-9

    toffoli_count = 2 * control_count - 3
    counts = lowered.count_ops()
    assert counts.get("t", 0) + counts.get("tdg", 0) == 7 * toffoli_count


@pytest.mark.parametrize("control_count", range(6))
def test_no_ancilla_mcx_matches_qiskit_oracle(control_count: int) -> None:
    semantic = QuantumCircuit(control_count + 1)
    if control_count == 0:
        semantic.x(0)
    else:
        semantic.append(MCXGate(control_count), range(control_count + 1))

    policy = Level3Policy(mcx=NoAncillaMCX())
    lowered = SingleControlLowerer(policy).lower(semantic)

    assert lowered.num_qubits == semantic.num_qubits
    assert unitary_error(semantic, lowered) < 1e-9
    assert_level_3_contract(lowered)


def test_no_ancilla_mcx_requests_no_workspace() -> None:
    decomposition = NoAncillaMCX()
    assert decomposition.clean_ancillas(0) == 0
    assert decomposition.clean_ancillas(8) == 0
    with pytest.raises(ValueError, match="control count cannot be negative"):
        decomposition.clean_ancillas(-1)


def test_mcx_rejects_invalid_work_allocations() -> None:
    decomposition = CleanAncillaMCX()
    probe = QuantumCircuit(6)
    cases = [
        ([], "need 2 clean adder_work qubits"),
        ([0, 5], "must not overlap"),
        ([4, 4], "must be distinct"),
    ]
    for work, message in cases:
        with pytest.raises(ValueError, match=message):
            decomposition.append(probe, [0, 1, 2, 3], 4, work)


def test_default_level_3_components_are_independent() -> None:
    policy = Level3Policy()
    assert isinstance(policy.mcx, NoAncillaMCX)
    assert isinstance(policy.rotations, GrayCodeMCR)
    assert isinstance(policy.phases, RecursiveMCPhase)


def test_default_no_ancilla_policy_metadata() -> None:
    policy = Level3Policy()
    assert policy.metadata()["mcx"] == "no_ancilla_recursive_phase"
