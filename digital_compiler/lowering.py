"""Audited Level-2 to Level-3 lowering pass."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import MCPhaseGate, PhaseGate

from .policies import Level3Policy


def work_register(circuit: QuantumCircuit) -> list:
    matches = [register for register in circuit.qregs if register.name == "adder_work"]
    if len(matches) > 1:
        raise ValueError("circuit contains multiple adder_work registers")
    return [] if not matches else list(matches[0])


def _mcx_work_lane(
    circuit: QuantumCircuit,
    target,
    available_work: list,
) -> list:
    """Return lane-local work for compiler-generated height-register MCX gates."""

    metadata = circuit.metadata or {}
    lane_count = metadata.get("parallel_lanes")
    selector_width = metadata.get("height_selector_qubits")
    work_per_lane = metadata.get("workspace_qubits_per_lane")
    if lane_count is None or selector_width is None or work_per_lane is None:
        return available_work

    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (lane_count, selector_width, work_per_lane)
    ):
        raise ValueError("compiler lane metadata must contain integer widths")
    if lane_count < 1 or selector_width < 1 or work_per_lane < 0:
        raise ValueError("compiler lane metadata contains invalid widths")
    if len(available_work) != lane_count * work_per_lane:
        raise ValueError("adder_work size does not match compiler lane metadata")
    if work_per_lane == 0:
        return []

    height_matches = [register for register in circuit.qregs if register.name == "height"]
    if len(height_matches) != 1:
        raise ValueError("a scheduled compiler circuit must contain one height register")
    height = list(height_matches[0])
    if len(height) != lane_count * selector_width:
        raise ValueError("height size does not match compiler lane metadata")
    try:
        height_index = height.index(target)
    except ValueError:
        raise ValueError(
            "a compiler-generated MCX must target a height-lane qubit"
        ) from None
    lane = height_index // selector_width
    start = lane * work_per_lane
    return available_work[start : start + work_per_lane]


def assert_level_2_contract(circuit: QuantumCircuit) -> None:
    """Verify semantic gate families and idle lowering workspace at Level 2."""

    work = set(work_register(circuit))
    allowed_uncontrolled = {
        "x",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "p",
        "cx",
        "measure",
        "barrier",
    }
    for instruction in circuit.data:
        operation = instruction.operation
        if any(qubit in work for qubit in instruction.qubits):
            raise AssertionError("adder_work must remain idle in Level 2")
        if operation.name in allowed_uncontrolled:
            continue
        if isinstance(operation, ControlledGate):
            if operation.base_gate.name in {"x", "rx", "ry", "rz", "p"}:
                continue
        raise AssertionError(f"operation {operation.name!r} violates the Level-2 contract")


def assert_level_3_contract(circuit: QuantumCircuit) -> None:
    """Verify that Level 3 contains no multiply controlled operation."""

    allowed = {
        "x",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "cx",
        "crx",
        "cry",
        "crz",
        "measure",
        "barrier",
    }
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name not in allowed:
            raise AssertionError(f"operation {operation.name!r} violates Level 3")
        if isinstance(operation, ControlledGate) and operation.num_ctrl_qubits > 1:
            raise AssertionError("Level 3 cannot contain a multiply controlled gate")


def _assert_all_ones_control(operation: ControlledGate) -> None:
    expected = (1 << operation.num_ctrl_qubits) - 1
    if operation.ctrl_state != expected:
        raise ValueError("the audited lowerer supports only all-ones control states")


class SingleControlLowerer:
    """Lower semantic Level-2 operations using one composable policy bundle."""

    def __init__(self, policy: Level3Policy):
        self.policy = policy

    def lower(
        self,
        level_2_circuit: QuantumCircuit,
        name: str | None = None,
    ) -> QuantumCircuit:
        assert_level_2_contract(level_2_circuit)
        level_3 = QuantumCircuit(
            *level_2_circuit.qregs,
            *level_2_circuit.cregs,
            name=name or f"level_3_single_control({level_2_circuit.name})",
        )
        level_3.global_phase = float(level_2_circuit.global_phase)
        metadata = dict(level_2_circuit.metadata or {})
        metadata.update(
            {
                "compiler_level": 3,
                "source_level": 2,
                "gate_contract": "single_control",
                "lowering_policies": self.policy.metadata(),
            }
        )
        level_3.metadata = metadata
        available_work = work_register(level_2_circuit)

        for instruction in level_2_circuit.data:
            operation = instruction.operation
            qubits = list(instruction.qubits)
            clbits = list(instruction.clbits)

            if isinstance(operation, MCPhaseGate):
                _assert_all_ones_control(operation)
                self.policy.phases.append(
                    level_3,
                    qubits,
                    float(operation.params[0]),
                    self.policy.rotations,
                )
                continue
            if isinstance(operation, PhaseGate):
                self.policy.phases.append(
                    level_3,
                    qubits,
                    float(operation.params[0]),
                    self.policy.rotations,
                )
                continue
            if isinstance(operation, ControlledGate):
                _assert_all_ones_control(operation)
                controls = qubits[: operation.num_ctrl_qubits]
                target = qubits[-1]
                base_name = operation.base_gate.name
                if base_name == "x" and operation.num_ctrl_qubits >= 2:
                    self.policy.mcx.append(
                        level_3,
                        controls,
                        target,
                        _mcx_work_lane(
                            level_2_circuit,
                            target,
                            available_work,
                        ),
                    )
                    continue
                if base_name in {"rx", "ry", "rz"} and operation.num_ctrl_qubits >= 2:
                    self.policy.rotations.append(
                        level_3,
                        base_name[-1],
                        controls,
                        target,
                        float(operation.base_gate.params[0]),
                    )
                    continue
                if base_name == "p":
                    self.policy.phases.append(
                        level_3,
                        qubits,
                        float(operation.base_gate.params[0]),
                        self.policy.rotations,
                    )
                    continue
            level_3.append(operation, qubits, clbits)

        assert_level_3_contract(level_3)
        return level_3
