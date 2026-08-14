"""Reusable exact circuit primitives used by AJL synthesis policies."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Gate
from qiskit.circuit.library import MCPhaseGate, MCXGate, PhaseGate, RXGate, RYGate, RZGate

from .model import TOL, AJLPathModel, BraidGenerator, HadamardPart

RotationAxis = Literal["x", "y", "z"]


def gray_code_rotation_angles(angle_list: Sequence[float]) -> tuple[float, ...]:
    """Return physical angles for a cyclic reflected-Gray-code multiplexor."""

    desired = tuple(float(angle) for angle in angle_list)
    count = len(desired)
    if count == 0 or count & (count - 1):
        raise ValueError("uniformly controlled rotation needs a nonempty power-of-two angle list")

    transformed = []
    for physical_index in range(count):
        gray_word = physical_index ^ (physical_index >> 1)
        coefficient = 0.0
        for basis_index, desired_angle in enumerate(desired):
            parity = (basis_index & gray_word).bit_count() & 1
            coefficient += (-1.0 if parity else 1.0) * desired_angle
        transformed.append(coefficient / count)
    return tuple(transformed)


def append_uniformly_controlled_rotation(
    circuit: QuantumCircuit,
    controls,
    target,
    angle_list: Sequence[float],
    axis: Literal["y", "z"],
) -> None:
    """Emit an exact little-endian UCR with a cyclic Gray-code CNOT ladder."""

    controls = list(controls)
    desired = tuple(float(angle) for angle in angle_list)
    if axis not in {"y", "z"}:
        raise ValueError("uniform rotation axis must be 'y' or 'z'")
    if len(desired) != 1 << len(controls):
        raise ValueError("angle-list length must equal 2**len(controls)")

    append_rotation = circuit.ry if axis == "y" else circuit.rz
    if not controls:
        if abs(desired[0]) >= TOL:
            append_rotation(desired[0], target)
        return

    physical_angles = gray_code_rotation_angles(desired)
    final_index = len(physical_angles) - 1
    for index, angle in enumerate(physical_angles):
        if abs(angle) >= TOL:
            append_rotation(angle, target)
        if index == final_index:
            changed_bit = len(controls) - 1
        else:
            step = index + 1
            changed_bit = (step & -step).bit_length() - 1
        circuit.cx(controls[changed_bit], target)


def append_uniformly_controlled_ry(
    circuit: QuantumCircuit,
    controls,
    target,
    angle_list: Sequence[float],
) -> None:
    append_uniformly_controlled_rotation(circuit, controls, target, angle_list, axis="y")


def append_controlled_rotation(
    circuit: QuantumCircuit,
    axis: RotationAxis,
    controls,
    target,
    angle: float,
) -> None:
    """Append one semantic rotation with zero or more all-ones controls."""

    controls = list(controls)
    if abs(angle) < TOL:
        return
    gates = {"x": RXGate, "y": RYGate, "z": RZGate}
    if axis not in gates:
        raise ValueError(f"unsupported rotation axis {axis!r}")
    gate = gates[axis](float(angle))
    if controls:
        circuit.append(
            gate.control(len(controls), annotated=False),
            [*controls, target],
        )
    else:
        circuit.append(gate, [target])


def append_phase_on_all_ones(circuit: QuantumCircuit, qubits, angle: float) -> None:
    """Apply ``exp(i*angle)`` only to the all-ones state of ``qubits``."""

    qubits = list(qubits)
    if abs(angle) < TOL:
        return
    if not qubits:
        circuit.global_phase += float(angle)
    elif len(qubits) == 1:
        circuit.append(PhaseGate(float(angle)), qubits)
    else:
        circuit.append(MCPhaseGate(float(angle), len(qubits) - 1), qubits)


def append_phase_on_one(
    circuit: QuantumCircuit,
    controls,
    target,
    angle: float,
) -> None:
    append_phase_on_all_ones(circuit, [*list(controls), target], angle)


def append_rank_one_braid_phase(
    circuit: QuantumCircuit,
    model: AJLPathModel,
    controls,
    left,
    right,
    height: int,
    sign: int,
) -> None:
    """Append the height-fixed rank-one portion of one AJL crossing."""

    alpha = model.projector_angle(height)
    _, relative_angle = model.phase_angles(sign)
    controls = list(controls)
    if alpha <= math.pi / 4.0:
        parity, phase_target = left, right
        alignment_angle = 2.0 * alpha
        parity_check = (right, left)
    else:
        parity, phase_target = right, left
        alignment_angle = math.pi - 2.0 * alpha
        parity_check = (left, right)

    circuit.cx(*parity_check)
    append_controlled_rotation(
        circuit,
        "y",
        [parity],
        phase_target,
        alignment_angle,
    )
    append_phase_on_one(
        circuit,
        [*controls, parity],
        phase_target,
        relative_angle,
    )
    append_controlled_rotation(
        circuit,
        "y",
        [parity],
        phase_target,
        -alignment_angle,
    )
    circuit.cx(*parity_check)


def append_fixed_height_braid(
    circuit: QuantumCircuit,
    model: AJLPathModel,
    controls,
    left,
    right,
    height: int,
    sign: int,
) -> None:
    """Append one semantic crossing when its prefix height is classically known."""

    controls = list(controls)
    base_angle, _ = model.phase_angles(sign)
    append_phase_on_all_ones(circuit, controls, base_angle)
    append_rank_one_braid_phase(
        circuit,
        model,
        controls,
        left,
        right,
        height,
        sign,
    )


def append_standard_toffoli(circuit: QuantumCircuit, a, b, c) -> None:
    """Append the exact 6-CNOT, 7-T-like decomposition of ``CCX(a,b;c)``."""

    circuit.h(c)
    circuit.cx(b, c)
    circuit.tdg(c)
    circuit.cx(a, c)
    circuit.t(c)
    circuit.cx(b, c)
    circuit.tdg(c)
    circuit.cx(a, c)
    circuit.t(b)
    circuit.t(c)
    circuit.h(c)
    circuit.cx(a, b)
    circuit.t(a)
    circuit.tdg(b)
    circuit.cx(a, b)


class QuantumAdder:
    """Little-endian reversible +/-1 adder over semantic MCX gates."""

    def __init__(self, width: int):
        if width < 1:
            raise ValueError("adder width must be positive")
        self.width = int(width)

    @staticmethod
    def _append_multi_controlled_x(circuit, controls, target) -> None:
        controls = list(controls)
        if not controls:
            circuit.x(target)
        elif len(controls) == 1:
            circuit.cx(controls[0], target)
        else:
            circuit.append(MCXGate(len(controls)), [*controls, target])

    def increment(self, circuit, register, control=None) -> None:
        register = list(register)
        prefix = [] if control is None else [control]
        for target_index in range(self.width - 1, 0, -1):
            controls = [*prefix, *register[:target_index]]
            self._append_multi_controlled_x(circuit, controls, register[target_index])
        if control is None:
            circuit.x(register[0])
        else:
            circuit.cx(control, register[0])

    def decrement(self, circuit, register, control=None) -> None:
        register = list(register)
        prefix = [] if control is None else [control]
        if control is None:
            circuit.x(register[0])
        else:
            circuit.cx(control, register[0])
        for target_index in range(1, self.width):
            controls = [*prefix, *register[:target_index]]
            self._append_multi_controlled_x(
                circuit,
                controls,
                register[target_index],
            )

    def add_path_step(self, circuit, bit, register) -> None:
        register = list(register)
        if self.width == 1:
            circuit.x(register[0])
            return

        self.decrement(circuit, register)
        QuantumAdder(self.width - 1).increment(
            circuit,
            register[1:],
            control=bit,
        )

    def subtract_path_step(self, circuit, bit, register) -> None:
        register = list(register)
        if self.width == 1:
            circuit.x(register[0])
            return

        self.increment(circuit, register)
        QuantumAdder(self.width - 1).decrement(
            circuit,
            register[1:],
            control=bit,
        )


class PrefixAdderGate(Gate):
    """Opaque Level-1 block for one reversible prefix-height transition."""

    def __init__(
        self,
        path_indices: Sequence[int],
        height_qubits: int,
        *,
        inverse: bool = False,
    ) -> None:
        indices = tuple(int(index) for index in path_indices)
        if not indices:
            raise ValueError("a prefix Adder block needs at least one path qubit")
        if any(index < 0 for index in indices):
            raise ValueError("path indices must be nonnegative")
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise ValueError("a prefix Adder block needs contiguous path indices")
        if height_qubits < 1:
            raise ValueError("a prefix Adder block needs at least one height qubit")

        self.path_indices = indices
        self.height_qubits = int(height_qubits)
        self.is_inverse = bool(inverse)
        direction = "minus" if self.is_inverse else "plus"
        suffix = "_".join(str(index + 1) for index in indices)
        label_indices = ",".join(str(index + 1) for index in indices)
        dagger = "†" if self.is_inverse else ""
        super().__init__(
            f"level_1_adder_{direction}_{suffix}",
            len(indices) + self.height_qubits,
            [],
            label=f"Adder{dagger}[{label_indices}]",
        )

    def _define(self) -> None:
        definition = QuantumCircuit(self.num_qubits, name=self.name)
        path = definition.qubits[: len(self.path_indices)]
        height = definition.qubits[len(self.path_indices) :]
        adder = QuantumAdder(self.height_qubits)
        if self.is_inverse:
            for bit in reversed(path):
                adder.subtract_path_step(definition, bit, height)
        else:
            for bit in path:
                adder.add_path_step(definition, bit, height)
        self.definition = definition


def prepare_basis_path(circuit: QuantumCircuit, path_register, path_bits: Sequence[int]) -> None:
    for position, bit in enumerate(path_bits):
        if bit == 1:
            circuit.x(path_register[position])


def append_hadamard_readout(
    circuit: QuantumCircuit,
    control,
    part: HadamardPart,
    measurement=None,
) -> None:
    if part == "imag":
        circuit.sdg(control)
    circuit.h(control)
    if measurement is not None:
        circuit.measure(control, measurement)


def controlled_varphi_gate(
    generator: BraidGenerator,
    target_qubits: int,
) -> ControlledGate:
    """Create the legacy full-target opaque controlled AJL block.

    Deprecated: Level-1 compilation builds local blocks with
    :func:`local_controlled_varphi_gate`, which takes a height-lane width
    rather than a whole-register target count.
    """

    warnings.warn(
        "controlled_varphi_gate(target_qubits=...) is deprecated; use "
        "local_controlled_varphi_gate(generator, height_qubits)",
        DeprecationWarning,
        stacklevel=2,
    )
    if target_qubits < 1:
        raise ValueError("a varphi block needs at least one target qubit")
    return _controlled_varphi_gate(generator, int(target_qubits))


def local_controlled_varphi_gate(
    generator: BraidGenerator,
    height_qubits: int,
) -> ControlledGate:
    """Create an opaque Level-1 block on one local path pair and height lane.

    ``height_qubits=0`` selects the block for a classically known height.
    """

    if height_qubits < 0:
        raise ValueError("a local varphi block cannot have negative height qubits")
    return _controlled_varphi_gate(generator, 2 + int(height_qubits))


def _controlled_varphi_gate(
    generator: BraidGenerator,
    target_qubits: int,
) -> ControlledGate:
    sign_name = "plus" if generator.sign == 1 else "minus"
    operation_name = f"c_varphi_sigma_{generator.index}_{sign_name}"
    exponent = "" if generator.sign == 1 else "⁻¹"
    label = f"c-φ(σ_{generator.index}{exponent})"
    base = Gate(
        operation_name.removeprefix("c_"),
        target_qubits,
        [],
        label=label.removeprefix("c-"),
    )
    return ControlledGate(
        name=operation_name,
        num_qubits=target_qubits + 1,
        params=[],
        label=label,
        num_ctrl_qubits=1,
        base_gate=base,
    )
