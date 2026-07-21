"""Composable synthesis policies for the AJL compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate

from .model import TOL, AJLPathModel
from .primitives import (
    RotationAxis,
    append_fixed_height_braid,
    append_phase_on_all_ones,
    append_phase_on_one,
    append_standard_toffoli,
    append_uniformly_controlled_rotation,
    append_uniformly_controlled_ry,
)


class HeightSynthesisPolicy(Protocol):
    """Strategy for applying the height-dependent local AJL crossing."""

    name: str

    def append_braid(
        self,
        circuit: QuantumCircuit,
        model: AJLPathModel,
        path,
        height,
        index: int,
        sign: int,
        projector_alignment_angles: tuple[float, ...],
        extra_controls=(),
    ) -> None: ...


@dataclass(frozen=True)
class MultiplexedHeightSynthesis:
    """One Gray-code UCRy sandwich indexed by the complete height register."""

    name: str = "multiplexed"

    def append_braid(
        self,
        circuit: QuantumCircuit,
        model: AJLPathModel,
        path,
        height,
        index: int,
        sign: int,
        projector_alignment_angles: tuple[float, ...],
        extra_controls=(),
    ) -> None:
        left = path[index - 1]
        right = path[index]
        base_angle, relative_angle = model.phase_angles(sign)
        extra_controls = list(extra_controls)

        # The base and rank-one eigenphases are independent of a valid AJL height.
        append_phase_on_all_ones(circuit, extra_controls, base_angle)
        selector = list(height)

        circuit.cx(left, right)
        append_uniformly_controlled_ry(
            circuit,
            selector,
            left,
            projector_alignment_angles,
        )
        circuit.x(left)
        append_phase_on_one(circuit, [*extra_controls, right], left, relative_angle)
        circuit.x(left)
        append_uniformly_controlled_ry(
            circuit,
            selector,
            left,
            tuple(-angle for angle in projector_alignment_angles),
        )
        circuit.cx(left, right)


@dataclass(frozen=True)
class SwitchCaseHeightSynthesis:
    """Direct equality-selected height baseline."""

    name: str = "switch_case"

    def append_braid(
        self,
        circuit: QuantumCircuit,
        model: AJLPathModel,
        path,
        height,
        index: int,
        sign: int,
        projector_alignment_angles: tuple[float, ...],
        extra_controls=(),
    ) -> None:
        del projector_alignment_angles
        left = path[index - 1]
        right = path[index]
        for selected_height in model.valid_heights:
            binary_height = [
                (selected_height >> bit) & 1 for bit in range(len(height))
            ]
            for bit, value in enumerate(binary_height):
                if value == 0:
                    circuit.x(height[bit])

            append_fixed_height_braid(
                circuit,
                model,
                [*extra_controls, *height],
                left,
                right,
                selected_height,
                sign,
            )

            for bit, value in enumerate(binary_height):
                if value == 0:
                    circuit.x(height[bit])


class MCXDecomposer(Protocol):
    """Strategy for lowering an all-ones multi-controlled X."""

    name: str

    def clean_ancillas(self, control_count: int) -> int: ...

    def append(self, circuit: QuantumCircuit, controls, target, work=()) -> None: ...


@dataclass(frozen=True)
class CleanAncillaMCX:
    """Exact clean-ancilla MCX ladder over standard Toffoli decompositions."""

    name: str = "clean_ancilla_toffoli_ladder"

    @staticmethod
    def clean_ancillas(control_count: int) -> int:
        if control_count < 0:
            raise ValueError("control count cannot be negative")
        return max(0, int(control_count) - 2)

    def append(self, circuit: QuantumCircuit, controls, target, work=()) -> None:
        controls = list(controls)
        available_work = list(work)
        if len(set(available_work)) != len(available_work):
            raise ValueError("MCX work qubits must be distinct")
        if any(qubit in controls or qubit == target for qubit in available_work):
            raise ValueError("MCX work qubits must not overlap controls or target")

        control_count = len(controls)
        if control_count == 0:
            circuit.x(target)
            return
        if control_count == 1:
            circuit.cx(controls[0], target)
            return
        if control_count == 2:
            append_standard_toffoli(circuit, controls[0], controls[1], target)
            return

        required = self.clean_ancillas(control_count)
        if len(available_work) < required:
            raise ValueError(
                f"need {required} clean adder_work qubits to lower a "
                f"{control_count}-controlled X, found {len(available_work)}"
            )
        ladder = available_work[:required]
        append_standard_toffoli(circuit, controls[0], controls[1], ladder[0])
        for index in range(2, control_count - 1):
            append_standard_toffoli(
                circuit,
                ladder[index - 2],
                controls[index],
                ladder[index - 1],
            )
        append_standard_toffoli(circuit, ladder[-1], controls[-1], target)
        for index in reversed(range(2, control_count - 1)):
            append_standard_toffoli(
                circuit,
                ladder[index - 2],
                controls[index],
                ladder[index - 1],
            )
        append_standard_toffoli(circuit, controls[0], controls[1], ladder[0])


class MCRDecomposer(Protocol):
    """Strategy for lowering a multi-controlled axis rotation."""

    name: str

    def append(
        self,
        circuit: QuantumCircuit,
        axis: RotationAxis,
        controls,
        target,
        angle: float,
    ) -> None: ...


@dataclass(frozen=True)
class GrayCodeMCR:
    """Ancilla-free sparse multiplexor decomposition of an ordinary MCR."""

    name: str = "gray_code_sparse_multiplexor"

    def append(
        self,
        circuit: QuantumCircuit,
        axis: RotationAxis,
        controls,
        target,
        angle: float,
    ) -> None:
        controls = list(controls)
        gates = {"x": RXGate, "y": RYGate, "z": RZGate}
        if axis not in gates:
            raise ValueError(f"unsupported rotation axis {axis!r}")
        if not controls:
            circuit.append(gates[axis](float(angle)), [target])
            return
        if len(controls) == 1:
            circuit.append(
                gates[axis](float(angle)).control(1, annotated=False),
                [controls[0], target],
            )
            return

        desired = [0.0] * (1 << len(controls))
        desired[-1] = float(angle)
        if axis == "x":
            circuit.h(target)
            append_uniformly_controlled_rotation(
                circuit,
                controls,
                target,
                desired,
                axis="z",
            )
            circuit.h(target)
        else:
            append_uniformly_controlled_rotation(
                circuit,
                controls,
                target,
                desired,
                axis=axis,
            )


class MCPhaseDecomposer(Protocol):
    """Strategy for lowering a phase on the all-ones projector."""

    name: str

    def append(
        self,
        circuit: QuantumCircuit,
        qubits,
        angle: float,
        rotations: MCRDecomposer,
    ) -> None: ...


@dataclass(frozen=True)
class RecursiveMCPhase:
    """Exact recursive phase correction using selected multi-controlled Rz gates."""

    name: str = "recursive_corrected_rz"

    def append(
        self,
        circuit: QuantumCircuit,
        qubits,
        angle: float,
        rotations: MCRDecomposer,
    ) -> None:
        qubits = list(qubits)
        if abs(angle) < TOL:
            return
        if not qubits:
            circuit.global_phase += float(angle)
            return
        if len(qubits) == 1:
            circuit.rz(float(angle), qubits[0])
            circuit.global_phase += float(angle) / 2.0
            return

        controls, target = qubits[:-1], qubits[-1]
        rotations.append(circuit, "z", controls, target, angle)
        self.append(circuit, controls, angle / 2.0, rotations)


@dataclass(frozen=True)
class NoAncillaMCX:
    """Exact ancilla-free MCX using recursive phase and Gray-code multiplexing."""

    name: str = "no_ancilla_recursive_phase"

    @staticmethod
    def clean_ancillas(control_count: int) -> int:
        if control_count < 0:
            raise ValueError("control count cannot be negative")
        return 0

    def append(self, circuit: QuantumCircuit, controls, target, work=()) -> None:
        del work
        controls = list(controls)
        control_count = len(controls)
        if control_count == 0:
            circuit.x(target)
            return
        if control_count == 1:
            circuit.cx(controls[0], target)
            return
        if control_count == 2:
            append_standard_toffoli(circuit, controls[0], controls[1], target)
            return

        circuit.h(target)
        RecursiveMCPhase().append(
            circuit,
            [*controls, target],
            math.pi,
            GrayCodeMCR(),
        )
        circuit.h(target)


@dataclass(frozen=True)
class Level3Policy:
    """Independent decomposition choices used by the Level-3 lowering pass."""

    mcx: MCXDecomposer = field(default_factory=NoAncillaMCX)
    rotations: MCRDecomposer = field(default_factory=GrayCodeMCR)
    phases: MCPhaseDecomposer = field(default_factory=RecursiveMCPhase)

    def metadata(self) -> dict[str, str]:
        return {
            "mcx": self.mcx.name,
            "multi_controlled_rotations": self.rotations.name,
            "multi_controlled_phase": self.phases.name,
        }


@dataclass(frozen=True)
class CompilerConfig:
    """Immutable collection of independently replaceable compiler policies."""

    height: HeightSynthesisPolicy = field(default_factory=MultiplexedHeightSynthesis)
    level3: Level3Policy = field(default_factory=Level3Policy)

    def metadata(self) -> dict[str, object]:
        return {
            "height_synthesis": self.height.name,
            "level_3": self.level3.metadata(),
        }
