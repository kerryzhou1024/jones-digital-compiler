"""Composable synthesis policies for the AJL compiler."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from typing import Protocol

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate

from .model import TOL, AJLPathModel, BraidWord
from .primitives import (
    RotationAxis,
    append_fixed_height_braid,
    append_phase_on_all_ones,
    append_phase_on_one,
    append_standard_toffoli,
    append_uniformly_controlled_rotation,
    append_uniformly_controlled_ry,
)

GeneratorSchedule = tuple[tuple[int, ...], ...]


class GeneratorSchedulingPolicy(Protocol):
    """Strategy for assigning braid-word positions to executable layers."""

    name: str

    def lane_capacity(self, strands: int) -> int: ...

    def schedule(self, word: BraidWord, strands: int) -> GeneratorSchedule: ...

    def metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SerialGeneratorScheduling:
    """Preserve the braid word exactly, one generator per layer."""

    name: str = "serial"

    @staticmethod
    def lane_capacity(strands: int) -> int:
        del strands
        return 1

    @staticmethod
    def schedule(word: BraidWord, strands: int) -> GeneratorSchedule:
        del strands
        return tuple((position,) for position in range(word.crossings))

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "max_lanes": 1}


@dataclass(frozen=True)
class CommutingLayerScheduling:
    """Critical-path scheduling of pairwise-distant braid generators."""

    max_lanes: int | None = None
    name: str = "commuting_layers"

    def __post_init__(self) -> None:
        if self.max_lanes is None:
            return
        if isinstance(self.max_lanes, bool):
            raise ValueError("max_lanes must be a positive integer or None")
        try:
            value = int(operator.index(self.max_lanes))
        except TypeError:
            raise ValueError("max_lanes must be a positive integer or None") from None
        if value <= 0:
            raise ValueError("max_lanes must be a positive integer or None")
        object.__setattr__(self, "max_lanes", value)

    def lane_capacity(self, strands: int) -> int:
        theoretical_maximum = max(1, int(strands) // 2)
        if self.max_lanes is None:
            return theoretical_maximum
        return min(self.max_lanes, theoretical_maximum)

    def schedule(self, word: BraidWord, strands: int) -> GeneratorSchedule:
        capacity = self.lane_capacity(strands)
        predecessors: list[set[int]] = [set() for _ in word.generators]
        successors: list[set[int]] = [set() for _ in word.generators]
        last_position_by_index: dict[int, int] = {}
        for position, generator in enumerate(word.generators):
            for index in (
                generator.index - 1,
                generator.index,
                generator.index + 1,
            ):
                predecessor = last_position_by_index.get(index)
                if predecessor is None:
                    continue
                predecessors[position].add(predecessor)
                successors[predecessor].add(position)
            last_position_by_index[generator.index] = position

        remaining_depth = [1] * word.crossings
        for position in reversed(range(word.crossings)):
            if successors[position]:
                remaining_depth[position] = 1 + max(
                    remaining_depth[successor] for successor in successors[position]
                )

        remaining_predecessors = [
            len(position_predecessors) for position_predecessors in predecessors
        ]
        ready = [
            (-remaining_depth[position], position)
            for position, count in enumerate(remaining_predecessors)
            if count == 0
        ]
        heapify(ready)
        layers: list[tuple[int, ...]] = []
        scheduled = 0

        while ready:
            selected = [heappop(ready)[1] for _ in range(min(capacity, len(ready)))]
            layers.append(tuple(sorted(selected)))
            scheduled += len(selected)

            for position in selected:
                for successor in successors[position]:
                    remaining_predecessors[successor] -= 1
                    if remaining_predecessors[successor] == 0:
                        heappush(
                            ready,
                            (-remaining_depth[successor], successor),
                        )

        if scheduled != word.crossings:
            raise AssertionError("generator dependency graph must be acyclic")
        return tuple(layers)

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "max_lanes": self.max_lanes}


class ControlDistributionPolicy(Protocol):
    """Strategy for distributing one coherent experiment control across lanes."""

    name: str

    def control_ancillas(self, lane_capacity: int) -> int: ...

    def prepare(
        self,
        circuit: QuantumCircuit,
        control,
        ancillas,
        active_width: int,
    ) -> tuple: ...

    def unprepare(
        self,
        circuit: QuantumCircuit,
        control,
        ancillas,
        active_width: int,
    ) -> None: ...

    def metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SharedControl:
    """Reuse one experiment control across lanes without clean ancillas."""

    name: str = "shared"

    @staticmethod
    def control_ancillas(lane_capacity: int) -> int:
        del lane_capacity
        return 0

    @staticmethod
    def prepare(
        circuit: QuantumCircuit,
        control,
        ancillas,
        active_width: int,
    ) -> tuple:
        del circuit
        if ancillas:
            raise ValueError("shared control distribution does not use ancillas")
        if active_width < 0:
            raise ValueError("active control width cannot be negative")
        return (control,) * active_width

    @staticmethod
    def unprepare(
        circuit: QuantumCircuit,
        control,
        ancillas,
        active_width: int,
    ) -> None:
        del circuit, control, active_width
        if ancillas:
            raise ValueError("shared control distribution does not use ancillas")

    def metadata(self) -> dict[str, object]:
        return {"name": self.name}


@dataclass(frozen=True)
class TreeControlFanout:
    """Distribute a coherent control with a logarithmic-depth CNOT tree."""

    name: str = "tree_fanout"

    @staticmethod
    def control_ancillas(lane_capacity: int) -> int:
        if lane_capacity < 1:
            raise ValueError("lane capacity must be positive")
        return lane_capacity - 1

    @staticmethod
    def _rounds(control, ancillas, active_width: int):
        ancillas = list(ancillas)
        if active_width < 0:
            raise ValueError("active control width cannot be negative")
        if active_width > 1 + len(ancillas):
            raise ValueError("active control width exceeds fanout capacity")
        if active_width == 0:
            return (), ()

        lane_controls = (control, *ancillas[: active_width - 1])
        rounds = []
        copied = 1
        while copied < len(lane_controls):
            new_count = min(copied, len(lane_controls) - copied)
            rounds.append(
                tuple(
                    (lane_controls[offset], lane_controls[copied + offset])
                    for offset in range(new_count)
                )
            )
            copied += new_count
        return lane_controls, tuple(rounds)

    @classmethod
    def prepare(
        cls,
        circuit: QuantumCircuit,
        control,
        ancillas,
        active_width: int,
    ) -> tuple:
        lane_controls, rounds = cls._rounds(
            control,
            ancillas,
            active_width,
        )
        for pairs in rounds:
            for source, target in pairs:
                circuit.cx(source, target)
        return lane_controls

    @classmethod
    def unprepare(
        cls,
        circuit: QuantumCircuit,
        control,
        ancillas,
        active_width: int,
    ) -> None:
        _, rounds = cls._rounds(control, ancillas, active_width)
        for pairs in reversed(rounds):
            for source, target in pairs:
                circuit.cx(source, target)

    def metadata(self) -> dict[str, object]:
        return {"name": self.name}


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
    scheduling: GeneratorSchedulingPolicy = field(
        default_factory=SerialGeneratorScheduling
    )
    control_distribution: ControlDistributionPolicy = field(
        default_factory=SharedControl
    )

    def metadata(self) -> dict[str, object]:
        return {
            "height_synthesis": self.height.name,
            "level_3": self.level3.metadata(),
            "generator_scheduling": self.scheduling.metadata(),
            "control_distribution": self.control_distribution.metadata(),
        }
