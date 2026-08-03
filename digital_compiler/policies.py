"""Composable synthesis policies for the AJL compiler."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from functools import cache
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


@dataclass(frozen=True)
class PrefixHeightTransition:
    """One clean/load, live/move, or live/unload height-lane transition."""

    lane: int
    source_index: int | None
    target_index: int | None

    @property
    def path_steps(self) -> int:
        if self.source_index is None:
            return 0 if self.target_index is None else self.target_index - 1
        if self.target_index is None:
            return self.source_index - 1
        return abs(self.target_index - self.source_index)


@dataclass(frozen=True)
class RoutedGenerator:
    """One scheduled braid-word position assigned to a physical height lane."""

    position: int
    lane: int


@dataclass(frozen=True)
class PrefixHeightLayerPlan:
    """Height transitions and generator assignments for one scheduled layer."""

    before: tuple[PrefixHeightTransition, ...]
    generators: tuple[RoutedGenerator, ...]
    after: tuple[PrefixHeightTransition, ...] = ()


@dataclass(frozen=True)
class PrefixHeightPlan:
    """Immutable prefix-height route for a complete scheduled braid word."""

    layers: tuple[PrefixHeightLayerPlan, ...]

    @property
    def transitions(self) -> tuple[PrefixHeightTransition, ...]:
        return tuple(
            transition
            for layer in self.layers
            for transition in (*layer.before, *layer.after)
        )

    @property
    def loads(self) -> int:
        return sum(
            transition.source_index is None and transition.target_index is not None
            for transition in self.transitions
        )

    @property
    def moves(self) -> int:
        return sum(
            transition.source_index is not None and transition.target_index is not None
            for transition in self.transitions
        )

    @property
    def unloads(self) -> int:
        return sum(
            transition.source_index is not None and transition.target_index is None
            for transition in self.transitions
        )

    @property
    def path_steps(self) -> int:
        return sum(transition.path_steps for transition in self.transitions)


class PrefixHeightPolicy(Protocol):
    """Strategy for routing prefix heights through scheduled generator layers."""

    name: str

    def route(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
        lane_capacity: int,
    ) -> PrefixHeightPlan: ...

    def metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class RecomputePrefixHeight:
    """Compute and erase every scheduled generator's prefix height."""

    name: str = "recompute"

    def route(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
        lane_capacity: int,
    ) -> PrefixHeightPlan:
        del lane_capacity
        layers = []
        for layer in schedule:
            routed = tuple(
                RoutedGenerator(position=position, lane=lane)
                for lane, position in enumerate(layer)
            )
            loads = tuple(
                PrefixHeightTransition(
                    lane=item.lane,
                    source_index=None,
                    target_index=word.generators[item.position].index,
                )
                for item in routed
            )
            unloads = tuple(
                PrefixHeightTransition(
                    lane=item.lane,
                    source_index=word.generators[item.position].index,
                    target_index=None,
                )
                for item in reversed(routed)
            )
            layers.append(
                PrefixHeightLayerPlan(
                    before=loads,
                    generators=routed,
                    after=unloads,
                )
            )
        return PrefixHeightPlan(tuple(layers))

    def metadata(self) -> dict[str, object]:
        return {"name": self.name}


@dataclass(frozen=True)
class _RoutingChoice:
    path_steps: int
    base_updates: int
    matches: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RollingPrefixHeight:
    """Retain height lanes and move them between nearby generator indices."""

    name: str = "rolling"

    @staticmethod
    def _match_existing_lanes(
        previous: tuple[tuple[int, int], ...],
        targets: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        @cache
        def best(
            previous_offset: int,
            target_offset: int,
        ) -> _RoutingChoice:
            if previous_offset == len(previous) and target_offset == len(targets):
                return _RoutingChoice(0, 0, ())

            candidates = []
            if previous_offset < len(previous) and target_offset < len(targets):
                suffix = best(previous_offset + 1, target_offset + 1)
                candidates.append(
                    _RoutingChoice(
                        path_steps=(
                            abs(
                                previous[previous_offset][0] - targets[target_offset][0]
                            )
                            + suffix.path_steps
                        ),
                        base_updates=suffix.base_updates,
                        matches=(
                            (previous_offset, target_offset),
                            *suffix.matches,
                        ),
                    )
                )
            if previous_offset < len(previous):
                suffix = best(previous_offset + 1, target_offset)
                candidates.append(
                    _RoutingChoice(
                        path_steps=(
                            previous[previous_offset][0] - 1 + suffix.path_steps
                        ),
                        base_updates=1 + suffix.base_updates,
                        matches=suffix.matches,
                    )
                )
            if target_offset < len(targets):
                suffix = best(previous_offset, target_offset + 1)
                candidates.append(
                    _RoutingChoice(
                        path_steps=targets[target_offset][0] - 1 + suffix.path_steps,
                        base_updates=1 + suffix.base_updates,
                        matches=suffix.matches,
                    )
                )

            def choice_key(choice: _RoutingChoice):
                matched_lanes_and_positions = tuple(
                    sorted(
                        (
                            previous[old_offset][1],
                            targets[new_offset][1],
                        )
                        for old_offset, new_offset in choice.matches
                    )
                )
                return (
                    choice.path_steps,
                    choice.base_updates,
                    matched_lanes_and_positions,
                )

            return min(candidates, key=choice_key)

        choice = best(0, 0)
        return tuple(
            (
                previous[previous_offset][1],
                targets[target_offset][1],
            )
            for previous_offset, target_offset in choice.matches
        )

    def route(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
        lane_capacity: int,
    ) -> PrefixHeightPlan:
        current: dict[int, int] = {}
        layers = []

        for layer in schedule:
            targets = tuple(
                sorted(
                    (
                        word.generators[position].index,
                        position,
                    )
                    for position in layer
                )
            )
            previous = tuple(sorted((index, lane) for lane, index in current.items()))
            matches = self._match_existing_lanes(previous, targets)
            matched_lanes = {lane for lane, _ in matches}
            matched_positions = {position for _, position in matches}

            assignment = {position: lane for lane, position in matches}
            available_lanes = (
                lane for lane in range(lane_capacity) if lane not in matched_lanes
            )
            for position in sorted(
                position for _, position in targets if position not in matched_positions
            ):
                assignment[position] = next(available_lanes)

            unloads = tuple(
                PrefixHeightTransition(
                    lane=lane,
                    source_index=current[lane],
                    target_index=None,
                )
                for lane in sorted(current)
                if lane not in matched_lanes
            )
            moves = tuple(
                PrefixHeightTransition(
                    lane=lane,
                    source_index=current[lane],
                    target_index=word.generators[position].index,
                )
                for lane, position in sorted(matches)
                if current[lane] != word.generators[position].index
            )
            loads = tuple(
                PrefixHeightTransition(
                    lane=assignment[position],
                    source_index=None,
                    target_index=word.generators[position].index,
                )
                for position in sorted(assignment)
                if assignment[position] not in matched_lanes
            )
            routed = tuple(
                RoutedGenerator(position=position, lane=assignment[position])
                for position in layer
            )
            layers.append(
                PrefixHeightLayerPlan(
                    before=(*unloads, *moves, *loads),
                    generators=routed,
                )
            )
            current = {
                assignment[position]: word.generators[position].index
                for position in layer
            }

        if layers:
            last = layers[-1]
            layers[-1] = PrefixHeightLayerPlan(
                before=last.before,
                generators=last.generators,
                after=tuple(
                    PrefixHeightTransition(
                        lane=lane,
                        source_index=current[lane],
                        target_index=None,
                    )
                    for lane in sorted(current, reverse=True)
                ),
            )

        return PrefixHeightPlan(tuple(layers))

    def metadata(self) -> dict[str, object]:
        return {"name": self.name}


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
    """Apply a local AJL crossing selected by a register encoding ``height - 1``.

    AJL model methods continue to use the mathematical vertex labels
    ``1, ..., k - 1``. The circuit selector and its alignment-angle table use
    the compact zero-based codes ``0, ..., k - 2``.
    """

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
            encoded_height = selected_height - 1
            binary_height = [
                (encoded_height >> bit) & 1 for bit in range(len(height))
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
class CliffordTConfig:
    """Configuration for approximate Level-4 Clifford+T synthesis."""

    synthesis_error_budget: float
    optimization_level: int = 2
    seed_transpiler: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.synthesis_error_budget, bool):
            raise ValueError(
                "synthesis_error_budget must be finite and in (0, 1]"
            )
        try:
            error = float(self.synthesis_error_budget)
        except (TypeError, ValueError):
            raise ValueError(
                "synthesis_error_budget must be finite and in (0, 1]"
            ) from None
        if not math.isfinite(error) or not 0.0 < error <= 1.0:
            raise ValueError(
                "synthesis_error_budget must be finite and in (0, 1]"
            )

        if isinstance(self.optimization_level, bool):
            raise ValueError("optimization_level must be an integer in 0..3")
        try:
            optimization_level = int(operator.index(self.optimization_level))
        except TypeError:
            raise ValueError(
                "optimization_level must be an integer in 0..3"
            ) from None
        if optimization_level not in range(4):
            raise ValueError("optimization_level must be an integer in 0..3")

        if isinstance(self.seed_transpiler, bool):
            raise ValueError("seed_transpiler must be a non-negative integer")
        try:
            seed_transpiler = int(operator.index(self.seed_transpiler))
        except TypeError:
            raise ValueError(
                "seed_transpiler must be a non-negative integer"
            ) from None
        if seed_transpiler < 0:
            raise ValueError("seed_transpiler must be a non-negative integer")

        object.__setattr__(self, "synthesis_error_budget", error)
        object.__setattr__(self, "optimization_level", optimization_level)
        object.__setattr__(self, "seed_transpiler", seed_transpiler)

    def metadata(self) -> dict[str, object]:
        return {
            "synthesis_error_budget": self.synthesis_error_budget,
            "optimization_level": self.optimization_level,
            "seed_transpiler": self.seed_transpiler,
            "allocation": "uniform",
            "cache_error": 0.0,
            "method": "qiskit_gridsynth",
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
    prefix_height: PrefixHeightPolicy = field(default_factory=RollingPrefixHeight)
    level4: CliffordTConfig | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "height_synthesis": self.height.name,
            "level_3": self.level3.metadata(),
            "generator_scheduling": self.scheduling.metadata(),
            "control_distribution": self.control_distribution.metadata(),
            "prefix_height": self.prefix_height.metadata(),
            "level_4": None if self.level4 is None else self.level4.metadata(),
        }
