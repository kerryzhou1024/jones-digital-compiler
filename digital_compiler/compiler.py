"""Policy-driven AJL circuit compiler."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Qubit

from .fault_tolerance import (
    CliffordTCompilation,
    CliffordTCompiler,
    LogicalResourceReport,
)
from .lowering import SingleControlLowerer, assert_level_2_contract
from .model import AJLPathModel, BraidGenerator, BraidWord, HadamardPart
from .policies import (
    CompilerConfig,
    FinalHeightPolicy,
    GeneratorSchedule,
    PrefixHeightPlan,
    PrefixHeightTransition,
    RoutedGenerator,
)
from .primitives import (
    PrefixAdderGate,
    QuantumAdder,
    append_hadamard_readout,
    local_controlled_varphi_gate,
    prepare_basis_path,
)

LEVEL_4_STATUS = (
    "Level 4 is not configured; supply CompilerConfig(level4=CliffordTConfig(...))."
)

CompilerLevel = Literal[1, 2, 3, 4]
_HEIGHT_ENCODING = "vertex_minus_one"


def register_signature(
    circuit: QuantumCircuit,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    return (
        tuple((register.name, register.size) for register in circuit.qregs),
        tuple((register.name, register.size) for register in circuit.cregs),
    )


@dataclass(frozen=True)
class HadamardTestCompilation:
    """Matched circuit representations for one Hadamard-test component."""

    word: BraidWord
    initial_path: tuple[int, ...]
    part: HadamardPart
    level_1_varphi: QuantumCircuit
    level_2_multicontrolled: QuantumCircuit
    level_3_single_control: QuantumCircuit
    config: CompilerConfig
    level_4_clifford_t: QuantumCircuit | None = None
    level_4_resources: LogicalResourceReport | None = None
    level_4_status: str = LEVEL_4_STATUS

    def __post_init__(self) -> None:
        if (self.level_4_clifford_t is None) != (self.level_4_resources is None):
            raise ValueError(
                "Level 4 circuit and resource report must be provided together"
            )
        circuits = [circuit for _, circuit in self.levels]
        core_signatures = {
            (
                tuple(
                    (register.name, register.size)
                    for register in circuit.qregs
                    if register.name in {"ctrl", "path"}
                ),
                tuple((register.name, register.size) for register in circuit.cregs),
            )
            for circuit in circuits
        }
        if len(core_signatures) != 1:
            raise ValueError(
                "compiler levels must share control, path, and measurement registers"
            )
        expected_path_size = len(self.initial_path)
        for circuit in circuits:
            path_registers = [
                register for register in circuit.qregs if register.name == "path"
            ]
            control_registers = [
                register for register in circuit.qregs if register.name == "ctrl"
            ]
            if (
                len(path_registers) != 1
                or path_registers[0].size != expected_path_size
                or len(control_registers) != 1
                or control_registers[0].size != 1
            ):
                raise ValueError(
                    "compiler levels must contain one matching path register and control"
                )

    @property
    def path_label(self) -> str:
        return "".join(str(bit) for bit in self.initial_path)

    @property
    def levels(self) -> tuple[tuple[CompilerLevel, QuantumCircuit], ...]:
        """Return every compiled level of this component in compiler order."""

        levels: list[tuple[CompilerLevel, QuantumCircuit]] = [
            (1, self.level_1_varphi),
            (2, self.level_2_multicontrolled),
            (3, self.level_3_single_control),
        ]
        if self.level_4_clifford_t is not None:
            levels.append((4, self.level_4_clifford_t))
        return tuple(levels)

    @property
    def logical_qubits(self) -> int:
        """Return the width of the widest level; Level 1 carries no workspace."""

        return max(circuit.num_qubits for _, circuit in self.levels)


@dataclass(frozen=True)
class _Layout:
    """The schedule, height route, and register widths of one braid word.

    Lane-indexed registers are sized from the widest scheduled layer rather
    than from the policy's lane budget, so a word that never runs two
    generators together never pays for a second height lane.
    """

    word: BraidWord
    schedule: GeneratorSchedule
    plan: PrefixHeightPlan
    strands: int
    lanes: int
    height_qubits: int
    work_qubits_per_lane: int
    work_qubits: int
    fanout_qubits: int

    @property
    def logical_qubits(self) -> int:
        return (
            1
            + self.strands
            + self.height_qubits
            + self.work_qubits
            + self.fanout_qubits
        )


@dataclass(frozen=True)
class _Registers:
    """A freshly allocated circuit and the qubits its builder writes to.

    ``control`` and ``measurement`` are the single control qubit and readout
    bit rather than their registers, and the lowering workspace is absent
    because only the lowerer reaches it -- by register name, from the circuit.
    """

    circuit: QuantumCircuit
    path: QuantumRegister
    height: QuantumRegister
    control: Qubit | None = None
    fanout: tuple[Qubit, ...] = ()
    measurement: Clbit | None = None


class AJLCompiler:
    """Compile AJL braid words through explicit semantic and lowering layers."""

    def __init__(
        self,
        model: AJLPathModel,
        config: CompilerConfig | None = None,
    ):
        self.model = model
        self.config = CompilerConfig() if config is None else config
        self._clifford_t_compiler = (
            None
            if self.config.level4 is None
            else CliffordTCompiler(self.config.level4)
        )
        self.strands = model.strands
        self.level = model.level
        # Encode the k - 1 valid vertices h = 1, ..., k - 1 as h - 1.
        self.height_selector_qubits = (self.level - 2).bit_length()
        self.adder = QuantumAdder(self.height_selector_qubits)
        self.lane_capacity = self._validate_lane_capacity(
            self.config.scheduling.lane_capacity(self.strands)
        )
        max_adder_controls = self.height_selector_qubits - 1
        self.work_qubits_per_lane = int(
            self.config.level3.mcx.clean_ancillas(max_adder_controls)
        )
        if self.work_qubits_per_lane < 0:
            raise ValueError("an MCX policy cannot request negative workspace")
        self.lowerer = SingleControlLowerer(self.config.level3)
        (
            self.projector_basis_angles,
            self.projector_alignment_angles,
        ) = self._build_projector_alignment_angles()

    def _control_ancilla_count(self, lane_count: int) -> int:
        if lane_count == 0:
            return 0
        raw_control_ancillas = self.config.control_distribution.control_ancillas(
            lane_count
        )
        if isinstance(raw_control_ancillas, bool):
            raise ValueError(
                "a control distribution policy must request a nonnegative "
                "integer number of ancillas"
            )
        try:
            control_ancillas = int(operator.index(raw_control_ancillas))
        except TypeError:
            raise ValueError(
                "a control distribution policy must request a nonnegative "
                "integer number of ancillas"
            ) from None
        if control_ancillas < 0:
            raise ValueError(
                "a control distribution policy cannot request negative ancillas"
            )
        return control_ancillas

    def _build_projector_alignment_angles(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        basis_angles = [0.0] * (1 << self.height_selector_qubits)
        for height in self.model.valid_heights:
            basis_angles[height - 1] = self.model.projector_angle(height)
        return tuple(basis_angles), tuple(
            math.pi - 2.0 * angle for angle in basis_angles
        )

    def logical_qubits_for(self, word: BraidWord | str | Sequence[int]) -> int:
        """Return the Level-2 logical width required by one braid word."""

        return self._layout(word).logical_qubits

    def _validate_lane_capacity(self, raw_capacity: object) -> int:
        if isinstance(raw_capacity, bool):
            raise ValueError("a scheduling policy lane capacity must be a positive integer")
        try:
            capacity = int(operator.index(raw_capacity))
        except TypeError:
            raise ValueError(
                "a scheduling policy lane capacity must be a positive integer"
            ) from None
        theoretical_maximum = max(1, self.strands // 2)
        if capacity < 1 or capacity > theoretical_maximum:
            raise ValueError(
                "a scheduling policy lane capacity must be in "
                f"1..{theoretical_maximum} for {self.strands} strands"
            )
        return capacity

    def _config_metadata(self, layout: _Layout) -> dict[str, object]:
        metadata = self.config.metadata()
        metadata["workspace_qubits"] = layout.work_qubits
        metadata["workspace_qubits_per_lane"] = layout.work_qubits_per_lane
        metadata["height_encoding"] = _HEIGHT_ENCODING
        metadata["height_selector_qubits"] = self.height_selector_qubits
        metadata["height_register_qubits"] = layout.height_qubits
        metadata["lane_capacity"] = self.lane_capacity
        metadata["parallel_lanes"] = layout.lanes
        metadata["control_fanout_qubits"] = layout.fanout_qubits
        return metadata

    def _new_circuit(
        self,
        name: str,
        layout: _Layout,
        *,
        controlled: bool,
        readout: bool,
    ) -> _Registers:
        """Allocate a lowerable circuit: control, path, height, workspace, fanout.

        ``readout`` allocates the 1-bit readout register that makes this a
        Hadamard-test component; whether the measurement is applied is the
        caller's choice.
        """

        control = QuantumRegister(1, "ctrl") if controlled else None
        path = QuantumRegister(self.strands, "path")
        height = QuantumRegister(layout.height_qubits, "height")
        work = QuantumRegister(layout.work_qubits, "adder_work")
        fanout = (
            QuantumRegister(layout.fanout_qubits, "ctrl_fanout")
            if controlled and layout.fanout_qubits
            else None
        )
        measurement = ClassicalRegister(1, "meas") if readout else None
        return self._assemble(
            name,
            (control, path, height, work, fanout),
            measurement,
        )

    def _new_level_1_circuit(self, name: str, layout: _Layout) -> _Registers:
        """Allocate a semantic circuit: no lowering workspace, fanout beside the control."""

        control = QuantumRegister(1, "ctrl")
        fanout = (
            QuantumRegister(layout.fanout_qubits, "ctrl_fanout")
            if layout.fanout_qubits
            else None
        )
        path = QuantumRegister(self.strands, "path")
        height = QuantumRegister(layout.height_qubits, "height")
        return self._assemble(
            name,
            (control, fanout, path, height),
            ClassicalRegister(1, "meas"),
        )

    @staticmethod
    def _assemble(
        name: str,
        quantum: Sequence[QuantumRegister | None],
        measurement: ClassicalRegister | None,
    ) -> _Registers:
        """Build a circuit whose qubit order is the given register order."""

        allocated = [register for register in quantum if register is not None]
        by_name = {register.name: register for register in allocated}
        circuit = QuantumCircuit(
            *allocated,
            *(() if measurement is None else (measurement,)),
            name=name,
        )
        control = by_name.get("ctrl")
        fanout = by_name.get("ctrl_fanout")
        return _Registers(
            circuit=circuit,
            path=by_name["path"],
            height=by_name["height"],
            control=None if control is None else control[0],
            fanout=() if fanout is None else tuple(fanout),
            measurement=None if measurement is None else measurement[0],
        )

    @staticmethod
    def validate_part(part: str) -> HadamardPart:
        if part not in {"real", "imag"}:
            raise ValueError("part must be 'real' or 'imag'")
        return part

    def _schedule(self, word: BraidWord) -> GeneratorSchedule:
        raw_schedule = self.config.scheduling.schedule(word, self.strands)
        try:
            raw_layers = tuple(raw_schedule)
        except TypeError:
            raise ValueError("a generator schedule must be an iterable of layers") from None

        layers: list[tuple[int, ...]] = []
        for raw_layer in raw_layers:
            try:
                raw_positions = tuple(raw_layer)
            except TypeError:
                raise ValueError("each generator schedule layer must be iterable") from None
            if not raw_positions:
                raise ValueError("generator schedule layers cannot be empty")
            positions = []
            for raw_position in raw_positions:
                if isinstance(raw_position, bool):
                    raise ValueError("generator schedule positions must be integers")
                try:
                    position = int(operator.index(raw_position))
                except TypeError:
                    raise ValueError(
                        "generator schedule positions must be integers"
                    ) from None
                positions.append(position)
            layers.append(tuple(positions))

        schedule = tuple(layers)
        self._validate_schedule(word, schedule)
        return schedule

    def _validate_schedule(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
    ) -> None:
        if any(len(layer) > self.lane_capacity for layer in schedule):
            raise ValueError(
                "a generator schedule layer exceeds the configured lane capacity"
            )

        flattened = tuple(position for layer in schedule for position in layer)
        expected_positions = tuple(range(word.crossings))
        if any(
            position < 0 or position >= word.crossings for position in flattened
        ):
            raise ValueError("a generator schedule position is out of range")
        if len(flattened) != word.crossings or sorted(flattened) != list(
            expected_positions
        ):
            raise ValueError(
                "a generator schedule must contain every braid-word position exactly once"
            )

        layer_by_position: dict[int, int] = {}
        for layer_index, layer in enumerate(schedule):
            for position in layer:
                layer_by_position[position] = layer_index
            for left_offset, left_position in enumerate(layer):
                left_index = word.generators[left_position].index
                for right_position in layer[left_offset + 1 :]:
                    right_index = word.generators[right_position].index
                    if abs(left_index - right_index) < 2:
                        raise ValueError(
                            "generators in one schedule layer must be pairwise distant"
                        )

        for earlier_position, earlier in enumerate(word.generators):
            for later_position in range(earlier_position + 1, word.crossings):
                later = word.generators[later_position]
                if (
                    abs(earlier.index - later.index) <= 1
                    and layer_by_position[earlier_position]
                    >= layer_by_position[later_position]
                ):
                    raise ValueError(
                        "a generator schedule cannot reorder noncommuting generators"
                    )

    def _schedule_metadata(
        self,
        layout: _Layout,
        final_height_strategy: str,
    ) -> dict[str, object]:
        signed_layers = tuple(
            tuple(layout.word.generators[position].signed_index for position in layer)
            for layer in layout.schedule
        )
        plan = layout.plan
        return {
            "generator_scheduling": self.config.scheduling.name,
            "control_distribution": self.config.control_distribution.name,
            "generator_layers": signed_layers,
            "parallel_lanes": layout.lanes,
            "height_encoding": _HEIGHT_ENCODING,
            "height_selector_qubits": self.height_selector_qubits,
            "workspace_qubits_per_lane": layout.work_qubits_per_lane,
            "prefix_height_strategy": self.config.prefix_height.name,
            "final_height_strategy": final_height_strategy,
            "prefix_height_loads": plan.loads,
            "prefix_height_moves": plan.moves,
            "prefix_height_unloads": plan.unloads,
            "prefix_height_path_steps": plan.path_steps,
        }

    def _circuit_metadata(
        self,
        layout: _Layout,
        final_height_strategy: str,
        **extra: object,
    ) -> dict[str, object]:
        return {
            **extra,
            "compiler_config": self._config_metadata(layout),
            **self._schedule_metadata(layout, final_height_strategy),
        }

    def _height_lane(self, height, lane: int) -> list:
        start = lane * self.height_selector_qubits
        stop = start + self.height_selector_qubits
        return list(height[start:stop])

    def _layout(
        self,
        word: BraidWord | str | Sequence[int],
        final_height: FinalHeightPolicy | None = None,
    ) -> _Layout:
        """Schedule one braid word and size its lane-indexed registers."""

        braid_word = self.model.as_braid_word(word)
        schedule = self._schedule(braid_word)
        # Every valid plan keeps exactly one live lane per generator in a
        # layer, so the widest layer is the peak lane occupancy.
        lanes = max((len(layer) for layer in schedule), default=0)
        plan = self._prefix_height_plan(braid_word, schedule, lanes, final_height)
        # MCX workspace only exists to lower the prefix adders. A route with no
        # path steps emits no adder -- every generator sits at index 1, whose
        # prefix is empty -- so the register is not allocated at all. Workspace
        # stays uniform per lane above that, so a lane that happens to route
        # only index-1 generators still carries its (idle) share.
        work_per_lane = self.work_qubits_per_lane if plan.path_steps else 0
        return _Layout(
            word=braid_word,
            schedule=schedule,
            plan=plan,
            strands=self.strands,
            lanes=lanes,
            height_qubits=self.height_selector_qubits * lanes,
            work_qubits_per_lane=work_per_lane,
            work_qubits=work_per_lane * lanes,
            fanout_qubits=self._control_ancilla_count(lanes),
        )

    def _prefix_height_plan(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
        lanes: int,
        final_height: FinalHeightPolicy | None = None,
    ) -> PrefixHeightPlan:
        plan = self.config.prefix_height.route(word, schedule, lanes)
        self._validate_prefix_height_plan(word, schedule, lanes, plan)
        if final_height is None:
            return plan

        finalized = final_height.finalize(plan)
        self._validate_prefix_height_plan(
            word,
            schedule,
            lanes,
            finalized,
            require_clean_completion=final_height.clean_at_completion,
            allow_retained_inactive=not final_height.clean_at_completion,
        )
        return finalized

    def _validate_prefix_height_plan(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
        lanes: int,
        plan: PrefixHeightPlan,
        *,
        require_clean_completion: bool = True,
        allow_retained_inactive: bool = False,
    ) -> None:
        if not isinstance(plan, PrefixHeightPlan):
            raise ValueError("a prefix-height policy must return a PrefixHeightPlan")
        if len(plan.layers) != len(schedule):
            raise ValueError(
                "a prefix-height plan must contain one route for each generator layer"
            )

        lane_state: list[int | None] = [None] * lanes

        def apply_transition(transition: PrefixHeightTransition) -> None:
            lane = transition.lane
            if (
                isinstance(lane, bool)
                or not isinstance(lane, int)
                or lane < 0
                or lane >= lanes
            ):
                raise ValueError("a prefix-height transition has an invalid lane")
            if transition.source_index != lane_state[lane]:
                raise ValueError(
                    "a prefix-height transition source does not match its lane state"
                )
            if transition.target_index is not None and (
                transition.target_index < 1 or transition.target_index >= self.strands
            ):
                raise ValueError("a prefix-height transition has an invalid target")
            if transition.source_index is None and transition.target_index is None:
                raise ValueError("a prefix-height transition cannot be a no-op")
            lane_state[lane] = transition.target_index

        for expected_layer, routed_layer in zip(
            schedule,
            plan.layers,
            strict=True,
        ):
            for transition in routed_layer.before:
                apply_transition(transition)

            positions = tuple(item.position for item in routed_layer.generators)
            if positions != expected_layer:
                raise ValueError(
                    "a prefix-height plan must preserve each scheduled generator layer"
                )
            layer_lanes = tuple(item.lane for item in routed_layer.generators)
            if len(set(layer_lanes)) != len(layer_lanes):
                raise ValueError(
                    "a prefix-height plan cannot reuse one lane within a layer"
                )
            for item in routed_layer.generators:
                if (
                    isinstance(item.lane, bool)
                    or not isinstance(item.lane, int)
                    or item.lane < 0
                    or item.lane >= lanes
                ):
                    raise ValueError("a routed generator has an invalid height lane")
                expected_index = word.generators[item.position].index
                if lane_state[item.lane] != expected_index:
                    raise ValueError(
                        "a routed generator does not have its required prefix height"
                    )

            active_lanes = set(layer_lanes)
            dirty_lanes = {
                lane for lane, index in enumerate(lane_state) if index is not None
            }
            inactive_lanes = dirty_lanes - active_lanes
            if inactive_lanes and not allow_retained_inactive:
                raise ValueError(
                    "a prefix-height plan must clean every inactive lane before a layer"
                )
            for lane in inactive_lanes:
                height_index = lane_state[lane]
                assert height_index is not None
                if any(
                    word.generators[item.position].index == height_index - 1
                    for item in routed_layer.generators
                ):
                    raise ValueError(
                        "a retained inactive prefix-height lane is invalidated by "
                        "a generator crossing its boundary"
                    )

            for transition in routed_layer.after:
                apply_transition(transition)

        if require_clean_completion and any(index is not None for index in lane_state):
            raise ValueError("a prefix-height plan must clean every lane at completion")

    def _compute_prefix_height(self, circuit, path, height, index: int) -> None:
        # The clean all-zero lane already encodes the initial AJL vertex h = 1.
        for prefix_bit in path[: index - 1]:
            self.adder.add_path_step(circuit, prefix_bit, height)

    def _uncompute_prefix_height(self, circuit, path, height, index: int) -> None:
        for prefix_bit in reversed(path[: index - 1]):
            self.adder.subtract_path_step(circuit, prefix_bit, height)

    def _transition_prefix_height(
        self,
        circuit,
        path,
        height,
        source_index: int | None,
        target_index: int | None,
    ) -> None:
        if source_index is None:
            assert target_index is not None
            self._compute_prefix_height(circuit, path, height, target_index)
            return
        if target_index is None:
            self._uncompute_prefix_height(circuit, path, height, source_index)
            return
        if target_index > source_index:
            for path_index in range(source_index - 1, target_index - 1):
                self.adder.add_path_step(circuit, path[path_index], height)
        elif target_index < source_index:
            for path_index in range(source_index - 2, target_index - 2, -1):
                self.adder.subtract_path_step(circuit, path[path_index], height)

    def _append_height_selected_generator(
        self,
        circuit,
        path,
        height,
        generator: BraidGenerator,
        experiment_control=None,
    ) -> None:
        if generator.index < 1 or generator.index >= self.strands:
            raise ValueError(f"generator index must be in 1..{self.strands - 1}")
        extra_controls = () if experiment_control is None else (experiment_control,)
        self.config.height.append_braid(
            circuit,
            self.model,
            path,
            height,
            generator.index,
            generator.sign,
            self.projector_alignment_angles,
            extra_controls=extra_controls,
        )

    def _append_logical_plan(
        self,
        circuit,
        path,
        height,
        word: BraidWord,
        plan: PrefixHeightPlan,
        lane_controls=(),
    ) -> None:
        controls = list(lane_controls)
        for event in self._plan_events(plan):
            if isinstance(event, PrefixHeightTransition):
                self._transition_prefix_height(
                    circuit,
                    path,
                    self._height_lane(height, event.lane),
                    event.source_index,
                    event.target_index,
                )
            else:
                if not isinstance(event, RoutedGenerator):
                    raise TypeError("a prefix-height plan contains an unknown event")
                experiment_control = None if not controls else controls[event.lane]
                self._append_height_selected_generator(
                    circuit,
                    path,
                    self._height_lane(height, event.lane),
                    word.generators[event.position],
                    experiment_control,
                )

    @staticmethod
    def _plan_events(plan: PrefixHeightPlan):
        for layer in plan.layers:
            yield from layer.before
            yield from layer.generators
            yield from layer.after

    @staticmethod
    def _transition_path_indices(
        transition: PrefixHeightTransition,
    ) -> tuple[tuple[int, ...], bool]:
        source = transition.source_index
        target = transition.target_index
        if source is None:
            assert target is not None
            return tuple(range(target - 1)), False
        if target is None:
            return tuple(range(source - 1)), True
        if target > source:
            return tuple(range(source - 1, target - 1)), False
        return tuple(range(target - 1, source - 1)), True

    def _append_level_1_plan(
        self,
        circuit,
        lane_controls,
        path,
        height,
        word: BraidWord,
        plan: PrefixHeightPlan,
    ) -> None:
        controls = list(lane_controls)
        for event in self._plan_events(plan):
            height_lane = self._height_lane(height, event.lane)
            if isinstance(event, PrefixHeightTransition):
                path_indices, inverse = self._transition_path_indices(event)
                if not path_indices:
                    continue
                circuit.append(
                    PrefixAdderGate(
                        path_indices,
                        self.height_selector_qubits,
                        inverse=inverse,
                    ),
                    [*(path[index] for index in path_indices), *height_lane],
                )
                continue

            if not isinstance(event, RoutedGenerator):
                raise TypeError("a prefix-height plan contains an unknown event")
            generator = word.generators[event.position]
            circuit.append(
                local_controlled_varphi_gate(
                    generator,
                    self.height_selector_qubits,
                ),
                [
                    controls[event.lane],
                    path[generator.index - 1],
                    path[generator.index],
                    *height_lane,
                ],
            )

    def lower_to_level_3(
        self,
        level_2_circuit: QuantumCircuit,
        name: str | None = None,
    ) -> QuantumCircuit:
        return self.lowerer.lower(level_2_circuit, name)

    def lower_to_level_4(
        self,
        level_3_circuit: QuantumCircuit,
    ) -> CliffordTCompilation:
        if self._clifford_t_compiler is None:
            raise ValueError(
                "circuit_level=4 requires "
                "CompilerConfig(level4=CliffordTConfig(...))"
            )
        return self._clifford_t_compiler.compile(level_3_circuit)

    def _braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
        *,
        controlled: bool,
    ) -> QuantumCircuit:
        """Build one standalone braid unitary with clean final heights."""

        layout = self._layout(word)
        policy_name = self.config.height.name
        prefix = "controlled_" if controlled else ""
        registers = self._new_circuit(
            f"{prefix}level_2_braid_{policy_name}({layout.word})",
            layout,
            controlled=controlled,
            readout=False,
        )
        circuit = registers.circuit
        circuit.metadata = self._circuit_metadata(
            layout,
            "uncompute",
            compiler_level=2,
            height_strategy=policy_name,
            gate_contract="ajl_multicontrolled",
        )
        lane_controls = ()
        if controlled:
            lane_controls = self.config.control_distribution.prepare(
                circuit,
                registers.control,
                registers.fanout,
                layout.lanes,
            )
        self._append_logical_plan(
            circuit,
            registers.path,
            registers.height,
            layout.word,
            layout.plan,
            lane_controls,
        )
        if controlled:
            self.config.control_distribution.unprepare(
                circuit,
                registers.control,
                registers.fanout,
                layout.lanes,
            )
        assert_level_2_contract(circuit)
        return circuit

    def level_2_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        return self._braid_circuit(word, controlled=False)

    def controlled_level_2_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        return self._braid_circuit(word, controlled=True)

    def level_3_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        level_2 = self.level_2_braid_circuit(word)
        return self.lower_to_level_3(
            level_2,
            level_2.name.replace("level_2_braid", "level_3_braid"),
        )

    def controlled_level_3_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        level_2 = self.controlled_level_2_braid_circuit(word)
        return self.lower_to_level_3(
            level_2,
            level_2.name.replace("level_2_braid", "level_3_braid"),
        )

    def level_1_varphi_circuit(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
        measure: bool = True,
    ) -> QuantumCircuit:
        layout = self._layout(word, self.config.final_height)
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        registers = self._new_level_1_circuit(f"level_1_varphi_{part}", layout)
        circuit = registers.circuit
        circuit.metadata = self._circuit_metadata(
            layout,
            self.config.final_height.name,
            compiler_level=1,
            gate_contract="ajl_level_1_semantic_blocks",
        )
        prepare_basis_path(circuit, registers.path, path_bits)
        circuit.h(registers.control)
        lane_controls = self.config.control_distribution.prepare(
            circuit,
            registers.control,
            registers.fanout,
            layout.lanes,
        )
        self._append_level_1_plan(
            circuit,
            lane_controls,
            registers.path,
            registers.height,
            layout.word,
            layout.plan,
        )
        self.config.control_distribution.unprepare(
            circuit,
            registers.control,
            registers.fanout,
            layout.lanes,
        )
        append_hadamard_readout(
            circuit,
            registers.control,
            part,
            registers.measurement if measure else None,
        )
        return circuit

    def level_2_multicontrolled_circuit(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
        measure: bool = True,
    ) -> QuantumCircuit:
        layout = self._layout(word, self.config.final_height)
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        policy_name = self.config.height.name
        registers = self._new_circuit(
            f"level_2_multicontrolled_{policy_name}_{part}",
            layout,
            controlled=True,
            readout=True,
        )
        circuit = registers.circuit
        circuit.metadata = self._circuit_metadata(
            layout,
            self.config.final_height.name,
            compiler_level=2,
            height_strategy=policy_name,
            gate_contract="ajl_multicontrolled",
        )
        prepare_basis_path(circuit, registers.path, path_bits)
        circuit.h(registers.control)
        lane_controls = self.config.control_distribution.prepare(
            circuit,
            registers.control,
            registers.fanout,
            layout.lanes,
        )
        self._append_logical_plan(
            circuit,
            registers.path,
            registers.height,
            layout.word,
            layout.plan,
            lane_controls,
        )
        self.config.control_distribution.unprepare(
            circuit,
            registers.control,
            registers.fanout,
            layout.lanes,
        )
        append_hadamard_readout(
            circuit,
            registers.control,
            part,
            registers.measurement if measure else None,
        )
        assert_level_2_contract(circuit)
        return circuit

    def level_3_single_control_circuit(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
        measure: bool = True,
    ) -> QuantumCircuit:
        level_2 = self.level_2_multicontrolled_circuit(
            word,
            initial_path,
            part,
            measure=measure,
        )
        return self._lower_component(level_2)

    def _lower_component(self, level_2: QuantumCircuit) -> QuantumCircuit:
        return self.lower_to_level_3(
            level_2,
            level_2.name.replace(
                "level_2_multicontrolled",
                "level_3_single_control",
            ),
        )

    def level_4_clifford_t_circuit(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
        measure: bool = True,
    ) -> QuantumCircuit:
        level_3 = self.level_3_single_control_circuit(
            word,
            initial_path,
            part,
            measure=measure,
        )
        return self.lower_to_level_4(level_3).circuit

    def compile_component(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart,
        *,
        circuit_level: CompilerLevel = 3,
        measure: bool = True,
    ) -> QuantumCircuit:
        """Compile one real or imaginary Hadamard-test circuit at one level."""

        if circuit_level == 1:
            return self.level_1_varphi_circuit(
                word,
                initial_path,
                part,
                measure=measure,
            )
        if circuit_level == 2:
            return self.level_2_multicontrolled_circuit(
                word,
                initial_path,
                part,
                measure=measure,
            )
        if circuit_level == 3:
            return self.level_3_single_control_circuit(
                word,
                initial_path,
                part,
                measure=measure,
            )
        if circuit_level == 4:
            return self.level_4_clifford_t_circuit(
                word,
                initial_path,
                part,
                measure=measure,
            )
        raise ValueError("circuit_level must be 1, 2, 3, or 4")

    def compile_hadamard_test(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
        *,
        measure: bool = True,
    ) -> HadamardTestCompilation:
        braid_word = self.model.as_braid_word(word)
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        level_2 = self.level_2_multicontrolled_circuit(
            braid_word,
            path_bits,
            part,
            measure=measure,
        )
        level_3 = self._lower_component(level_2)
        level_4 = (
            None
            if self._clifford_t_compiler is None
            else self.lower_to_level_4(level_3)
        )
        return HadamardTestCompilation(
            word=braid_word,
            initial_path=path_bits,
            part=part,
            level_1_varphi=self.level_1_varphi_circuit(
                braid_word,
                path_bits,
                part,
                measure=measure,
            ),
            level_2_multicontrolled=level_2,
            level_3_single_control=level_3,
            config=self.config,
            level_4_clifford_t=None if level_4 is None else level_4.circuit,
            level_4_resources=None if level_4 is None else level_4.resources,
            level_4_status=(
                LEVEL_4_STATUS
                if level_4 is None
                else "Compiled approximate logical Clifford+T circuit."
            ),
        )
