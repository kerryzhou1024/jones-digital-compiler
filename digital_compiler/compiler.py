"""Policy-driven AJL circuit compiler."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .fault_tolerance import (
    CliffordTCompilation,
    CliffordTCompiler,
    LogicalResourceReport,
)
from .lowering import SingleControlLowerer, assert_level_2_contract
from .model import AJLPathModel, BraidGenerator, BraidWord, HadamardPart
from .policies import (
    CompilerConfig,
    GeneratorSchedule,
    PrefixHeightPlan,
    PrefixHeightTransition,
)
from .primitives import (
    QuantumAdder,
    append_hadamard_readout,
    controlled_varphi_gate,
    prepare_basis_path,
)

LEVEL_4_STATUS = (
    "Level 4 is not configured; supply CompilerConfig(level4=CliffordTConfig(...))."
)

CompilerLevel = Literal[1, 2, 3, 4]


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
        circuits = [
            self.level_1_varphi,
            self.level_2_multicontrolled,
            self.level_3_single_control,
        ]
        if self.level_4_clifford_t is not None:
            circuits.append(self.level_4_clifford_t)
        signatures = {register_signature(circuit) for circuit in circuits}
        if len(signatures) != 1:
            raise ValueError("compiler levels must have identical register layouts")
        if (self.level_4_clifford_t is None) != (
            self.level_4_resources is None
        ):
            raise ValueError(
                "Level 4 circuit and resource report must be provided together"
            )

    @property
    def path_label(self) -> str:
        return "".join(str(bit) for bit in self.initial_path)

    @property
    def logical_qubits(self) -> int:
        return self.level_1_varphi.num_qubits

    @property
    def height_policy_label(self) -> str:
        return self.config.height.name

    @property
    def scheduling_policy_label(self) -> str:
        return self.config.scheduling.name

    @property
    def prefix_height_policy_label(self) -> str:
        return self.config.prefix_height.name

    @property
    def control_distribution_policy_label(self) -> str:
        return self.config.control_distribution.name


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
        self.height_selector_qubits = max(1, math.ceil(math.log2(self.level)))
        self.height_qubits = self.height_selector_qubits
        self.adder = QuantumAdder(self.height_qubits)
        self.parallel_lanes = self._validate_lane_capacity(
            self.config.scheduling.lane_capacity(self.strands)
        )
        self.height_register_qubits = self.height_qubits * self.parallel_lanes
        max_adder_controls = self.height_qubits - 1
        self.work_qubits_per_lane = int(
            self.config.level3.mcx.clean_ancillas(max_adder_controls)
        )
        if self.work_qubits_per_lane < 0:
            raise ValueError("an MCX policy cannot request negative workspace")
        self.work_qubits = self.work_qubits_per_lane * self.parallel_lanes
        raw_control_ancillas = self.config.control_distribution.control_ancillas(
            self.parallel_lanes
        )
        if isinstance(raw_control_ancillas, bool):
            raise ValueError(
                "a control distribution policy must request a nonnegative "
                "integer number of ancillas"
            )
        try:
            self.control_fanout_qubits = int(operator.index(raw_control_ancillas))
        except TypeError:
            raise ValueError(
                "a control distribution policy must request a nonnegative "
                "integer number of ancillas"
            ) from None
        if self.control_fanout_qubits < 0:
            raise ValueError(
                "a control distribution policy cannot request negative ancillas"
            )
        self.lowerer = SingleControlLowerer(self.config.level3)
        (
            self.projector_basis_angles,
            self.projector_alignment_angles,
        ) = self._build_projector_alignment_angles()

    def _build_projector_alignment_angles(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        basis_angles = [0.0] * (1 << self.height_selector_qubits)
        for height in self.model.valid_heights:
            basis_angles[height] = self.model.projector_angle(height)
        return tuple(basis_angles), tuple(-2.0 * angle for angle in basis_angles)

    @property
    def logical_qubits(self) -> int:
        return (
            1
            + self.strands
            + self.height_register_qubits
            + self.work_qubits
            + self.control_fanout_qubits
        )

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

    def _config_metadata(self) -> dict[str, object]:
        metadata = self.config.metadata()
        metadata["workspace_qubits"] = self.work_qubits
        metadata["workspace_qubits_per_lane"] = self.work_qubits_per_lane
        metadata["height_selector_qubits"] = self.height_selector_qubits
        metadata["height_register_qubits"] = self.height_register_qubits
        metadata["parallel_lanes"] = self.parallel_lanes
        metadata["control_fanout_qubits"] = self.control_fanout_qubits
        return metadata

    def _new_hadamard_circuit(self, name: str):
        control = QuantumRegister(1, "ctrl")
        path = QuantumRegister(self.strands, "path")
        height = QuantumRegister(self.height_register_qubits, "height")
        work = QuantumRegister(self.work_qubits, "adder_work")
        control_fanout = (
            QuantumRegister(self.control_fanout_qubits, "ctrl_fanout")
            if self.control_fanout_qubits
            else None
        )
        measurement = ClassicalRegister(1, "meas")
        registers = [control, path, height, work]
        if control_fanout is not None:
            registers.append(control_fanout)
        circuit = QuantumCircuit(*registers, measurement, name=name)
        fanout_qubits = [] if control_fanout is None else list(control_fanout)
        return circuit, control, path, height, work, fanout_qubits, measurement

    def _new_braid_circuit(self, name: str, controlled: bool = False):
        registers = []
        control = None
        control_fanout = None
        if controlled:
            control = QuantumRegister(1, "ctrl")
            registers.append(control)
        path = QuantumRegister(self.strands, "path")
        height = QuantumRegister(self.height_register_qubits, "height")
        work = QuantumRegister(self.work_qubits, "adder_work")
        registers.extend([path, height, work])
        if controlled and self.control_fanout_qubits:
            control_fanout = QuantumRegister(
                self.control_fanout_qubits,
                "ctrl_fanout",
            )
            registers.append(control_fanout)
        circuit = QuantumCircuit(*registers, name=name)
        fanout_qubits = [] if control_fanout is None else list(control_fanout)
        return circuit, control, path, height, work, fanout_qubits

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
        if any(len(layer) > self.parallel_lanes for layer in schedule):
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
        word: BraidWord,
        schedule: GeneratorSchedule,
        prefix_height_plan: PrefixHeightPlan,
    ) -> dict[str, object]:
        signed_layers = tuple(
            tuple(word.generators[position].signed_index for position in layer)
            for layer in schedule
        )
        return {
            "generator_scheduling": self.config.scheduling.name,
            "control_distribution": self.config.control_distribution.name,
            "generator_layers": signed_layers,
            "parallel_lanes": self.parallel_lanes,
            "active_parallel_width": max((len(layer) for layer in schedule), default=0),
            "height_selector_qubits": self.height_selector_qubits,
            "workspace_qubits_per_lane": self.work_qubits_per_lane,
            "prefix_height_strategy": self.config.prefix_height.name,
            "prefix_height_loads": prefix_height_plan.loads,
            "prefix_height_moves": prefix_height_plan.moves,
            "prefix_height_unloads": prefix_height_plan.unloads,
            "prefix_height_path_steps": prefix_height_plan.path_steps,
        }

    def _height_lane(self, height, lane: int) -> list:
        start = lane * self.height_selector_qubits
        stop = start + self.height_selector_qubits
        return list(height[start:stop])

    def _prefix_height_plan(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
    ) -> PrefixHeightPlan:
        plan = self.config.prefix_height.route(
            word,
            schedule,
            self.parallel_lanes,
        )
        self._validate_prefix_height_plan(word, schedule, plan)
        return plan

    def _validate_prefix_height_plan(
        self,
        word: BraidWord,
        schedule: GeneratorSchedule,
        plan: PrefixHeightPlan,
    ) -> None:
        if not isinstance(plan, PrefixHeightPlan):
            raise ValueError("a prefix-height policy must return a PrefixHeightPlan")
        if len(plan.layers) != len(schedule):
            raise ValueError(
                "a prefix-height plan must contain one route for each generator layer"
            )

        lane_state: list[int | None] = [None] * self.parallel_lanes

        def apply_transition(transition: PrefixHeightTransition) -> None:
            lane = transition.lane
            if (
                isinstance(lane, bool)
                or not isinstance(lane, int)
                or lane < 0
                or lane >= self.parallel_lanes
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
            lanes = tuple(item.lane for item in routed_layer.generators)
            if len(set(lanes)) != len(lanes):
                raise ValueError(
                    "a prefix-height plan cannot reuse one lane within a layer"
                )
            for item in routed_layer.generators:
                if (
                    isinstance(item.lane, bool)
                    or not isinstance(item.lane, int)
                    or item.lane < 0
                    or item.lane >= self.parallel_lanes
                ):
                    raise ValueError("a routed generator has an invalid height lane")
                expected_index = word.generators[item.position].index
                if lane_state[item.lane] != expected_index:
                    raise ValueError(
                        "a routed generator does not have its required prefix height"
                    )

            active_lanes = set(lanes)
            dirty_lanes = {
                lane for lane, index in enumerate(lane_state) if index is not None
            }
            if dirty_lanes != active_lanes:
                raise ValueError(
                    "a prefix-height plan must clean every inactive lane before a layer"
                )

            for transition in routed_layer.after:
                apply_transition(transition)

        if any(index is not None for index in lane_state):
            raise ValueError("a prefix-height plan must clean every lane at completion")

    def controlled_varphi_gate(
        self,
        generator: BraidGenerator,
        include_workspace: bool = True,
    ):
        workspace_qubits = (
            self.height_register_qubits + self.work_qubits
            if include_workspace
            else 0
        )
        return controlled_varphi_gate(generator, self.strands + workspace_qubits)

    def _compute_prefix_height(self, circuit, path, height, index: int) -> None:
        self.adder.increment(circuit, height)
        for prefix_bit in path[: index - 1]:
            self.adder.add_path_step(circuit, prefix_bit, height)

    def _uncompute_prefix_height(self, circuit, path, height, index: int) -> None:
        for prefix_bit in reversed(path[: index - 1]):
            self.adder.subtract_path_step(circuit, prefix_bit, height)
        self.adder.decrement(circuit, height)

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
        for layer in plan.layers:
            for transition in layer.before:
                self._transition_prefix_height(
                    circuit,
                    path,
                    self._height_lane(height, transition.lane),
                    transition.source_index,
                    transition.target_index,
                )

            for item in layer.generators:
                experiment_control = None if not controls else controls[item.lane]
                self._append_height_selected_generator(
                    circuit,
                    path,
                    self._height_lane(height, item.lane),
                    word.generators[item.position],
                    experiment_control,
                )

            for transition in layer.after:
                self._transition_prefix_height(
                    circuit,
                    path,
                    self._height_lane(height, transition.lane),
                    transition.source_index,
                    transition.target_index,
                )

    def lower_to_level_3(self, level_2_circuit: QuantumCircuit) -> QuantumCircuit:
        return self.lowerer.lower(level_2_circuit)

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

    def level_2_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        braid_word = self.model.as_braid_word(word)
        schedule = self._schedule(braid_word)
        prefix_height_plan = self._prefix_height_plan(braid_word, schedule)
        policy_name = self.config.height.name
        circuit, _, path, height, _, _ = self._new_braid_circuit(
            f"level_2_braid_{policy_name}({braid_word})",
            controlled=False,
        )
        circuit.metadata = {
            "compiler_level": 2,
            "height_strategy": policy_name,
            "gate_contract": "ajl_multicontrolled",
            "compiler_config": self._config_metadata(),
            **self._schedule_metadata(
                braid_word,
                schedule,
                prefix_height_plan,
            ),
        }
        self._append_logical_plan(
            circuit,
            path,
            height,
            braid_word,
            prefix_height_plan,
        )
        assert_level_2_contract(circuit)
        return circuit

    def controlled_level_2_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        braid_word = self.model.as_braid_word(word)
        schedule = self._schedule(braid_word)
        prefix_height_plan = self._prefix_height_plan(braid_word, schedule)
        policy_name = self.config.height.name
        circuit, control, path, height, _, control_fanout = self._new_braid_circuit(
            f"controlled_level_2_braid_{policy_name}({braid_word})",
            controlled=True,
        )
        circuit.metadata = {
            "compiler_level": 2,
            "height_strategy": policy_name,
            "gate_contract": "ajl_multicontrolled",
            "compiler_config": self._config_metadata(),
            **self._schedule_metadata(
                braid_word,
                schedule,
                prefix_height_plan,
            ),
        }
        active_width = max((len(layer) for layer in schedule), default=0)
        lane_controls = self.config.control_distribution.prepare(
            circuit,
            control[0],
            control_fanout,
            active_width,
        )
        self._append_logical_plan(
            circuit,
            path,
            height,
            braid_word,
            prefix_height_plan,
            lane_controls,
        )
        self.config.control_distribution.unprepare(
            circuit,
            control[0],
            control_fanout,
            active_width,
        )
        assert_level_2_contract(circuit)
        return circuit

    def level_3_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        circuit = self.lower_to_level_3(self.level_2_braid_circuit(word))
        circuit.name = circuit.name.replace(
            "level_3_single_control(level_2",
            "level_3",
        )[:-1]
        return circuit

    def controlled_level_3_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        circuit = self.lower_to_level_3(self.controlled_level_2_braid_circuit(word))
        circuit.name = circuit.name.replace(
            "level_3_single_control(controlled_level_2",
            "controlled_level_3",
        )[:-1]
        return circuit

    def level_1_varphi_circuit(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
        measure: bool = True,
    ) -> QuantumCircuit:
        braid_word = self.model.as_braid_word(word)
        schedule = self._schedule(braid_word)
        prefix_height_plan = self._prefix_height_plan(braid_word, schedule)
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        (
            circuit,
            control,
            path,
            height,
            work,
            _,
            measurement,
        ) = self._new_hadamard_circuit(f"level_1_varphi_{part}")
        circuit.metadata = {
            "compiler_level": 1,
            "gate_contract": "ajl_varphi_blocks",
            "compiler_config": self._config_metadata(),
            **self._schedule_metadata(
                braid_word,
                schedule,
                prefix_height_plan,
            ),
        }
        prepare_basis_path(circuit, path, path_bits)
        circuit.h(control[0])
        for generator in braid_word.generators:
            circuit.append(
                self.controlled_varphi_gate(generator),
                [control[0], *path, *height, *work],
            )
        append_hadamard_readout(
            circuit,
            control[0],
            part,
            measurement[0] if measure else None,
        )
        return circuit

    def level_2_multicontrolled_circuit(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
        measure: bool = True,
    ) -> QuantumCircuit:
        braid_word = self.model.as_braid_word(word)
        schedule = self._schedule(braid_word)
        prefix_height_plan = self._prefix_height_plan(braid_word, schedule)
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        policy_name = self.config.height.name
        (
            circuit,
            control,
            path,
            height,
            _,
            control_fanout,
            measurement,
        ) = self._new_hadamard_circuit(f"level_2_multicontrolled_{policy_name}_{part}")
        circuit.metadata = {
            "compiler_level": 2,
            "height_strategy": policy_name,
            "gate_contract": "ajl_multicontrolled",
            "compiler_config": self._config_metadata(),
            **self._schedule_metadata(
                braid_word,
                schedule,
                prefix_height_plan,
            ),
        }
        prepare_basis_path(circuit, path, path_bits)
        circuit.h(control[0])
        active_width = max((len(layer) for layer in schedule), default=0)
        lane_controls = self.config.control_distribution.prepare(
            circuit,
            control[0],
            control_fanout,
            active_width,
        )
        self._append_logical_plan(
            circuit,
            path,
            height,
            braid_word,
            prefix_height_plan,
            lane_controls,
        )
        self.config.control_distribution.unprepare(
            circuit,
            control[0],
            control_fanout,
            active_width,
        )
        append_hadamard_readout(
            circuit,
            control[0],
            part,
            measurement[0] if measure else None,
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
        level_3 = self.lower_to_level_3(level_2)
        level_3.name = level_2.name.replace(
            "level_2_multicontrolled",
            "level_3_single_control",
        )
        return level_3

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
        level_3 = self.lower_to_level_3(level_2)
        level_3.name = level_2.name.replace(
            "level_2_multicontrolled",
            "level_3_single_control",
        )
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
