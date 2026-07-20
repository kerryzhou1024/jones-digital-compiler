"""Policy-driven AJL circuit compiler."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .lowering import SingleControlLowerer, assert_level_2_contract
from .model import AJLPathModel, BraidGenerator, BraidWord, HadamardPart
from .policies import CompilerConfig
from .primitives import (
    QuantumAdder,
    append_hadamard_readout,
    controlled_varphi_gate,
    prepare_basis_path,
)

LEVEL_4_STATUS = (
    "Not implemented: choose a rotation-synthesis algorithm, total approximation "
    "tolerance, and error-allocation policy before producing logical Clifford+T. "
    "Surface-code mapping and scheduling are downstream tasks."
)


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
    level_4_status: str = LEVEL_4_STATUS

    def __post_init__(self) -> None:
        signatures = {
            register_signature(self.level_1_varphi),
            register_signature(self.level_2_multicontrolled),
            register_signature(self.level_3_single_control),
        }
        if len(signatures) != 1:
            raise ValueError("Levels 1, 2, and 3 must have identical register layouts")
        if self.level_4_clifford_t is not None:
            raise ValueError("Level 4 is intentionally not implemented")

    @property
    def path_label(self) -> str:
        return "".join(str(bit) for bit in self.initial_path)

    @property
    def logical_qubits(self) -> int:
        return self.level_1_varphi.num_qubits

    @property
    def height_policy_label(self) -> str:
        return self.config.height.name


class AJLCompiler:
    """Compile AJL braid words through explicit semantic and lowering layers."""

    def __init__(
        self,
        model: AJLPathModel,
        config: CompilerConfig | None = None,
    ):
        self.model = model
        self.config = CompilerConfig() if config is None else config
        self.strands = model.strands
        self.level = model.level
        self.height_selector_qubits = max(1, math.ceil(math.log2(self.level)))
        self.height_qubits = self.height_selector_qubits
        self.adder = QuantumAdder(self.height_qubits)
        self.work_qubits = int(
            self.config.level3.mcx.clean_ancillas(self.height_qubits)
        )
        if self.work_qubits < 0:
            raise ValueError("an MCX policy cannot request negative workspace")
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
        return 1 + self.strands + self.height_qubits + self.work_qubits

    def _config_metadata(self) -> dict[str, object]:
        metadata = self.config.metadata()
        metadata["workspace_qubits"] = self.work_qubits
        return metadata

    def _new_hadamard_circuit(self, name: str):
        control = QuantumRegister(1, "ctrl")
        path = QuantumRegister(self.strands, "path")
        height = QuantumRegister(self.height_qubits, "height")
        work = QuantumRegister(self.work_qubits, "adder_work")
        measurement = ClassicalRegister(1, "meas")
        circuit = QuantumCircuit(control, path, height, work, measurement, name=name)
        return circuit, control, path, height, work, measurement

    def _new_braid_circuit(self, name: str, controlled: bool = False):
        registers = []
        control = None
        if controlled:
            control = QuantumRegister(1, "ctrl")
            registers.append(control)
        path = QuantumRegister(self.strands, "path")
        height = QuantumRegister(self.height_qubits, "height")
        work = QuantumRegister(self.work_qubits, "adder_work")
        registers.extend([path, height, work])
        circuit = QuantumCircuit(*registers, name=name)
        return circuit, control, path, height, work

    @staticmethod
    def validate_part(part: str) -> HadamardPart:
        if part not in {"real", "imag"}:
            raise ValueError("part must be 'real' or 'imag'")
        return part

    def controlled_varphi_gate(
        self,
        generator: BraidGenerator,
        include_workspace: bool = True,
    ):
        workspace_qubits = self.height_qubits + self.work_qubits if include_workspace else 0
        return controlled_varphi_gate(generator, self.strands + workspace_qubits)

    def _compute_prefix_height(self, circuit, path, height, index: int) -> None:
        self.adder.increment(circuit, height)
        for prefix_bit in path[: index - 1]:
            self.adder.add_path_step(circuit, prefix_bit, height)

    def _uncompute_prefix_height(self, circuit, path, height, index: int) -> None:
        for prefix_bit in reversed(path[: index - 1]):
            self.adder.subtract_path_step(circuit, prefix_bit, height)
        self.adder.decrement(circuit, height)

    def append_logical_generator(
        self,
        circuit,
        path,
        height,
        generator: BraidGenerator,
        experiment_control=None,
    ) -> None:
        if generator.index < 1 or generator.index >= self.strands:
            raise ValueError(f"generator index must be in 1..{self.strands - 1}")
        self._compute_prefix_height(circuit, path, height, generator.index)
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
        self._uncompute_prefix_height(circuit, path, height, generator.index)

    def lower_to_level_3(self, level_2_circuit: QuantumCircuit) -> QuantumCircuit:
        return self.lowerer.lower(level_2_circuit)

    def level_2_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        braid_word = self.model.as_braid_word(word)
        policy_name = self.config.height.name
        circuit, _, path, height, _ = self._new_braid_circuit(
            f"level_2_braid_{policy_name}({braid_word})",
            controlled=False,
        )
        circuit.metadata = {
            "compiler_level": 2,
            "height_strategy": policy_name,
            "gate_contract": "ajl_multicontrolled",
            "compiler_config": self._config_metadata(),
        }
        for generator in braid_word.generators:
            self.append_logical_generator(circuit, path, height, generator)
        assert_level_2_contract(circuit)
        return circuit

    def controlled_level_2_braid_circuit(
        self,
        word: BraidWord | str | Sequence[int],
    ) -> QuantumCircuit:
        braid_word = self.model.as_braid_word(word)
        policy_name = self.config.height.name
        circuit, control, path, height, _ = self._new_braid_circuit(
            f"controlled_level_2_braid_{policy_name}({braid_word})",
            controlled=True,
        )
        circuit.metadata = {
            "compiler_level": 2,
            "height_strategy": policy_name,
            "gate_contract": "ajl_multicontrolled",
            "compiler_config": self._config_metadata(),
        }
        for generator in braid_word.generators:
            self.append_logical_generator(
                circuit,
                path,
                height,
                generator,
                experiment_control=control[0],
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
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        circuit, control, path, height, work, measurement = self._new_hadamard_circuit(
            f"level_1_varphi_{part}"
        )
        circuit.metadata = {
            "compiler_level": 1,
            "gate_contract": "ajl_varphi_blocks",
            "compiler_config": self._config_metadata(),
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
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        policy_name = self.config.height.name
        circuit, control, path, height, _, measurement = self._new_hadamard_circuit(
            f"level_2_multicontrolled_{policy_name}_{part}"
        )
        circuit.metadata = {
            "compiler_level": 2,
            "height_strategy": policy_name,
            "gate_contract": "ajl_multicontrolled",
            "compiler_config": self._config_metadata(),
        }
        prepare_basis_path(circuit, path, path_bits)
        circuit.h(control[0])
        for generator in braid_word.generators:
            self.append_logical_generator(
                circuit,
                path,
                height,
                generator,
                experiment_control=control[0],
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

    def compile_hadamard_test(
        self,
        word: BraidWord | str | Sequence[int],
        initial_path: str | Sequence[int],
        part: HadamardPart = "real",
    ) -> HadamardTestCompilation:
        braid_word = self.model.as_braid_word(word)
        path_bits = self.model.coerce_path(initial_path)
        part = self.validate_part(part)
        level_2 = self.level_2_multicontrolled_circuit(
            braid_word,
            path_bits,
            part,
            measure=True,
        )
        level_3 = self.lower_to_level_3(level_2)
        level_3.name = level_2.name.replace(
            "level_2_multicontrolled",
            "level_3_single_control",
        )
        return HadamardTestCompilation(
            word=braid_word,
            initial_path=path_bits,
            part=part,
            level_1_varphi=self.level_1_varphi_circuit(
                braid_word,
                path_bits,
                part,
                measure=True,
            ),
            level_2_multicontrolled=level_2,
            level_3_single_control=level_3,
            config=self.config,
        )
