from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from digital_compiler import (
    AJLCompiler,
    AJLPathModel,
    BraidWord,
    CleanAncillaMCX,
    CommutingLayerScheduling,
    CompilerConfig,
    DenseAJLReference,
    Level3Policy,
    MultiplexedHeightSynthesis,
    NoAncillaMCX,
    RecomputePrefixHeight,
    RollingPrefixHeight,
    SerialGeneratorScheduling,
    SharedControl,
    SwitchCaseHeightSynthesis,
    TreeControlFanout,
    compilation_summary,
    register_signature,
)

TOL = 1e-9


class StaticScheduling:
    """Deliberately unrestricted policy used to exercise compiler validation."""

    name = "test_static"

    def __init__(self, schedule, capacity: int = 2):
        self.schedule_value = schedule
        self.capacity = capacity

    def lane_capacity(self, strands: int) -> int:
        del strands
        return self.capacity

    def schedule(self, word: BraidWord, strands: int):
        del word, strands
        return self.schedule_value

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "capacity": self.capacity}


def parallel_config(
    *,
    height=None,
    mcx=None,
    max_lanes: int | None = None,
    control_distribution=None,
    prefix_height=None,
) -> CompilerConfig:
    return CompilerConfig(
        height=MultiplexedHeightSynthesis() if height is None else height,
        level3=Level3Policy(mcx=NoAncillaMCX() if mcx is None else mcx),
        scheduling=CommutingLayerScheduling(max_lanes=max_lanes),
        control_distribution=(
            SharedControl() if control_distribution is None else control_distribution
        ),
        prefix_height=(
            RollingPrefixHeight() if prefix_height is None else prefix_height
        ),
    )


def assert_braid_matches_dense(
    compiler: AJLCompiler,
    word: str,
    *,
    level: int,
    controlled: bool,
) -> None:
    reference = DenseAJLReference(compiler.model)
    matrix = reference.compile_matrix(word)
    if level == 2:
        builder = (
            compiler.controlled_level_2_braid_circuit
            if controlled
            else compiler.level_2_braid_circuit
        )
    else:
        builder = (
            compiler.controlled_level_3_braid_circuit
            if controlled
            else compiler.level_3_braid_circuit
        )
    circuit = builder(word)
    clean_width = compiler.strands + (1 if controlled else 0)
    clean_dimension = 1 << clean_width

    for control_bit in ((0, 1) if controlled else (0,)):
        for column, input_path in enumerate(reference.paths):
            path_basis = reference.little_endian_index(input_path)
            input_basis = (
                control_bit | (path_basis << 1) if controlled else path_basis
            )
            state = Statevector.from_int(
                input_basis,
                2**circuit.num_qubits,
            ).evolve(circuit)
            expected = np.zeros(clean_dimension, dtype=complex)
            if controlled and control_bit == 0:
                expected[input_basis] = 1.0
            else:
                for row, output_path in enumerate(reference.paths):
                    output_path_basis = reference.little_endian_index(output_path)
                    output_basis = (
                        control_bit | (output_path_basis << 1)
                        if controlled
                        else output_path_basis
                    )
                    expected[output_basis] = matrix[row, column]

            np.testing.assert_allclose(
                state.data[:clean_dimension],
                expected,
                atol=TOL,
            )
            assert np.linalg.norm(state.data[clean_dimension:]) < TOL


def test_serial_and_commuting_schedule_construction() -> None:
    word = BraidWord.parse("1 3 2 4 1 3")
    assert SerialGeneratorScheduling().schedule(word, strands=5) == (
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
    )
    assert CommutingLayerScheduling().schedule(word, strands=5) == (
        (0, 1),
        (2, 3),
        (4, 5),
    )
    assert CommutingLayerScheduling(max_lanes=1).schedule(word, strands=5) == (
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
    )

    repeated = BraidWord.parse("1 3 1 2 2 4")
    repeated_layers = CommutingLayerScheduling().schedule(repeated, strands=5)
    layer_by_position = {
        position: layer_index
        for layer_index, layer in enumerate(repeated_layers)
        for position in layer
    }
    assert layer_by_position[0] < layer_by_position[2]
    assert layer_by_position[2] < layer_by_position[3]
    assert layer_by_position[3] < layer_by_position[4]


def test_critical_path_priority_avoids_a_greedy_extra_layer() -> None:
    word = BraidWord.parse("1 3 5 4")
    policy = CommutingLayerScheduling(max_lanes=2)

    assert policy.schedule(word, strands=6) == ((1, 2), (0, 3))

    model = AJLPathModel(6, 5)
    critical = AJLCompiler(
        model,
        CompilerConfig(
            scheduling=policy,
            control_distribution=SharedControl(),
            prefix_height=RecomputePrefixHeight(),
        ),
    ).compile_hadamard_test(word, "101010")
    former_greedy = AJLCompiler(
        model,
        CompilerConfig(
            scheduling=StaticScheduling(((0, 1), (2,), (3,)), capacity=2),
            control_distribution=SharedControl(),
            prefix_height=RecomputePrefixHeight(),
        ),
    ).compile_hadamard_test(word, "101010")

    assert critical.level_2_multicontrolled.metadata["generator_layers"] == (
        (3, 5),
        (1, 4),
    )
    assert critical.logical_qubits == former_greedy.logical_qubits
    assert (
        critical.level_3_single_control.depth()
        < former_greedy.level_3_single_control.depth()
    )


def test_parallel_lane_capacity_is_configurable_and_clamped() -> None:
    assert CommutingLayerScheduling().lane_capacity(6) == 3
    assert CommutingLayerScheduling(max_lanes=2).lane_capacity(6) == 2
    assert CommutingLayerScheduling(max_lanes=99).lane_capacity(6) == 3
    assert CommutingLayerScheduling().lane_capacity(2) == 1

    for invalid in (True, False, 0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="max_lanes"):
            CommutingLayerScheduling(max_lanes=invalid)


@pytest.mark.parametrize(
    "schedule,capacity,message",
    [
        (((0,),), 2, "exactly once"),
        (((0, 1), (1,)), 2, "exactly once"),
        (((0, 2),), 2, "out of range"),
        (((), (0, 1)), 2, "cannot be empty"),
        ((("0",), (1,)), 2, "must be integers"),
        (((0, 1),), 1, "exceeds"),
    ],
)
def test_compiler_rejects_malformed_schedules(schedule, capacity, message) -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        CompilerConfig(scheduling=StaticScheduling(schedule, capacity)),
    )
    with pytest.raises(ValueError, match=message):
        compiler.level_2_braid_circuit("1 3")


def test_compiler_rejects_adjacent_layers_and_noncommuting_reordering() -> None:
    model = AJLPathModel(3, 5)
    adjacent = AJLCompiler(
        model,
        CompilerConfig(scheduling=StaticScheduling(((0, 1),), capacity=1)),
    )
    with pytest.raises(ValueError, match="exceeds"):
        adjacent.level_2_braid_circuit("1 2")

    adjacent_same_width = AJLCompiler(
        AJLPathModel(4, 5),
        CompilerConfig(scheduling=StaticScheduling(((0, 1),), capacity=2)),
    )
    with pytest.raises(ValueError, match="pairwise distant"):
        adjacent_same_width.level_2_braid_circuit("1 2")

    reordered = AJLCompiler(
        model,
        CompilerConfig(scheduling=StaticScheduling(((1,), (0,)), capacity=1)),
    )
    with pytest.raises(ValueError, match="cannot reorder"):
        reordered.level_2_braid_circuit("1 2")


def test_compiler_rejects_invalid_custom_lane_capacities() -> None:
    for capacity in (True, 0, 3, 1.5):
        with pytest.raises(ValueError, match="lane capacity"):
            AJLCompiler(
                AJLPathModel(4, 5),
                CompilerConfig(scheduling=StaticScheduling((), capacity)),
            )


@pytest.mark.parametrize("word", ["1 3", "1 -3", "-1 3", "-1 -3"])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize(
    "control_distribution",
    [SharedControl(), TreeControlFanout()],
)
def test_parallel_mixed_sign_generators_match_dense_level_2(
    word: str,
    controlled: bool,
    control_distribution,
) -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        parallel_config(control_distribution=control_distribution),
    )
    assert_braid_matches_dense(compiler, word, level=2, controlled=controlled)


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize(
    "prefix_height",
    [RollingPrefixHeight(), RecomputePrefixHeight()],
)
def test_multiple_parallel_layers_match_dense(level: int, prefix_height) -> None:
    compiler = AJLCompiler(
        AJLPathModel(5, 5),
        parallel_config(max_lanes=2, prefix_height=prefix_height),
    )
    assert_braid_matches_dense(
        compiler,
        "1 3 2 4 1 3",
        level=level,
        controlled=True,
    )


@pytest.mark.parametrize("max_lanes", [1, 2, None])
def test_lane_caps_preserve_semantics(max_lanes: int | None) -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        parallel_config(max_lanes=max_lanes),
    )
    assert_braid_matches_dense(compiler, "1 -3", level=2, controlled=False)


@pytest.mark.parametrize(
    "height_policy",
    [MultiplexedHeightSynthesis(), SwitchCaseHeightSynthesis()],
)
@pytest.mark.parametrize("mcx_policy", [NoAncillaMCX(), CleanAncillaMCX()])
@pytest.mark.parametrize(
    "control_distribution",
    [SharedControl(), TreeControlFanout()],
)
@pytest.mark.parametrize("controlled", [False, True])
def test_parallel_level_3_matches_dense_across_synthesis_policies(
    height_policy,
    mcx_policy,
    control_distribution,
    controlled: bool,
) -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        parallel_config(
            height=height_policy,
            mcx=mcx_policy,
            control_distribution=control_distribution,
        ),
    )
    assert_braid_matches_dense(compiler, "1 -3", level=3, controlled=controlled)


@pytest.mark.parametrize("part", ["real", "imag"])
@pytest.mark.parametrize(
    "control_distribution",
    [SharedControl(), TreeControlFanout()],
)
def test_parallel_hadamard_test_matches_dense_and_cleans_scratch(
    part: str,
    control_distribution,
) -> None:
    model = AJLPathModel(4, 5)
    compiler = AJLCompiler(
        model,
        parallel_config(control_distribution=control_distribution),
    )
    compilation = compiler.compile_hadamard_test("1 -3", "1010", part)
    amplitude = DenseAJLReference(model).path_amplitude("1 -3", "1010")
    expected = amplitude.real if part == "real" else amplitude.imag
    clean_dimension = 1 << (1 + model.strands)

    for circuit in (
        compilation.level_2_multicontrolled,
        compilation.level_3_single_control,
    ):
        state = Statevector.from_instruction(
            circuit.remove_final_measurements(inplace=False)
        )
        probabilities = state.probabilities(qargs=[0])
        observed = float(probabilities[0] - probabilities[1])
        assert abs(observed - expected) < TOL
        assert np.linalg.norm(state.data[clean_dimension:]) < TOL

    assert register_signature(compilation.level_1_varphi) == register_signature(
        compilation.level_2_multicontrolled
    ) == register_signature(compilation.level_3_single_control)


def test_parallel_clean_ancilla_workspace_is_partitioned_by_lane() -> None:
    compiler = AJLCompiler(
        AJLPathModel(5, 5),
        parallel_config(mcx=CleanAncillaMCX(), max_lanes=2),
    )
    level_3 = compiler.level_3_braid_circuit("2 4")
    work = next(register for register in level_3.qregs if register.name == "adder_work")

    assert compiler.work_qubits_per_lane == 1
    assert compiler.work_qubits == 2
    assert work.size == 2
    assert all(
        any(qubit in instruction.qubits for instruction in level_3.data)
        for qubit in work
    )


def test_identity_and_two_strands_degenerate_cleanly() -> None:
    two_strand_model = AJLPathModel(2, 5)
    serial = AJLCompiler(two_strand_model)
    parallel = AJLCompiler(two_strand_model, parallel_config())
    serial_circuit = serial.level_2_multicontrolled_circuit("1 -1", "10")
    parallel_circuit = parallel.level_2_multicontrolled_circuit("1 -1", "10")

    assert parallel.parallel_lanes == 1
    assert register_signature(serial_circuit) == register_signature(parallel_circuit)
    assert serial_circuit.count_ops() == parallel_circuit.count_ops()
    assert serial_circuit.depth() == parallel_circuit.depth()

    four_strand = AJLCompiler(AJLPathModel(4, 5), parallel_config())
    identity = four_strand.level_2_multicontrolled_circuit(
        BraidWord.identity(),
        "1010",
        measure=False,
    )
    assert identity.metadata["generator_layers"] == ()
    assert identity.metadata["active_parallel_width"] == 0
    assert identity.count_ops().get("cx", 0) == 0
    state = Statevector.from_instruction(identity)
    assert np.linalg.norm(state.data[1 << 5 :]) < TOL


def test_shared_control_reuses_one_qubit_without_ancillas_or_gates() -> None:
    policy = SharedControl()
    circuit = QuantumCircuit(1)
    controls = policy.prepare(
        circuit,
        circuit.qubits[0],
        (),
        active_width=3,
    )
    policy.unprepare(
        circuit,
        circuit.qubits[0],
        (),
        active_width=3,
    )

    assert controls == (circuit.qubits[0],) * 3
    assert circuit.data == []


def test_control_fanout_uses_a_logarithmic_depth_tree_and_uncomputes() -> None:
    policy = TreeControlFanout()
    circuit = QuantumCircuit(4)
    controls = policy.prepare(
        circuit,
        circuit.qubits[0],
        circuit.qubits[1:],
        active_width=4,
    )
    assert len(controls) == 4
    assert circuit.depth() == 2
    policy.unprepare(
        circuit,
        circuit.qubits[0],
        circuit.qubits[1:],
        active_width=4,
    )
    assert circuit.depth() == 4
    assert circuit.count_ops().get("cx", 0) == 6

    state = Statevector.from_int(1, 2**4).evolve(circuit)
    np.testing.assert_allclose(state.data, Statevector.from_int(1, 2**4).data)


def test_parallel_policy_reduces_sigma1_sigma3_depth_and_reports_schedule() -> None:
    model = AJLPathModel(4, 5)
    serial = AJLCompiler(model).compile_hadamard_test("1 3", "1010")
    parallel_compiler = AJLCompiler(model, parallel_config())
    parallel = parallel_compiler.compile_hadamard_test("1 3", "1010")
    summary = compilation_summary(parallel)

    assert parallel_compiler.parallel_lanes == 2
    assert parallel_compiler.height_qubits == 3
    assert parallel_compiler.height_register_qubits == 6
    assert parallel_compiler.control_fanout_qubits == 0
    assert parallel.logical_qubits == 11
    assert all(
        register.name != "ctrl_fanout"
        for register in parallel.level_2_multicontrolled.qregs
    )
    assert parallel.level_2_multicontrolled.depth() < serial.level_2_multicontrolled.depth()
    assert parallel.level_3_single_control.depth() < serial.level_3_single_control.depth()
    assert summary["generator_scheduling"] == "commuting_layers"
    assert summary["control_distribution"] == "shared"
    assert summary["prefix_height_strategy"] == "rolling"
    assert summary["generator_layers"] == ((1, 3),)
    assert summary["parallel_lanes"] == 2
    assert summary["active_parallel_width"] == 2
    assert summary["logical_qubits_each_level"] == 11


def test_explicit_tree_fanout_preserves_the_former_parallel_layout() -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        parallel_config(control_distribution=TreeControlFanout()),
    )
    compilation = compiler.compile_hadamard_test("1 3", "1010")

    assert compiler.control_fanout_qubits == 1
    assert compilation.logical_qubits == 12
    assert (
        register_signature(compilation.level_1_varphi)
        == register_signature(compilation.level_2_multicontrolled)
        == register_signature(compilation.level_3_single_control)
    )
    assert any(
        register.name == "ctrl_fanout" and register.size == 1
        for register in compilation.level_2_multicontrolled.qregs
    )
    assert compilation_summary(compilation)["control_distribution"] == "tree_fanout"


def test_parallel_level_2_and_level_3_are_equivalent() -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        parallel_config(height=SwitchCaseHeightSynthesis(), mcx=CleanAncillaMCX()),
    )
    level_2 = compiler.controlled_level_2_braid_circuit("1 -3")
    level_3 = compiler.lower_to_level_3(level_2)
    basis = 1 | (DenseAJLReference.little_endian_index((1, 0, 1, 0)) << 1)
    state_2 = Statevector.from_int(basis, 2**level_2.num_qubits).evolve(level_2)
    state_3 = Statevector.from_int(basis, 2**level_3.num_qubits).evolve(level_3)

    np.testing.assert_allclose(state_2.data, state_3.data, atol=TOL)
    assert math.sqrt(float(np.sum(np.abs(state_3.data[1 << 5 :]) ** 2))) < TOL
