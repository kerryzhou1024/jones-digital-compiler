from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from digital_compiler import (
    AJLCompiler,
    AJLJonesEvaluator,
    AJLPathModel,
    BraidWord,
    CommutingLayerScheduling,
    CompiledCircuit,
    CompilerConfig,
    JonesProblem,
    evaluate_jones,
)

TOL = 1e-9


def test_problem_normalizes_and_exposes_one_immutable_domain_object() -> None:
    problem = JonesProblem(
        [1, 1],
        closure="trace",
        strands=2,
        k=5,
    )

    assert problem.word == BraidWord.power(1, 2)
    assert problem.closure == "trace"
    assert problem.writhe is None
    assert problem.strands == 2
    assert problem.k == 5
    assert problem.A == problem.model.A
    assert problem.d == problem.model.d
    assert problem.valid_paths == ((1, 0), (1, 1))
    assert problem.evaluation_paths == problem.valid_paths
    with pytest.raises(FrozenInstanceError):
        problem.k = 7


def test_problem_construction_does_not_enumerate_paths_or_compile(
    monkeypatch,
) -> None:
    valid_path_calls = 0
    compile_calls = 0
    original_valid_paths = AJLPathModel.valid_paths
    original_compile_component = AJLCompiler.compile_component

    def recording_valid_paths(model):
        nonlocal valid_path_calls
        valid_path_calls += 1
        return original_valid_paths(model)

    def recording_compile_component(compiler, *args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile_component(compiler, *args, **kwargs)

    monkeypatch.setattr(AJLPathModel, "valid_paths", recording_valid_paths)
    monkeypatch.setattr(AJLCompiler, "compile_component", recording_compile_component)

    problem = JonesProblem("s1^2", strands=2)
    workload = problem.circuits()

    assert valid_path_calls == 0
    assert compile_calls == 0

    first = next(workload)
    assert first.path == (1, 0)
    assert first.part == "real"
    assert valid_path_calls == 1
    assert compile_calls == 1


def test_plat_problem_selects_only_the_alternating_evaluation_path() -> None:
    problem = JonesProblem(
        "s2^2",
        closure="plat",
        writhe=2,
        strands=4,
        k=5,
    )

    assert problem.valid_paths == problem.model.valid_paths()
    assert problem.evaluation_paths == ((1, 0, 1, 0),)
    assert [(item.path, item.part) for item in problem.circuits()] == [
        ((1, 0, 1, 0), "real"),
        ((1, 0, 1, 0), "imag"),
    ]


def test_plat_problem_can_compile_any_model_valid_path() -> None:
    problem = JonesProblem(
        "s2^2",
        closure="plat",
        writhe=2,
        strands=4,
    )

    circuit = problem.circuit("1100", "imag", circuit_level=1)

    assert circuit.metadata["compiler_level"] == 1
    assert circuit.name.endswith("_imag")
    assert circuit.count_ops().get("measure", 0) == 0
    with pytest.raises(ValueError, match="not a valid AJL path"):
        problem.circuit("1001", "real")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"closure": "invalid"}, "closure must be"),
        ({"closure": []}, "closure must be"),
        ({"closure": "plat"}, "writhe is required"),
        ({"writhe": 0}, "writhe can only"),
        ({"closure": "plat", "writhe": 1.5}, "writhe must be"),
        ({"closure": "plat", "writhe": True}, "writhe must be"),
        ({"closure": "plat", "writhe": 0, "strands": 3}, "even number"),
        ({"config": object()}, "config must be"),
    ],
)
def test_problem_rejects_invalid_configuration(kwargs, message: str) -> None:
    options = {"strands": 2, **kwargs}
    with pytest.raises((TypeError, ValueError), match=message):
        JonesProblem("s1", **options)


@pytest.mark.parametrize("circuit_level", [1, 2, 3])
@pytest.mark.parametrize("part", ["real", "imag"])
def test_problem_circuit_matches_the_low_level_compiler(
    circuit_level: int,
    part: str,
) -> None:
    problem = JonesProblem("s1^2", strands=2)
    expected = AJLCompiler(problem.model, problem.config).compile_component(
        problem.word,
        "10",
        part,
        circuit_level=circuit_level,
        measure=False,
    )

    observed = problem.circuit(
        "10",
        part,
        circuit_level=circuit_level,
    )

    assert isinstance(observed, CompiledCircuit)
    assert observed.circuit == expected
    assert str(observed) == str(expected)
    assert observed.path_label == "10"
    assert observed.part == part
    assert observed.circuit_level == circuit_level
    assert observed.measured is False
    assert observed.metadata["compiler_level"] == circuit_level
    assert observed.count_ops().get("measure", 0) == 0


def test_problem_circuit_defaults_to_the_first_valid_path() -> None:
    problem = JonesProblem("s1^2", strands=2)

    circuit = problem.circuit(part="imag", circuit_level=2)

    assert circuit.path == problem.valid_paths[0]
    assert circuit.path_label == "10"
    assert circuit.part == "imag"
    assert circuit.circuit_level == 2


def test_problem_compile_builds_matched_unmeasured_levels() -> None:
    problem = JonesProblem("s1^2", strands=2)
    compilation = problem.compile("10", "imag")

    assert compilation.word == problem.word
    assert compilation.initial_path == (1, 0)
    assert compilation.part == "imag"
    assert compilation.config == problem.config
    for circuit in (
        compilation.level_1_varphi,
        compilation.level_2_multicontrolled,
        compilation.level_3_single_control,
    ):
        assert circuit.count_ops().get("measure", 0) == 0


def test_problem_circuits_are_labeled_ordered_and_optionally_measured() -> None:
    problem = JonesProblem("s1^2", strands=2)
    circuits = tuple(problem.circuits(circuit_level=2, measure=True))

    assert [
        (item.path, item.part, item.circuit_level, item.measured)
        for item in circuits
    ] == [
        ((1, 0), "real", 2, True),
        ((1, 0), "imag", 2, True),
        ((1, 1), "real", 2, True),
        ((1, 1), "imag", 2, True),
    ]
    assert [item.path_label for item in circuits] == ["10", "10", "11", "11"]
    assert all(item.circuit.count_ops().get("measure", 0) == 1 for item in circuits)


def test_problem_circuits_filter_path_part_and_iterate_all_levels() -> None:
    problem = JonesProblem("s1^2", strands=2)

    circuits = tuple(
        problem.circuits(
            path="11",
            part="imag",
            circuit_level="all",
        )
    )

    assert [
        (item.path_label, item.part, item.circuit_level)
        for item in circuits
    ] == [
        ("11", "imag", 1),
        ("11", "imag", 2),
        ("11", "imag", 3),
    ]
    assert all(item.count_ops().get("measure", 0) == 0 for item in circuits)


def test_problem_circuit_filters_reject_invalid_selections() -> None:
    problem = JonesProblem("s1^2", strands=2)

    with pytest.raises(ValueError, match="part must be"):
        tuple(problem.circuits(part="magnitude"))
    with pytest.raises(ValueError, match="circuit_level must be"):
        tuple(problem.circuits(circuit_level=4))


def test_problem_propagates_compiler_policy_to_circuits_and_results() -> None:
    config = CompilerConfig(
        scheduling=CommutingLayerScheduling(max_lanes=2),
    )
    problem = JonesProblem("1 3", strands=4, config=config)
    circuit = problem.circuit("1010", "real", circuit_level=2)
    result = problem.evaluate(circuit_level=2)

    assert problem.config is config
    assert circuit.metadata["generator_layers"] == ((1, 3),)
    assert circuit.metadata["compiler_config"] == config.metadata() | {
        "workspace_qubits": 0,
        "workspace_qubits_per_lane": 0,
        "height_selector_qubits": 3,
        "height_register_qubits": 6,
        "parallel_lanes": 2,
        "control_fanout_qubits": 1,
    }
    assert result.config is config


def test_problem_statevector_and_legacy_wrapper_match_the_evaluator() -> None:
    problem = JonesProblem("s1^2", strands=2, k=5)
    direct = AJLJonesEvaluator(problem.model, problem.config).evaluate(
        problem.word,
        circuit_level=2,
    )
    facade = problem.evaluate(circuit_level=2)
    legacy = evaluate_jones("s1^2", strands=2, k=5, circuit_level=2)

    assert facade.value == direct.value
    assert facade.path_estimates == direct.path_estimates
    assert legacy.value == facade.value
    assert legacy.path_estimates == facade.path_estimates


def test_problem_shot_evaluation_is_reproducible() -> None:
    problem = JonesProblem("s1^2", strands=2)
    first = problem.evaluate(
        method="shots",
        circuit_level=2,
        shots=256,
        seed=19,
    )
    second = problem.evaluate(
        method="shots",
        circuit_level=2,
        shots=256,
        seed=19,
    )

    assert first.value == second.value
    assert first.path_estimates == second.path_estimates
    assert first.total_shots == 4 * 256


@pytest.mark.parametrize(
    "problem",
    [
        JonesProblem("s1^2", closure="trace", strands=2),
        JonesProblem("s2^2", closure="plat", writhe=2, strands=4),
    ],
)
def test_dense_reference_value_matches_statevector_evaluation(
    problem: JonesProblem,
) -> None:
    assert abs(problem.reference_value() - problem.evaluate().value) < TOL
