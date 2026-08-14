from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from qiskit.primitives import BaseSamplerV2, StatevectorSampler
from qiskit.primitives.containers import SamplerPub

from digital_compiler import (
    AJL_SUCCESS_PROBABILITY,
    DEFAULT_SHOTS,
    AJLJonesEvaluator,
    AJLPathModel,
    BraidWord,
    CliffordTConfig,
    CompilerConfig,
    JonesProblem,
    assert_clifford_t_contract,
)

TOL = 1e-9


class RecordingSampler(BaseSamplerV2):
    """SamplerV2 test double that records batching before delegating locally."""

    def __init__(self, seed: int = 17):
        self.delegate = StatevectorSampler(seed=seed)
        self.calls: list[tuple[int, int | None, tuple[int | None, ...]]] = []
        self.circuits = []

    def run(self, pubs, *, shots=None):
        pubs = list(pubs)
        normalized = [SamplerPub.coerce(pub, shots) for pub in pubs]
        pub_shots = tuple(pub.shots for pub in normalized)
        self.circuits.extend(pub.circuit for pub in normalized)
        self.calls.append((len(pubs), shots, pub_shots))
        return self.delegate.run(pubs, shots=shots)


class WrongShotSampler(BaseSamplerV2):
    """Sampler double that intentionally ignores per-pub shot counts."""

    def run(self, pubs, *, shots=None):
        circuits = [
            SamplerPub.coerce(pub, shots).circuit
            for pub in pubs
        ]
        return StatevectorSampler(seed=7).run(circuits, shots=1)


@pytest.mark.parametrize("circuit_level", [2, 3])
def test_exact_hopf_evaluation_matches_analytic_value(circuit_level: int) -> None:
    result = JonesProblem("s1^2", strands=2, k=5).evaluate(circuit_level=circuit_level)
    expected = -(result.model.A**10) - result.model.A**2

    assert abs(result.value - expected) < TOL
    assert result.method == "statevector"
    assert result.closure == "trace"
    assert result.writhe == result.word.writhe == 2
    assert result.plat_writhe is None
    assert result.markov_trace is not None
    assert result.plat_amplitude is None
    assert result.circuit_level == circuit_level
    assert result.circuit_count == 4
    assert result.path_sampling == "enumerated"
    assert result.trace_samples is None
    assert result.shots_per_circuit is None
    assert result.shots_per_component is None
    assert result.total_shots == 0
    assert result.sampler_name is None
    assert result.real_standard_error == 0.0
    assert result.imag_standard_error == 0.0
    assert result.markov_trace_additive_error_bound is None
    assert result.value_additive_error_bound is None
    assert result.synthesis_error_budget_per_circuit is None
    assert result.value_synthesis_error_bound is None
    assert result.ajl_success_probability is None
    assert tuple(result.path_amplitudes) == result.model.valid_paths()
    for estimate in result.path_estimates:
        assert estimate.real.counts is None
        assert estimate.imag.counts is None
        assert estimate.real.shots == 0
        assert estimate.imag.shots == 0


def test_exact_sigma3_squared_four_strand_evaluation() -> None:
    result = JonesProblem("s3^2", strands=4, k=5).evaluate()
    expected = result.model.d**2 * (
        -(result.model.A**10) - result.model.A**2
    )

    assert abs(result.value - expected) < TOL
    assert len(result.path_estimates) == 5
    assert result.circuit_count == 10


def test_identity_braid_evaluates_the_unlink() -> None:
    evaluator = AJLJonesEvaluator(AJLPathModel(3, 5))
    result = evaluator.evaluate(BraidWord.identity(), circuit_level=2)

    assert abs(result.value - evaluator.model.d**2) < TOL
    assert all(
        abs(estimate.amplitude - 1.0) < TOL
        for estimate in result.path_estimates
    )


@pytest.mark.parametrize("circuit_level", [2, 3])
def test_exact_four_strand_plat_uses_only_the_alternating_path(
    circuit_level: int,
) -> None:
    result = JonesProblem(
        BraidWord.identity(),
        strands=4,
        k=5,
        closure="plat",
        writhe=0,
    ).evaluate(circuit_level=circuit_level)

    assert result.value == pytest.approx(result.model.d)
    assert result.closure == "plat"
    assert result.writhe == 0
    assert result.plat_writhe == 0
    assert result.markov_trace is None
    assert result.plat_amplitude == pytest.approx(1.0)
    assert result.path_sampling == "fixed_plat"
    assert tuple(result.path_amplitudes) == (result.model.plat_path(),)
    assert result.circuit_count == 2
    assert result.total_shots == 0


@pytest.mark.parametrize(
    "closure_options",
    [
        {},
        {"closure": "plat", "writhe": -1},
    ],
)
def test_level_4_statevector_evaluation_respects_synthesis_bound(
    closure_options: dict[str, object],
) -> None:
    config = CompilerConfig(level4=CliffordTConfig(1e-3))
    word = "s1^2" if not closure_options else "s1"
    level_3 = JonesProblem(word, strands=2, **closure_options).evaluate(circuit_level=3)
    level_4 = JonesProblem(
        word,
        strands=2,
        config=config,
        **closure_options,
    ).evaluate(circuit_level=4)

    assert level_4.value_synthesis_error_bound is not None
    assert (
        abs(level_4.value - level_3.value)
        <= level_4.value_synthesis_error_bound
    )
    assert level_4.synthesis_error_budget_per_circuit == 1e-3
    assert level_4.real_standard_error == 0.0
    assert level_4.imag_standard_error == 0.0


def test_exact_level_4_evaluation_reports_zero_synthesis_bound() -> None:
    result = JonesProblem(
        BraidWord.identity(),
        strands=2,
        config=CompilerConfig(level4=CliffordTConfig(1e-3)),
    ).evaluate(circuit_level=4)

    assert result.synthesis_error_budget_per_circuit == 1e-3
    assert result.value_synthesis_error_bound == 0.0


def test_level_4_trace_shots_preserve_batching_and_sampling() -> None:
    sampler = RecordingSampler(seed=29)
    target_additive_error = 1.0
    confidence_factor = math.log(4.0) - math.log1p(-AJL_SUCCESS_PROBABILITY)
    closure_magnitude = AJLPathModel(2, 5).d
    expected_shots = math.ceil(
        4.0
        * closure_magnitude**2
        * confidence_factor
        / target_additive_error**2
    )
    result = JonesProblem(
        "s1^2",
        strands=2,
        config=CompilerConfig(level4=CliffordTConfig(1e-3)),
    ).evaluate(
        method="shots",
        circuit_level=4,
        target_additive_error=target_additive_error,
        path_seed=13,
        sampler=sampler,
    )

    assert len(sampler.calls) == 1
    assert sampler.calls[0][0] == result.circuit_count
    assert (
        sum(shot for shot in sampler.calls[0][2] if shot is not None)
        == 2 * expected_shots
    )
    assert result.total_shots == 2 * expected_shots
    assert result.trace_samples is not None
    assert result.value_additive_error_bound <= target_additive_error
    assert result.value_synthesis_error_bound is not None
    assert all(
        sample.circuit_count < sample.shots
        for sample in result.trace_samples
    )
    for circuit in sampler.circuits:
        assert_clifford_t_contract(circuit)


def test_level_4_plat_shots_are_seeded_and_use_two_circuits() -> None:
    problem = JonesProblem(
        "s1",
        strands=2,
        closure="plat",
        writhe=-1,
        config=CompilerConfig(level4=CliffordTConfig(1e-3)),
    )
    kwargs = {"method": "shots", "circuit_level": 4, "shots": 64, "seed": 41}
    first = problem.evaluate(**kwargs)
    second = problem.evaluate(**kwargs)

    assert first.value == second.value
    assert first.path_estimates == second.path_estimates
    assert first.circuit_count == 2
    assert first.total_shots == 128
    assert first.value_synthesis_error_bound is not None


def test_plat_uses_oriented_closure_writhe_not_braid_exponent_sum() -> None:
    result = JonesProblem("s1", strands=2, closure="plat", writhe=-1).evaluate(
        circuit_level=2,
    )

    assert result.word.writhe == 1
    assert result.plat_writhe == -1
    assert result.value == pytest.approx(1.0, abs=TOL)


def test_seeded_shot_evaluation_is_reproducible_and_reports_cost() -> None:
    problem = JonesProblem("s1^2", strands=2, k=5)
    kwargs = {"method": "shots", "circuit_level": 2, "shots": 10_000, "seed": 23}
    first = problem.evaluate(**kwargs)
    second = problem.evaluate(**kwargs)
    expected = -(first.model.A**10) - first.model.A**2

    assert first.value == second.value
    assert first.trace_samples == second.trace_samples
    assert first.sampler_name == "StatevectorSampler"
    assert first.path_sampling == "ajl_weighted"
    assert first.path_estimates is None
    assert first.path_amplitudes is None
    assert first.shots_per_circuit is None
    assert first.shots_per_component == 10_000
    assert first.circuit_count == 4
    assert first.total_shots == 20_000
    assert first.ajl_success_probability == AJL_SUCCESS_PROBABILITY
    expected_markov_bound = math.sqrt(16.0 * math.log(2.0) / 10_000)
    assert first.markov_trace_additive_error_bound == pytest.approx(
        expected_markov_bound
    )
    assert first.value_additive_error_bound == pytest.approx(
        first.model.d * expected_markov_bound
    )
    assert abs(first.value - expected) <= first.value_additive_error_bound
    assert abs(first.value.real - expected.real) <= 5.0 * first.real_standard_error
    assert abs(first.value.imag - expected.imag) <= 5.0 * first.imag_standard_error
    assert first.trace_samples is not None
    assert first.trace_samples[0].path_counts != first.trace_samples[1].path_counts
    for sample in first.trace_samples:
        assert sample.shots == 10_000
        assert sum(sample.counts.values()) == 10_000
        assert sum(sample.path_counts.values()) == 10_000
        assert sample.standard_error >= 0.0
        with pytest.raises(TypeError):
            sample.counts["0"] = 0
        with pytest.raises(TypeError):
            sample.path_counts[(1, 0)] = 0
    with pytest.raises(FrozenInstanceError):
        first.value = 0j


def test_explicit_shots_report_the_bound_at_requested_confidence() -> None:
    shots = 512
    success_probability = 0.99
    result = JonesProblem("s1^2", strands=2).evaluate(
        method="shots",
        circuit_level=2,
        shots=shots,
        success_probability=success_probability,
        seed=47,
    )
    confidence_factor = math.log(4.0) - math.log1p(-success_probability)
    expected_markov_bound = math.sqrt(4.0 * confidence_factor / shots)

    assert result.ajl_success_probability == success_probability
    assert result.markov_trace_additive_error_bound == pytest.approx(
        expected_markov_bound
    )
    assert result.value_additive_error_bound == pytest.approx(
        result.model.d * expected_markov_bound
    )


def test_trace_target_additive_error_selects_minimal_shots() -> None:
    target_additive_error = 1.5
    success_probability = 0.9
    model = AJLPathModel(2, 5)
    closure_magnitude = model.d
    confidence_factor = math.log(4.0) - math.log1p(-success_probability)
    expected_shots = math.ceil(
        4.0
        * closure_magnitude**2
        * confidence_factor
        / target_additive_error**2
    )
    result = AJLJonesEvaluator(model).evaluate(
        "s1^2",
        method="shots",
        circuit_level=2,
        target_additive_error=target_additive_error,
        success_probability=success_probability,
        path_seed=59,
        sampler=RecordingSampler(seed=61),
    )

    assert result.shots_per_component == expected_shots
    assert result.total_shots == 2 * expected_shots
    assert result.value_additive_error_bound <= target_additive_error
    assert closure_magnitude * math.sqrt(
        4.0 * confidence_factor / (expected_shots - 1)
    ) > target_additive_error


def test_shot_uncertainties_are_propagated_to_jones_components() -> None:
    result = JonesProblem("s1^2", strands=2).evaluate(
        method="shots",
        circuit_level=2,
        shots=2048,
        seed=31,
    )
    assert result.trace_samples is not None
    trace_real_variance = result.trace_samples[0].standard_error**2
    trace_imag_variance = result.trace_samples[1].standard_error**2
    factor = (-(result.model.A**3)) ** result.word.writhe
    factor *= result.model.d ** (result.model.strands - 1)
    expected_real_error = math.sqrt(
        factor.real**2 * trace_real_variance
        + factor.imag**2 * trace_imag_variance
    )
    expected_imag_error = math.sqrt(
        factor.imag**2 * trace_real_variance
        + factor.real**2 * trace_imag_variance
    )

    assert result.real_standard_error == pytest.approx(expected_real_error)
    assert result.imag_standard_error == pytest.approx(expected_imag_error)


def test_custom_sampler_receives_one_batched_job() -> None:
    sampler = RecordingSampler()
    result = JonesProblem("s1^2", strands=2).evaluate(
        method="shots",
        shots=128,
        sampler=sampler,
    )

    assert len(sampler.calls) == 1
    pub_count, global_shots, pub_shots = sampler.calls[0]
    assert pub_count == result.circuit_count
    assert global_shots is None
    assert result.trace_samples is not None
    real_circuits = result.trace_samples[0].circuit_count
    assert sum(pub_shots[:real_circuits]) == 128
    assert sum(pub_shots[real_circuits:]) == 128
    assert result.sampler_name == "RecordingSampler"
    assert result.total_shots == 256


def test_trace_shots_do_not_enumerate_valid_paths(monkeypatch) -> None:
    def fail_enumeration(_model):
        raise AssertionError("trace shot evaluation enumerated valid paths")

    monkeypatch.setattr(AJLPathModel, "valid_paths", fail_enumeration)
    result = JonesProblem(BraidWord.identity(), strands=4).evaluate(
        method="shots",
        circuit_level=2,
        shots=32,
        seed=11,
    )

    assert result.path_sampling == "ajl_weighted"
    assert result.total_shots == 64
    assert result.trace_samples is not None
    assert all(
        result.model.is_valid_path(path)
        for sample in result.trace_samples
        for path in sample.path_counts
    )


def test_custom_sampler_can_use_an_independent_path_seed() -> None:
    first = JonesProblem("s1^2", strands=2).evaluate(
        method="shots",
        shots=128,
        path_seed=37,
        sampler=RecordingSampler(seed=5),
    )
    second = JonesProblem("s1^2", strands=2).evaluate(
        method="shots",
        shots=128,
        path_seed=37,
        sampler=RecordingSampler(seed=5),
    )

    assert first.trace_samples == second.trace_samples
    assert first.value == second.value


def test_trace_sampler_rejects_wrong_per_pub_shot_counts() -> None:
    with pytest.raises(ValueError, match="shots but .* were requested"):
        JonesProblem("s1^2", strands=2).evaluate(
            method="shots",
            shots=128,
            path_seed=13,
            sampler=WrongShotSampler(),
        )


def test_sampled_component_rejects_a_malformed_result() -> None:
    with pytest.raises(ValueError, match="expected 'meas'"):
        AJLJonesEvaluator._sampled_component(object())


def test_sampled_plat_is_reproducible_and_propagates_its_normalization() -> None:
    problem = JonesProblem("s1", strands=2, closure="plat", writhe=-1)
    kwargs = {"method": "shots", "circuit_level": 2, "shots": 2048, "seed": 41}
    first = problem.evaluate(**kwargs)
    second = problem.evaluate(**kwargs)
    estimate = first.path_estimates[0]
    factor = first.model.plat_closure_jones(1.0, writhe=-1)
    expected_real_error = math.sqrt(
        factor.real**2 * estimate.real.standard_error**2
        + factor.imag**2 * estimate.imag.standard_error**2
    )
    expected_imag_error = math.sqrt(
        factor.imag**2 * estimate.real.standard_error**2
        + factor.real**2 * estimate.imag.standard_error**2
    )

    assert first.path_estimates == second.path_estimates
    assert first.path_sampling == "fixed_plat"
    assert first.trace_samples is None
    assert first.circuit_count == 2
    assert first.shots_per_circuit == 2048
    assert first.shots_per_component == 2048
    assert first.total_shots == 4096
    assert first.real_standard_error == pytest.approx(expected_real_error)
    assert first.imag_standard_error == pytest.approx(expected_imag_error)
    expected_value_bound = abs(factor) * math.sqrt(
        16.0 * math.log(2.0) / 2048
    )
    assert first.markov_trace_additive_error_bound is None
    assert first.value_additive_error_bound == pytest.approx(
        expected_value_bound
    )
    assert first.ajl_success_probability == AJL_SUCCESS_PROBABILITY


def test_plat_target_additive_error_selects_minimal_shots() -> None:
    target_additive_error = 1.0
    success_probability = 0.75
    confidence_factor = math.log(4.0) - math.log1p(-success_probability)
    expected_shots = math.ceil(
        4.0 * confidence_factor / target_additive_error**2
    )
    result = JonesProblem("s1", strands=2, closure="plat", writhe=-1).evaluate(
        method="shots",
        circuit_level=2,
        target_additive_error=target_additive_error,
        success_probability=success_probability,
        seed=67,
    )

    assert result.shots_per_circuit == expected_shots
    assert result.shots_per_component == expected_shots
    assert result.total_shots == 2 * expected_shots
    assert result.markov_trace_additive_error_bound is None
    assert result.value_additive_error_bound <= target_additive_error
    assert math.sqrt(
        4.0 * confidence_factor / (expected_shots - 1)
    ) > target_additive_error
    assert result.ajl_success_probability == success_probability


def test_custom_sampler_batches_only_two_plat_circuits() -> None:
    sampler = RecordingSampler()
    result = JonesProblem("s1", strands=2, closure="plat", writhe=-1).evaluate(
        method="shots",
        circuit_level=2,
        shots=128,
        sampler=sampler,
    )

    assert sampler.calls == [(2, 128, (128, 128))]
    assert result.total_shots == 256


def test_shot_mode_uses_documented_default() -> None:
    result = JonesProblem(BraidWord.identity(), strands=2).evaluate(
        method="shots",
        circuit_level=2,
        seed=3,
    )
    assert result.shots_per_circuit is None
    assert result.shots_per_component == DEFAULT_SHOTS
    assert result.total_shots == 2 * DEFAULT_SHOTS
    assert result.ajl_success_probability == AJL_SUCCESS_PROBABILITY
    assert result.markov_trace_additive_error_bound == pytest.approx(
        math.sqrt(16.0 * math.log(2.0) / DEFAULT_SHOTS)
    )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"method": "invalid"}, "method must be"),
        ({"circuit_level": 1}, "circuit_level must be"),
        ({"method": "shots", "shots": 0}, "shots must be"),
        ({"method": "shots", "shots": 1.5}, "shots must be"),
        ({"shots": 10}, "shots can only"),
        ({"target_additive_error": 1.0}, "target_additive_error can only"),
        ({"success_probability": 0.9}, "success_probability can only"),
        (
            {
                "method": "shots",
                "shots": 10,
                "target_additive_error": 1.0,
            },
            "mutually exclusive",
        ),
        (
            {"method": "shots", "target_additive_error": 0.0},
            "target_additive_error must be",
        ),
        (
            {"method": "shots", "target_additive_error": -1.0},
            "target_additive_error must be",
        ),
        (
            {"method": "shots", "target_additive_error": float("nan")},
            "target_additive_error must be",
        ),
        (
            {"method": "shots", "target_additive_error": float("inf")},
            "target_additive_error must be",
        ),
        (
            {"method": "shots", "target_additive_error": True},
            "target_additive_error must be",
        ),
        (
            {"method": "shots", "target_additive_error": "1.0"},
            "target_additive_error must be",
        ),
        (
            {"method": "shots", "target_additive_error": 1e-300},
            "unrepresentable shot count",
        ),
        (
            {"method": "shots", "success_probability": 0.0},
            "success_probability must be",
        ),
        (
            {"method": "shots", "success_probability": 1.0},
            "success_probability must be",
        ),
        (
            {"method": "shots", "success_probability": float("nan")},
            "success_probability must be",
        ),
        (
            {"method": "shots", "success_probability": float("inf")},
            "success_probability must be",
        ),
        (
            {"method": "shots", "success_probability": True},
            "success_probability must be",
        ),
        (
            {"method": "shots", "success_probability": "0.9"},
            "success_probability must be",
        ),
        ({"seed": 4}, "seed can only"),
        ({"path_seed": 4}, "path_seed can only"),
        ({"sampler": RecordingSampler()}, "sampler can only"),
        (
            {"method": "shots", "path_seed": -1},
            "path_seed must be",
        ),
        (
            {"method": "shots", "path_seed": 1.5},
            "path_seed must be",
        ),
        (
            {
                "method": "shots",
                "sampler": RecordingSampler(),
                "seed": 4,
            },
            "seed configures only",
        ),
    ],
)
def test_invalid_evaluation_options_are_rejected(kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        JonesProblem("s1^2", strands=2).evaluate(**kwargs)


def test_plat_shots_reject_a_path_seed() -> None:
    plat = JonesProblem("s1^2", strands=2, closure="plat", writhe=0)

    with pytest.raises(ValueError, match="path_seed can only"):
        plat.evaluate(method="shots", path_seed=4)


def test_declared_strands_must_support_the_braid_word() -> None:
    with pytest.raises(ValueError, match="needs 4 strands"):
        JonesProblem("s3", strands=3).evaluate()


def test_plat_closure_rejects_odd_strand_counts() -> None:
    with pytest.raises(ValueError, match="even number of strands"):
        JonesProblem("s1^2", strands=3, closure="plat", writhe=0).evaluate()
