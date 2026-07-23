from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from qiskit.primitives import BaseSamplerV2, StatevectorSampler

from digital_compiler import (
    DEFAULT_SHOTS,
    AJLJonesEvaluator,
    AJLPathModel,
    BraidWord,
    evaluate_jones,
)

TOL = 1e-9


class RecordingSampler(BaseSamplerV2):
    """SamplerV2 test double that records batching before delegating locally."""

    def __init__(self, seed: int = 17):
        self.delegate = StatevectorSampler(seed=seed)
        self.calls: list[tuple[int, int | None]] = []

    def run(self, pubs, *, shots=None):
        pubs = list(pubs)
        self.calls.append((len(pubs), shots))
        return self.delegate.run(pubs, shots=shots)


@pytest.mark.parametrize("circuit_level", [2, 3])
def test_exact_hopf_evaluation_matches_analytic_value(circuit_level: int) -> None:
    result = evaluate_jones(
        "s1^2",
        strands=2,
        k=5,
        circuit_level=circuit_level,
    )
    expected = -(result.model.A**10) - result.model.A**2

    assert abs(result.value - expected) < TOL
    assert result.method == "statevector"
    assert result.closure == "trace"
    assert result.plat_writhe is None
    assert result.markov_trace is not None
    assert result.plat_amplitude is None
    assert result.circuit_level == circuit_level
    assert result.circuit_count == 4
    assert result.shots_per_circuit is None
    assert result.total_shots == 0
    assert result.sampler_name is None
    assert result.real_standard_error == 0.0
    assert result.imag_standard_error == 0.0
    assert tuple(result.path_amplitudes) == result.model.valid_paths()
    for estimate in result.path_estimates:
        assert estimate.real.counts is None
        assert estimate.imag.counts is None
        assert estimate.real.shots == 0
        assert estimate.imag.shots == 0


def test_exact_sigma3_squared_four_strand_evaluation() -> None:
    result = evaluate_jones("s3^2", strands=4, k=5)
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
    result = evaluate_jones(
        BraidWord.identity(),
        strands=4,
        k=5,
        closure="plat",
        plat_writhe=0,
        circuit_level=circuit_level,
    )

    assert result.value == pytest.approx(result.model.d)
    assert result.closure == "plat"
    assert result.plat_writhe == 0
    assert result.markov_trace is None
    assert result.plat_amplitude == pytest.approx(1.0)
    assert tuple(result.path_amplitudes) == (result.model.plat_path(),)
    assert result.circuit_count == 2
    assert result.total_shots == 0


def test_plat_uses_oriented_closure_writhe_not_braid_exponent_sum() -> None:
    result = evaluate_jones(
        "s1",
        strands=2,
        closure="plat",
        plat_writhe=-1,
        circuit_level=2,
    )

    assert result.word.writhe == 1
    assert result.plat_writhe == -1
    assert result.value == pytest.approx(1.0, abs=TOL)


def test_seeded_shot_evaluation_is_reproducible_and_reports_cost() -> None:
    kwargs = {
        "strands": 2,
        "k": 5,
        "method": "shots",
        "circuit_level": 2,
        "shots": 10_000,
        "seed": 23,
    }
    first = evaluate_jones("s1^2", **kwargs)
    second = evaluate_jones("s1^2", **kwargs)
    expected = -(first.model.A**10) - first.model.A**2

    assert first.value == second.value
    assert first.path_estimates == second.path_estimates
    assert first.sampler_name == "StatevectorSampler"
    assert first.shots_per_circuit == 10_000
    assert first.circuit_count == 4
    assert first.total_shots == 40_000
    assert abs(first.value.real - expected.real) <= 5.0 * first.real_standard_error
    assert abs(first.value.imag - expected.imag) <= 5.0 * first.imag_standard_error
    for estimate in first.path_estimates:
        for component in (estimate.real, estimate.imag):
            assert component.counts is not None
            assert sum(component.counts.values()) == 10_000
            assert component.standard_error >= 0.0
            with pytest.raises(TypeError):
                component.counts["0"] = 0
    with pytest.raises(FrozenInstanceError):
        first.value = 0j


def test_shot_uncertainties_are_propagated_to_jones_components() -> None:
    result = evaluate_jones(
        "s1^2",
        strands=2,
        method="shots",
        circuit_level=2,
        shots=2048,
        seed=31,
    )
    total_weight = math.fsum(
        estimate.endpoint_weight for estimate in result.path_estimates
    )
    trace_real_variance = math.fsum(
        (estimate.endpoint_weight / total_weight) ** 2
        * estimate.real.standard_error**2
        for estimate in result.path_estimates
    )
    trace_imag_variance = math.fsum(
        (estimate.endpoint_weight / total_weight) ** 2
        * estimate.imag.standard_error**2
        for estimate in result.path_estimates
    )
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
    result = evaluate_jones(
        "s1^2",
        strands=2,
        method="shots",
        shots=128,
        sampler=sampler,
    )

    assert sampler.calls == [(4, 128)]
    assert result.sampler_name == "RecordingSampler"
    assert result.total_shots == 512


def test_sampled_plat_is_reproducible_and_propagates_its_normalization() -> None:
    kwargs = {
        "strands": 2,
        "closure": "plat",
        "plat_writhe": -1,
        "method": "shots",
        "circuit_level": 2,
        "shots": 2048,
        "seed": 41,
    }
    first = evaluate_jones("s1", **kwargs)
    second = evaluate_jones("s1", **kwargs)
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
    assert first.circuit_count == 2
    assert first.total_shots == 4096
    assert first.real_standard_error == pytest.approx(expected_real_error)
    assert first.imag_standard_error == pytest.approx(expected_imag_error)


def test_custom_sampler_batches_only_two_plat_circuits() -> None:
    sampler = RecordingSampler()
    result = evaluate_jones(
        "s1",
        strands=2,
        closure="plat",
        plat_writhe=-1,
        method="shots",
        circuit_level=2,
        shots=128,
        sampler=sampler,
    )

    assert sampler.calls == [(2, 128)]
    assert result.total_shots == 256


def test_shot_mode_uses_documented_default() -> None:
    result = evaluate_jones(
        BraidWord.identity(),
        strands=2,
        method="shots",
        circuit_level=2,
        seed=3,
    )
    assert result.shots_per_circuit == DEFAULT_SHOTS
    assert result.total_shots == result.circuit_count * DEFAULT_SHOTS


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"method": "invalid"}, "method must be"),
        ({"circuit_level": 1}, "circuit_level must be"),
        ({"closure": "invalid"}, "closure must be"),
        ({"closure": "plat"}, "plat_writhe is required"),
        ({"plat_writhe": 0}, "plat_writhe can only"),
        (
            {"closure": "plat", "plat_writhe": 1.5},
            "plat_writhe must be an integer",
        ),
        (
            {"closure": "plat", "plat_writhe": True},
            "plat_writhe must be an integer",
        ),
        ({"method": "shots", "shots": 0}, "shots must be"),
        ({"method": "shots", "shots": 1.5}, "shots must be"),
        ({"shots": 10}, "shots can only"),
        ({"seed": 4}, "seed can only"),
        ({"sampler": RecordingSampler()}, "sampler can only"),
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
        evaluate_jones("s1^2", strands=2, **kwargs)


def test_declared_strands_must_support_the_braid_word() -> None:
    with pytest.raises(ValueError, match="needs 4 strands"):
        evaluate_jones("s3", strands=3)


def test_plat_closure_rejects_odd_strand_counts() -> None:
    with pytest.raises(ValueError, match="even number of strands"):
        evaluate_jones(
            "s1^2",
            strands=3,
            closure="plat",
            plat_writhe=0,
        )
