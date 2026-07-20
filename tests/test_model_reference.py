from __future__ import annotations

import math

import numpy as np
import pytest

from digital_compiler import (
    AJLPathModel,
    BraidGenerator,
    BraidWord,
    DenseAJLReference,
)


def test_braid_word_parsing_and_properties() -> None:
    word = BraidWord.parse("sigma_1 s2^-2 -s1")
    assert word.signed_indices() == (1, -2, -2, -1)
    assert word.crossings == 4
    assert word.writhe == -2
    assert word.strands_needed == 3
    assert BraidWord.power(2, 0) == BraidWord.identity()


@pytest.mark.parametrize("value", [0, "s0"])
def test_invalid_braid_generator_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        if isinstance(value, str):
            BraidWord.parse(value)
        else:
            BraidWord.from_signed_indices([value])


def test_path_model_validates_boundaries_without_dense_enumeration() -> None:
    model = AJLPathModel(strands=3, level=5)
    assert model.lambda_at(0) == 0.0
    assert model.lambda_at(model.level) == 0.0
    assert model.valid_heights == range(1, 5)
    assert model.coerce_path("|101>") == (1, 0, 1)
    assert model.vertices((1, 0, 1)) == (1, 2, 1, 2)
    assert model.is_valid_path((1, 0, 1))
    assert not hasattr(model, "paths")

    with pytest.raises(ValueError, match="not a valid AJL path"):
        model.coerce_path("000")
    with pytest.raises(ValueError, match="needs 4 strands"):
        model.as_braid_word("s3")


@pytest.mark.parametrize(
    "strands,level",
    [
        (2.5, 5),
        (2, 5.5),
        (True, 5),
        (2, False),
    ],
)
def test_path_model_rejects_non_integral_parameters(strands, level) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        AJLPathModel(strands=strands, level=level)


@pytest.mark.parametrize("path", [(1, 0.0, 1), (1, "0", 1), (1, 2, 1)])
def test_path_model_rejects_non_binary_integer_steps(path) -> None:
    model = AJLPathModel(strands=3, level=5)
    with pytest.raises(ValueError, match="path bits"):
        model.coerce_path(path)


@pytest.mark.parametrize(
    "strands,level,expected",
    [
        (2, 5, ((1, 0), (1, 1))),
        (3, 5, ((1, 0, 1), (1, 1, 0), (1, 1, 1))),
        (4, 3, ((1, 0, 1, 0),)),
    ],
)
def test_valid_paths_are_boundary_pruned_and_deterministic(
    strands: int,
    level: int,
    expected: tuple[tuple[int, ...], ...],
) -> None:
    model = AJLPathModel(strands, level)
    assert model.valid_paths() == expected
    assert all(model.is_valid_path(path) for path in expected)


def test_endpoint_vertices_and_weights_preserve_path_order() -> None:
    model = AJLPathModel(2, 5)
    paths = model.valid_paths()
    assert tuple(model.endpoint_vertex(path) for path in paths) == (1, 3)
    assert model.endpoint_weights(paths) == pytest.approx(
        (model.lambda_at(1), model.lambda_at(3))
    )
    assert model.endpoint_weights(reversed(paths)) == pytest.approx(
        (model.lambda_at(3), model.lambda_at(1))
    )


@pytest.mark.parametrize("level", [3, 5, 6])
def test_projector_alignment(level: int) -> None:
    model = AJLPathModel(strands=3, level=level)
    for height in model.valid_heights:
        state = model.projector_state(height)
        angle = -2.0 * model.projector_angle(height)
        rotation = np.array(
            [
                [math.cos(angle / 2.0), -math.sin(angle / 2.0)],
                [math.sin(angle / 2.0), math.cos(angle / 2.0)],
            ]
        )
        np.testing.assert_allclose(rotation @ state, np.array([1.0, 0.0]), atol=1e-10)
        np.testing.assert_allclose(
            model.temperley_lieb_block(height),
            model.d * np.outer(state, state),
            atol=1e-10,
        )


def test_markov_trace_normalizes_paths_and_checks_complete_coverage() -> None:
    model = AJLPathModel(2, 5)
    amplitudes = {"10": 1.0 + 0.5j, (1, 1): -0.25j}
    weights = model.endpoint_weights()
    expected = (weights[0] * amplitudes["10"] + weights[1] * amplitudes[(1, 1)]) / sum(
        weights
    )
    assert model.markov_trace(amplitudes) == pytest.approx(expected)

    with pytest.raises(ValueError, match="missing valid paths: 11"):
        model.markov_trace({"10": 1.0})
    with pytest.raises(ValueError, match="duplicate representations of 10"):
        model.markov_trace({"10": 1.0, (1, 0): 1.0, "11": 1.0})
    with pytest.raises(ValueError, match="not a valid AJL path"):
        model.markov_trace({"10": 1.0, "00": 1.0})


def test_dense_oracle_composes_the_lightweight_model() -> None:
    model = AJLPathModel(strands=2, level=5)
    reference = DenseAJLReference(model)
    assert reference.model is model
    assert reference.basis_labels() == ["10", "11"]

    forward = reference.generator_matrix(BraidGenerator(1, 1))
    inverse = reference.generator_matrix(BraidGenerator(1, -1))
    np.testing.assert_allclose(
        forward,
        np.diag((-(model.A**3), model.A**-1)),
        atol=1e-10,
    )
    np.testing.assert_allclose(inverse @ forward, np.eye(reference.dimension), atol=1e-10)
    assert reference.markov_trace_observable(forward) == pytest.approx(
        model.markov_trace(dict(zip(model.valid_paths(), np.diag(forward), strict=True)))
    )
