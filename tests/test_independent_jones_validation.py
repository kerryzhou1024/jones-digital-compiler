"""Independent Kauffman-bracket checks of exact compiler evaluations."""

from __future__ import annotations

import cmath
import math
from itertools import product

import pytest

from digital_compiler import JonesProblem

LaurentPolynomial = dict[int, int]
TOL = 1e-9
K5_A = 1j * cmath.exp(-1j * math.pi / 10)
DELTA: LaurentPolynomial = {2: -1, -2: -1}


def _add(left: LaurentPolynomial, right: LaurentPolynomial) -> LaurentPolynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, 0) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _multiply(left: LaurentPolynomial, right: LaurentPolynomial) -> LaurentPolynomial:
    result: LaurentPolynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = left_exponent + right_exponent
            result[exponent] = result.get(exponent, 0) + left_coefficient * right_coefficient
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def _power(base: LaurentPolynomial, exponent: int) -> LaurentPolynomial:
    result = {0: 1}
    for _ in range(exponent):
        result = _multiply(result, base)
    return result


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _loops_in_trace_closure(
    strands: int,
    word: tuple[int, ...],
    smoothings: tuple[str, ...],
) -> int:
    layers = len(word)
    disjoint = _DisjointSet((layers + 1) * strands)

    def node(layer: int, position: int) -> int:
        return layer * strands + position

    for layer, (signed_index, smoothing) in enumerate(zip(word, smoothings, strict=True)):
        index = abs(signed_index) - 1
        for position in range(strands):
            if position not in (index, index + 1):
                disjoint.union(node(layer, position), node(layer + 1, position))
        if smoothing == "I":
            disjoint.union(node(layer, index), node(layer + 1, index))
            disjoint.union(node(layer, index + 1), node(layer + 1, index + 1))
        else:
            disjoint.union(node(layer, index), node(layer, index + 1))
            disjoint.union(node(layer + 1, index), node(layer + 1, index + 1))

    for position in range(strands):
        disjoint.union(node(0, position), node(layers, position))

    return len(
        {disjoint.find(value) for value in range((layers + 1) * strands)}
    )


def _jones_laurent(strands: int, word: tuple[int, ...]) -> LaurentPolynomial:
    """Enumerate bracket smoothings and apply the oriented writhe factor."""

    bracket: LaurentPolynomial = {}
    for smoothings in product(("I", "E"), repeat=len(word)):
        exponent = 0
        for signed_index, smoothing in zip(word, smoothings, strict=True):
            positive = signed_index > 0
            exponent += (
                (-1 if smoothing == "I" else 1)
                if positive
                else (1 if smoothing == "I" else -1)
            )
        loops = _loops_in_trace_closure(strands, word, smoothings)
        bracket = _add(
            bracket,
            _multiply({exponent: 1}, _power(DELTA, loops - 1)),
        )

    writhe = sum(1 if generator > 0 else -1 for generator in word)
    return _multiply({3 * writhe: -1 if writhe % 2 else 1}, bracket)


def _evaluate(poly: LaurentPolynomial, value: complex) -> complex:
    return complex(
        sum(coefficient * value**exponent for exponent, coefficient in poly.items())
    )


@pytest.mark.parametrize(
    "strands,word,expected_laurent",
    [
        (2, (1,), {0: 1}),
        (2, (1, 1), {2: -1, 10: -1}),
        (2, (1, 1, 1), {4: 1, 12: 1, 16: -1}),
        (2, (1, 1, 1, 1), {6: -1, 14: -1, 18: 1, 22: -1}),
        (3, (1, -2, 1, -2), {-8: 1, -4: -1, 0: 1, 4: -1, 8: 1}),
        (
            3,
            (1, -2, 1, -2, 1, -2),
            {-12: -1, -8: 3, -4: -2, 0: 4, 4: -2, 8: 3, 12: -1},
        ),
    ],
    ids=(
        "unknot",
        "hopf-link",
        "trefoil-knot",
        "solomon-link",
        "figure-eight-knot",
        "borromean-rings",
    ),
)
def test_level_3_value_matches_independent_kauffman_bracket(
    strands: int,
    word: tuple[int, ...],
    expected_laurent: LaurentPolynomial,
) -> None:
    oracle_laurent = _jones_laurent(strands, word)
    assert oracle_laurent == expected_laurent

    oracle_value = _evaluate(oracle_laurent, K5_A)
    compiler_value = JonesProblem(word, strands=strands, k=5).evaluate(
        method="statevector",
        circuit_level=3,
    ).value

    assert abs(compiler_value - oracle_value) < TOL
