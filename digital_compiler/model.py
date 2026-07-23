"""Non-dense AJL mathematics and braid-word domain objects.

The model owns path rules, local blocks, endpoint weights, and closure
normalization.  It never constructs a full path-space matrix, and potentially
expensive valid-path enumeration occurs only when explicitly requested.
"""

from __future__ import annotations

import cmath
import math
import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

TOL = 1e-10
HadamardPart = Literal["real", "imag"]


def _coerce_integer(value: object, name: str) -> int:
    """Return an exact integer without silently truncating numeric inputs."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None


def _coerce_path_bit(value: object) -> int:
    """Return one exact binary path step."""

    try:
        bit = int(operator.index(value))
    except TypeError:
        raise ValueError("path bits must be integers equal to 0 or 1") from None
    if bit not in (0, 1):
        raise ValueError("path bits must be 0 or 1")
    return bit


@dataclass(frozen=True)
class BraidGenerator:
    """One signed braid generator :math:`\sigma_i^{\pm1}`."""

    index: int
    sign: int = 1

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("braid-generator index must be at least 1")
        if self.sign not in (-1, 1):
            raise ValueError("braid-generator sign must be +1 or -1")

    @property
    def label(self) -> str:
        return f"sigma_{self.index}" if self.sign == 1 else f"sigma_{self.index}^-1"

    @property
    def signed_index(self) -> int:
        return self.sign * self.index


@dataclass(frozen=True)
class BraidWord:
    """A chronological tuple of braid generators."""

    generators: tuple[BraidGenerator, ...]

    @classmethod
    def identity(cls) -> BraidWord:
        return cls(())

    @classmethod
    def from_signed_indices(cls, signed_indices: Sequence[int]) -> BraidWord:
        generators = []
        for raw_value in signed_indices:
            value = int(raw_value)
            if value == 0:
                raise ValueError("0 is not a braid generator")
            generators.append(BraidGenerator(abs(value), 1 if value > 0 else -1))
        return cls(tuple(generators))

    @classmethod
    def power(cls, index: int, exponent: int) -> BraidWord:
        if exponent == 0:
            return cls.identity()
        sign = 1 if exponent > 0 else -1
        return cls(tuple(BraidGenerator(index, sign) for _ in range(abs(exponent))))

    @classmethod
    def parse(cls, text: str) -> BraidWord:
        generators: list[BraidGenerator] = []
        for raw in text.replace(",", " ").split():
            token = raw.lower().replace("sigma", "s")
            token = token.replace("_", "").replace("{", "").replace("}", "")

            prefix_sign = 1
            if token.startswith("-"):
                prefix_sign = -1
                token = token[1:]

            match = re.fullmatch(r"s?(\d+)(?:\^(-?\d+))?", token)
            if match is None:
                raise ValueError(f"could not parse braid token {raw!r}")

            index = int(match.group(1))
            exponent = int(match.group(2) or "1")
            if exponent == 0:
                continue
            sign = prefix_sign * (1 if exponent > 0 else -1)
            generators.extend(BraidGenerator(index, sign) for _ in range(abs(exponent)))
        return cls(tuple(generators))

    @property
    def crossings(self) -> int:
        return len(self.generators)

    @property
    def writhe(self) -> int:
        """Return the braid exponent sum, equal to the standard trace writhe."""

        return sum(generator.sign for generator in self.generators)

    @property
    def strands_needed(self) -> int:
        return 1 if not self.generators else max(g.index for g in self.generators) + 1

    def signed_indices(self) -> tuple[int, ...]:
        return tuple(generator.signed_index for generator in self.generators)

    def __str__(self) -> str:
        if not self.generators:
            return "identity"
        return " ".join(generator.label for generator in self.generators)


def coerce_braid_word(word: BraidWord | str | Sequence[int]) -> BraidWord:
    """Normalize any supported braid-word input into :class:`BraidWord`."""

    if isinstance(word, BraidWord):
        return word
    if isinstance(word, str):
        return BraidWord.parse(word)
    return BraidWord.from_signed_indices(word)


@dataclass(frozen=True)
class AJLPathModel:
    """Lightweight source of truth for non-dense AJL path-model mathematics."""

    strands: int
    level: int = 5
    lambdas: tuple[float, ...] = field(init=False, repr=False)
    d: float = field(init=False)
    A: complex = field(init=False)

    def __post_init__(self) -> None:
        strands = _coerce_integer(self.strands, "strands")
        level = _coerce_integer(self.level, "level")
        if strands < 2:
            raise ValueError("strands must be at least 2")
        if level < 3:
            raise ValueError("level k must be at least 3")

        lambdas = [math.sin(math.pi * j / level) for j in range(level + 1)]
        lambdas[0] = lambdas[-1] = 0.0
        object.__setattr__(self, "strands", strands)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "lambdas", tuple(lambdas))
        object.__setattr__(self, "d", float(2.0 * math.cos(math.pi / level)))
        object.__setattr__(
            self,
            "A",
            complex(1j * cmath.exp(-1j * math.pi / (2.0 * level))),
        )

    @property
    def valid_heights(self) -> range:
        """Interior path-graph vertices that can occur as AJL heights."""

        return range(1, self.level)

    def lambda_at(self, vertex: int) -> float:
        vertex = _coerce_integer(vertex, "vertex")
        if vertex < 0 or vertex > self.level:
            return 0.0
        return self.lambdas[vertex]

    def vertices(self, bits: Sequence[int]) -> tuple[int, ...]:
        vertex = 1
        vertices = [vertex]
        for raw_bit in bits:
            bit = _coerce_path_bit(raw_bit)
            vertex += 1 if bit == 1 else -1
            vertices.append(vertex)
        return tuple(vertices)

    def is_valid_path(self, bits: Sequence[int]) -> bool:
        return all(1 <= vertex <= self.level - 1 for vertex in self.vertices(bits))

    def coerce_path(self, path: str | Sequence[int]) -> tuple[int, ...]:
        if isinstance(path, str):
            text = path.strip().replace("|", "").replace(">", "")
            if any(character not in "01" for character in text):
                raise ValueError("path strings must contain only 0 and 1")
            bits = tuple(int(character) for character in text)
        else:
            bits = tuple(_coerce_path_bit(bit) for bit in path)

        if len(bits) != self.strands:
            raise ValueError(f"path must have {self.strands} bits")
        if not self.is_valid_path(bits):
            raise ValueError(f"{self.bitstring(bits)} is not a valid AJL path")
        return bits

    @staticmethod
    def bitstring(bits: Sequence[int]) -> str:
        return "".join(str(_coerce_path_bit(bit)) for bit in bits)

    def valid_paths(self) -> tuple[tuple[int, ...], ...]:
        """Enumerate complete valid paths, pruning boundary violations eagerly.

        Enumeration is deliberately on demand: constructing a model or compiler
        never pays the potentially exponential cost of listing its path basis.
        """

        paths: list[tuple[int, ...]] = []

        def visit(position: int, vertex: int, prefix: tuple[int, ...]) -> None:
            if position == self.strands:
                paths.append(prefix)
                return
            if vertex > 1:
                visit(position + 1, vertex - 1, (*prefix, 0))
            if vertex < self.level - 1:
                visit(position + 1, vertex + 1, (*prefix, 1))

        visit(0, 1, ())
        return tuple(paths)

    def plat_path(self) -> tuple[int, ...]:
        """Return the alternating AJL path used for an even-strand plat closure."""

        if self.strands % 2:
            raise ValueError("plat closure requires an even number of strands")
        return (1, 0) * (self.strands // 2)

    def endpoint_vertex(self, path: str | Sequence[int]) -> int:
        """Return the terminal graph vertex of one complete valid path."""

        bits = self.coerce_path(path)
        return self.vertices(bits)[-1]

    def endpoint_weight(self, path: str | Sequence[int]) -> float:
        """Return the AJL Markov weight associated with a path endpoint."""

        return self.lambda_at(self.endpoint_vertex(path))

    def endpoint_weights(
        self,
        paths: Iterable[str | Sequence[int]] | None = None,
    ) -> tuple[float, ...]:
        """Return endpoint weights in exactly the supplied path order."""

        selected_paths = self.valid_paths() if paths is None else paths
        return tuple(self.endpoint_weight(path) for path in selected_paths)

    def as_braid_word(self, word: BraidWord | str | Sequence[int]) -> BraidWord:
        braid_word = coerce_braid_word(word)
        if braid_word.strands_needed > self.strands:
            raise ValueError(
                f"word needs {braid_word.strands_needed} strands but model has {self.strands}"
            )
        return braid_word

    def projector_state(self, height: int) -> np.ndarray:
        """Return the normalized local projector state for one valid height."""

        lambda_height = self.lambda_at(height)
        if abs(lambda_height) < TOL:
            raise ValueError(f"lambda_{height} is zero; invalid prefix vertex")
        return np.array(
            [
                math.sqrt(
                    max(0.0, self.lambda_at(height - 1) / (self.d * lambda_height))
                ),
                math.sqrt(
                    max(0.0, self.lambda_at(height + 1) / (self.d * lambda_height))
                ),
            ],
            dtype=float,
        )

    def projector_angle(self, height: int) -> float:
        state = self.projector_state(height)
        return math.atan2(float(state[1]), float(state[0]))

    def temperley_lieb_block(self, height: int) -> np.ndarray:
        """Return the local TL block in the basis ``[01, 10]``."""

        state = self.projector_state(height)
        return self.d * np.outer(state, state)

    def phase_angles(self, sign: int) -> tuple[float, float]:
        """Return the base and rank-one relative phase for a signed crossing."""

        if sign == 1:
            base_phase = self.A**-1
            rank_one_phase = -(self.A**3)
        elif sign == -1:
            base_phase = self.A
            rank_one_phase = -(self.A**-3)
        else:
            raise ValueError("generator sign must be +1 or -1")
        return float(np.angle(base_phase)), float(np.angle(rank_one_phase / base_phase))

    def markov_trace(
        self,
        path_amplitudes: Mapping[str | tuple[int, ...], complex],
    ) -> complex:
        """Return the normalized endpoint-weighted trace from path amplitudes."""

        normalized: dict[tuple[int, ...], complex] = {}
        for raw_path, amplitude in path_amplitudes.items():
            if not isinstance(raw_path, (str, Sequence)):
                raise ValueError("path-amplitude keys must be path strings or bit sequences")
            path = self.coerce_path(raw_path)
            if path in normalized:
                raise ValueError(
                    f"path amplitudes contain duplicate representations of {self.bitstring(path)}"
                )
            normalized[path] = complex(amplitude)

        paths = self.valid_paths()
        missing = tuple(path for path in paths if path not in normalized)
        if missing:
            labels = ", ".join(self.bitstring(path) for path in missing)
            raise ValueError(f"path amplitudes are missing valid paths: {labels}")
        if len(normalized) != len(paths):
            raise ValueError("path amplitudes contain paths outside the valid path basis")

        weights = self.endpoint_weights(paths)
        normalization = math.fsum(weights)
        if normalization <= TOL:
            raise ValueError("endpoint weights have zero normalization")
        weighted_amplitudes = (
            weight * normalized[path] for path, weight in zip(paths, weights, strict=True)
        )
        return complex(sum(weighted_amplitudes) / normalization)

    def trace_closure_jones(
        self,
        word: BraidWord | str | Sequence[int],
        path_amplitudes: Mapping[str | tuple[int, ...], complex],
    ) -> complex:
        """Return the Jones value of the braid's trace closure in the AJL convention."""

        braid_word = self.as_braid_word(word)
        markov_trace = self.markov_trace(path_amplitudes)
        normalization = (-(self.A**3)) ** braid_word.writhe
        normalization *= self.d ** (self.strands - 1)
        return complex(normalization * markov_trace)

    def plat_closure_jones(self, amplitude: complex, *, writhe: int) -> complex:
        """Return an oriented plat-closure Jones value from its single amplitude."""

        self.plat_path()
        plat_writhe = _coerce_integer(writhe, "writhe")
        normalization = (-(self.A**3)) ** plat_writhe
        normalization *= self.d ** (self.strands // 2 - 1)
        return complex(normalization * complex(amplitude))


def clean_complex(value: complex, tol: float = TOL) -> complex:
    """Remove numerical noise from a complex scalar for compact reports."""

    real = 0.0 if abs(value.real) < tol else value.real
    imag = 0.0 if abs(value.imag) < tol else value.imag
    return complex(real, imag)
