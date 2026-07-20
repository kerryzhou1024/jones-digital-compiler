"""Dense correctness oracle for small AJL instances."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .model import TOL, AJLPathModel, BraidGenerator, BraidWord


class DenseAJLReference:
    """Dense AJL path-model matrices for validation, never circuit construction."""

    def __init__(self, model: AJLPathModel):
        self.model = model
        self.paths = model.valid_paths()
        self.path_index = {path: index for index, path in enumerate(self.paths)}
        if not self.paths:
            raise ValueError("no valid AJL paths for these parameters")

    @property
    def dimension(self) -> int:
        return len(self.paths)

    def basis_labels(self) -> list[str]:
        return [self.model.bitstring(path) for path in self.paths]

    def temperley_lieb_matrix(self, index: int) -> np.ndarray:
        if index < 1 or index >= self.model.strands:
            raise ValueError(f"generator index must be in 1..{self.model.strands - 1}")

        phi = np.zeros((self.dimension, self.dimension), dtype=complex)
        local_basis = {(0, 1): 0, (1, 0): 1}
        local_pairs = [(0, 1), (1, 0)]

        for column, path in enumerate(self.paths):
            local_pair = (path[index - 1], path[index])
            if local_pair not in local_basis:
                continue
            height = self.model.vertices(path)[index - 1]
            block = self.model.temperley_lieb_block(height)
            local_column = local_basis[local_pair]

            for local_row, output_pair in enumerate(local_pairs):
                output_path = list(path)
                output_path[index - 1], output_path[index] = output_pair
                output_tuple = tuple(output_path)
                coefficient = block[local_row, local_column]
                if abs(coefficient) > TOL and output_tuple in self.path_index:
                    phi[self.path_index[output_tuple], column] += coefficient
        return phi

    def generator_matrix(
        self,
        generator: BraidGenerator | int,
        sign: int | None = None,
    ) -> np.ndarray:
        if isinstance(generator, BraidGenerator):
            index = generator.index
            generator_sign = generator.sign
        else:
            index = int(generator)
            generator_sign = 1 if sign is None else int(sign)
        if generator_sign not in (-1, 1):
            raise ValueError("generator sign must be +1 or -1")

        phi = self.temperley_lieb_matrix(index)
        identity = np.eye(self.dimension, dtype=complex)
        if generator_sign == 1:
            return self.model.A * phi + (self.model.A**-1) * identity
        return (self.model.A**-1) * phi + self.model.A * identity

    def compile_matrix(self, word: BraidWord | str | Sequence[int]) -> np.ndarray:
        braid_word = self.model.as_braid_word(word)
        matrix = np.eye(self.dimension, dtype=complex)
        for generator in braid_word.generators:
            matrix = self.generator_matrix(generator) @ matrix
        return matrix

    def path_amplitude(
        self,
        word: BraidWord | str | Sequence[int],
        path: str | Sequence[int],
    ) -> complex:
        bits = self.model.coerce_path(path)
        index = self.path_index[bits]
        return complex(self.compile_matrix(word)[index, index])

    def markov_trace_observable(self, matrix: np.ndarray) -> complex:
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("matrix has the wrong shape for this reference")
        amplitudes = dict(zip(self.paths, np.diag(matrix), strict=True))
        return self.model.markov_trace(amplitudes)

    @staticmethod
    def little_endian_index(bits: Sequence[int]) -> int:
        return sum(int(bit) << position for position, bit in enumerate(bits))

    def embed_valid_subspace(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("matrix has the wrong shape for this reference")
        full = np.eye(2**self.model.strands, dtype=complex)
        full_indices = [self.little_endian_index(path) for path in self.paths]
        for row, full_row in enumerate(full_indices):
            for column, full_column in enumerate(full_indices):
                full[full_row, full_column] = matrix[row, column]
        return full

    def full_generator_matrix(
        self,
        generator: BraidGenerator | int,
        sign: int | None = None,
    ) -> np.ndarray:
        return self.embed_valid_subspace(self.generator_matrix(generator, sign))

    def full_braid_matrix(self, word: BraidWord | str | Sequence[int]) -> np.ndarray:
        braid_word = self.model.as_braid_word(word)
        matrix = np.eye(2**self.model.strands, dtype=complex)
        for generator in braid_word.generators:
            matrix = self.full_generator_matrix(generator) @ matrix
        return matrix
