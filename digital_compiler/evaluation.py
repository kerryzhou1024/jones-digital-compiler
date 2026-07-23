"""One-call circuit-based evaluation of AJL trace and plat closures."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from qiskit.primitives import BaseSamplerV2, StatevectorSampler
from qiskit.quantum_info import Statevector

from .compiler import AJLCompiler
from .model import AJLPathModel, BraidWord
from .policies import CompilerConfig

EvaluationMethod = Literal["statevector", "shots"]
EvaluationCircuitLevel = Literal[2, 3]
ClosureType = Literal["trace", "plat"]
DEFAULT_SHOTS = 4096


@dataclass(frozen=True)
class HadamardComponentEstimate:
    """One real or imaginary Hadamard-test expectation."""

    expectation: float
    standard_error: float
    shots: int = 0
    counts: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.counts is not None:
            object.__setattr__(
                self,
                "counts",
                MappingProxyType(dict(self.counts)),
            )


@dataclass(frozen=True)
class PathAmplitudeEstimate:
    """Circuit estimates of both components of one diagonal path amplitude."""

    path: tuple[int, ...]
    endpoint_weight: float
    real: HadamardComponentEstimate
    imag: HadamardComponentEstimate

    @property
    def amplitude(self) -> complex:
        return complex(self.real.expectation, self.imag.expectation)


@dataclass(frozen=True)
class JonesEvaluation:
    """Complete numerical Jones evaluation and its sampling metadata."""

    model: AJLPathModel
    word: BraidWord
    closure: ClosureType
    plat_writhe: int | None
    method: EvaluationMethod
    circuit_level: EvaluationCircuitLevel
    config: CompilerConfig
    path_estimates: tuple[PathAmplitudeEstimate, ...]
    markov_trace: complex | None
    value: complex
    real_standard_error: float
    imag_standard_error: float
    circuit_count: int
    shots_per_circuit: int | None
    total_shots: int
    sampler_name: str | None

    @property
    def strands(self) -> int:
        return self.model.strands

    @property
    def k(self) -> int:
        return self.model.level

    @property
    def path_amplitudes(self) -> Mapping[tuple[int, ...], complex]:
        return MappingProxyType(
            {estimate.path: estimate.amplitude for estimate in self.path_estimates}
        )

    @property
    def plat_amplitude(self) -> complex | None:
        """Return the single plat matrix element, or ``None`` for trace closure."""

        if self.closure != "plat" or len(self.path_estimates) != 1:
            return None
        return self.path_estimates[0].amplitude


class AJLJonesEvaluator:
    """Evaluate small AJL closures by executing compiled Hadamard circuits."""

    def __init__(
        self,
        model: AJLPathModel,
        config: CompilerConfig | None = None,
    ):
        self.model = model
        self.config = CompilerConfig() if config is None else config
        self.compiler = AJLCompiler(model, self.config)

    def _hadamard_circuit(
        self,
        word: BraidWord,
        path: tuple[int, ...],
        part: Literal["real", "imag"],
        circuit_level: EvaluationCircuitLevel,
        *,
        measure: bool,
    ):
        if circuit_level == 2:
            return self.compiler.level_2_multicontrolled_circuit(
                word,
                path,
                part,
                measure=measure,
            )
        return self.compiler.level_3_single_control_circuit(
            word,
            path,
            part,
            measure=measure,
        )

    @staticmethod
    def _statevector_component(circuit) -> HadamardComponentEstimate:
        state = Statevector.from_instruction(circuit)
        probabilities = state.probabilities(qargs=[0])
        expectation = float(probabilities[0] - probabilities[1])
        return HadamardComponentEstimate(
            expectation=expectation,
            standard_error=0.0,
        )

    @staticmethod
    def _sampled_component(pub_result) -> HadamardComponentEstimate:
        try:
            bit_array = pub_result.data.meas
        except AttributeError:
            raise ValueError(
                "sampler result does not contain the expected 'meas' classical register"
            ) from None

        counts = {str(bitstring): int(count) for bitstring, count in bit_array.get_counts().items()}
        unexpected = set(counts) - {"0", "1"}
        if unexpected:
            raise ValueError(
                "the evaluator expects one-bit measurement results, found "
                f"{sorted(unexpected)}"
            )
        shots = sum(counts.values())
        if shots <= 0:
            raise ValueError("sampler returned no shots")
        expectation = float((counts.get("0", 0) - counts.get("1", 0)) / shots)
        standard_error = math.sqrt(max(0.0, 1.0 - expectation**2) / shots)
        return HadamardComponentEstimate(
            expectation=expectation,
            standard_error=standard_error,
            shots=shots,
            counts=counts,
        )

    def _statevector_estimates(
        self,
        word: BraidWord,
        paths: tuple[tuple[int, ...], ...],
        circuit_level: EvaluationCircuitLevel,
    ) -> dict[tuple[tuple[int, ...], str], HadamardComponentEstimate]:
        estimates = {}
        for path in paths:
            for part in ("real", "imag"):
                circuit = self._hadamard_circuit(
                    word,
                    path,
                    part,
                    circuit_level,
                    measure=False,
                )
                estimates[(path, part)] = self._statevector_component(circuit)
        return estimates

    def _sampled_estimates(
        self,
        word: BraidWord,
        paths: tuple[tuple[int, ...], ...],
        circuit_level: EvaluationCircuitLevel,
        shots: int,
        sampler: BaseSamplerV2,
    ) -> dict[tuple[tuple[int, ...], str], HadamardComponentEstimate]:
        labels = []
        circuits = []
        for path in paths:
            for part in ("real", "imag"):
                labels.append((path, part))
                circuits.append(
                    self._hadamard_circuit(
                        word,
                        path,
                        part,
                        circuit_level,
                        measure=True,
                    )
                )

        results = sampler.run(circuits, shots=shots).result()
        if len(results) != len(labels):
            raise ValueError(
                f"sampler returned {len(results)} results for {len(labels)} circuits"
            )
        return {
            label: self._sampled_component(pub_result)
            for label, pub_result in zip(labels, results, strict=True)
        }

    def evaluate(
        self,
        word: BraidWord | str | Sequence[int],
        *,
        closure: ClosureType = "trace",
        plat_writhe: int | None = None,
        method: EvaluationMethod = "statevector",
        circuit_level: EvaluationCircuitLevel = 3,
        shots: int | None = None,
        seed: int | None = None,
        sampler: BaseSamplerV2 | None = None,
    ) -> JonesEvaluation:
        """Evaluate the numerical Jones value of a braid closure."""

        if method not in {"statevector", "shots"}:
            raise ValueError("method must be 'statevector' or 'shots'")
        if circuit_level not in {2, 3}:
            raise ValueError("circuit_level must be 2 or 3")

        normalized_plat_writhe = _validate_closure_options(closure, plat_writhe)

        braid_word = self.model.as_braid_word(word)
        paths = (
            self.model.valid_paths()
            if closure == "trace"
            else (self.model.plat_path(),)
        )
        sampler_name = None
        shots_per_circuit = None

        if method == "statevector":
            if shots is not None:
                raise ValueError("shots can only be specified when method='shots'")
            if seed is not None:
                raise ValueError("seed can only be specified when method='shots'")
            if sampler is not None:
                raise ValueError("sampler can only be specified when method='shots'")
            components = self._statevector_estimates(
                braid_word,
                paths,
                circuit_level,
            )
        else:
            shots_per_circuit = _positive_shots(
                DEFAULT_SHOTS if shots is None else shots
            )
            if sampler is None:
                active_sampler: BaseSamplerV2 = StatevectorSampler(seed=seed)
            else:
                if seed is not None:
                    raise ValueError(
                        "seed configures only the default StatevectorSampler; "
                        "configure a custom sampler directly"
                    )
                if not isinstance(sampler, BaseSamplerV2):
                    raise TypeError("sampler must implement Qiskit's BaseSamplerV2")
                active_sampler = sampler
            sampler_name = type(active_sampler).__name__
            components = self._sampled_estimates(
                braid_word,
                paths,
                circuit_level,
                shots_per_circuit,
                active_sampler,
            )

        path_estimates = tuple(
            PathAmplitudeEstimate(
                path=path,
                endpoint_weight=self.model.endpoint_weight(path),
                real=components[(path, "real")],
                imag=components[(path, "imag")],
            )
            for path in paths
        )
        amplitudes = {
            estimate.path: estimate.amplitude for estimate in path_estimates
        }
        if closure == "trace":
            markov_trace = self.model.markov_trace(amplitudes)
            value = self.model.trace_closure_jones(braid_word, amplitudes)
        else:
            markov_trace = None
            value = self.model.plat_closure_jones(
                path_estimates[0].amplitude,
                writhe=normalized_plat_writhe,
            )
        real_error, imag_error = self._propagate_standard_errors(
            braid_word,
            path_estimates,
            closure,
            normalized_plat_writhe,
        )
        total_shots = sum(
            estimate.real.shots + estimate.imag.shots
            for estimate in path_estimates
        )
        return JonesEvaluation(
            model=self.model,
            word=braid_word,
            closure=closure,
            plat_writhe=normalized_plat_writhe,
            method=method,
            circuit_level=circuit_level,
            config=self.config,
            path_estimates=path_estimates,
            markov_trace=markov_trace,
            value=value,
            real_standard_error=real_error,
            imag_standard_error=imag_error,
            circuit_count=2 * len(paths),
            shots_per_circuit=shots_per_circuit,
            total_shots=total_shots,
            sampler_name=sampler_name,
        )

    def _propagate_standard_errors(
        self,
        word: BraidWord,
        estimates: tuple[PathAmplitudeEstimate, ...],
        closure: ClosureType,
        plat_writhe: int | None,
    ) -> tuple[float, float]:
        if closure == "trace":
            total_weight = math.fsum(
                estimate.endpoint_weight for estimate in estimates
            )
            raw_real_variance = math.fsum(
                (estimate.endpoint_weight / total_weight) ** 2
                * estimate.real.standard_error**2
                for estimate in estimates
            )
            raw_imag_variance = math.fsum(
                (estimate.endpoint_weight / total_weight) ** 2
                * estimate.imag.standard_error**2
                for estimate in estimates
            )
            closure_factor = (-(self.model.A**3)) ** word.writhe
            closure_factor *= self.model.d ** (self.model.strands - 1)
        else:
            estimate = estimates[0]
            raw_real_variance = estimate.real.standard_error**2
            raw_imag_variance = estimate.imag.standard_error**2
            closure_factor = self.model.plat_closure_jones(
                1.0,
                writhe=plat_writhe,
            )

        real_variance = (
            closure_factor.real**2 * raw_real_variance
            + closure_factor.imag**2 * raw_imag_variance
        )
        imag_variance = (
            closure_factor.imag**2 * raw_real_variance
            + closure_factor.real**2 * raw_imag_variance
        )
        return math.sqrt(real_variance), math.sqrt(imag_variance)


def _validate_closure_options(
    closure: object,
    plat_writhe: object,
) -> int | None:
    if closure not in {"trace", "plat"}:
        raise ValueError("closure must be 'trace' or 'plat'")
    if closure == "trace":
        if plat_writhe is not None:
            raise ValueError("plat_writhe can only be specified for plat closure")
        return None
    if plat_writhe is None:
        raise ValueError("plat_writhe is required for plat closure")
    if isinstance(plat_writhe, bool):
        raise ValueError("plat_writhe must be an integer")
    try:
        return int(operator.index(plat_writhe))
    except TypeError:
        raise ValueError("plat_writhe must be an integer") from None


def _positive_shots(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("shots must be a positive integer")
    try:
        shots = int(operator.index(value))
    except TypeError:
        raise ValueError("shots must be a positive integer") from None
    if shots <= 0:
        raise ValueError("shots must be a positive integer")
    return shots


def evaluate_jones(
    word: BraidWord | str | Sequence[int],
    *,
    strands: int,
    k: int = 5,
    closure: ClosureType = "trace",
    plat_writhe: int | None = None,
    method: EvaluationMethod = "statevector",
    circuit_level: EvaluationCircuitLevel = 3,
    shots: int | None = None,
    seed: int | None = None,
    config: CompilerConfig | None = None,
    sampler: BaseSamplerV2 | None = None,
) -> JonesEvaluation:
    """Numerically evaluate a trace- or plat-closure Jones value."""

    evaluator = AJLJonesEvaluator(
        AJLPathModel(strands=strands, level=k),
        config,
    )
    return evaluator.evaluate(
        word,
        closure=closure,
        plat_writhe=plat_writhe,
        method=method,
        circuit_level=circuit_level,
        shots=shots,
        seed=seed,
        sampler=sampler,
    )
