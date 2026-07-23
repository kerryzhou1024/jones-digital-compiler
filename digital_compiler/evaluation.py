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
from .model import AJLPathModel, BraidWord, HadamardPart
from .policies import CompilerConfig

EvaluationMethod = Literal["statevector", "shots"]
EvaluationCircuitLevel = Literal[2, 3]
ClosureType = Literal["trace", "plat"]
DEFAULT_SHOTS = 4096


@dataclass(frozen=True)
class CircuitTask:
    """One path/component circuit required by a Jones evaluation."""

    path: tuple[int, ...]
    part: HadamardPart

    @property
    def path_label(self) -> str:
        return "".join(str(bit) for bit in self.path)


def circuit_tasks(
    paths: Sequence[tuple[int, ...]],
    *,
    part: HadamardPart | None = None,
) -> tuple[CircuitTask, ...]:
    """Expand paths into deterministic, optionally filtered component tasks."""

    if part is None:
        parts: tuple[HadamardPart, ...] = ("real", "imag")
    elif part == "real" or part == "imag":
        parts = (part,)
    else:
        raise ValueError("part must be 'real' or 'imag'")
    return tuple(
        CircuitTask(path=path, part=selected_part)
        for path in paths
        for selected_part in parts
    )


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
class _TraceClosure:
    kind: Literal["trace"] = "trace"
    writhe: None = None

    @staticmethod
    def paths(model: AJLPathModel) -> tuple[tuple[int, ...], ...]:
        return model.valid_paths()

    @staticmethod
    def evaluate(
        model: AJLPathModel,
        word: BraidWord,
        amplitudes: Mapping[tuple[int, ...], complex],
    ) -> tuple[complex, complex]:
        markov_trace = model.markov_trace(amplitudes)
        return markov_trace, model.trace_closure_jones(word, amplitudes)

    @staticmethod
    def raw_component_variances(
        estimates: tuple[PathAmplitudeEstimate, ...],
    ) -> tuple[float, float]:
        total_weight = math.fsum(estimate.endpoint_weight for estimate in estimates)
        return (
            math.fsum(
                (estimate.endpoint_weight / total_weight) ** 2
                * estimate.real.standard_error**2
                for estimate in estimates
            ),
            math.fsum(
                (estimate.endpoint_weight / total_weight) ** 2
                * estimate.imag.standard_error**2
                for estimate in estimates
            ),
        )

    @staticmethod
    def normalization_factor(model: AJLPathModel, word: BraidWord) -> complex:
        factor = (-(model.A**3)) ** word.writhe
        return complex(factor * model.d ** (model.strands - 1))


@dataclass(frozen=True)
class _PlatClosure:
    writhe: int
    kind: Literal["plat"] = "plat"

    @staticmethod
    def paths(model: AJLPathModel) -> tuple[tuple[int, ...], ...]:
        return (model.plat_path(),)

    def evaluate(
        self,
        model: AJLPathModel,
        word: BraidWord,
        amplitudes: Mapping[tuple[int, ...], complex],
    ) -> tuple[None, complex]:
        del word
        amplitude = amplitudes[model.plat_path()]
        return None, model.plat_closure_jones(amplitude, writhe=self.writhe)

    @staticmethod
    def raw_component_variances(
        estimates: tuple[PathAmplitudeEstimate, ...],
    ) -> tuple[float, float]:
        estimate = estimates[0]
        return (
            estimate.real.standard_error**2,
            estimate.imag.standard_error**2,
        )

    def normalization_factor(self, model: AJLPathModel, word: BraidWord) -> complex:
        del word
        return model.plat_closure_jones(1.0, writhe=self.writhe)


ClosureSpec = _TraceClosure | _PlatClosure


def _normalize_closure(
    closure: object,
    writhe: object,
    *,
    writhe_name: str = "writhe",
) -> ClosureSpec:
    if closure != "trace" and closure != "plat":
        raise ValueError("closure must be 'trace' or 'plat'")
    if closure == "trace":
        if writhe is not None:
            raise ValueError(
                f"{writhe_name} can only be specified for plat closure"
            )
        return _TraceClosure()
    if writhe is None:
        raise ValueError(f"{writhe_name} is required for plat closure")
    if isinstance(writhe, bool):
        raise ValueError(f"{writhe_name} must be an integer")
    try:
        normalized_writhe = int(operator.index(writhe))
    except TypeError:
        raise ValueError(f"{writhe_name} must be an integer") from None
    return _PlatClosure(normalized_writhe)


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
        tasks: tuple[CircuitTask, ...],
        circuit_level: EvaluationCircuitLevel,
    ) -> dict[tuple[tuple[int, ...], str], HadamardComponentEstimate]:
        estimates = {}
        for task in tasks:
            circuit = self.compiler.compile_component(
                word,
                task.path,
                task.part,
                circuit_level=circuit_level,
                measure=False,
            )
            estimates[(task.path, task.part)] = self._statevector_component(circuit)
        return estimates

    def _sampled_estimates(
        self,
        word: BraidWord,
        tasks: tuple[CircuitTask, ...],
        circuit_level: EvaluationCircuitLevel,
        shots: int,
        sampler: BaseSamplerV2,
    ) -> dict[tuple[tuple[int, ...], str], HadamardComponentEstimate]:
        circuits = [
            self.compiler.compile_component(
                word,
                task.path,
                task.part,
                circuit_level=circuit_level,
                measure=True,
            )
            for task in tasks
        ]

        results = sampler.run(circuits, shots=shots).result()
        if len(results) != len(tasks):
            raise ValueError(
                f"sampler returned {len(results)} results for {len(tasks)} circuits"
            )
        return {
            (task.path, task.part): self._sampled_component(pub_result)
            for task, pub_result in zip(tasks, results, strict=True)
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

        braid_word = self.model.as_braid_word(word)
        closure_spec = _normalize_closure(
            closure,
            plat_writhe,
            writhe_name="plat_writhe",
        )
        paths = closure_spec.paths(self.model)
        tasks = circuit_tasks(paths)
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
                tasks,
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
                tasks,
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
        markov_trace, value = closure_spec.evaluate(
            self.model,
            braid_word,
            amplitudes,
        )
        real_error, imag_error = self._propagate_standard_errors(
            braid_word,
            path_estimates,
            closure_spec,
        )
        total_shots = sum(
            estimate.real.shots + estimate.imag.shots
            for estimate in path_estimates
        )
        return JonesEvaluation(
            model=self.model,
            word=braid_word,
            closure=closure_spec.kind,
            plat_writhe=closure_spec.writhe,
            method=method,
            circuit_level=circuit_level,
            config=self.config,
            path_estimates=path_estimates,
            markov_trace=markov_trace,
            value=value,
            real_standard_error=real_error,
            imag_standard_error=imag_error,
            circuit_count=len(tasks),
            shots_per_circuit=shots_per_circuit,
            total_shots=total_shots,
            sampler_name=sampler_name,
        )

    def _propagate_standard_errors(
        self,
        word: BraidWord,
        estimates: tuple[PathAmplitudeEstimate, ...],
        closure: ClosureSpec,
    ) -> tuple[float, float]:
        raw_real_variance, raw_imag_variance = (
            closure.raw_component_variances(estimates)
        )
        closure_factor = closure.normalization_factor(self.model, word)

        real_variance = (
            closure_factor.real**2 * raw_real_variance
            + closure_factor.imag**2 * raw_imag_variance
        )
        imag_variance = (
            closure_factor.imag**2 * raw_real_variance
            + closure_factor.real**2 * raw_imag_variance
        )
        return math.sqrt(real_variance), math.sqrt(imag_variance)


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

    from .problem import JonesProblem

    _normalize_closure(
        closure,
        plat_writhe,
        writhe_name="plat_writhe",
    )
    return JonesProblem(
        word,
        strands=strands,
        k=k,
        closure=closure,
        writhe=plat_writhe,
        config=config,
    ).evaluate(
        method=method,
        circuit_level=circuit_level,
        shots=shots,
        seed=seed,
        sampler=sampler,
    )
