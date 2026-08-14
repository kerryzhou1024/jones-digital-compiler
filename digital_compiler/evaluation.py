"""One-call circuit-based evaluation of AJL trace and plat closures."""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Literal

import numpy as np
from qiskit.primitives import BaseSamplerV2, StatevectorSampler
from qiskit.quantum_info import Statevector

from .compiler import AJLCompiler
from .model import AJLPathModel, BraidWord, HadamardPart
from .policies import CompilerConfig

EvaluationMethod = Literal["statevector", "shots"]
EvaluationCircuitLevel = Literal[2, 3, 4]
ClosureType = Literal["trace", "plat"]
PathSampling = Literal["enumerated", "ajl_weighted", "fixed_plat"]
DEFAULT_SHOTS = 4096
AJL_SUCCESS_PROBABILITY = 0.75


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
class TraceComponentSample:
    """One AJL-sampled real or imaginary normalized-trace estimate."""

    part: HadamardPart
    expectation: float
    standard_error: float
    shots: int
    counts: Mapping[str, int]
    path_counts: Mapping[tuple[int, ...], int]

    def __post_init__(self) -> None:
        if self.part not in {"real", "imag"}:
            raise ValueError("part must be 'real' or 'imag'")
        if self.shots <= 0:
            raise ValueError("shots must be a positive integer")

        counts = {str(bitstring): int(count) for bitstring, count in self.counts.items()}
        unexpected = set(counts) - {"0", "1"}
        if unexpected:
            raise ValueError(
                "trace component counts must contain only '0' and '1', found "
                f"{sorted(unexpected)}"
            )
        if any(count < 0 for count in counts.values()) or sum(counts.values()) != self.shots:
            raise ValueError("trace component counts must sum to shots")

        path_counts = {
            tuple(path): int(count) for path, count in self.path_counts.items()
        }
        if (
            any(count <= 0 for count in path_counts.values())
            or sum(path_counts.values()) != self.shots
        ):
            raise ValueError("trace component path counts must be positive and sum to shots")

        object.__setattr__(self, "counts", MappingProxyType(counts))
        object.__setattr__(self, "path_counts", MappingProxyType(path_counts))

    @property
    def circuit_count(self) -> int:
        """Return the number of unique path circuits used for this component."""

        return len(self.path_counts)


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
class _RunOptions:
    """Validated shot, confidence, seeding, and sampler choices for one call."""

    method: EvaluationMethod
    circuit_level: EvaluationCircuitLevel
    component_shots: int | None
    success_probability: float | None
    path_seed: int | None
    sampler: BaseSamplerV2 | None

    @classmethod
    def build(
        cls,
        *,
        closure: ClosureSpec,
        closure_factor: complex,
        method: EvaluationMethod,
        circuit_level: EvaluationCircuitLevel,
        shots: int | None,
        target_additive_error: float | None,
        success_probability: float | None,
        seed: int | None,
        path_seed: int | None,
        sampler: BaseSamplerV2 | None,
    ) -> _RunOptions:
        if path_seed is not None and (method != "shots" or closure.kind != "trace"):
            raise ValueError(
                "path_seed can only be specified for shot-based trace evaluation"
            )
        if method == "statevector":
            for value, name in (
                (shots, "shots"),
                (target_additive_error, "target_additive_error"),
                (success_probability, "success_probability"),
                (seed, "seed"),
                (sampler, "sampler"),
            ):
                if value is not None:
                    raise ValueError(f"{name} can only be specified when method='shots'")
            return cls(method, circuit_level, None, None, None, None)

        if shots is not None and target_additive_error is not None:
            raise ValueError("shots and target_additive_error are mutually exclusive")
        confidence = _success_probability(success_probability)
        if target_additive_error is None:
            component_shots = _positive_shots(DEFAULT_SHOTS if shots is None else shots)
        else:
            component_shots = _shots_for_additive_error(
                target_additive_error,
                closure_factor=closure_factor,
                success_probability=confidence,
            )

        if sampler is None:
            active_sampler: BaseSamplerV2 = StatevectorSampler(seed=seed)
        elif seed is not None:
            raise ValueError(
                "seed configures only the default StatevectorSampler; "
                "configure a custom sampler directly"
            )
        elif not isinstance(sampler, BaseSamplerV2):
            raise TypeError("sampler must implement Qiskit's BaseSamplerV2")
        else:
            active_sampler = sampler

        return cls(
            method=method,
            circuit_level=circuit_level,
            component_shots=component_shots,
            success_probability=confidence,
            # Only a sampled trace draws paths, so only it seeds them.
            path_seed=(
                _nonnegative_path_seed(seed if path_seed is None else path_seed)
                if closure.kind == "trace"
                else None
            ),
            sampler=active_sampler,
        )

    @property
    def sampler_name(self) -> str | None:
        return None if self.sampler is None else type(self.sampler).__name__


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
    path_sampling: PathSampling
    path_estimates: tuple[PathAmplitudeEstimate, ...] | None
    trace_samples: tuple[TraceComponentSample, TraceComponentSample] | None
    markov_trace: complex | None
    value: complex
    real_standard_error: float
    imag_standard_error: float
    circuit_count: int
    shots_per_circuit: int | None
    shots_per_component: int | None
    total_shots: int
    sampler_name: str | None
    markov_trace_additive_error_bound: float | None
    value_additive_error_bound: float | None
    synthesis_error_budget_per_circuit: float | None
    value_synthesis_error_bound: float | None
    ajl_success_probability: float | None

    @property
    def strands(self) -> int:
        return self.model.strands

    @property
    def k(self) -> int:
        return self.model.level

    @property
    def writhe(self) -> int:
        """Return the oriented diagram writhe used for normalization."""

        if self.closure == "trace":
            return self.word.writhe
        assert self.plat_writhe is not None
        return self.plat_writhe

    @property
    def path_amplitudes(self) -> Mapping[tuple[int, ...], complex] | None:
        if self.path_estimates is None:
            return None
        return MappingProxyType(
            {estimate.path: estimate.amplitude for estimate in self.path_estimates}
        )

    @property
    def plat_amplitude(self) -> complex | None:
        """Return the single plat matrix element, or ``None`` for trace closure."""

        if (
            self.closure != "plat"
            or self.path_estimates is None
            or len(self.path_estimates) != 1
        ):
            return None
        return self.path_estimates[0].amplitude


class AJLJonesEvaluator:
    """Evaluate AJL closures by executing compiled Hadamard circuits."""

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
        control = circuit.find_bit(circuit.qubits[0]).registers[0][0]
        if control.name != "ctrl":
            raise ValueError("a Hadamard-test circuit must start with its control")
        state = Statevector.from_instruction(circuit)
        probabilities = state.probabilities(qargs=[0])
        expectation = float(probabilities[0] - probabilities[1])
        return HadamardComponentEstimate(
            expectation=expectation,
            standard_error=0.0,
        )

    @staticmethod
    def _sampled_component(
        pub_result,
        *,
        expected_shots: int | None = None,
    ) -> HadamardComponentEstimate:
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
        if expected_shots is not None and shots != expected_shots:
            raise ValueError(
                f"sampler returned {shots} shots but {expected_shots} were requested"
            )
        expectation = float((counts.get("0", 0) - counts.get("1", 0)) / shots)
        standard_error = math.sqrt(max(0.0, 1.0 - expectation**2) / shots)
        return HadamardComponentEstimate(
            expectation=expectation,
            standard_error=standard_error,
            shots=shots,
            counts=counts,
        )

    @staticmethod
    def _effective_synthesis_budget(circuit) -> float:
        metadata = circuit.metadata or {}
        synthesis = metadata.get("clifford_t_synthesis")
        if not isinstance(synthesis, Mapping):
            return 0.0
        if int(synthesis.get("arbitrary_rotation_count", 0)) == 0:
            return 0.0
        return float(synthesis["synthesis_error_budget"])

    def _statevector_estimates(
        self,
        word: BraidWord,
        tasks: tuple[CircuitTask, ...],
        circuit_level: EvaluationCircuitLevel,
    ) -> tuple[
        dict[tuple[tuple[int, ...], str], HadamardComponentEstimate],
        float,
    ]:
        estimates = {}
        effective_synthesis_budget = 0.0
        for task in tasks:
            circuit = self.compiler.compile_component(
                word,
                task.path,
                task.part,
                circuit_level=circuit_level,
                measure=False,
            )
            effective_synthesis_budget = max(
                effective_synthesis_budget,
                self._effective_synthesis_budget(circuit),
            )
            estimates[(task.path, task.part)] = self._statevector_component(circuit)
        return estimates, effective_synthesis_budget

    def _sampled_fixed_path_estimates(
        self,
        word: BraidWord,
        tasks: tuple[CircuitTask, ...],
        circuit_level: EvaluationCircuitLevel,
        shots: int,
        sampler: BaseSamplerV2,
    ) -> tuple[
        dict[tuple[tuple[int, ...], str], HadamardComponentEstimate],
        float,
    ]:
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
        return (
            {
                (task.path, task.part): self._sampled_component(pub_result)
                for task, pub_result in zip(tasks, results, strict=True)
            },
            max(
                (
                    self._effective_synthesis_budget(circuit)
                    for circuit in circuits
                ),
                default=0.0,
            ),
        )

    def _sampled_trace_components(
        self,
        word: BraidWord,
        circuit_level: EvaluationCircuitLevel,
        shots: int,
        sampler: BaseSamplerV2,
        path_seed: int | None,
    ) -> tuple[TraceComponentSample, TraceComponentSample, float]:
        seed_sequence = np.random.SeedSequence(path_seed)
        real_seed, imag_seed = seed_sequence.spawn(2)
        sampled_paths = {
            "real": self.model.sample_paths(
                shots,
                rng=np.random.default_rng(real_seed),
            ),
            "imag": self.model.sample_paths(
                shots,
                rng=np.random.default_rng(imag_seed),
            ),
        }
        path_counts = {
            part: dict(sorted(Counter(paths).items()))
            for part, paths in sampled_paths.items()
        }

        specifications = [
            (part, path, multiplicity)
            for part in ("real", "imag")
            for path, multiplicity in path_counts[part].items()
        ]
        circuits = [
            self.compiler.compile_component(
                word,
                path,
                part,
                circuit_level=circuit_level,
                measure=True,
            )
            for part, path, _multiplicity in specifications
        ]
        pubs = [
            (circuit, None, multiplicity)
            for circuit, (_part, _path, multiplicity) in zip(
                circuits,
                specifications,
                strict=True,
            )
        ]
        results = sampler.run(pubs).result()
        if len(results) != len(specifications):
            raise ValueError(
                f"sampler returned {len(results)} results for "
                f"{len(specifications)} sampled trace circuits"
            )

        aggregate_counts = {
            "real": Counter(),
            "imag": Counter(),
        }
        for (part, _path, multiplicity), pub_result in zip(
            specifications,
            results,
            strict=True,
        ):
            component = self._sampled_component(
                pub_result,
                expected_shots=multiplicity,
            )
            aggregate_counts[part].update(component.counts)

        estimates = []
        for part in ("real", "imag"):
            counts = dict(aggregate_counts[part])
            returned_shots = sum(counts.values())
            if returned_shots != shots:
                raise ValueError(
                    f"sampler returned {returned_shots} aggregate {part} shots "
                    f"but {shots} were requested"
                )
            expectation = float(
                (counts.get("0", 0) - counts.get("1", 0)) / returned_shots
            )
            standard_error = math.sqrt(
                max(0.0, 1.0 - expectation**2) / returned_shots
            )
            estimates.append(
                TraceComponentSample(
                    part=part,
                    expectation=expectation,
                    standard_error=standard_error,
                    shots=returned_shots,
                    counts=counts,
                    path_counts=path_counts[part],
                )
            )
        return (
            estimates[0],
            estimates[1],
            max(
                (
                    self._effective_synthesis_budget(circuit)
                    for circuit in circuits
                ),
                default=0.0,
            ),
        )

    def evaluate(
        self,
        word: BraidWord | str | Sequence[int],
        *,
        closure: ClosureType = "trace",
        plat_writhe: int | None = None,
        method: EvaluationMethod = "statevector",
        circuit_level: EvaluationCircuitLevel = 3,
        shots: int | None = None,
        target_additive_error: float | None = None,
        success_probability: float | None = None,
        seed: int | None = None,
        path_seed: int | None = None,
        sampler: BaseSamplerV2 | None = None,
    ) -> JonesEvaluation:
        """Evaluate the numerical Jones value of a braid closure."""

        if method not in {"statevector", "shots"}:
            raise ValueError("method must be 'statevector' or 'shots'")
        if circuit_level not in {2, 3, 4}:
            raise ValueError("circuit_level must be 2, 3, or 4")
        if circuit_level == 4 and self.config.level4 is None:
            raise ValueError(
                "circuit_level=4 requires "
                "CompilerConfig(level4=CliffordTConfig(...))"
            )

        braid_word = self.model.as_braid_word(word)
        closure_spec = _normalize_closure(
            closure,
            plat_writhe,
            writhe_name="plat_writhe",
        )
        closure_factor = closure_spec.normalization_factor(self.model, braid_word)
        options = _RunOptions.build(
            closure=closure_spec,
            closure_factor=closure_factor,
            method=method,
            circuit_level=circuit_level,
            shots=shots,
            target_additive_error=target_additive_error,
            success_probability=success_probability,
            seed=seed,
            path_seed=path_seed,
            sampler=sampler,
        )

        if options.method == "shots" and closure_spec.kind == "trace":
            return self._sampled_trace_evaluation(
                braid_word,
                closure_spec,
                closure_factor,
                options,
            )
        return self._fixed_path_evaluation(
            braid_word,
            closure_spec,
            closure_factor,
            options,
        )

    def _shared_evaluation_fields(
        self,
        word: BraidWord,
        closure: ClosureSpec,
        closure_factor: complex,
        options: _RunOptions,
        synthesis_budget: float,
    ) -> dict[str, object]:
        """Return the fields every evaluation reports identically.

        Level-4 synthesis replaces each component circuit ``C`` by ``C_tilde``
        with ``norm(C - C_tilde) <= budget``, so each measured component moves
        by at most ``2 * budget``, a complex amplitude by ``2 * sqrt(2) *
        budget``, and the closure-normalized Jones value by that times
        ``abs(closure_factor)``.  Averaging over paths does not enlarge it.
        """

        level_4 = self.config.level4 if options.circuit_level == 4 else None
        return {
            "model": self.model,
            "word": word,
            "closure": closure.kind,
            "plat_writhe": closure.writhe,
            "method": options.method,
            "circuit_level": options.circuit_level,
            "config": self.config,
            "sampler_name": options.sampler_name,
            "ajl_success_probability": options.success_probability,
            "synthesis_error_budget_per_circuit": (
                None if level_4 is None else level_4.synthesis_error_budget
            ),
            "value_synthesis_error_bound": (
                None
                if level_4 is None
                else 2.0 * math.sqrt(2.0) * abs(closure_factor) * synthesis_budget
            ),
        }

    def _sampled_trace_evaluation(
        self,
        word: BraidWord,
        closure: ClosureSpec,
        closure_factor: complex,
        options: _RunOptions,
    ) -> JonesEvaluation:
        """Estimate a trace closure from AJL endpoint-weighted path samples."""

        shots = options.component_shots
        assert shots is not None and options.sampler is not None
        assert options.success_probability is not None
        real_sample, imag_sample, synthesis_budget = self._sampled_trace_components(
            word,
            options.circuit_level,
            shots,
            options.sampler,
            options.path_seed,
        )
        markov_trace = complex(real_sample.expectation, imag_sample.expectation)
        real_error, imag_error = self._propagate_component_standard_errors(
            word,
            closure,
            real_sample.standard_error,
            imag_sample.standard_error,
        )
        markov_error_bound = _complex_additive_error_bound(
            shots,
            options.success_probability,
        )
        return JonesEvaluation(
            **self._shared_evaluation_fields(
                word,
                closure,
                closure_factor,
                options,
                synthesis_budget,
            ),
            path_sampling="ajl_weighted",
            path_estimates=None,
            trace_samples=(real_sample, imag_sample),
            markov_trace=markov_trace,
            value=complex(closure_factor * markov_trace),
            real_standard_error=real_error,
            imag_standard_error=imag_error,
            circuit_count=real_sample.circuit_count + imag_sample.circuit_count,
            shots_per_circuit=None,
            shots_per_component=shots,
            total_shots=2 * shots,
            markov_trace_additive_error_bound=markov_error_bound,
            value_additive_error_bound=abs(closure_factor) * markov_error_bound,
        )

    def _fixed_path_evaluation(
        self,
        word: BraidWord,
        closure: ClosureSpec,
        closure_factor: complex,
        options: _RunOptions,
    ) -> JonesEvaluation:
        """Estimate a closure from its complete, fixed set of path circuits."""

        paths = closure.paths(self.model)
        tasks = circuit_tasks(paths)
        if options.method == "statevector":
            components, synthesis_budget = self._statevector_estimates(
                word,
                tasks,
                options.circuit_level,
            )
        else:
            assert options.component_shots is not None and options.sampler is not None
            components, synthesis_budget = self._sampled_fixed_path_estimates(
                word,
                tasks,
                options.circuit_level,
                options.component_shots,
                options.sampler,
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
        markov_trace, value = closure.evaluate(
            self.model,
            word,
            {estimate.path: estimate.amplitude for estimate in path_estimates},
        )
        real_error, imag_error = self._propagate_standard_errors(
            word,
            path_estimates,
            closure,
        )
        value_additive_error_bound = None
        if options.component_shots is not None:
            assert options.success_probability is not None
            value_additive_error_bound = abs(
                closure_factor
            ) * _complex_additive_error_bound(
                options.component_shots,
                options.success_probability,
            )
        return JonesEvaluation(
            **self._shared_evaluation_fields(
                word,
                closure,
                closure_factor,
                options,
                synthesis_budget,
            ),
            path_sampling="enumerated" if closure.kind == "trace" else "fixed_plat",
            path_estimates=path_estimates,
            trace_samples=None,
            markov_trace=markov_trace,
            value=value,
            real_standard_error=real_error,
            imag_standard_error=imag_error,
            circuit_count=len(tasks),
            shots_per_circuit=options.component_shots,
            shots_per_component=options.component_shots,
            total_shots=sum(
                estimate.real.shots + estimate.imag.shots
                for estimate in path_estimates
            ),
            markov_trace_additive_error_bound=None,
            value_additive_error_bound=value_additive_error_bound,
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
        return self._propagate_component_standard_errors(
            word,
            closure,
            math.sqrt(raw_real_variance),
            math.sqrt(raw_imag_variance),
        )

    def _propagate_component_standard_errors(
        self,
        word: BraidWord,
        closure: ClosureSpec,
        raw_real_error: float,
        raw_imag_error: float,
    ) -> tuple[float, float]:
        closure_factor = closure.normalization_factor(self.model, word)

        real_variance = (
            closure_factor.real**2 * raw_real_error**2
            + closure_factor.imag**2 * raw_imag_error**2
        )
        imag_variance = (
            closure_factor.imag**2 * raw_real_error**2
            + closure_factor.real**2 * raw_imag_error**2
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


def _success_probability(value: object | None) -> float:
    if value is None:
        return AJL_SUCCESS_PROBABILITY
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("success_probability must be a finite number between 0 and 1")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("success_probability must be a finite number between 0 and 1")
    return probability


def _confidence_log_factor(success_probability: float) -> float:
    return math.log(4.0) - math.log1p(-success_probability)


def _complex_additive_error_bound(
    shots: int,
    success_probability: float,
) -> float:
    return math.sqrt(
        4.0 * _confidence_log_factor(success_probability) / shots
    )


def _shots_for_additive_error(
    target_additive_error: object,
    *,
    closure_factor: complex,
    success_probability: float,
) -> int:
    if isinstance(target_additive_error, bool) or not isinstance(
        target_additive_error,
        Real,
    ):
        raise ValueError("target_additive_error must be a finite positive number")
    target = float(target_additive_error)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_additive_error must be a finite positive number")

    try:
        required_shots = (
            4.0
            * abs(closure_factor) ** 2
            * _confidence_log_factor(success_probability)
            / target**2
        )
    except (OverflowError, ZeroDivisionError):
        raise ValueError(
            "target_additive_error requires an unrepresentable shot count"
        ) from None
    if not math.isfinite(required_shots):
        raise ValueError(
            "target_additive_error requires an unrepresentable shot count"
        )
    return max(1, math.ceil(required_shots))


def _nonnegative_path_seed(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("path_seed must be a non-negative integer or None")
    try:
        seed = int(operator.index(value))
    except TypeError:
        raise ValueError("path_seed must be a non-negative integer or None") from None
    if seed < 0:
        raise ValueError("path_seed must be a non-negative integer or None")
    return seed

