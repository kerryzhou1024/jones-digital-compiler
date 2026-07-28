"""High-level façade for one configured Jones-polynomial evaluation problem."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2

from .compiler import (
    AJLCompiler,
    CompilerLevel,
    HadamardTestCompilation,
)
from .evaluation import (
    AJLJonesEvaluator,
    ClosureSpec,
    ClosureType,
    EvaluationCircuitLevel,
    EvaluationMethod,
    JonesEvaluation,
    _normalize_closure,
    circuit_tasks,
)
from .model import AJLPathModel, BraidWord, HadamardPart
from .policies import CompilerConfig
from .reference import DenseAJLReference

if TYPE_CHECKING:
    from .reporting import CircuitInfo

CircuitLevelSelection = CompilerLevel | Literal["all"]


@dataclass(frozen=True)
class _CircuitProvenance:
    word: BraidWord
    closure: ClosureType
    strands: int
    k: int
    config: CompilerConfig


@dataclass(frozen=True)
class CompiledCircuit:
    """One labeled, displayable Qiskit circuit from a problem."""

    path: tuple[int, ...]
    part: HadamardPart
    circuit_level: CompilerLevel
    measured: bool
    circuit: QuantumCircuit = field(repr=False, compare=False)
    _provenance: _CircuitProvenance | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def path_label(self) -> str:
        return "".join(str(bit) for bit in self.path)

    def __str__(self) -> str:
        return str(self.circuit)

    def __getattr__(self, name: str) -> Any:
        """Forward convenience access to the underlying Qiskit circuit."""

        return getattr(self.circuit, name)

    def display(
        self,
        *,
        title: str | None = None,
        max_lines: int | None = None,
    ) -> None:
        """Display this circuit in a horizontally scrollable notebook panel."""

        try:
            from .notebook import show_scrollable_circuit
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "circuit display requires the package's 'notebook' extra"
            ) from error

        active_title = (
            title
            if title is not None
            else (
                f"path |{self.path_label}> — {self.part} — "
                f"Level {self.circuit_level}"
            )
        )
        show_scrollable_circuit(
            self.circuit,
            active_title,
            max_lines=max_lines,
        )

    def info(self) -> CircuitInfo:
        """Return a fresh immutable provenance and logical-resource report."""

        from .reporting import circuit_info

        return circuit_info(self)


@dataclass(frozen=True, init=False)
class JonesProblem:
    """One immutable braid, closure, AJL root, and compiler configuration."""

    word: BraidWord
    closure: ClosureType
    strands: int
    k: int
    writhe: int | None
    config: CompilerConfig
    model: AJLPathModel = field(repr=False, compare=False)
    _closure_spec: ClosureSpec = field(repr=False, compare=False)

    def __init__(
        self,
        word: BraidWord | str | Sequence[int],
        *,
        closure: ClosureType = "trace",
        strands: int,
        k: int = 5,
        writhe: int | None = None,
        config: CompilerConfig | None = None,
    ) -> None:
        model = AJLPathModel(strands=strands, level=k)
        normalized_word = model.as_braid_word(word)
        closure_spec = _normalize_closure(closure, writhe)
        if closure_spec.kind == "plat":
            model.plat_path()

        active_config = CompilerConfig() if config is None else config
        if not isinstance(active_config, CompilerConfig):
            raise TypeError("config must be a CompilerConfig or None")

        object.__setattr__(self, "word", normalized_word)
        object.__setattr__(self, "closure", closure_spec.kind)
        object.__setattr__(self, "strands", model.strands)
        object.__setattr__(self, "k", model.level)
        object.__setattr__(self, "writhe", closure_spec.writhe)
        object.__setattr__(self, "config", active_config)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "_closure_spec", closure_spec)

    @property
    def A(self) -> complex:
        return self.model.A

    @property
    def d(self) -> float:
        return self.model.d

    @cached_property
    def valid_paths(self) -> tuple[tuple[int, ...], ...]:
        return self.model.valid_paths()

    @cached_property
    def evaluation_paths(self) -> tuple[tuple[int, ...], ...]:
        if self.closure == "trace":
            return self.valid_paths
        return self._closure_spec.paths(self.model)

    @cached_property
    def _evaluator(self) -> AJLJonesEvaluator:
        return AJLJonesEvaluator(self.model, self.config)

    @property
    def _compiler(self) -> AJLCompiler:
        return self._evaluator.compiler

    def _circuit_provenance(self) -> _CircuitProvenance:
        return _CircuitProvenance(
            word=self.word,
            closure=self.closure,
            strands=self.strands,
            k=self.k,
            config=self.config,
        )

    def compile(
        self,
        path: str | Sequence[int],
        part: HadamardPart,
        *,
        measure: bool = False,
    ) -> HadamardTestCompilation:
        """Compile one component into matched Level 1, 2, and 3 circuits."""

        return self._compiler.compile_hadamard_test(
            self.word,
            path,
            part,
            measure=measure,
        )

    def circuit(
        self,
        path: str | Sequence[int] | None = None,
        part: HadamardPart = "real",
        *,
        circuit_level: CompilerLevel = 3,
        measure: bool = True,
    ) -> CompiledCircuit:
        """Compile one component, defaulting to the first valid path."""

        normalized_path = (
            self.valid_paths[0]
            if path is None
            else self.model.coerce_path(path)
        )
        normalized_part = self._compiler.validate_part(part)
        circuit = self._compiler.compile_component(
            self.word,
            normalized_path,
            normalized_part,
            circuit_level=circuit_level,
            measure=measure,
        )
        return CompiledCircuit(
            path=normalized_path,
            part=normalized_part,
            circuit_level=circuit_level,
            measured=measure,
            circuit=circuit,
            _provenance=self._circuit_provenance(),
        )

    def circuits(
        self,
        *,
        path: str | Sequence[int] | None = None,
        part: HadamardPart | None = None,
        circuit_level: CircuitLevelSelection = 3,
        measure: bool = False,
    ) -> Iterator[CompiledCircuit]:
        """Lazily compile a filtered path, component, and level workload."""

        selected_paths = (
            self.evaluation_paths
            if path is None
            else (self.model.coerce_path(path),)
        )
        tasks = circuit_tasks(selected_paths, part=part)

        if circuit_level == "all":
            for task in tasks:
                compilation = self.compile(
                    task.path,
                    task.part,
                    measure=measure,
                )
                selected_circuits: list[tuple[CompilerLevel, QuantumCircuit]] = [
                    (1, compilation.level_1_varphi),
                    (2, compilation.level_2_multicontrolled),
                    (3, compilation.level_3_single_control),
                ]
                if compilation.level_4_clifford_t is not None:
                    selected_circuits.append(
                        (4, compilation.level_4_clifford_t)
                    )
                for level, circuit in selected_circuits:
                    yield CompiledCircuit(
                        path=task.path,
                        part=task.part,
                        circuit_level=level,
                        measured=measure,
                        circuit=circuit,
                        _provenance=self._circuit_provenance(),
                    )
            return

        for task in tasks:
            yield self.circuit(
                task.path,
                task.part,
                circuit_level=circuit_level,
                measure=measure,
            )

    def evaluate(
        self,
        *,
        method: EvaluationMethod = "statevector",
        circuit_level: EvaluationCircuitLevel = 3,
        shots: int | None = None,
        seed: int | None = None,
        path_seed: int | None = None,
        sampler: BaseSamplerV2 | None = None,
    ) -> JonesEvaluation:
        """Evaluate the Jones value at the AJL root selected by ``k``."""

        return self._evaluator.evaluate(
            self.word,
            closure=self.closure,
            plat_writhe=self.writhe,
            method=method,
            circuit_level=circuit_level,
            shots=shots,
            seed=seed,
            path_seed=path_seed,
            sampler=sampler,
        )

    def reference_value(self) -> complex:
        """Return the dense-oracle value for a suitably small path space."""

        reference = DenseAJLReference(self.model)
        matrix = reference.compile_matrix(self.word)
        selected_paths = (
            reference.paths
            if self.closure == "trace"
            else self.evaluation_paths
        )
        amplitudes = {
            path: complex(
                matrix[reference.path_index[path], reference.path_index[path]]
            )
            for path in selected_paths
        }
        return self._closure_spec.evaluate(
            self.model,
            self.word,
            amplitudes,
        )[1]
