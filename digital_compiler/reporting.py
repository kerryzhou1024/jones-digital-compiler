"""Small, deterministic compiler summaries for notebooks and scripts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Gate

from .compiler import CompilerLevel, HadamardTestCompilation
from .fault_tolerance import (
    CLIFFORD_GATE_NAMES,
    T_GATE_NAMES,
    t_layer_widths,
)

if TYPE_CHECKING:
    from .problem import CompiledCircuit


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _text_table(
    headers: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> str:
    text_rows = [tuple(str(value) for value in row) for row in rows]
    widths = [
        max(
            len(header),
            *(len(row[index]) for row in text_rows),
        )
        for index, header in enumerate(headers)
    ]
    header = "  ".join(
        f"{value:<{width}}" for value, width in zip(headers, widths, strict=True)
    )
    rule = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(
            f"{value:<{width}}"
            for value, width in zip(row, widths, strict=True)
        )
        for row in text_rows
    ]
    return "\n".join((header, rule, *body))


def _html_table(
    headers: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> str:
    header = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<table style="border-collapse:collapse">'
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
    )


@dataclass(frozen=True)
class CircuitInfo:
    """Immutable provenance and logical-resource report for one circuit."""

    word: str
    signed_generators: tuple[int, ...]
    closure: str
    strands: int
    k: int
    path: tuple[int, ...]
    part: str
    compiler_level: CompilerLevel
    measured: bool
    logical_qubits: int
    classical_bits: int
    quantum_registers: tuple[tuple[str, int], ...]
    classical_registers: tuple[tuple[str, int], ...]
    quantum_gate_count: int
    quantum_gate_depth: int
    circuit_depth: int
    measurement_count: int
    non_gate_operations: Mapping[str, int]
    gate_families: Mapping[str, int]
    exact_gate_stats: Mapping[str, Mapping[str, int]]
    level_4_resources: Mapping[str, object] | None
    compiler_policies: Mapping[str, object]
    scope_notes: tuple[str, ...]

    @property
    def path_label(self) -> str:
        return "".join(str(bit) for bit in self.path)

    def as_dict(self) -> dict[str, object]:
        return {
            "identity": {
                "word": self.word,
                "signed_generators": self.signed_generators,
                "closure": self.closure,
                "strands": self.strands,
                "k": self.k,
                "path": self.path,
                "path_label": self.path_label,
                "part": self.part,
                "compiler_level": self.compiler_level,
                "measured": self.measured,
            },
            "resources": {
                "logical_qubits": self.logical_qubits,
                "classical_bits": self.classical_bits,
                "quantum_registers": self.quantum_registers,
                "classical_registers": self.classical_registers,
                "quantum_gate_count": self.quantum_gate_count,
                "quantum_gate_depth": self.quantum_gate_depth,
                "circuit_depth": self.circuit_depth,
                "measurement_count": self.measurement_count,
                "non_gate_operations": _thaw(self.non_gate_operations),
            },
            "gates": {
                "families": _thaw(self.gate_families),
                "exact": _thaw(self.exact_gate_stats),
                "clifford_t": _thaw(self.level_4_resources),
            },
            "compiler": _thaw(self.compiler_policies),
            "scope_notes": self.scope_notes,
        }

    def _summary_rows(self) -> list[tuple[object, ...]]:
        rows = [
            ("word", self.word),
            ("closure", self.closure),
            ("path", f"|{self.path_label}>"),
            ("component", self.part),
            ("compiler level", self.compiler_level),
            ("strands / k", f"{self.strands} / {self.k}"),
            ("logical qubits", self.logical_qubits),
            ("classical bits", self.classical_bits),
            ("quantum gates", self.quantum_gate_count),
            ("quantum gate depth", self.quantum_gate_depth),
            ("Qiskit circuit depth", self.circuit_depth),
            ("measurements", self.measurement_count),
        ]
        if self.level_4_resources is not None:
            rows.extend(
                (
                    ("T count", self.level_4_resources["t_count"]),
                    ("T depth", self.level_4_resources["t_depth"]),
                    ("CX count", self.level_4_resources["cx_count"]),
                    ("CX depth", self.level_4_resources["cx_depth"]),
                )
            )
        return rows

    def _policy_rows(self) -> list[tuple[object, ...]]:
        rows: list[tuple[object, ...]] = []
        for name, value in self.compiler_policies.items():
            if name == "configuration" or value is None:
                continue
            label = name.replace("_", " ")
            if isinstance(value, Mapping):
                rows.extend(
                    (
                        f"{label}.{str(child).replace('_', ' ')}",
                        child_value,
                    )
                    for child, child_value in value.items()
                )
            else:
                rows.append((label, value))
        return rows

    def __str__(self) -> str:
        family_rows = [
            (family, count)
            for family, count in self.gate_families.items()
        ]
        policy_rows = self._policy_rows()
        sections = [
            "Circuit info",
            _text_table(("field", "value"), self._summary_rows()),
            "\nGate families",
            _text_table(("family", "count"), family_rows),
        ]
        if policy_rows:
            sections.extend(
                (
                    "\nCompiler policy",
                    _text_table(("field", "value"), policy_rows),
                )
            )
        sections.extend(("\nScope", *(f"- {note}" for note in self.scope_notes)))
        return "\n".join(sections)

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        del cycle
        printer.text(str(self))

    def _repr_html_(self) -> str:
        family_rows = [
            (family, count)
            for family, count in self.gate_families.items()
        ]
        policy_rows = self._policy_rows()
        policies = (
            "<h4>Compiler policy</h4>"
            + _html_table(("field", "value"), policy_rows)
            if policy_rows
            else ""
        )
        notes = "".join(f"<li>{escape(note)}</li>" for note in self.scope_notes)
        return (
            '<div class="digital-compiler-circuit-info">'
            "<h3>Circuit info</h3>"
            + _html_table(("field", "value"), self._summary_rows())
            + "<h4>Gate families</h4>"
            + _html_table(("family", "count"), family_rows)
            + policies
            + f"<h4>Scope</h4><ul>{notes}</ul></div>"
        )


@dataclass(frozen=True)
class CircuitComparison:
    """Immutable side-by-side comparison of labeled circuit reports."""

    reports: tuple[tuple[str, CircuitInfo], ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "circuits": {
                label: report.as_dict() for label, report in self.reports
            },
            "warnings": self.warnings,
        }

    def _comparison_rows(self) -> list[tuple[object, ...]]:
        labels = tuple(label for label, _ in self.reports)
        fields = (
            ("word", lambda report: report.word),
            ("closure", lambda report: report.closure),
            ("path", lambda report: f"|{report.path_label}>"),
            ("component", lambda report: report.part),
            ("compiler level", lambda report: report.compiler_level),
            ("logical qubits", lambda report: report.logical_qubits),
            ("classical bits", lambda report: report.classical_bits),
            ("quantum gates", lambda report: report.quantum_gate_count),
            ("quantum gate depth", lambda report: report.quantum_gate_depth),
            ("Qiskit circuit depth", lambda report: report.circuit_depth),
            ("measurements", lambda report: report.measurement_count),
        )
        rows = [
            (name, *(getter(report) for _, report in self.reports))
            for name, getter in fields
        ]
        families = sorted(
            {
                family
                for _, report in self.reports
                for family in report.gate_families
            }
        )
        rows.extend(
            (
                f"gate: {family}",
                *(
                    report.gate_families.get(family, 0)
                    for _, report in self.reports
                ),
            )
            for family in families
        )
        if len(labels) != len(set(labels)):
            raise ValueError("comparison labels must be unique")
        return rows

    def __str__(self) -> str:
        headers = ("resource", *(label for label, _ in self.reports))
        sections = [
            "Circuit comparison",
            _text_table(headers, self._comparison_rows()),
        ]
        if self.warnings:
            sections.extend(
                ("\nNotes", *(f"- {warning}" for warning in self.warnings))
            )
        return "\n".join(sections)

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        del cycle
        printer.text(str(self))

    def _repr_html_(self) -> str:
        headers = ("resource", *(label for label, _ in self.reports))
        notes = "".join(f"<li>{escape(note)}</li>" for note in self.warnings)
        return (
            '<div class="digital-compiler-circuit-comparison">'
            "<h3>Circuit comparison</h3>"
            + _html_table(headers, self._comparison_rows())
            + (f"<h4>Notes</h4><ul>{notes}</ul>" if notes else "")
            + "</div>"
        )


_ELEMENTARY_FAMILIES = {
    "x": "X",
    "h": "H",
    "s": "S/Sdg",
    "sdg": "S/Sdg",
    "t": "T/Tdg",
    "tdg": "T/Tdg",
    "rx": "Rx",
    "ry": "Ry",
    "rz": "Rz",
    "p": "Phase",
    "cx": "CNOT",
    "crx": "CRx",
    "cry": "CRy",
    "crz": "CRz",
}


def _level_1_gate_family(operation: Gate) -> str:
    if operation.name.startswith("c_varphi_sigma_"):
        return "controlled varphi"
    if operation.name.startswith("level_1_adder_"):
        return "prefix Adder"
    return _ELEMENTARY_FAMILIES.get(operation.name, operation.name)


def _level_2_gate_family(operation: Gate, *, strict: bool) -> str:
    name = operation.name
    if name == "cx":
        return "CNOT"
    if isinstance(operation, ControlledGate):
        base = operation.base_gate.name
        controls = operation.num_ctrl_qubits
        if base == "x" and controls >= 2:
            return f"MCX[{controls}]"
        if base == "p":
            return f"MCPhase[{controls}]"
        if base in {"rx", "ry", "rz"}:
            return (
                f"C{base[1:].upper()}"
                if controls == 1
                else f"MC{base[1:].upper()}[{controls}]"
            )
        if strict:
            raise AssertionError(f"unexpected Level-2 controlled gate {name!r}")
    family = _ELEMENTARY_FAMILIES.get(name)
    if family is not None:
        return family
    if strict:
        raise AssertionError(f"unexpected Level-2 gate {name!r}")
    return name


def _level_3_gate_family(operation: Gate, *, strict: bool) -> str:
    family = _ELEMENTARY_FAMILIES.get(operation.name)
    if family is not None:
        return family
    if strict:
        raise AssertionError(f"unexpected Level-3 gate {operation.name!r}")
    return operation.name


def _level_4_gate_family(operation: Gate, *, strict: bool) -> str:
    if operation.name in CLIFFORD_GATE_NAMES:
        return "Clifford"
    if operation.name in T_GATE_NAMES:
        return "T"
    if strict:
        raise AssertionError(f"unexpected Level-4 gate {operation.name!r}")
    return operation.name


def _gate_family(
    operation: Gate,
    compiler_level: CompilerLevel,
    *,
    strict: bool = False,
) -> str:
    if compiler_level == 1:
        return _level_1_gate_family(operation)
    if compiler_level == 2:
        return _level_2_gate_family(operation, strict=strict)
    if compiler_level == 3:
        return _level_3_gate_family(operation, strict=strict)
    return _level_4_gate_family(operation, strict=strict)


def _exact_gate_stats(
    circuit: QuantumCircuit,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    counts: dict[str, int] = {}
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, Gate):
            counts[operation.name] = counts.get(operation.name, 0) + 1

    total_depth = circuit.depth(
        lambda instruction: isinstance(instruction.operation, Gate)
    )
    by_gate = {
        name: {
            "count": count,
            "depth": circuit.depth(
                lambda instruction, name=name: (
                    isinstance(instruction.operation, Gate)
                    and instruction.operation.name == name
                )
            ),
        }
        for name, count in sorted(counts.items())
    }
    return sum(counts.values()), total_depth, by_gate


def _gate_family_counts(
    circuit: QuantumCircuit,
    compiler_level: CompilerLevel,
) -> dict[str, int]:
    families: dict[str, int] = (
        {"Clifford": 0, "T": 0} if compiler_level == 4 else {}
    )
    for instruction in circuit.data:
        operation = instruction.operation
        if not isinstance(operation, Gate):
            continue
        family = _gate_family(
            operation,
            compiler_level,
            strict=compiler_level == 4,
        )
        families[family] = families.get(family, 0) + 1
    return dict(sorted(families.items()))


def _non_gate_operation_counts(circuit: QuantumCircuit) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, Gate):
            continue
        counts[operation.name] = counts.get(operation.name, 0) + 1
    return dict(sorted(counts.items()))


def _compiler_policy_summary(
    circuit: QuantumCircuit,
    fallback_configuration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = circuit.metadata or {}
    return {
        "height_strategy": metadata.get("height_strategy"),
        "height_encoding": metadata.get("height_encoding"),
        "generator_scheduling": metadata.get("generator_scheduling"),
        "control_distribution": metadata.get("control_distribution"),
        "prefix_height_strategy": metadata.get("prefix_height_strategy"),
        "final_height_strategy": metadata.get("final_height_strategy"),
        "prefix_height_loads": metadata.get("prefix_height_loads"),
        "prefix_height_moves": metadata.get("prefix_height_moves"),
        "prefix_height_unloads": metadata.get("prefix_height_unloads"),
        "prefix_height_path_steps": metadata.get("prefix_height_path_steps"),
        "generator_layers": metadata.get("generator_layers"),
        "parallel_lanes": metadata.get("parallel_lanes"),
        "active_parallel_width": metadata.get("active_parallel_width"),
        "lowering_policies": metadata.get("lowering_policies"),
        "clifford_t_synthesis": metadata.get("clifford_t_synthesis"),
        "configuration": metadata.get(
            "compiler_config",
            fallback_configuration,
        ),
    }


def circuit_info(compiled: CompiledCircuit) -> CircuitInfo:
    """Build a fresh immutable report for one compiled circuit."""

    circuit = compiled.circuit
    total_count, quantum_depth, exact_stats = _exact_gate_stats(circuit)
    non_gate_operations = _non_gate_operation_counts(circuit)
    provenance = compiled._provenance
    if provenance is None:
        word = "unknown"
        signed_generators: tuple[int, ...] = ()
        closure = "unknown"
        strands = len(compiled.path)
        k = 0
        fallback_configuration = None
    else:
        word = str(provenance.word)
        signed_generators = provenance.word.signed_indices()
        # A bare compilation has no closure or AJL root; those are evaluation
        # concepts supplied by JonesProblem.
        closure = "n/a" if provenance.closure is None else provenance.closure
        strands = provenance.strands
        k = 0 if provenance.k is None else provenance.k
        fallback_configuration = provenance.config.metadata()

    scope_notes = [
        "Single compiler-level logical circuit; not the complete closure workload.",
        "Before backend transpilation and physical or surface-code mapping.",
    ]
    if (circuit.metadata or {}).get("final_height_strategy") == "retain":
        scope_notes.append(
            "Final height selectors are retained and may remain entangled; "
            "treat this as a terminal basis-path Hadamard-test component."
        )
    if compiled.circuit_level == 3:
        scope_notes.append(
            "Arbitrary rotations remain unsynthesized; this is not a final "
            "Clifford+T estimate."
        )
    if compiled.circuit_level == 4:
        scope_notes.append(
            "Approximate logical Clifford+T circuit; excludes physical "
            "mapping, routing, surface-code cycles, and factories."
        )

    level_4_resources = None
    if compiled.circuit_level == 4:
        names = [
            instruction.operation.name
            for instruction in circuit.data
            if isinstance(instruction.operation, Gate)
        ]
        layers = t_layer_widths(circuit)
        synthesis = (circuit.metadata or {}).get(
            "clifford_t_synthesis",
            {},
        )
        level_4_resources = {
            "clifford_count": sum(
                name in CLIFFORD_GATE_NAMES for name in names
            ),
            "clifford_depth": circuit.depth(
                lambda instruction: (
                    isinstance(instruction.operation, Gate)
                    and instruction.operation.name in CLIFFORD_GATE_NAMES
                )
            ),
            "cx_count": sum(name == "cx" for name in names),
            "cx_depth": circuit.depth(
                lambda instruction: instruction.operation.name == "cx"
            ),
            "t_gate_count": sum(name == "t" for name in names),
            "tdg_gate_count": sum(name == "tdg" for name in names),
            "t_count": sum(name in T_GATE_NAMES for name in names),
            "t_depth": len(layers),
            "t_layer_widths": layers,
            "original_rz_count": synthesis.get("original_rz_count"),
            "arbitrary_rotation_count": synthesis.get(
                "arbitrary_rotation_count"
            ),
            "synthesis_error_budget": synthesis.get(
                "synthesis_error_budget"
            ),
            "per_rotation_error": synthesis.get("per_rotation_error"),
        }

    return CircuitInfo(
        word=word,
        signed_generators=signed_generators,
        closure=closure,
        strands=strands,
        k=k,
        path=compiled.path,
        part=compiled.part,
        compiler_level=compiled.circuit_level,
        measured=non_gate_operations.get("measure", 0) > 0,
        logical_qubits=circuit.num_qubits,
        classical_bits=circuit.num_clbits,
        quantum_registers=tuple(
            (register.name, register.size) for register in circuit.qregs
        ),
        classical_registers=tuple(
            (register.name, register.size) for register in circuit.cregs
        ),
        quantum_gate_count=total_count,
        quantum_gate_depth=quantum_depth,
        circuit_depth=circuit.depth(),
        measurement_count=non_gate_operations.get("measure", 0),
        non_gate_operations=_freeze(non_gate_operations),
        gate_families=_freeze(
            _gate_family_counts(circuit, compiled.circuit_level)
        ),
        exact_gate_stats=_freeze(exact_stats),
        level_4_resources=(
            None
            if level_4_resources is None
            else _freeze(level_4_resources)
        ),
        compiler_policies=_freeze(
            _compiler_policy_summary(circuit, fallback_configuration)
        ),
        scope_notes=tuple(scope_notes),
    )


def compare_circuits(
    circuits: Mapping[str, CompiledCircuit],
) -> CircuitComparison:
    """Return a deterministic side-by-side comparison of labeled circuits."""

    if not circuits:
        raise ValueError("compare_circuits requires at least one labeled circuit")

    from .problem import CompiledCircuit

    reports = []
    for label, compiled in circuits.items():
        if not isinstance(label, str) or not label:
            raise ValueError("comparison labels must be non-empty strings")
        if not isinstance(compiled, CompiledCircuit):
            raise TypeError("compare_circuits values must be CompiledCircuit objects")
        reports.append((label, circuit_info(compiled)))

    warnings = [
        "Each column describes one compiled circuit, not a complete closure workload."
    ]
    if len({report.compiler_level for _, report in reports}) > 1:
        warnings.append(
            "Compiler levels differ; gate families use each circuit's own level."
        )
    policy_dicts = [
        _thaw(report.compiler_policies.get("configuration"))
        for _, report in reports
    ]
    if any(policy != policy_dicts[0] for policy in policy_dicts[1:]):
        warnings.append("Compiler policies differ between compared circuits.")
    return CircuitComparison(
        reports=tuple(reports),
        warnings=tuple(warnings),
    )


def circuit_gate_count_depth(circuit: QuantumCircuit) -> dict[str, object]:
    """Return total and exact per-gate counts alongside parallel depths.

    Only Qiskit ``Gate`` operations contribute to the total, so measurements,
    barriers, resets, and delays are not accidentally reported as quantum gates.
    """

    total_count, total_depth, by_gate = _exact_gate_stats(circuit)
    return {
        "total": {
            "count": total_count,
            "depth": total_depth,
        },
        "by_gate": by_gate,
    }


def level_1_varphi_names(circuit: QuantumCircuit) -> tuple[str, ...]:
    return tuple(
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name.startswith("c_varphi_sigma_")
    )


def _is_measured(circuit: QuantumCircuit) -> bool:
    return any(
        instruction.operation.name == "measure" for instruction in circuit.data
    )


def compilation_info(compilation: HadamardTestCompilation) -> CircuitComparison:
    """Report every compiler level of one Hadamard-test compilation.

    This is the ``AJLCompiler`` counterpart to :func:`compare_circuits`: one
    column per compiler level of the same component, built from the same
    :class:`CircuitInfo` reports. A bare compilation carries no closure or AJL
    root, so those identity fields read ``n/a``; use ``JonesProblem`` when the
    report should describe a closure workload.
    """

    from .problem import CompiledCircuit, _CircuitProvenance

    provenance = _CircuitProvenance(
        word=compilation.word,
        strands=len(compilation.initial_path),
        config=compilation.config,
    )
    levels: list[tuple[CompilerLevel, QuantumCircuit]] = [
        (1, compilation.level_1_varphi),
        (2, compilation.level_2_multicontrolled),
        (3, compilation.level_3_single_control),
    ]
    if compilation.level_4_clifford_t is not None:
        levels.append((4, compilation.level_4_clifford_t))

    reports = [
        (
            f"Level {level}",
            circuit_info(
                CompiledCircuit(
                    path=compilation.initial_path,
                    part=compilation.part,
                    circuit_level=level,
                    measured=_is_measured(circuit),
                    circuit=circuit,
                    _provenance=provenance,
                )
            ),
        )
        for level, circuit in levels
    ]
    warnings = [
        "Each column is one compiler level of the same Hadamard-test component, "
        "not a complete closure workload.",
        "Gate families use each level's own vocabulary.",
    ]
    if compilation.level_4_clifford_t is None:
        warnings.append(f"Level 4: {compilation.level_4_status}")
    return CircuitComparison(reports=tuple(reports), warnings=tuple(warnings))
