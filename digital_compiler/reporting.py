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

from .compiler import CompilerLevel, HadamardTestCompilation, register_signature

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
            },
            "compiler": _thaw(self.compiler_policies),
            "scope_notes": self.scope_notes,
        }

    def _summary_rows(self) -> list[tuple[object, ...]]:
        return [
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
    return _level_3_gate_family(operation, strict=strict)


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
    families: dict[str, int] = {}
    for instruction in circuit.data:
        operation = instruction.operation
        if not isinstance(operation, Gate):
            continue
        family = _gate_family(operation, compiler_level)
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


def _legacy_gate_family_counts(
    circuit: QuantumCircuit,
    compiler_level: CompilerLevel,
) -> dict[str, int]:
    families: dict[str, int] = {}
    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        if name == "measure":
            family = "measurement"
        elif name == "barrier":
            family = "barrier"
        elif isinstance(operation, Gate):
            family = _gate_family(operation, compiler_level, strict=True)
        else:
            raise AssertionError(
                f"unexpected Level-{compiler_level} operation {name!r}"
            )
        families[family] = families.get(family, 0) + 1
    return families


def _compiler_policy_summary(
    circuit: QuantumCircuit,
    fallback_configuration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = circuit.metadata or {}
    return {
        "height_strategy": metadata.get("height_strategy"),
        "generator_scheduling": metadata.get("generator_scheduling"),
        "generator_layers": metadata.get("generator_layers"),
        "parallel_lanes": metadata.get("parallel_lanes"),
        "active_parallel_width": metadata.get("active_parallel_width"),
        "lowering_policies": metadata.get("lowering_policies"),
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
        closure = provenance.closure
        strands = provenance.strands
        k = provenance.k
        fallback_configuration = provenance.config.metadata()

    scope_notes = [
        "Single compiler-level logical circuit; not the complete closure workload.",
        "Before backend transpilation and physical or surface-code mapping.",
    ]
    if compiled.circuit_level == 3:
        scope_notes.append(
            "Arbitrary rotations remain unsynthesized; this is not a final "
            "Clifford+T estimate."
        )

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


def level_2_gate_family_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return _legacy_gate_family_counts(circuit, 2)


def level_3_gate_family_counts(circuit: QuantumCircuit) -> dict[str, int]:
    return _legacy_gate_family_counts(circuit, 3)


def compilation_summary(compilation: HadamardTestCompilation) -> dict[str, object]:
    level_2_metadata = compilation.level_2_multicontrolled.metadata or {}
    level_3_metadata = compilation.level_3_single_control.metadata or {}
    return {
        "word": str(compilation.word),
        "signed_generators": compilation.word.signed_indices(),
        "initial_path": compilation.path_label,
        "part": compilation.part,
        "height_strategy": compilation.height_policy_label,
        "generator_scheduling": compilation.scheduling_policy_label,
        "generator_layers": level_2_metadata.get("generator_layers"),
        "parallel_lanes": level_2_metadata.get("parallel_lanes"),
        "active_parallel_width": level_2_metadata.get("active_parallel_width"),
        "level_1_compiler_level": compilation.level_1_varphi.metadata.get("compiler_level"),
        "level_2_compiler_level": compilation.level_2_multicontrolled.metadata.get(
            "compiler_level"
        ),
        "level_3_compiler_level": level_3_metadata.get("compiler_level"),
        "level_3_source_level": level_3_metadata.get("source_level"),
        "level_3_lowering_policies": level_3_metadata.get("lowering_policies"),
        "logical_qubits_each_level": compilation.logical_qubits,
        "registers": register_signature(compilation.level_1_varphi),
        "level_1_varphi_blocks": len(level_1_varphi_names(compilation.level_1_varphi)),
        "level_1_depth": compilation.level_1_varphi.depth(),
        "level_2_depth": compilation.level_2_multicontrolled.depth(),
        "level_2_gate_families": level_2_gate_family_counts(
            compilation.level_2_multicontrolled
        ),
        "level_3_depth": compilation.level_3_single_control.depth(),
        "level_3_gate_families": level_3_gate_family_counts(
            compilation.level_3_single_control
        ),
        "level_1_gate_count_depth": circuit_gate_count_depth(
            compilation.level_1_varphi
        ),
        "level_2_gate_count_depth": circuit_gate_count_depth(
            compilation.level_2_multicontrolled
        ),
        "level_3_gate_count_depth": circuit_gate_count_depth(
            compilation.level_3_single_control
        ),
        "level_4": compilation.level_4_status,
    }


def _print_gate_count_depth_table(
    label: str,
    report: dict[str, object],
) -> None:
    total = report["total"]
    by_gate = report["by_gate"]
    if not isinstance(total, dict) or not isinstance(by_gate, dict):
        raise TypeError("gate count/depth report has an invalid structure")

    rows = [("TOTAL quantum gates", total), *by_gate.items()]
    name_width = max(len("gate"), *(len(name) for name, _ in rows))
    print(f"\n{label}")
    print(f"{'gate':<{name_width}}  {'count':>7} | {'depth':>5}")
    print(f"{'-' * name_width}  {'-' * 7}-+-{'-' * 5}")
    for name, resources in rows:
        if not isinstance(resources, dict):
            raise TypeError("gate resource row has an invalid structure")
        print(
            f"{name:<{name_width}}  {resources['count']:>7} | "
            f"{resources['depth']:>5}"
        )


def print_compilation_summary(compilation: HadamardTestCompilation) -> None:
    summary = compilation_summary(compilation)
    table_keys = {
        "level_1_gate_count_depth",
        "level_2_gate_count_depth",
        "level_3_gate_count_depth",
    }
    hidden_keys = {
        "level_1_depth",
        "level_2_depth",
        "level_2_gate_families",
        "level_3_depth",
        "level_3_gate_families",
        "level_4",
        *table_keys,
    }
    for key, value in summary.items():
        if key in hidden_keys:
            continue
        print(f"{key}: {value}")

    _print_gate_count_depth_table(
        "Level 1 — exact gate count | depth",
        summary["level_1_gate_count_depth"],
    )
    _print_gate_count_depth_table(
        "Level 2 — exact gate count | depth",
        summary["level_2_gate_count_depth"],
    )
    _print_gate_count_depth_table(
        "Level 3 — exact gate count | depth",
        summary["level_3_gate_count_depth"],
    )
    print(f"\nlevel_4: {summary['level_4']}")
    print()
