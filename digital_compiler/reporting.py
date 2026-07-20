"""Small, deterministic compiler summaries for notebooks and scripts."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Gate

from .compiler import HadamardTestCompilation, register_signature


def circuit_gate_count_depth(circuit: QuantumCircuit) -> dict[str, object]:
    """Return total and exact per-gate counts alongside parallel depths.

    Only Qiskit ``Gate`` operations contribute to the total, so measurements,
    barriers, resets, and delays are not accidentally reported as quantum gates.
    """

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
    return {
        "total": {
            "count": sum(counts.values()),
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
    families: dict[str, int] = {}
    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        if name == "measure":
            family = "measurement"
        elif name == "cx":
            family = "CNOT"
        elif isinstance(operation, ControlledGate):
            base = operation.base_gate.name
            controls = operation.num_ctrl_qubits
            if base == "x" and controls >= 2:
                family = f"MCX[{controls}]"
            elif base == "p":
                family = f"MCPhase[{controls}]"
            elif base in {"rx", "ry", "rz"}:
                family = (
                    f"C{base[1:].upper()}"
                    if controls == 1
                    else f"MC{base[1:].upper()}[{controls}]"
                )
            else:
                raise AssertionError(f"unexpected Level-2 controlled gate {name!r}")
        else:
            family = {
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
                "barrier": "barrier",
            }.get(name)
            if family is None:
                raise AssertionError(f"unexpected Level-2 gate {name!r}")
        families[family] = families.get(family, 0) + 1
    return families


def level_3_gate_family_counts(circuit: QuantumCircuit) -> dict[str, int]:
    mapping = {
        "x": "X",
        "h": "H",
        "s": "S/Sdg",
        "sdg": "S/Sdg",
        "t": "T/Tdg",
        "tdg": "T/Tdg",
        "rx": "Rx",
        "ry": "Ry",
        "rz": "Rz",
        "cx": "CNOT",
        "crx": "CRx",
        "cry": "CRy",
        "crz": "CRz",
        "measure": "measurement",
        "barrier": "barrier",
    }
    families: dict[str, int] = {}
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in mapping:
            raise AssertionError(f"unexpected Level-3 gate {name!r}")
        family = mapping[name]
        families[family] = families.get(family, 0) + 1
    return families


def compilation_summary(compilation: HadamardTestCompilation) -> dict[str, object]:
    level_3_metadata = compilation.level_3_single_control.metadata or {}
    return {
        "word": str(compilation.word),
        "signed_generators": compilation.word.signed_indices(),
        "initial_path": compilation.path_label,
        "part": compilation.part,
        "height_strategy": compilation.height_policy_label,
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
