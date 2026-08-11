"""Optional IPython display helpers; importing the compiler does not require them."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from IPython.display import HTML, display
from qiskit import QuantumCircuit

from .model import HadamardPart
from .primitives import PrefixAdderGate
from .problem import CircuitLevelSelection, JonesProblem

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _subscript(value: int) -> str:
    return str(value).translate(_SUBSCRIPT)


def _named_register(circuit: QuantumCircuit, name: str):
    matches = [register for register in circuit.qregs if register.name == name]
    return matches[0] if len(matches) == 1 else None


def _varphi_label(operation_name: str) -> tuple[int, int] | None:
    parts = operation_name.split("_")
    if len(parts) != 5 or parts[:3] != ["c", "varphi", "sigma"]:
        return None
    try:
        index = int(parts[3])
    except ValueError:
        return None
    if index < 1 or parts[4] not in {"plus", "minus"}:
        return None
    return index, 1 if parts[4] == "plus" else -1


def _level_1_serial_svg(circuit: QuantumCircuit) -> str | None:
    """Return a symbolic SVG for a well-formed serial Level-1 circuit."""

    metadata = circuit.metadata or {}
    if metadata.get("compiler_level") != 1:
        return None
    if metadata.get("generator_scheduling") != "serial":
        return None

    control = _named_register(circuit, "ctrl")
    path = _named_register(circuit, "path")
    height = _named_register(circuit, "height")
    if control is None or control.size != 1 or path is None or height is None:
        return None
    if set(register.name for register in circuit.qregs) != {"ctrl", "path", "height"}:
        return None

    path_qubits = set(path)
    operations = []
    started = False
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple(instruction.qubits)
        if not started and operation.name == "x" and len(qubits) == 1:
            if qubits[0] not in path_qubits:
                return None
            continue
        if operation.name == "barrier":
            continue
        if operation.name == "h" and qubits == (control[0],):
            started = True
            operations.append(instruction)
            continue
        if operation.name == "sdg" and qubits == (control[0],):
            operations.append(instruction)
            continue
        if operation.name == "measure" and qubits == (control[0],):
            operations.append(instruction)
            continue
        if isinstance(operation, PrefixAdderGate):
            operations.append(instruction)
            continue
        if _varphi_label(operation.name) is not None:
            operations.append(instruction)
            continue
        return None

    if not operations or operations[0].operation.name != "h":
        return None

    control_y = 48
    path_start_y = 104
    row_gap = 44
    height_start_y = path_start_y + path.size * row_gap + 40
    path_y = {qubit: path_start_y + index * row_gap for index, qubit in enumerate(path)}
    height_y = {
        qubit: height_start_y + index * row_gap for index, qubit in enumerate(height)
    }
    qubit_y = {control[0]: control_y, **path_y, **height_y}

    column_gap = 125
    first_column_x = 185
    width = max(540, first_column_x + column_gap * len(operations))
    wire_start_x = 132
    wire_end_x = width - 30
    height_px = max(height_y.values()) + 52

    svg = [
        (
            f'<svg class="level-1-ajl-circuit" xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height_px}" viewBox="0 0 {width} {height_px}" '
            'role="img" aria-label="Serial Level-1 AJL Hadamard-test circuit">'
        ),
        "<title>Serial Level-1 AJL Hadamard-test circuit</title>",
        (
            "<style>"
            ".wire{stroke:#1f2328;stroke-width:1.8}"
            ".connector{stroke:#1f2328;stroke-width:1.8}"
            ".gate{fill:#fff;stroke:#1f2328;stroke-width:1.8}"
            ".control,.selector{fill:#1f2328}"
            ".label{font:17px serif;fill:#1f2328}"
            ".small{font:15px serif;fill:#1f2328}"
            ".group{font:18px serif;fill:#1f2328}"
            "</style>"
        ),
    ]

    all_wires = [(control[0], "|0⟩")]
    all_wires.extend(
        (qubit, f"|a{_subscript(index)}⟩") for index, qubit in enumerate(path, start=1)
    )
    all_wires.extend(
        (qubit, f"|h{_subscript(index)}⟩") for index, qubit in enumerate(height, start=1)
    )
    for qubit, label in all_wires:
        y = qubit_y[qubit]
        svg.append(
            f'<line class="wire" x1="{wire_start_x}" y1="{y}" '
            f'x2="{wire_end_x}" y2="{y}"/>'
        )
        svg.append(
            f'<text class="label" x="{wire_start_x - 10}" y="{y + 6}" '
            f'text-anchor="end">{escape(label)}</text>'
        )

    def group_bracket(label: str, top: int, bottom: int) -> None:
        middle = (top + bottom) / 2
        svg.append(
            f'<path class="wire" fill="none" d="M 76 {top} Q 66 {top} 66 {top + 10} '
            f'L 66 {bottom - 10} Q 66 {bottom} 76 {bottom}"/>'
        )
        svg.append(
            f'<text class="group" x="56" y="{middle + 6}" text-anchor="end">'
            f'{escape(label)}</text>'
        )

    group_bracket("Path", min(path_y.values()) - 18, max(path_y.values()) + 18)
    group_bracket("Height", min(height_y.values()) - 18, max(height_y.values()) + 18)

    def box(
        x: int,
        top: int,
        bottom: int,
        label: str,
        css_class: str,
        *,
        box_width: int = 74,
    ) -> None:
        svg.append(
            f'<g class="operation {css_class}"><rect class="gate" '
            f'x="{x - box_width // 2}" y="{top}" width="{box_width}" '
            f'height="{bottom - top}"/>'
        )
        svg.append(
            f'<text class="small" x="{x}" y="{(top + bottom) / 2 + 5}" '
            f'text-anchor="middle">{escape(label)}</text></g>'
        )

    for column, instruction in enumerate(operations):
        x = first_column_x + column * column_gap
        operation = instruction.operation
        qubits = tuple(instruction.qubits)
        if operation.name in {"h", "sdg"}:
            label = "H" if operation.name == "h" else "S†"
            box(x, control_y - 18, control_y + 18, label, operation.name)
            continue
        if operation.name == "measure":
            box(x, control_y - 18, control_y + 18, "M", "measure")
            continue

        if isinstance(operation, PrefixAdderGate):
            selected_path = [qubit for qubit in qubits if qubit in path_y]
            selected_height = [qubit for qubit in qubits if qubit in height_y]
            if not selected_path or set(selected_height) != set(height):
                return None
            svg.append(
                f'<line class="connector" x1="{x}" y1="{min(path_y[q] for q in selected_path)}" '
                f'x2="{x}" y2="{max(height_y.values())}"/>'
            )
            for qubit in selected_path:
                svg.append(
                    f'<rect class="selector" x="{x - 5}" y="{path_y[qubit] - 5}" '
                    'width="10" height="10"/>'
                )
            indices = ",".join(_subscript(index + 1) for index in operation.path_indices)
            dagger = "†" if operation.is_inverse else ""
            box(
                x,
                min(height_y.values()) - 21,
                max(height_y.values()) + 21,
                f"Adder{dagger}{indices}",
                "prefix-adder",
                box_width=108,
            )
            continue

        parsed = _varphi_label(operation.name)
        if parsed is None or len(qubits) != 3 + height.size:
            return None
        generator_index, sign = parsed
        left = path[generator_index - 1]
        right = path[generator_index]
        if qubits[:3] != (control[0], left, right) or set(qubits[3:]) != set(height):
            return None
        svg.append(
            f'<line class="connector" x1="{x}" y1="{control_y}" '
            f'x2="{x}" y2="{max(height_y.values())}"/>'
        )
        svg.append(f'<circle class="control" cx="{x}" cy="{control_y}" r="5"/>')
        exponent = "⁻¹" if sign == -1 else ""
        height_label = "" if generator_index == 1 else "⁽ᶻ⁾"
        box(
            x,
            path_y[left] - 21,
            path_y[right] + 21,
            f"φ(σ{_subscript(generator_index)}{exponent}){height_label}",
            "controlled-varphi",
        )
        for qubit in height:
            svg.append(
                f'<rect class="selector" x="{x - 5}" y="{height_y[qubit] - 5}" '
                'width="10" height="10"/>'
            )

    svg.append("</svg>")
    return "".join(svg)


def show_scrollable_circuit(
    circuit: QuantumCircuit,
    title: str,
    max_lines: int | None = None,
) -> None:
    """Display one unwrapped text circuit in a horizontally scrollable container."""

    level_1_svg = _level_1_serial_svg(circuit)
    if level_1_svg is not None:
        display(
            HTML(
                f"""
                <div style="margin:0.5rem 0 1.25rem 0;">
                  <div style="font-weight:600; margin-bottom:0.35rem;">{escape(title)}</div>
                  <div style="overflow-x:auto; max-width:100%; border:1px solid #d0d7de;
                              border-radius:6px; padding:0.75rem; background:#fff;">
                    {level_1_svg}
                  </div>
                </div>
                """
            )
        )
        return

    drawing = str(
        circuit.draw(
            output="text",
            fold=-1,
            idle_wires=True,
            vertical_compression="high",
            cregbundle=False,
        )
    )
    lines = drawing.splitlines()
    if max_lines is not None and len(lines) > max_lines:
        omitted = len(lines) - max_lines
        drawing = (
            "\n".join(lines[:max_lines])
            + f"\n... {omitted} more lines omitted. Pass max_lines=None to show all."
        )
    container_style = (
        "overflow-x:auto; max-width:100%; border:1px solid #d0d7de; "
        "border-radius:6px; padding:0.75rem; background:#fafbfc;"
    )
    pre_style = (
        "margin:0; width:max-content; min-width:100%; white-space:pre; "
        "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; "
        "line-height:1.15;"
    )
    display(
        HTML(
            f"""
            <div style="margin:0.5rem 0 1.25rem 0;">
              <div style="font-weight:600; margin-bottom:0.35rem;">{escape(title)}</div>
              <div style="{container_style}">
                <pre style="{pre_style}">{escape(drawing)}</pre>
              </div>
            </div>
            """
        )
    )


def show_problem_circuits(
    problem: JonesProblem,
    *,
    title: str | None = None,
    path: str | Sequence[int] | None = None,
    part: HadamardPart | None = None,
    circuit_level: CircuitLevelSelection = 3,
    measure: bool = False,
    max_lines: int | None = None,
) -> None:
    """Display a filtered problem workload without retaining its circuits."""

    base_title = (
        title
        if title is not None
        else f"{problem.closure.title()} closure — {problem.word}"
    )
    for compiled in problem.circuits(
        path=path,
        part=part,
        circuit_level=circuit_level,
        measure=measure,
    ):
        show_scrollable_circuit(
            compiled.circuit,
            (
                f"{base_title} — path |{compiled.path_label}> — "
                f"{compiled.part} — Level {compiled.circuit_level}"
            ),
            max_lines=max_lines,
        )
