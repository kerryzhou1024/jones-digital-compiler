"""Optional IPython display helpers; importing the compiler does not require them."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from IPython.display import HTML, display
from qiskit import QuantumCircuit

from .model import HadamardPart
from .problem import CircuitLevelSelection, JonesProblem


def show_scrollable_circuit(
    circuit: QuantumCircuit,
    title: str,
    max_lines: int | None = None,
) -> None:
    """Display one unwrapped text circuit in a horizontally scrollable container."""

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
