"""Optional IPython display helpers; importing the compiler does not require them."""

from __future__ import annotations

from html import escape

from IPython.display import HTML, display
from qiskit import QuantumCircuit


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
