from __future__ import annotations

import pytest

from digital_compiler import JonesProblem

notebook_helpers = pytest.importorskip("digital_compiler.notebook")


def test_compiled_circuit_displays_itself(monkeypatch) -> None:
    calls = []

    def record(circuit, title, max_lines=None):
        calls.append((circuit, title, max_lines))

    monkeypatch.setattr(notebook_helpers, "show_scrollable_circuit", record)

    compiled = JonesProblem("s1^2", strands=2).circuit(
        "10",
        "real",
        circuit_level=3,
    )
    result = compiled.display(title="", max_lines=7)

    assert result is None
    assert calls == [(compiled.circuit, "", 7)]


def test_show_problem_circuits_displays_the_lazy_labeled_workload(
    monkeypatch,
) -> None:
    calls = []

    def record(circuit, title, max_lines=None):
        calls.append((circuit, title, max_lines))

    monkeypatch.setattr(notebook_helpers, "show_scrollable_circuit", record)

    problem = JonesProblem("s1^2", strands=2)
    result = notebook_helpers.show_problem_circuits(
        problem,
        title="Trace Hopf",
        circuit_level=2,
        max_lines=12,
    )

    assert result is None
    assert [title for _, title, _ in calls] == [
        "Trace Hopf — path |10> — real — Level 2",
        "Trace Hopf — path |10> — imag — Level 2",
        "Trace Hopf — path |11> — real — Level 2",
        "Trace Hopf — path |11> — imag — Level 2",
    ]
    assert all(max_lines == 12 for _, _, max_lines in calls)
    assert all(circuit.count_ops().get("measure", 0) == 0 for circuit, _, _ in calls)


def test_show_problem_circuits_matches_shot_measurement_shape(
    monkeypatch,
) -> None:
    calls = []

    def record(circuit, title, max_lines=None):
        calls.append((circuit, title, max_lines))

    monkeypatch.setattr(notebook_helpers, "show_scrollable_circuit", record)

    problem = JonesProblem(
        "s1",
        closure="plat",
        writhe=-1,
        strands=2,
    )
    notebook_helpers.show_problem_circuits(
        problem,
        circuit_level=3,
        measure=True,
    )

    assert len(calls) == 2
    assert all(circuit.count_ops().get("measure", 0) == 1 for circuit, _, _ in calls)
    assert calls[0][1].startswith("Plat closure — sigma_1 — path |10>")


def test_show_problem_circuits_accepts_path_part_and_all_level_filters(
    monkeypatch,
) -> None:
    calls = []

    def record(circuit, title, max_lines=None):
        calls.append((circuit, title, max_lines))

    monkeypatch.setattr(notebook_helpers, "show_scrollable_circuit", record)

    problem = JonesProblem("s1^2", strands=2)
    notebook_helpers.show_problem_circuits(
        problem,
        path="11",
        part="imag",
        circuit_level="all",
    )

    assert [title for _, title, _ in calls] == [
        "Trace closure — sigma_1 sigma_1 — path |11> — imag — Level 1",
        "Trace closure — sigma_1 sigma_1 — path |11> — imag — Level 2",
        "Trace closure — sigma_1 sigma_1 — path |11> — imag — Level 3",
    ]
