"""Optional IPython display helpers; importing the compiler does not require them."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from html import escape

from IPython.display import HTML, display
from qiskit import QuantumCircuit

from .model import HadamardPart
from .primitives import PrefixAdderGate
from .problem import CircuitLevelSelection, JonesProblem

_log = logging.getLogger(__name__)
_LANE_COLORS = ("#c43d4b", "#2f6fdf", "#16866f", "#8a56b3", "#d27a19")
_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _subscript(value: int) -> str:
    return str(value).translate(_SUBSCRIPT)


def _sigma(signed_index: int) -> str:
    exponent = "⁻¹" if signed_index < 0 else ""
    return f"σ{_subscript(abs(signed_index))}{exponent}"


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


def _register_offsets(circuit, instruction, register_name: str) -> tuple[int, ...]:
    offsets = []
    for qubit in instruction.qubits:
        for register, offset in circuit.find_bit(qubit).registers:
            if register.name == register_name:
                offsets.append(offset)
    return tuple(offsets)


def _height_lane(circuit, instruction, selector_width: int) -> int:
    offsets = _register_offsets(circuit, instruction, "height")
    if not offsets:
        raise ValueError("a Level-1 semantic block must use one height lane")
    lane = min(offsets) // selector_width
    if any(offset // selector_width != lane for offset in offsets):
        raise ValueError("a Level-1 semantic block cannot span height lanes")
    return lane


def _pack_disjoint(instructions):
    columns: list[tuple[list, set]] = []
    for instruction in instructions:
        operands = set(instruction.qubits)
        for column, used in columns:
            if operands.isdisjoint(used):
                column.append(instruction)
                used.update(operands)
                break
        else:
            columns.append(([instruction], set(operands)))
    return tuple(tuple(column) for column, _ in columns)


def _semantic_stages(instructions, layers):
    cursor = 0
    stages = []
    for signed_layer in layers:
        route = []
        while cursor < len(instructions) and isinstance(
            instructions[cursor].operation,
            PrefixAdderGate,
        ):
            route.append(instructions[cursor])
            cursor += 1

        varphi = instructions[cursor : cursor + len(signed_layer)]
        if len(varphi) != len(signed_layer):
            raise ValueError("a Level-1 layer is missing controlled varphi blocks")
        observed_layer = []
        for instruction in varphi:
            parsed = _varphi_label(instruction.operation.name)
            if parsed is None:
                raise ValueError("a Level-1 layer contains a non-varphi block")
            index, sign = parsed
            observed_layer.append(sign * index)
        if tuple(observed_layer) != signed_layer:
            raise ValueError("Level-1 operations disagree with generator metadata")
        cursor += len(varphi)
        stages.append((tuple(route), tuple(varphi), signed_layer))

    tail = tuple(instructions[cursor:])
    if any(not isinstance(item.operation, PrefixAdderGate) for item in tail):
        raise ValueError("Level-1 semantic operations remain after the final layer")
    return tuple(stages), tail


def _level_1_svg(circuit: QuantumCircuit) -> str | None:
    """Return a symbolic SVG for a well-formed Level-1 circuit.

    Anything the symbolic renderer does not recognize falls back to Qiskit's
    own drawing, logged at debug level so a genuine Level-1 regression is
    still discoverable rather than silently redrawn.
    """

    try:
        return _render_level_1_svg(circuit)
    except (IndexError, KeyError, TypeError, ValueError) as error:
        _log.debug("symbolic Level-1 rendering declined: %s", error)
        return None


def _render_level_1_svg(circuit: QuantumCircuit) -> str:
    metadata = circuit.metadata or {}
    if metadata.get("compiler_level") != 1:
        raise ValueError("not a Level-1 circuit")

    control = _named_register(circuit, "ctrl")
    path = _named_register(circuit, "path")
    height = _named_register(circuit, "height")
    fanout = _named_register(circuit, "ctrl_fanout")
    if control is None or control.size != 1 or path is None or height is None:
        raise ValueError("Level 1 requires control, path, and height registers")

    distribution = metadata.get("control_distribution")
    if distribution not in {"shared", "tree_fanout"}:
        raise ValueError("unsupported Level-1 control distribution")
    expected_names = {"ctrl", "path", "height"}
    if fanout is not None:
        expected_names.add("ctrl_fanout")
    if {register.name for register in circuit.qregs} != expected_names:
        raise ValueError("Level 1 contains an unsupported quantum register")
    if distribution == "shared" and fanout is not None:
        raise ValueError("shared control cannot contain fanout qubits")

    selector_width = metadata.get("height_selector_qubits")
    lane_count = metadata.get("parallel_lanes")
    raw_layers = metadata.get("generator_layers")
    if (
        isinstance(selector_width, bool)
        or not isinstance(selector_width, int)
        or selector_width < 1
        or isinstance(lane_count, bool)
        or not isinstance(lane_count, int)
        or lane_count < 1
        or height.size != selector_width * lane_count
    ):
        raise ValueError("invalid Level-1 lane metadata")
    try:
        layers = tuple(tuple(int(value) for value in layer) for layer in raw_layers)
    except (TypeError, ValueError):
        raise ValueError("invalid Level-1 generator layers") from None
    # Lanes are allocated from the widest layer, so the two must agree.
    if max((len(layer) for layer in layers), default=0) != lane_count:
        raise ValueError("lane count disagrees with generator layers")

    if distribution == "tree_fanout":
        expected_fanout = max(0, lane_count - 1)
        if expected_fanout == 0 and fanout is not None:
            raise ValueError("inactive tree fanout must not allocate qubits")
        if expected_fanout and (fanout is None or fanout.size != expected_fanout):
            raise ValueError("tree fanout register has the wrong width")

    path_qubits = set(path)
    operations = [
        instruction for instruction in circuit.data if instruction.operation.name != "barrier"
    ]
    while operations and operations[0].operation.name == "x":
        instruction = operations.pop(0)
        if len(instruction.qubits) != 1 or instruction.qubits[0] not in path_qubits:
            raise ValueError("only path preparation may precede the Hadamard")
    if not operations or operations.pop(0).operation.name != "h":
        raise ValueError("Level 1 must begin with a Hadamard")

    measurement = None
    if operations and operations[-1].operation.name == "measure":
        measurement = operations.pop()
        if tuple(measurement.qubits) != (control[0],):
            raise ValueError("Level-1 measurement must target the control")
    if not operations or operations[-1].operation.name != "h":
        raise ValueError("Level 1 must end with a Hadamard readout")
    readout_h = operations.pop()
    if tuple(readout_h.qubits) != (control[0],):
        raise ValueError("Level-1 readout must target the control")
    phase = None
    if operations and operations[-1].operation.name == "sdg":
        phase = operations.pop()
        if tuple(phase.qubits) != (control[0],):
            raise ValueError("Level-1 imaginary readout must target the control")

    fanout_operations = []
    while operations and operations[0].operation.name == "cx":
        fanout_operations.append(operations.pop(0))
    unfanout_operations = []
    while operations and operations[-1].operation.name == "cx":
        unfanout_operations.append(operations.pop())
    unfanout_operations.reverse()

    control_qubits = [control[0], *(fanout or ())]
    control_set = set(control_qubits)
    for instruction in (*fanout_operations, *unfanout_operations):
        if len(instruction.qubits) != 2 or not set(instruction.qubits) <= control_set:
            raise ValueError("fanout CNOTs must stay within the control register set")
    fanout_pairs = sorted(
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        for instruction in fanout_operations
    )
    unfanout_pairs = sorted(
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        for instruction in unfanout_operations
    )
    if distribution == "shared" and (fanout_operations or unfanout_operations):
        raise ValueError("shared control cannot contain fanout CNOTs")
    if distribution == "tree_fanout" and lane_count > 1:
        if not fanout_operations or fanout_pairs != unfanout_pairs:
            raise ValueError("tree fanout must be reversibly uncomputed")

    for instruction in operations:
        operation = instruction.operation
        if not isinstance(operation, PrefixAdderGate) and _varphi_label(operation.name) is None:
            raise ValueError("unsupported Level-1 semantic operation")
        lane = _height_lane(circuit, instruction, selector_width)
        if lane >= lane_count:
            raise ValueError("Level-1 operation uses an invalid height lane")
        height_offsets = _register_offsets(circuit, instruction, "height")
        expected_height = tuple(range(lane * selector_width, (lane + 1) * selector_width))
        if tuple(sorted(height_offsets)) != expected_height:
            raise ValueError("Level-1 operation must use one complete height lane")
        if isinstance(operation, PrefixAdderGate):
            if not _register_offsets(circuit, instruction, "path"):
                raise ValueError("a prefix Adder must use path controls")
            continue
        parsed = _varphi_label(operation.name)
        assert parsed is not None
        index, _ = parsed
        expected_control = control[0]
        if distribution == "tree_fanout" and lane > 0:
            assert fanout is not None
            expected_control = fanout[lane - 1]
        expected_path = (path[index - 1], path[index])
        if tuple(instruction.qubits[:3]) != (expected_control, *expected_path):
            raise ValueError("controlled varphi uses the wrong lane or path pair")

    stages, tail_route = _semantic_stages(operations, layers)
    units = [("h", None)]
    if fanout_operations:
        units.append(("fanout", tuple(fanout_operations)))
    for layer_number, (route, varphi, signed_layer) in enumerate(stages, start=1):
        units.extend(("route", column) for column in _pack_disjoint(route))
        units.append(("varphi", (layer_number, varphi, signed_layer)))
    units.extend(("route", column) for column in _pack_disjoint(tail_route))
    if unfanout_operations:
        units.append(("unfanout", tuple(unfanout_operations)))
    if phase is not None:
        units.append(("sdg", phase))
    units.append(("h", readout_h))
    if measurement is not None:
        units.append(("measure", measurement))

    control_y = 58
    fanout_y = {qubit: 98 + 40 * offset for offset, qubit in enumerate(fanout or ())}
    control_y_by_qubit = {control[0]: control_y, **fanout_y}
    last_control_y = max(control_y_by_qubit.values())
    path_start = last_control_y + 72
    path_y = {qubit: path_start + 42 * offset for offset, qubit in enumerate(path)}
    height_start = max(path_y.values()) + 84
    height_lanes = {}
    cursor_y = height_start
    for lane in range(lane_count):
        lane_qubits = tuple(height[lane * selector_width : (lane + 1) * selector_width])
        height_lanes[lane] = {qubit: cursor_y + 34 * bit for bit, qubit in enumerate(lane_qubits)}
        cursor_y = max(height_lanes[lane].values()) + 58
    height_y = {qubit: y for lane in height_lanes.values() for qubit, y in lane.items()}
    canvas_height = cursor_y + 25

    wire_start = 190
    first_x = 250
    spacing = 146
    unit_x = tuple(first_x + spacing * offset for offset in range(len(units)))
    canvas_width = max(540, unit_x[-1] + 105)
    wire_end = canvas_width - 45
    parallel = any(len(layer) > 1 for layer in layers)

    def color(lane: int) -> str:
        if not parallel:
            return "#1f2328"
        return _LANE_COLORS[lane % len(_LANE_COLORS)]

    def lane_x(x: float, lane: int) -> float:
        if not parallel:
            return x
        return x + (lane - (lane_count - 1) / 2) * 14

    svg = [
        (
            f'<svg class="level-1-ajl-circuit" xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_width}" height="{canvas_height}" '
            f'viewBox="0 0 {canvas_width} {canvas_height}" role="img" '
            'aria-label="Level-1 AJL Hadamard-test circuit">'
        ),
        "<title>Level-1 AJL Hadamard-test circuit</title>",
        (
            "<style>"
            "text{font-family:STIX Two Text,Times New Roman,serif;fill:#171a21}"
            ".wire{stroke:#262a33;stroke-width:1.8}"
            ".gate{fill:#fff;stroke:#20242c;stroke-width:1.8}"
            ".small{font-family:ui-sans-serif,system-ui,sans-serif;fill:#656d7b}"
            "</style>"
        ),
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="white"/>',
    ]

    def wire(qubit, label: str, y: float) -> None:
        svg.append(
            f'<text x="174" y="{y + 6}" font-size="18" text-anchor="end">{escape(label)}</text>'
        )
        svg.append(f'<line class="wire" x1="{wire_start}" y1="{y}" x2="{wire_end}" y2="{y}"/>')

    wire(control[0], "|0⟩", control_y)
    for index, qubit in enumerate(fanout or (), start=1):
        wire(qubit, f"|0⟩ f{_subscript(index)}", fanout_y[qubit])

    def group_bracket(label: str, top: float, bottom: float, x: int) -> None:
        svg.append(f'<path d="M{x + 10} {top} H{x} V{bottom} H{x + 10}" fill="none" class="wire"/>')
        svg.append(
            f'<text x="{x - 29}" y="{(top + bottom) / 2 + 6}" '
            f'font-size="20" text-anchor="middle">{escape(label)}</text>'
        )

    group_bracket("Path", min(path_y.values()) - 19, max(path_y.values()) + 19, 102)
    for index, qubit in enumerate(path, start=1):
        wire(qubit, f"|a{_subscript(index)}⟩", path_y[qubit])

    group_bracket(
        "Height",
        min(height_y.values()) - 18,
        max(height_y.values()) + 18,
        102,
    )
    for lane, lane_rows in height_lanes.items():
        if parallel:
            lane_top = min(lane_rows.values()) - 14
            lane_bottom = max(lane_rows.values()) + 14
            svg.append(
                f'<line x1="126" y1="{lane_top}" x2="126" y2="{lane_bottom}" '
                f'stroke="{color(lane)}" stroke-width="4" stroke-linecap="round"/>'
            )
            svg.append(
                f'<text x="119" y="{(lane_top + lane_bottom) / 2 + 5}" '
                f'font-size="13" text-anchor="end" fill="{color(lane)}">'
                f"H{str(lane + 1).translate(_SUPERSCRIPT)}</text>"
            )
        for bit, (qubit, y) in enumerate(lane_rows.items(), start=1):
            lane_label = f"⁽{str(lane + 1).translate(_SUPERSCRIPT)}⁾" if parallel else ""
            wire(qubit, f"|h{_subscript(bit)}{lane_label}⟩", y)

    fanout_indices = [index for index, (kind, _) in enumerate(units) if kind == "fanout"]
    unfanout_indices = [index for index, (kind, _) in enumerate(units) if kind == "unfanout"]
    if fanout_indices and unfanout_indices:
        color_start = unit_x[fanout_indices[-1]] + 12
        color_end = unit_x[unfanout_indices[0]] - 12
        for lane, qubit in enumerate(control_qubits[:lane_count]):
            svg.append(
                f'<line x1="{color_start}" y1="{control_y_by_qubit[qubit]}" '
                f'x2="{color_end}" y2="{control_y_by_qubit[qubit]}" '
                f'stroke="{color(lane)}" stroke-width="3"/>'
            )

    def box(
        x: float,
        top: float,
        bottom: float,
        label: str,
        css_class: str,
        *,
        stroke: str = "#20242c",
        width: int = 96,
        lane: int | None = None,
    ) -> None:
        lane_attribute = "" if lane is None else f' data-lane="{lane}"'
        svg.append(
            f'<g class="operation {css_class}"{lane_attribute}>'
            f'<rect x="{x - width / 2}" y="{top}" width="{width}" '
            f'height="{bottom - top}" rx="3" fill="white" '
            f'stroke="{stroke}" stroke-width="2.2"/>'
        )
        svg.append(
            f'<text x="{x}" y="{(top + bottom) / 2 + 6}" font-size="18" '
            f'text-anchor="middle">{escape(label)}</text></g>'
        )

    def draw_h(x: float, label: str = "H", css_class: str = "h") -> None:
        box(
            x,
            control_y - 20,
            control_y + 20,
            label,
            css_class,
            width=40,
        )

    def draw_fanout(x: float, css_class: str) -> None:
        targets = control_qubits[1:lane_count]
        if not targets:
            return
        svg.append(f'<g class="operation {css_class}">')
        svg.append(
            f'<line class="fanout-connector" x1="{x}" y1="{control_y}" '
            f'x2="{x}" y2="{control_y_by_qubit[targets[-1]]}" '
            'stroke="#20242c" stroke-width="2"/>'
        )
        svg.append(
            f'<circle class="fanout-control" cx="{x}" cy="{control_y}" '
            'r="5.5" fill="#20242c"/>'
        )
        for target in targets:
            target_y = control_y_by_qubit[target]
            svg.append(
                f'<circle class="fanout-target" cx="{x}" cy="{target_y}" '
                'r="10" fill="white" '
                'stroke="#20242c" stroke-width="2"/>'
            )
            svg.append(
                f'<line x1="{x - 7}" y1="{target_y}" x2="{x + 7}" '
                f'y2="{target_y}" stroke="#20242c" stroke-width="2"/>'
            )
            svg.append(
                f'<line x1="{x}" y1="{target_y - 7}" x2="{x}" '
                f'y2="{target_y + 7}" stroke="#20242c" stroke-width="2"/>'
            )
        svg.append("</g>")
        caption = "fanout" if css_class == "fanout-cnot" else "unfanout"
        svg.append(
            f'<text class="small" x="{x}" y="26" font-size="12" '
            f'text-anchor="middle">{caption}</text>'
        )

    def draw_adder(x: float, instruction) -> None:
        lane = _height_lane(circuit, instruction, selector_width)
        gate_x = lane_x(x, lane)
        selected_path = _register_offsets(circuit, instruction, "path")
        lane_rows = height_lanes[lane]
        top = min(lane_rows.values()) - 17
        bottom = max(lane_rows.values()) + 17
        svg.append(
            f'<line x1="{gate_x}" y1="{path_y[path[min(selected_path)]]}" '
            f'x2="{gate_x}" y2="{top}" stroke="{color(lane)}" '
            'stroke-width="2.2"/>'
        )
        for offset in selected_path:
            svg.append(
                f'<rect x="{gate_x - 5}" y="{path_y[path[offset]] - 5}" '
                f'width="10" height="10" fill="{color(lane)}"/>'
            )
        indices = ",".join(_subscript(index + 1) for index in instruction.operation.path_indices)
        dagger = "†" if instruction.operation.is_inverse else ""
        box(
            gate_x,
            top,
            bottom,
            f"Adder{dagger}{indices}",
            "prefix-adder",
            stroke=color(lane),
            width=98,
            lane=lane,
        )

    def draw_varphi(x: float, instruction) -> None:
        lane = _height_lane(circuit, instruction, selector_width)
        gate_x = lane_x(x, lane)
        selected_path = tuple(
            path[offset] for offset in _register_offsets(circuit, instruction, "path")
        )
        lane_rows = height_lanes[lane]
        parsed = _varphi_label(instruction.operation.name)
        assert parsed is not None
        index, sign = parsed
        control_qubit = instruction.qubits[0]
        svg.append(
            f'<line x1="{gate_x}" y1="{control_y_by_qubit[control_qubit]}" '
            f'x2="{gate_x}" y2="{max(lane_rows.values())}" '
            f'stroke="{color(lane)}" stroke-width="2.6"/>'
        )
        svg.append(
            f'<circle cx="{gate_x}" cy="{control_y_by_qubit[control_qubit]}" '
            f'r="6" fill="{color(lane)}"/>'
        )
        height_label = "⁽ᶻ⁾" if not parallel and index > 1 else ""
        box(
            gate_x,
            min(path_y[qubit] for qubit in selected_path) - 17,
            max(path_y[qubit] for qubit in selected_path) + 17,
            f"φ({_sigma(sign * index)}){height_label}",
            "controlled-varphi",
            stroke=color(lane),
            lane=lane,
        )
        for y in lane_rows.values():
            svg.append(
                f'<rect x="{gate_x - 5}" y="{y - 5}" width="10" height="10" fill="{color(lane)}"/>'
            )

    for x, (kind, payload) in zip(unit_x, units, strict=True):
        if kind == "h":
            draw_h(x)
        elif kind == "sdg":
            draw_h(x, "S†", "sdg")
        elif kind == "fanout":
            draw_fanout(x, "fanout-cnot")
        elif kind == "unfanout":
            draw_fanout(x, "unfanout-cnot")
        elif kind == "route":
            for instruction in payload:
                draw_adder(x, instruction)
        elif kind == "varphi":
            layer_number, instructions, signed_layer = payload
            if parallel:
                layer_label = " ∥ ".join(_sigma(value) for value in signed_layer)
                svg.append(
                    f'<text class="small" x="{x}" y="26" font-size="13" '
                    f'text-anchor="middle">Layer {layer_number}: '
                    f"{escape(layer_label)}</text>"
                )
            for instruction in instructions:
                draw_varphi(x, instruction)
        elif kind == "measure":
            svg.append('<g class="operation measure">')
            svg.append(
                f'<rect class="gate" x="{x - 23}" y="{control_y - 20}" '
                'width="46" height="40" rx="2"/>'
            )
            svg.append(
                f'<path d="M{x - 13} {control_y + 8} Q{x} {control_y - 7} '
                f'{x + 13} {control_y + 8}" fill="none" stroke="#20242c" '
                'stroke-width="1.8"/>'
            )
            svg.append(
                f'<line x1="{x}" y1="{control_y + 5}" x2="{x + 11}" '
                f'y2="{control_y - 7}" stroke="#20242c" stroke-width="1.8"/>'
            )
            svg.append("</g>")

    svg.append("</svg>")
    return "".join(svg)


def show_scrollable_circuit(
    circuit: QuantumCircuit,
    title: str,
    max_lines: int | None = None,
) -> None:
    """Display one unwrapped text circuit in a horizontally scrollable container."""

    level_1_svg = _level_1_svg(circuit)
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
        title if title is not None else f"{problem.closure.title()} closure — {problem.word}"
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
