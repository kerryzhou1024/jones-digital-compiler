from __future__ import annotations

import pytest

from digital_compiler import (
    AJLCompiler,
    AJLPathModel,
    BraidWord,
    CommutingLayerScheduling,
    CompilerConfig,
    RecomputePrefixHeight,
    RetainFinalHeight,
    RollingPrefixHeight,
    SerialGeneratorScheduling,
    UncomputeFinalHeight,
)
from digital_compiler.policies import (
    PrefixHeightLayerPlan,
    PrefixHeightPlan,
    PrefixHeightTransition,
    RoutedGenerator,
)


def serial_plan(policy, word: str, strands: int):
    braid_word = BraidWord.parse(word)
    schedule = SerialGeneratorScheduling().schedule(braid_word, strands)
    return policy.route(braid_word, schedule, lanes=1)


@pytest.mark.parametrize("policy", [RollingPrefixHeight(), RecomputePrefixHeight()])
def test_identity_and_single_generator_routes(policy) -> None:
    identity = serial_plan(policy, "", strands=4)
    assert identity.layers == ()
    assert (
        identity.loads,
        identity.moves,
        identity.unloads,
        identity.path_steps,
    ) == (0, 0, 0, 0)

    single = serial_plan(policy, "3", strands=4)
    assert (
        single.loads,
        single.moves,
        single.unloads,
        single.path_steps,
    ) == (1, 0, 1, 4)


@pytest.mark.parametrize(
    "prefix_policy",
    [RollingPrefixHeight(), RecomputePrefixHeight()],
)
@pytest.mark.parametrize("word", ["", "1", "3"])
def test_final_height_policies_change_only_safe_terminal_unloads(
    prefix_policy,
    word: str,
) -> None:
    clean = serial_plan(prefix_policy, word, strands=4)
    retained = RetainFinalHeight().finalize(clean)
    uncomputed = UncomputeFinalHeight().finalize(clean)

    assert uncomputed == clean
    if not clean.layers:
        assert retained == clean
        return

    terminal_steps = sum(
        transition.path_steps for transition in clean.layers[-1].after
    )
    assert retained.layers[:-1] == clean.layers[:-1]
    assert retained.layers[-1].before == clean.layers[-1].before
    assert retained.layers[-1].generators == clean.layers[-1].generators
    assert retained.layers[-1].after == ()
    assert retained.unloads == clean.unloads - len(clean.layers[-1].after)
    assert retained.path_steps == clean.path_steps - terminal_steps


@pytest.mark.parametrize(
    "prefix_policy,clean_resources,retained_resources",
    [
        (RollingPrefixHeight(), (3, 1, 3, 12), (3, 1, 0, 7)),
        (RecomputePrefixHeight(), (4, 0, 4, 14), (4, 0, 2, 9)),
    ],
)
def test_retain_prunes_only_safe_retired_lane_unloads(
    prefix_policy,
    clean_resources: tuple[int, int, int, int],
    retained_resources: tuple[int, int, int, int],
) -> None:
    word = BraidWord.parse("1 3 2 5")
    schedule = CommutingLayerScheduling().schedule(word, strands=6)
    clean = prefix_policy.route(word, schedule, lanes=3)
    retained = RetainFinalHeight().finalize(clean)

    assert schedule == ((0, 1, 3), (2,))
    assert (clean.loads, clean.moves, clean.unloads, clean.path_steps) == (
        clean_resources
    )
    assert (
        retained.loads,
        retained.moves,
        retained.unloads,
        retained.path_steps,
    ) == retained_resources

    unsafe_word = BraidWord.parse("1 3 2 1 1 3 2")
    unsafe_schedule = CommutingLayerScheduling().schedule(unsafe_word, strands=4)
    unsafe_clean = prefix_policy.route(unsafe_word, unsafe_schedule, lanes=2)
    unsafe_retained = RetainFinalHeight().finalize(unsafe_clean)

    # The lane holding H_3 must be erased before the final sigma_2 crosses its
    # prefix boundary, even though that lane will never be used again.
    assert any(
        transition.source_index == 3 and transition.target_index is None
        for layer in unsafe_retained.layers[:-1]
        for transition in (*layer.before, *layer.after)
    )


def test_default_final_height_policy_is_retain() -> None:
    config = CompilerConfig()

    assert config.final_height == RetainFinalHeight()
    assert config.metadata()["final_height"] == {"name": "retain"}


@pytest.mark.parametrize(
    "word,rolling_steps,recomputed_steps,rolling_counts",
    [
        ("100 99 101", 202, 594, (1, 2, 1)),
        ("100 -99 101", 202, 594, (1, 2, 1)),
        ("1 100 1 100", 396, 396, (1, 3, 1)),
        ("100 100 100 100", 198, 792, (1, 0, 1)),
    ],
)
def test_serial_rolling_routes_capture_generator_locality(
    word: str,
    rolling_steps: int,
    recomputed_steps: int,
    rolling_counts: tuple[int, int, int],
) -> None:
    rolling = serial_plan(RollingPrefixHeight(), word, strands=102)
    recomputed = serial_plan(RecomputePrefixHeight(), word, strands=102)

    assert rolling.path_steps == rolling_steps
    assert recomputed.path_steps == recomputed_steps
    assert (rolling.loads, rolling.moves, rolling.unloads) == rolling_counts


def test_parallel_route_uses_the_minimum_cost_lane_assignment() -> None:
    word = BraidWord.parse("100 99 101")
    scheduler = CommutingLayerScheduling(max_lanes=2)
    schedule = scheduler.schedule(word, strands=102)
    rolling = RollingPrefixHeight().route(word, schedule, lanes=2)
    recomputed = RecomputePrefixHeight().route(word, schedule, lanes=2)

    assert schedule == ((0,), (1, 2))
    assert tuple(
        (item.position, item.lane) for item in rolling.layers[1].generators
    ) == (
        (1, 1),
        (2, 0),
    )
    assert (
        rolling.loads,
        rolling.moves,
        rolling.unloads,
        rolling.path_steps,
    ) == (2, 1, 2, 396)
    assert (
        recomputed.loads,
        recomputed.moves,
        recomputed.unloads,
        recomputed.path_steps,
    ) == (3, 0, 3, 594)

    compiler = AJLCompiler(
        AJLPathModel(102, 5),
        CompilerConfig(scheduling=scheduler),
    )
    circuit = compiler.level_2_braid_circuit(word)
    assert circuit.metadata["prefix_height_strategy"] == "rolling"
    assert circuit.metadata["prefix_height_path_steps"] == 396
    assert sum(circuit.count_ops().values()) == 1239
    assert circuit.depth() == 430


def test_serial_signed_path_step_adder_resource_regression() -> None:
    circuit = AJLCompiler(AJLPathModel(102, 5)).level_2_braid_circuit(
        "100 99 101"
    )

    assert sum(circuit.count_ops().values()) == 657
    assert circuit.depth() == 447
    assert circuit.count_ops().get("mcx", 0) == 0


def test_rolling_route_cleans_lanes_when_parallel_width_shrinks() -> None:
    word = BraidWord.parse("1 3 5 2 4")
    plan = RollingPrefixHeight().route(
        word,
        ((0, 1, 2), (3, 4)),
        lanes=3,
    )

    second_layer = plan.layers[1]
    assert any(
        transition.source_index == 1 and transition.target_index is None
        for transition in second_layer.before
    )
    assert (
        plan.loads,
        plan.moves,
        plan.unloads,
        plan.path_steps,
    ) == (3, 2, 3, 12)


class DirtyPrefixHeight:
    name = "test_dirty"

    @staticmethod
    def route(word, schedule, lanes):
        del word, schedule, lanes
        return PrefixHeightPlan(
            (
                PrefixHeightLayerPlan(
                    before=(PrefixHeightTransition(0, None, 1),),
                    generators=(RoutedGenerator(0, 0),),
                ),
            )
        )

    def metadata(self):
        return {"name": self.name}


def test_compiler_rejects_a_prefix_height_plan_with_dirty_final_lanes() -> None:
    compiler = AJLCompiler(
        AJLPathModel(2, 5),
        CompilerConfig(prefix_height=DirtyPrefixHeight()),
    )

    with pytest.raises(ValueError, match="clean every lane at completion"):
        compiler.level_2_braid_circuit("1")


class ParkedPrefixHeight:
    """Keep an idle height lane live across a layer that does not use it.

    A live lane holding the prefix height of index ``m`` covers path steps
    ``0..m - 2``.  Applying ``sigma_j`` rewrites ``path[j - 1]`` and ``path[j]``
    while preserving their sum, so the retained total survives whenever both
    endpoints sit inside that range (``j <= m - 2``) or both sit outside
    (``j >= m``).  Only ``j == m - 1`` splits the pair and silently invalidates
    the parked height, which is why the compiler requires every lane that a
    layer does not use to be clean before that layer runs.
    """

    name = "test_parked"

    @staticmethod
    def route(word, schedule, lanes):
        del word, schedule, lanes
        return PrefixHeightPlan(
            (
                PrefixHeightLayerPlan(
                    before=(
                        PrefixHeightTransition(0, None, 1),
                        PrefixHeightTransition(1, None, 3),
                    ),
                    generators=(RoutedGenerator(0, 0), RoutedGenerator(1, 1)),
                ),
                PrefixHeightLayerPlan(
                    before=(PrefixHeightTransition(0, 1, 2),),
                    generators=(RoutedGenerator(2, 0),),
                    after=(
                        PrefixHeightTransition(0, 2, None),
                        PrefixHeightTransition(1, 3, None),
                    ),
                ),
            )
        )

    def metadata(self):
        return {"name": self.name}


def test_compiler_rejects_a_prefix_height_plan_that_parks_an_idle_lane() -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        CompilerConfig(
            scheduling=CommutingLayerScheduling(max_lanes=2),
            prefix_height=ParkedPrefixHeight(),
        ),
    )

    # sigma_1 and sigma_3 share the first layer, so sigma_2 runs alone in the
    # second while the lane holding the prefix height of index 3 sits idle.
    with pytest.raises(ValueError, match="clean every inactive lane before a layer"):
        compiler.level_2_braid_circuit("1 3 2")


class UnsafeBoundaryRetention:
    """Deliberately retain H_3 across a later sigma_2 boundary crossing."""

    name = "test_unsafe_boundary_retention"
    clean_at_completion = False

    @staticmethod
    def finalize(plan):
        return PrefixHeightPlan(
            tuple(
                PrefixHeightLayerPlan(
                    before=tuple(
                        transition
                        for transition in layer.before
                        if not (
                            transition.source_index == 3
                            and transition.target_index is None
                        )
                    ),
                    generators=layer.generators,
                    after=tuple(
                        transition
                        for transition in layer.after
                        if not (
                            transition.source_index == 3
                            and transition.target_index is None
                        )
                    ),
                )
                for layer in plan.layers
            )
        )

    def metadata(self):
        return {"name": self.name}


def test_compiler_rejects_unsafe_retained_height_boundary() -> None:
    compiler = AJLCompiler(
        AJLPathModel(4, 5),
        CompilerConfig(
            scheduling=CommutingLayerScheduling(max_lanes=2),
            final_height=UnsafeBoundaryRetention(),
        ),
    )

    with pytest.raises(ValueError, match="generator crossing its boundary"):
        compiler.level_2_multicontrolled_circuit(
            "1 3 2 1 1 3 2",
            "1010",
            measure=False,
        )


def test_rolling_routes_never_park_an_idle_lane_across_a_layer() -> None:
    word = BraidWord.parse("1 3 5 2 4 1 5 3")
    for schedule, lanes in (
        (((0, 1, 2), (3, 4), (5, 6), (7,)), 3),
        (((0, 1), (2, 3), (4, 5), (6, 7)), 2),
        (tuple((position,) for position in range(word.crossings)), 1),
    ):
        plan = RollingPrefixHeight().route(word, schedule, lanes)
        live: dict[int, int] = {}
        for layer, routed in zip(schedule, plan.layers, strict=True):
            for transition in routed.before:
                if transition.target_index is None:
                    del live[transition.lane]
                else:
                    live[transition.lane] = transition.target_index
            assert set(live) == {item.lane for item in routed.generators}
            assert len(live) == len(layer)
