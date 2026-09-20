"""Issue #130's acceptance criteria: real numbers, or words, never a fake zero.

Every test here fails if ``whiteout score`` goes back to printing a constant.
``fixtures/episodes/demo.jsonl`` is a committed 400-tick episode, so the
assertions about it are assertions about a real log rather than about a
hand-built one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whiteout.log import SCHEMA_VERSION, read_episode_log
from whiteout.score import AXES, ScoreError, score_episode, unscorable_criteria
from whiteout.types import (
    BeliefDigest,
    Contact,
    EpisodeRecord,
    FleetIntent,
    Pose,
    Truth,
    WaypointIntent,
    WorldObservation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "fixtures" / "episodes" / "demo.jsonl"
WEIGHTS = REPO_ROOT / "fixtures" / "weights" / "equal.json"


def _record(
    tick: int,
    *,
    covered: float,
    energy: float,
    contacts: tuple[Contact, ...] = (),
    tasks: tuple[str, ...] = ("task-a",),
) -> EpisodeRecord:
    t = float(tick)
    return EpisodeRecord(
        schema_version=SCHEMA_VERSION,
        t=t,
        observation=WorldObservation(
            t=t,
            poses=(
                Pose(
                    asset_id="quadcopter",
                    cls="quad",
                    t=t,
                    lat=71.99,
                    lon=-94.84,
                    z=40.0,
                    heading=0.0,
                    speed=6.0,
                    energy_used=energy,
                ),
            ),
            reports=(),
        ),
        intent=FleetIntent(
            t=t,
            intents=tuple(
                WaypointIntent(
                    asset_id="quadcopter",
                    t=t,
                    target_lat=72.0,
                    target_lon=-94.8,
                    target_z=120.0,
                    speed=6.0,
                    reason="search",
                    task_id=task,
                )
                for task in tasks
            ),
        ),
        belief_digest=BeliefDigest(
            t=t,
            entropy=6.0,
            mass=1.0,
            peak_lat=71.99,
            peak_lon=-94.84,
            peak_p=0.01,
            covered_fraction=covered,
            grid_shape=(63, 16),
        ),
        contacts=contacts,
        truth=Truth(t=t, targets=()),
    )


def _contact(tick: int) -> Contact:
    return Contact(
        contact_id="c-1",
        t=float(tick),
        state="confirming",
        lat=71.996,
        lon=-94.845,
        confidence=0.7,
        classification="vessel",
        assigned_asset_id="quadcopter",
    )


def _episode(*, detect_at: int | None, ticks: int = 10) -> list[EpisodeRecord]:
    return [
        _record(
            tick,
            covered=min(1.0, 0.1 * (tick + 1)),
            energy=10.0 * tick,
            contacts=() if detect_at is None or tick < detect_at else (_contact(tick),),
        )
        for tick in range(ticks)
    ]


# --- the committed demo episode ---------------------------------------------


def test_demo_episode_scores_non_zero_bounded_numbers() -> None:
    """The acceptance criterion, over the real 400-tick log.

    Coverage and search efficiency are the two axes ``demo.jsonl`` can answer
    for: it carries a belief digest and accumulating energy on every tick.
    Both must be real numbers inside ``(0, 1]`` — the old stub printed 0.0000
    for every axis, so a non-zero bound is exactly what distinguishes a
    measurement here from a constant.
    """
    scored = score_episode(read_episode_log(DEMO))
    assert scored.records == 400
    axes = scored.by_axis()
    for name in ("coverage", "search_efficiency"):
        value = axes[name].value
        assert value is not None, f"{name} was not derived from the demo episode"
        assert 0.0 < value <= 1.0, f"{name} is {value}, outside (0, 1]"


def test_demo_episode_names_a_source_for_every_axis() -> None:
    """Each axis names what it was derived from, so the number is falsifiable."""
    scored = score_episode(read_episode_log(DEMO))
    for axis in scored.axes:
        assert axis.derivation.strip(), f"{axis.axis} says nothing about where it came from"
    sources = {axis.axis: axis.derivation for axis in scored.axes}
    assert "belief_digest.covered_fraction" in sources["coverage"]
    assert "energy_used" in sources["search_efficiency"]
    assert "contact" in sources["tracking_duration"]
    assert "truth" in sources["accuracy"]


def test_demo_episode_reports_the_axes_it_cannot_answer_for_in_words() -> None:
    """Never ``0.0000`` for an axis nobody measured.

    ``demo.jsonl`` carries no contacts, so detection speed has nothing to
    time; no log carries truth, so accuracy has nothing to compare against.
    """
    axes = score_episode(read_episode_log(DEMO)).by_axis()
    assert axes["detection_speed"].value is None
    assert axes["detection_speed"].rendered() == "not detected"
    assert axes["accuracy"].value is None
    assert axes["accuracy"].rendered() == "not measured (no truth in log)"


def test_demo_episode_tracks_for_no_time_because_it_holds_nothing() -> None:
    """A measured zero is a different statement from an unmeasured one."""
    axes = score_episode(read_episode_log(DEMO)).by_axis()
    assert axes["tracking_duration"].value == 0.0
    assert "0 of 400 ticks" in axes["tracking_duration"].derivation


# --- the axes, one at a time -------------------------------------------------


def test_coverage_is_the_last_tick_not_the_peak() -> None:
    records = [
        _record(0, covered=0.9, energy=0.0),
        _record(1, covered=0.4, energy=10.0),
    ]
    assert score_episode(records).by_axis()["coverage"].value == pytest.approx(0.4)


def test_detecting_sooner_scores_higher_than_detecting_later() -> None:
    early = score_episode(_episode(detect_at=1)).by_axis()["detection_speed"]
    late = score_episode(_episode(detect_at=8)).by_axis()["detection_speed"]
    assert early.value is not None and late.value is not None
    assert 0.0 < late.value < early.value <= 1.0
    assert "first contact at t=1.0" in early.derivation


def test_tracking_duration_is_the_share_of_ticks_holding_a_contact() -> None:
    axis = score_episode(_episode(detect_at=5)).by_axis()["tracking_duration"]
    assert axis.value == pytest.approx(0.5)
    assert "5 of 10 ticks" in axis.derivation


def test_search_efficiency_rewards_covering_before_spending() -> None:
    """Same coverage, same energy, different order.

    The fleet that covers the strait early spends the rest of its energy at
    high coverage; the one that covers late spends most of it in the dark.
    """
    early = [_record(tick, covered=1.0 if tick else 0.0, energy=10.0 * tick) for tick in range(10)]
    late = [
        _record(tick, covered=1.0 if tick == 9 else 0.0, energy=10.0 * tick) for tick in range(10)
    ]
    early_axis = score_episode(early).by_axis()["search_efficiency"]
    late_axis = score_episode(late).by_axis()["search_efficiency"]
    assert early_axis.value is not None and late_axis.value is not None
    assert early_axis.value > late_axis.value
    assert 0.0 <= late_axis.value <= 1.0 and 0.0 <= early_axis.value <= 1.0
    assert "90 units" in early_axis.derivation


def test_search_efficiency_says_so_when_no_energy_was_spent() -> None:
    records = [_record(tick, covered=0.5, energy=0.0) for tick in range(4)]
    axis = score_episode(records).by_axis()["search_efficiency"]
    assert axis.value is None
    assert axis.rendered() == "no energy recorded"


def test_accuracy_is_never_a_number() -> None:
    """Truth in the log changes nothing: the comparison itself is #26's."""
    axis = score_episode(_episode(detect_at=2)).by_axis()["accuracy"]
    assert axis.value is None
    assert "truth" in axis.derivation


# --- the total and the criteria ---------------------------------------------


def test_the_total_weighs_only_the_axes_the_log_answered_for() -> None:
    """An unmeasured axis must neither lift the total nor drag it down."""
    scored = score_episode(_episode(detect_at=1))
    weights = {axis: 0.2 for axis in AXES}
    measured = scored.measured()
    assert len(measured) == 4, "accuracy should be the only axis without a number here"
    expected = sum(axis.value or 0.0 for axis in measured) / len(measured)
    assert scored.weighted_total(weights) == pytest.approx(expected)


def test_a_weight_can_silence_an_axis_without_dragging_the_total_down() -> None:
    scored = score_episode(_episode(detect_at=1))
    only_coverage = {axis: (1.0 if axis == "coverage" else 0.0) for axis in AXES}
    coverage = scored.by_axis()["coverage"].value
    assert coverage is not None
    assert scored.weighted_total(only_coverage) == pytest.approx(coverage)


def test_the_committed_weights_file_carries_exactly_the_axes() -> None:
    """``--weights`` is the tuner's and the demo's input; it must stay in step."""
    loaded = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    assert tuple(loaded) == AXES


def test_the_two_criteria_no_log_can_answer_for_are_named() -> None:
    """ARENA.md §5 has seven criteria; the mismatch is stated, not dropped."""
    said = unscorable_criteria().lower()
    assert "autonomy" in said and "collaboration" in said
    assert "seven" in said


def test_an_empty_episode_is_refused_rather_than_scored() -> None:
    with pytest.raises(ScoreError):
        score_episode([])
