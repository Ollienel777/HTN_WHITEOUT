"""The axes an episode log can already answer for, and the one it cannot.

Every axis here is derived from a field an :class:`~whiteout.types.EpisodeRecord`
already carries, and **names the field it came from**. That naming is the
point: :attr:`AxisScore.derivation` is what lets anyone holding the log
recompute the number by hand and disagree with it. A score a reader cannot
falsify is a claim, not a measurement.

``hackathon/ARENA.md`` §5 names seven criteria — search efficiency, coverage,
detection speed, tracking duration, accuracy, autonomy and collaboration.
Five of them are axes here. **Autonomy and collaboration are deliberately not**:
they are read off the fleet's behaviour and off the five-minute explanation,
not off a log, and inventing a log-derived proxy for them would be the kind of
number this module exists to stop printing. :func:`unscorable_criteria` says
so, and the CLI prints it, so the mismatch between the axis list and the seven
criteria is stated rather than silently dropped.

What each axis means:

``coverage``
    ``belief_digest.covered_fraction`` at the last tick — the share of the
    strait's water cells some sensor footprint has seen.

``detection_speed``
    How early the first contact appears: ``1 - (t_first - t_start) / span``,
    so a contact on the opening tick scores 1 and one on the closing tick
    scores 0. When no record carries a contact there is nothing to time, and
    the axis reports **not detected** in words.

``tracking_duration``
    The share of ticks that carry a contact **in a state that means we are
    holding it** (:data:`HELD_STATES`), with the equivalent in seconds in the
    derivation. A ``lost`` contact is not a tracked tick: ``Coordinator``
    keeps emitting one for every tick after the first sighting, so counting
    any non-empty ``contacts`` would measure time since first detection and
    print a near-perfect score for a target lost on sight. When no record
    carries a contact at all there was never anything to hold, and the axis
    reports **not detected** in words — the same absence ``detection_speed``
    reports, decided once. A zero here is therefore a measurement: contacts
    exist in the log and none of them was ever held.

``search_efficiency``
    *When* the fleet's energy went, weighed against the coverage standing at
    the time: ``sum(covered_fraction_i * energy_spent_in_tick_i) /
    total_energy_used``, differenced per ``asset_id`` so an asset that misses
    a tick does not mis-attribute its whole cumulative spend. Energy burned
    while the strait is still dark scores low; a fleet that covers early and
    then holds spends most of its energy at high coverage and scores high.
    Bounded in ``[0, 1]`` by construction, which a bare coverage-per-joule
    ratio is not. It is a convex combination, so it is **scale-invariant in
    energy**: a fleet burning ten times the energy on the identical coverage
    curve scores the same. It ranks promptness of spend, not thrift; the
    magnitude of the spend is in the derivation line, not in the number.
    With no energy recorded there is no denominator, and the axis says so in
    words.

``accuracy``
    **Not measured.** It is a comparison against ``truth``, and ``cmd_run``
    writes ``Truth(t=..., targets=())`` on every tick, so no episode log
    written so far carries a target to compare against. Populating truth
    during an arena run is its own ticket; the full scoring model is issue
    #26. Until then this axis prints words, never ``0.0000``, because a zero
    here reads as "we tracked it badly" rather than "we did not measure".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from whiteout.types import EpisodeRecord

#: Ordered scoring axes, as ``whiteout score`` prints them. Five of
#: ``ARENA.md`` §5's seven criteria; see :func:`unscorable_criteria` for the
#: two that no log can answer for. ``fixtures/weights/*.json`` carries exactly
#: these keys, and ``scripts/gate.py`` looks for exactly these lines.
AXES: tuple[str, str, str, str, str] = (
    "coverage",
    "detection_speed",
    "tracking_duration",
    "search_efficiency",
    "accuracy",
)

#: The :data:`~whiteout.types.CONTACT_STATES` that mean the fleet is holding
#: the target on this tick. ``unconfirmed`` and ``confirming`` are a hypothesis
#: not yet a track, and ``lost`` is the opposite of holding — and it is the
#: dangerous one, because ``Coordinator._contacts`` emits a ``lost`` contact on
#: every tick after the first sighting for the rest of the run.
HELD_STATES: frozenset[str] = frozenset({"tracked", "handed_off"})

_NO_TRUTH = "no episode log carries truth targets yet, so there is nothing to measure against"

_NOT_DETECTED = "not detected"


class ScoreError(ValueError):
    """An episode that cannot be scored, with the reason."""


@dataclass(frozen=True)
class AxisScore:
    """One axis: a number, or the words that stand in for one.

    ``value`` is a finite float in ``[0, 1]`` when the log could answer for
    this axis, and ``None`` when it could not. An axis with ``value is None``
    prints :attr:`words` instead of a number and is left out of the weighted
    total, so an unanswerable axis never masquerades as a bad score.
    """

    axis: str
    value: float | None
    words: str
    derivation: str

    @property
    def measured(self) -> bool:
        return self.value is not None

    def rendered(self) -> str:
        """What follows ``axis:`` on the printed line."""
        return self.words if self.value is None else f"{self.value:.4f}"


@dataclass(frozen=True)
class EpisodeScore:
    """Every axis for one episode, plus the tick count they were taken over."""

    records: int
    axes: tuple[AxisScore, ...]

    def by_axis(self) -> dict[str, AxisScore]:
        return {axis.axis: axis for axis in self.axes}

    def measured(self) -> tuple[AxisScore, ...]:
        return tuple(axis for axis in self.axes if axis.measured)

    def weighted_total(self, weights: Mapping[str, float]) -> float:
        """The weighted mean over the axes the log could answer for.

        Axes that reported words are dropped from numerator and denominator
        alike, so an axis nobody can compute yet neither lifts the total nor
        drags it down. With equal weights this is the plain mean of the
        measured axes.
        """
        measured = self.measured()
        divisor = sum(weights[axis.axis] for axis in measured)
        if not measured or divisor <= 0.0:
            return 0.0
        total = sum(weights[axis.axis] * (axis.value or 0.0) for axis in measured)
        return total / divisor


def unscorable_criteria() -> str:
    """The one line that reconciles :data:`AXES` with ``ARENA.md`` §5."""
    return (
        "ARENA.md section 5 names seven criteria; the five above are the ones a log "
        "can answer for. Autonomy and collaboration are read off the fleet's behaviour "
        "and the explanation, not off a log, so they are not axes here."
    )


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _coverage(records: Sequence[EpisodeRecord]) -> AxisScore:
    last = records[-1]
    return AxisScore(
        axis="coverage",
        value=_clamp(last.belief_digest.covered_fraction),
        words="",
        derivation=(
            f"belief_digest.covered_fraction at the last tick (t={last.t:.1f} s, "
            f"grid {last.belief_digest.grid_shape[0]}x{last.belief_digest.grid_shape[1]})"
        ),
    )


def _contact_ticks(records: Sequence[EpisodeRecord]) -> list[int]:
    """Ticks carrying any contact at all — a detection, whatever its state."""
    return [index for index, record in enumerate(records) if record.contacts]


def _held_ticks(records: Sequence[EpisodeRecord]) -> list[int]:
    """Ticks the fleet was actually holding a contact on (:data:`HELD_STATES`)."""
    return [
        index
        for index, record in enumerate(records)
        if any(contact.state in HELD_STATES for contact in record.contacts)
    ]


def _span(records: Sequence[EpisodeRecord]) -> float:
    return records[-1].t - records[0].t


def _detection_speed(records: Sequence[EpisodeRecord]) -> AxisScore:
    held = _contact_ticks(records)
    if not held:
        return AxisScore(
            axis="detection_speed",
            value=None,
            words=_NOT_DETECTED,
            derivation=(
                f"no record of {len(records)} carries a contact, so there is no first "
                "detection to time"
            ),
        )
    first = records[held[0]]
    span = _span(records)
    elapsed = first.t - records[0].t
    promptness = 1.0 if span <= 0.0 else 1.0 - elapsed / span
    return AxisScore(
        axis="detection_speed",
        value=_clamp(promptness),
        words="",
        derivation=(
            f"first contact at t={first.t:.1f} s, {elapsed:.1f} s into an episode "
            f"spanning {span:.1f} s (1.0 is the opening tick, 0.0 the closing one)"
        ),
    )


def _tracking_duration(records: Sequence[EpisodeRecord]) -> AxisScore:
    if not _contact_ticks(records):
        return AxisScore(
            axis="tracking_duration",
            value=None,
            words=_NOT_DETECTED,
            derivation=(
                f"no record of {len(records)} carries a contact, so there was never "
                "anything to hold"
            ),
        )
    held = _held_ticks(records)
    share = len(held) / len(records)
    span = _span(records)
    states = ", ".join(sorted(HELD_STATES))
    return AxisScore(
        axis="tracking_duration",
        value=_clamp(share),
        words="",
        derivation=(
            f"{len(held)} of {len(records)} ticks carry a contact in state {states} "
            f"({share * span:.1f} s of {span:.1f} s); a lost contact is not a held one"
        ),
    )


def _search_efficiency(records: Sequence[EpisodeRecord]) -> AxisScore:
    """Coverage weighted by *when* the energy went (see the module docstring).

    Scale-invariant in energy by construction: it ranks promptness of spend,
    not thrift.
    """
    spent = 0.0
    weighted = 0.0
    #: Last seen ``energy_used`` per asset. Differencing the fleet-wide sum
    #: would mis-read a silent asset: ``ArenaTransport.observe`` omits a pose
    #: for an asset that has not reported this tick, so the sum dips and then
    #: jumps, and that asset's whole cumulative spend lands on one tick's
    #: coverage. Per asset, a missed tick costs nothing and the increment is
    #: credited when it arrives.
    previous: dict[str, float] = {
        pose.asset_id: pose.energy_used for pose in records[0].observation.poses
    }
    opening = sum(previous.values())
    for record in records[1:]:
        delta = 0.0
        for pose in record.observation.poses:
            before = previous.get(pose.asset_id)
            increment = pose.energy_used if before is None else pose.energy_used - before
            previous[pose.asset_id] = pose.energy_used
            delta += max(0.0, increment)
        spent += delta
        weighted += record.belief_digest.covered_fraction * delta
    if spent <= 0.0:
        return AxisScore(
            axis="search_efficiency",
            value=None,
            words="no energy recorded",
            derivation=(
                "every observation.poses[].energy_used is unchanged across the episode, "
                "so there is no energy to weigh coverage against"
            ),
        )
    return AxisScore(
        axis="search_efficiency",
        value=_clamp(weighted / spent),
        words="",
        derivation=(
            f"coverage averaged over the {spent:.0f} units of "
            f"observation.poses[].energy_used spent after t={records[0].t:.1f} s, "
            f"differenced per asset (fleet total last reported: "
            f"{sum(previous.values()):.0f}, at the first tick: {opening:.0f}); "
            "this ranks when the energy went, not how much of it there was"
        ),
    )


def _accuracy() -> AxisScore:
    return AxisScore(
        axis="accuracy",
        value=None,
        words="not measured (no truth in log)",
        derivation=_NO_TRUTH,
    )


def score_episode(records: Sequence[EpisodeRecord]) -> EpisodeScore:
    """Score one episode. A pure function of ``records``; refuses an empty one.

    An empty log is refused rather than scored, because zeros over no data are
    exactly the output this scorer exists to stop printing.
    """
    if not records:
        raise ScoreError("episode log has no records")
    computed = {
        "coverage": _coverage(records),
        "detection_speed": _detection_speed(records),
        "tracking_duration": _tracking_duration(records),
        "search_efficiency": _search_efficiency(records),
        "accuracy": _accuracy(),
    }
    return EpisodeScore(records=len(records), axes=tuple(computed[name] for name in AXES))
