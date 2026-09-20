"""The motion gate: the vessel moves, the ice does not.

Issue #107. The two tests that carry this file are the first two — a static
candidate is never released however long it is seen, and one moving at the
vessel's speed is.
"""

from __future__ import annotations

import pytest

from whiteout.geo import GeoPoint, LocalPoint, local_to_geodetic
from whiteout.tracks.maintain import Sighting
from whiteout.types import WorldObservation
from whiteout.vision.motion import DEFAULT_MOTION_PARAMS, MotionGate, MotionParams

ORIGIN = GeoPoint(71.9900, -94.8400)


def _at(east_m: float, north_m: float) -> tuple[float, float]:
    point = local_to_geodetic(ORIGIN, LocalPoint(east_m=east_m, north_m=north_m))
    return point.lat_deg, point.lon_deg


class _Scripted:
    """Reports whatever it is handed, tick by tick."""

    def __init__(self, per_tick: dict[float, tuple[Sighting, ...]]) -> None:
        self.per_tick = per_tick

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        return self.per_tick.get(observation.t, ())


def _observation(t: float) -> WorldObservation:
    return WorldObservation(t=t, poses=(), reports=())


def _run(gate: MotionGate, ticks: int, step_s: float = 1.0) -> list[tuple[float, int]]:
    out = []
    for i in range(ticks):
        t = (i + 1) * step_s
        out.append((t, len(gate.sightings(_observation(t)))))
    return out


def _static(east: float = 0.0, north: float = 0.0, jitter: float = 0.0):
    """A floe: the same place every tick, give or take projection jitter."""

    def source(t: float) -> tuple[Sighting, ...]:
        wobble = jitter * (1 if int(t) % 2 else -1)
        lat, lon = _at(east + wobble, north + wobble)
        return (Sighting(t, lat, lon, "tower-1"),)

    return source


def _moving(speed_mps: float):
    """The vessel: heading north at ``speed_mps``."""

    def source(t: float) -> tuple[Sighting, ...]:
        lat, lon = _at(0.0, speed_mps * t)
        return (Sighting(t, lat, lon, "tower-1"),)

    return source


class _FromFunction:
    def __init__(self, fn) -> None:
        self.fn = fn

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        return self.fn(observation.t)


# -- the two that matter ----------------------------------------------------


def test_ice_is_never_released_however_long_it_is_watched() -> None:
    """A floe has a fixed lat/lon and keeps it however the camera moves.

    This is the whole point: flown live, the detector reported sightings on
    about 70% of ticks and none of them were the vessel.
    """
    gate = MotionGate(source=_FromFunction(_static()))
    passed = _run(gate, 40)
    assert sum(n for _, n in passed) == 0
    assert gate.released == 0
    assert gate.rejected > 0, "it must be judging, not merely never deciding"


def test_something_at_the_vessel_s_speed_is_released() -> None:
    gate = MotionGate(source=_FromFunction(_moving(3.0)))
    passed = _run(gate, 20)
    assert gate.released == 1
    assert sum(n for _, n in passed) > 0


def test_projection_jitter_does_not_make_ice_look_alive() -> None:
    """Jitter is larger than the vessel's per-tick travel, so it must not pass.

    The vessel moves about 2.3 m between ticks at the measured 1.3 Hz. A
    tick-to-tick speed would be noise; this gate measures net displacement
    across the whole span, and zero-mean jitter does not accumulate.
    """
    gate = MotionGate(source=_FromFunction(_static(jitter=25.0)))
    _run(gate, 40)
    assert gate.released == 0


# -- the band ---------------------------------------------------------------


@pytest.mark.parametrize("speed", [0.0, 0.2, 0.5])
def test_slow_drift_is_rejected(speed: float) -> None:
    gate = MotionGate(source=_FromFunction(_moving(speed)))
    _run(gate, 25)
    assert gate.released == 0


@pytest.mark.parametrize("speed", [1.5, 3.0, 6.0])
def test_the_band_covers_the_vessel_with_room_either_side(speed: float) -> None:
    """3.0 m/s is what the simulator uses; the band is wide around it.

    The measurement is a net displacement over a short span and carries
    projection error, so a band tight on 3.0 would reject the vessel on a bad
    span rather than reject the ice.
    """
    gate = MotionGate(source=_FromFunction(_moving(speed)))
    _run(gate, 25)
    assert gate.released == 1


def test_a_projection_that_jumps_is_rejected_as_too_fast() -> None:
    """Usually an attitude error on a turning aircraft, not a boat."""
    gate = MotionGate(source=_FromFunction(_moving(40.0)))
    _run(gate, 25)
    assert gate.released == 0


# -- latency, which is the price ---------------------------------------------


def test_nothing_is_released_before_the_span_has_elapsed() -> None:
    """Detection speed is scored, so this delay is a real cost, not free.

    It is paid because a fix on the wrong thing is worth less than nothing
    when accuracy is scored too.
    """
    gate = MotionGate(source=_FromFunction(_moving(3.0)))
    passed = _run(gate, 20)
    released_at = next(t for t, n in passed if n)
    assert released_at >= DEFAULT_MOTION_PARAMS.min_span_s


def test_once_released_a_track_is_not_re_judged_every_tick() -> None:
    """A vessel that slows for a moment must not blink out of its own track."""
    moving = _FromFunction(_moving(3.0))
    gate = MotionGate(source=moving)
    _run(gate, 15)
    assert gate.released == 1

    # Now it stops dead, at the place it had reached.
    stalled = _at(0.0, 3.0 * 15.0)
    gate.source = _FromFunction(lambda t: (Sighting(t, stalled[0], stalled[1], "tower-1"),))
    for i in range(8):
        found = gate.sightings(_observation(16.0 + i))
        assert found, "a released track keeps passing even when it stops"
    assert gate.rejected == 0


# -- bookkeeping -------------------------------------------------------------


def test_two_things_far_apart_are_watched_separately() -> None:
    """A floe must not be joined to the vessel a channel away."""
    vessel = _at(0.0, 0.0)
    floe = _at(2000.0, 0.0)
    gate = MotionGate(
        source=_Scripted(
            {
                1.0: (
                    Sighting(1.0, *vessel, "tower-1"),
                    Sighting(1.0, *floe, "tower-2"),
                )
            }
        )
    )
    gate.sightings(_observation(1.0))
    assert gate.watching == 2


def test_a_candidate_not_seen_again_is_forgotten() -> None:
    gate = MotionGate(source=_Scripted({1.0: (Sighting(1.0, *_at(0.0, 0.0), "t"),)}))
    gate.sightings(_observation(1.0))
    assert gate.watching == 1
    gate.sightings(_observation(1.0 + DEFAULT_MOTION_PARAMS.drop_after_s + 1.0))
    assert gate.watching == 0


def test_an_empty_tick_passes_nothing_and_breaks_nothing() -> None:
    gate = MotionGate(source=_Scripted({}))
    assert gate.sightings(_observation(1.0)) == ()


def test_a_nonsense_band_is_refused() -> None:
    with pytest.raises(ValueError):
        MotionParams(min_speed_mps=9.0, max_speed_mps=1.0)
    with pytest.raises(ValueError):
        MotionParams(min_span_s=0.0)
