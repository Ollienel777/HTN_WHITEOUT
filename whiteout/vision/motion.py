"""Telling the vessel from the ice, by the one thing that separates them.

Issue #107. Flown against the live arena, the detector produced sightings on
about seventy per cent of ticks and **none of them were the vessel**: they sat
one to three kilometres away, on the channel's ice.

The camera shows why. Bellot Strait is dark water heavily scattered with
bright floes, so the **gaps between floes** are dark compact blobs on a light
surround — which is exactly what a hull looks like to a single frame. The
detector's own ablation measured this and said so: no false alarms at 0.22 ice
cover, sixteen in thirty-two at 0.53, and "past about a third cover the
channel reads as a cracked sheet rather than water with ice in it". The
operating point was chosen at 0.22 on synthetic frames. The real channel is
well past a third.

No threshold fixes that, and the detector's own notes say which cue does:

    the vessel is the only thing in the world that moves.

Why this works at all
----------------------

Because sightings are **projected to the ground before they get here**. A
false alarm on a floe or a shadow has a fixed lat/lon and keeps it however the
camera moves; the vessel's moves at 3.0 m/s. So "does it move?" is a question
about the projected position and not about the pixel, and the camera's own
motion is already divided out.

Why a long baseline and not tick-to-tick
-----------------------------------------

The vessel travels about 2.3 m between ticks at the measured 1.3 Hz. Projection
jitter — attitude error on a moving aircraft, mostly — is comfortably larger
than that, so a tick-to-tick speed is noise. Over ten seconds the vessel moves
thirty metres and the jitter, being roughly zero-mean, does not. **Net
displacement across the whole span** is therefore the measurement, and a
candidate is judged only once it has one worth taking.

What it costs
--------------

**Latency.** Nothing is released until a candidate has been seen for
:attr:`MotionParams.min_span_s`, so the first true fix arrives that much after
the first true detection. *Detection speed* is a scored criterion, so this is
a real price and not a free win — it is paid because a fix on the wrong thing
is worth less than nothing when accuracy is also scored.

**Anything genuinely stationary is invisible.** A vessel hove to would be
rejected exactly as ice is. Against this target that is safe: the plugin moves
it at a constant 3.0 m/s and never stops. Against one that could stop, this
gate would be the wrong mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from whiteout.geo import GeoPoint, geodetic_to_local
from whiteout.tracks.maintain import Sighting
from whiteout.types import WorldObservation

__all__ = ["DEFAULT_MOTION_PARAMS", "MotionGate", "MotionParams", "Sightable"]


class Sightable(Protocol):
    """Anything that reports sightings — the shape this gate wraps.

    Declared here rather than imported from :mod:`whiteout.coordinate`, which
    is the run loop and sits *above* this package. The protocol is structural,
    so a :class:`MotionGate` satisfies the coordinator's ``SightingSource``
    and vice versa without either module importing the other.
    """

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        """Zero or more sightings, in the observation's timebase."""


@dataclass(frozen=True)
class MotionParams:
    """The band a candidate's motion must fall in to be believed.

    :param associate_m: how close a new candidate must be to an open
        candidate to be treated as the same thing. Wide enough to absorb
        projection jitter, narrow enough not to join two floes a channel
        apart.
    :param min_span_s: how long a candidate must have been watched before it
        is judged at all.
    :param min_speed_mps: below this it is not moving. Ice, shadow, shore.
    :param max_speed_mps: above this it is not the vessel. A projection that
        has jumped, usually because the aircraft was turning.
    :param drop_after_s: an unseen candidate is forgotten after this.
    """

    associate_m: float = 160.0
    min_span_s: float = 8.0
    min_speed_mps: float = 1.0
    max_speed_mps: float = 8.0
    drop_after_s: float = 20.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_speed_mps < self.max_speed_mps:
            raise ValueError(
                f"speed band must be ordered and non-negative, got "
                f"{self.min_speed_mps!r}..{self.max_speed_mps!r}"
            )
        for name in ("associate_m", "min_span_s", "drop_after_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")


#: The band the arena's vessel actually falls in. It does a constant 3.0 m/s
#: (``SHIP_SPEED``, and ``VesselPathPlugin.cc``'s ``speed{3.0}``), so the band
#: is set wide around that rather than tight on it: the measurement is a net
#: displacement over a short span and carries projection error with it.
DEFAULT_MOTION_PARAMS = MotionParams()


@dataclass
class _Candidate:
    """One thing that has been seen, and where it has been."""

    lat_deg: float
    lon_deg: float
    first_t: float
    last_t: float
    first_lat: float
    first_lon: float
    fixes: int = 1
    released: bool = False

    def span_s(self) -> float:
        return self.last_t - self.first_t

    def net_speed_mps(self) -> float:
        """Speed implied by the net displacement over the whole span."""
        span = self.span_s()
        if span <= 0.0:
            return 0.0
        offset = geodetic_to_local(
            GeoPoint(self.first_lat, self.first_lon), GeoPoint(self.lat_deg, self.lon_deg)
        )
        return math.hypot(offset.east_m, offset.north_m) / span


@dataclass
class MotionGate:
    """Wraps a sighting source and passes on only what moves like the vessel.

    It is a :class:`~whiteout.coordinate.SightingSource` itself, so it drops
    in wherever one goes and the run loop does not know it is there.
    """

    source: Sightable
    params: MotionParams = DEFAULT_MOTION_PARAMS
    _open: list[_Candidate] = field(default_factory=list)
    _rejected: int = 0
    _released: int = 0

    @property
    def rejected(self) -> int:
        """Candidates judged and found not to be moving. Mostly ice."""
        return self._rejected

    @property
    def released(self) -> int:
        """Candidates that moved like the vessel and were passed on."""
        return self._released

    @property
    def watching(self) -> int:
        """Candidates still being watched, not yet judged either way."""
        return len(self._open)

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        """Every candidate this tick that has earned being believed."""
        now = observation.t
        candidates = self.source.sightings(observation)
        self._forget(now)

        passed: list[Sighting] = []
        for sighting in candidates:
            candidate = self._associate(sighting)
            if candidate is None:
                self._open.append(
                    _Candidate(
                        lat_deg=sighting.lat_deg,
                        lon_deg=sighting.lon_deg,
                        first_t=sighting.t,
                        last_t=sighting.t,
                        first_lat=sighting.lat_deg,
                        first_lon=sighting.lon_deg,
                    )
                )
                continue
            candidate.lat_deg = sighting.lat_deg
            candidate.lon_deg = sighting.lon_deg
            candidate.last_t = sighting.t
            candidate.fixes += 1
            if self._believable(candidate):
                passed.append(sighting)
        return tuple(passed)

    def _associate(self, sighting: Sighting) -> _Candidate | None:
        """The open candidate this sighting belongs to, or ``None`` for a new one."""
        best: _Candidate | None = None
        best_range = self.params.associate_m
        here = GeoPoint(sighting.lat_deg, sighting.lon_deg)
        for candidate in self._open:
            offset = geodetic_to_local(GeoPoint(candidate.lat_deg, candidate.lon_deg), here)
            distance = math.hypot(offset.east_m, offset.north_m)
            if distance <= best_range:
                best, best_range = candidate, distance
        return best

    def _believable(self, candidate: _Candidate) -> bool:
        """Has this candidate moved like the vessel over a long enough span?

        Once released a candidate stays released: a track that has proved it
        moves does not have to re-prove it on every tick, and a vessel that
        slows for a moment must not blink out of the track that is holding it.
        """
        if candidate.released:
            return True
        if candidate.span_s() < self.params.min_span_s:
            return False
        speed = candidate.net_speed_mps()
        if self.params.min_speed_mps <= speed <= self.params.max_speed_mps:
            candidate.released = True
            self._released += 1
            return True
        # Judged and found wanting. Its window is reset rather than the
        # candidate dropped, so a floe re-seen for ever is judged again rather
        # than accumulating a span that never ends.
        self._rejected += 1
        candidate.first_t = candidate.last_t
        candidate.first_lat = candidate.lat_deg
        candidate.first_lon = candidate.lon_deg
        return False

    def _forget(self, now: float) -> None:
        cutoff = now - self.params.drop_after_s
        self._open = [c for c in self._open if c.last_t >= cutoff]
