"""The run loop: the one place every part is joined.

Every piece of this project worked on its own and none of them were joined.
The link brought poses in (#23), the policy decided where to go (#96), the
detector found the vessel in a frame (#66), the projection turned a pixel into
a lat/lon (#65), the hold kept a track alive (#67) and the client posted it
(#64) — and nothing called any of them in order, so the fleet sat disarmed and
the chain never started. This is that order.

One tick
---------

.. code-block:: text

    observe                 poses in, from the transport
      -> age the belief     dt seconds of the vessel's motion
      -> sightings          whatever saw the vessel this tick, if anything
      -> fold them in       belief update, and the track hold
      -> post               the hold decides whether there is anything new
      -> decide             the policy, given belief and any held contact
      -> command            waypoints out, through the transport

Where the seam sits, and why the coordinator is not the transport's problem
----------------------------------------------------------------------------

:class:`Coordinator` never touches the transport. It takes a
:class:`~whiteout.types.WorldObservation` and returns a
:class:`TickOutcome`; the caller does the I/O. That keeps ``SPEC.md`` §4's
seam intact from the other side — the policy and the belief cannot reach the
socket even by accident — and it is what lets the whole loop be tested against
a list of observations with no arena, no network and no clock.

Sightings are injected, not fetched
------------------------------------

A :class:`SightingSource` is asked what saw the vessel this tick. The default
is :class:`NoSightings`, which sees nothing, and under it the loop is a pure
search: the fleet sweeps the channel and posts nothing, which is the correct
behaviour when nothing has been detected and is exactly what the run does
before first contact.

That is a seam and not a placeholder. The vision path needs a JPEG decoder
that ``pyproject.toml`` does not declare (``whiteout/vision/imagery.py`` says
so, and calls it a dependency decision it will not take on its own), so a loop
that reached for frames itself would be a loop that could not run in a clean
checkout. Injecting the source means the search half runs everywhere, the
vision half plugs in where a decoder exists, and neither has to pretend about
the other.

**Nothing here posts a guess.** The hold decides what reaches the tracks API,
and it posts a fix only when it is new (#67). A tick with no sighting posts
nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from whiteout.belief.field import BeliefField
from whiteout.belief.geometry import DEFAULT_STRAIT, StraitGeometry
from whiteout.belief.grid import ChannelBeliefGrid
from whiteout.policy import DEFAULT_SEARCH_PARAMS, AssetRole, SearchParams, SearchPolicy
from whiteout.tracks.maintain import Sighting, TrackHold
from whiteout.types import BeliefDigest, Contact, FleetIntent, WorldObservation

__all__ = [
    "DEFAULT_DETECTION_SIGMA_M",
    "DEFAULT_TRACK_NAME",
    "Coordinator",
    "NoSightings",
    "SightingSource",
    "TickOutcome",
]

#: The name every fix is posted under. The tracks API keys on it, so it is the
#: difference between one long track and a great many one-fix ones (#64).
DEFAULT_TRACK_NAME = "Sierra One"

#: One-sigma position error folded into the belief on a sighting, metres. A
#: standing assumption until the projection's error is measured against the
#: arena, and a parameter rather than a constant for that reason.
DEFAULT_DETECTION_SIGMA_M = 150.0


class SightingSource(Protocol):
    """Whatever saw the vessel this tick."""

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        """Zero or more sightings, in the observation's timebase."""


class NoSightings:
    """Sees nothing, ever.

    The default, and the honest one: under it the loop is a pure search. It is
    also what the run genuinely looks like before first contact, so it is not
    a degraded mode — it is the first half of every episode.
    """

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        return ()


@dataclass(frozen=True)
class TickOutcome:
    """What one tick decided, in the shapes the episode log wants."""

    intent: FleetIntent
    digest: BeliefDigest
    contacts: tuple[Contact, ...]
    posted: bool
    track_state: str


class Coordinator:
    """Belief, policy and the track hold, driven one tick at a time.

    Deterministic: the same sequence of observations and sightings gives the
    same sequence of outcomes. There is no clock in here — ``dt`` comes from
    the observations' own ``t``, which the transport owns.
    """

    def __init__(
        self,
        roles: tuple[AssetRole, ...],
        geometry: StraitGeometry = DEFAULT_STRAIT,
        *,
        params: SearchParams = DEFAULT_SEARCH_PARAMS,
        belief: BeliefField | None = None,
        hold: TrackHold | None = None,
        sightings: SightingSource | None = None,
        detection_sigma_m: float = DEFAULT_DETECTION_SIGMA_M,
    ) -> None:
        self._geometry = geometry
        self._policy = SearchPolicy(roles, geometry, params)
        self._belief: BeliefField = belief if belief is not None else ChannelBeliefGrid(geometry)
        self._hold = hold
        self._sightings = sightings if sightings is not None else NoSightings()
        self._detection_sigma_m = float(detection_sigma_m)
        self._last_t: float | None = None
        self._ticks = 0

    @property
    def policy(self) -> SearchPolicy:
        """The search policy, for a caller that wants its counters."""
        return self._policy

    @property
    def belief(self) -> BeliefField:
        """The belief field this loop is reasoning over."""
        return self._belief

    def tick(self, observation: WorldObservation) -> TickOutcome:
        """Age the belief, fold in what was seen, decide, and report."""
        self._ticks += 1
        now = observation.t

        # Age the field by the gap since the last tick, not by the clock.
        #
        # The first tick's guard is defensive rather than load-bearing, and
        # saying so beats implying a test protects it: a uniform field is a
        # fixed point of diffusion and the field is always uniform on tick
        # one, so ageing it by `now` would be unobservable. Mutating this
        # guard away leaves the whole suite green. It stays because it is
        # clearer and cheaper than depending on that coincidence, and because
        # it stops being a coincidence the moment a prior is seeded.
        if self._last_t is not None:
            elapsed = now - self._last_t
            if elapsed > 0.0:
                self._belief.diffuse(elapsed)
        self._last_t = now

        seen = self._sightings.sightings(observation)
        for sighting in seen:
            self._belief.update_detection(
                sighting.lat_deg, sighting.lon_deg, sigma_m=self._detection_sigma_m
            )
            if self._hold is not None:
                self._hold.sight(sighting)

        posted = False
        held_lat: float | None = None
        held_lon: float | None = None
        state = "unseen"
        if self._hold is not None:
            posted = self._hold.tick(now)
            state = self._hold.state
            last = self._hold.last_sighting
            # A coasting track still pulls the quadcopter: the last fix is
            # where to look, even when there is nothing new to post. Only a
            # track that is genuinely lost releases the asset back to search.
            if last is not None and state != "lost":
                held_lat, held_lon = last.lat_deg, last.lon_deg

        intent = self._policy.decide(
            observation, self._belief, held_lat_deg=held_lat, held_lon_deg=held_lon
        )
        return TickOutcome(
            intent=intent,
            digest=self._digest(now),
            contacts=self._contacts(now, state),
            posted=posted,
            track_state=state,
        )

    def _digest(self, now: float) -> BeliefDigest:
        peak = self._belief.peak()
        shape = getattr(self._belief, "shape", (0, 0))
        # The coverage the policy itself acts on, not the scorer's axis: the
        # log records what the coordinator believed, not what was scored.
        covered = self._policy.covered_fraction(now)
        return BeliefDigest(
            t=now,
            entropy=self._belief.entropy(),
            mass=self._belief.mass(),
            peak_lat=peak.lat_deg,
            peak_lon=peak.lon_deg,
            peak_p=peak.probability,
            covered_fraction=covered,
            grid_shape=(int(shape[0]), int(shape[1])),
        )

    def _contacts(self, now: float, state: str) -> tuple[Contact, ...]:
        """The held track, as the log's contact record.

        ``TrackHold``'s states and ``CONTACT_STATES`` are different
        vocabularies on purpose — one is about whether we can still see the
        vessel, the other is the log's lifecycle — so the mapping is written
        out rather than assumed to coincide.
        """
        if self._hold is None or self._hold.last_sighting is None:
            return ()
        last = self._hold.last_sighting
        mapped = {
            "held": "tracked",
            "coasting": "tracked",
            "lost": "lost",
            "unseen": "unconfirmed",
        }[state]
        return (
            Contact(
                contact_id=self._hold.name,
                t=now,
                state=mapped,
                lat=last.lat_deg,
                lon=last.lon_deg,
                confidence=0.0 if state == "lost" else 1.0,
                classification="vessel",
                assigned_asset_id=self._hold.holder,
            ),
        )
