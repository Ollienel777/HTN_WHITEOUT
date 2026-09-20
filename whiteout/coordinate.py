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

from whiteout.belief.encode import belief_frame
from whiteout.belief.field import BeliefError, BeliefField
from whiteout.belief.geometry import DEFAULT_STRAIT, StraitGeometry
from whiteout.belief.grid import ChannelBeliefGrid
from whiteout.belief.negative import (
    CLASS_CAMERAS,
    DEFAULT_SWEEP,
    SweepParams,
    non_detection_likelihood,
)
from whiteout.geo import GeoPoint
from whiteout.policy import DEFAULT_SEARCH_PARAMS, AssetRole, SearchParams, SearchPolicy
from whiteout.tracks.maintain import Sighting, TrackHold
from whiteout.types import (
    BeliefDigest,
    BeliefFrame,
    Contact,
    FleetIntent,
    Pose,
    SightingRefusal,
    WorldObservation,
)
from whiteout.vision.camera import CAMERAS, VisionError
from whiteout.vision.projection import CameraPose, ProjectionError
from whiteout.vision.standoff import standoff_point, standoff_range_m

__all__ = [
    "DEFAULT_DETECTION_SIGMA_M",
    "DEFAULT_HOLD_ASSET",
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

#: Which asset holds a contact, and whose camera the stand-off is computed
#: for. The quadcopter, because it is the only one that can stop and stare.
DEFAULT_HOLD_ASSET = "quadcopter"


class SightingSource(Protocol):
    """Whatever saw the vessel this tick — and whatever it would not stand behind.

    **Both methods, not one.** ``refusals`` was briefly optional, discovered on
    the object, so that a source with nothing to refuse could stay a single
    method. That is the shape in which a *decorator* — the motion gate, which
    wraps a source and is deliberately invisible to this loop — silently
    swallows every refusal the source made, and the episode log goes back to
    showing a dark camera as an empty sea. Nothing fails; the log is just
    quietly empty again, which is the failure this whole seam exists to end.
    So the contract asks for both, every source in ``whiteout/`` answers both,
    and ``tests/test_coordinate.py`` fails the gate if one stops.
    """

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        """Zero or more sightings, in the observation's timebase."""

    def refusals(self) -> tuple[SightingRefusal, ...]:
        """What the last ``sightings`` call dropped, and why.

        Empty is the normal answer, and is what a source with nothing to
        refuse returns for ever. A source that *wraps* another forwards the
        inner source's refusals; swallowing them makes a camera's silence
        unreadable.
        """


class NoSightings:
    """Sees nothing, ever.

    The default, and the honest one: under it the loop is a pure search. It is
    also what the run genuinely looks like before first contact, so it is not
    a degraded mode — it is the first half of every episode.
    """

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        return ()

    def refusals(self) -> tuple[SightingRefusal, ...]:
        """Nothing was refused, because nothing was looked at."""
        return ()


@dataclass(frozen=True)
class TickOutcome:
    """What one tick decided, in the shapes the episode log wants.

    ``refusals`` is what the sighting source declined to use and why, straight
    onto the episode record. A source that refuses silently would leave a
    camera's going dark looking exactly like an empty sea, which is the one
    thing worse than an unqualified fix.

    ``field`` is the belief field quantised for drawing (#124), or ``None``
    from a field this build cannot encode. It is the digest's large sibling:
    the digest says how sharp the belief is, this says where it is.
    """

    intent: FleetIntent
    digest: BeliefDigest
    contacts: tuple[Contact, ...]
    posted: bool
    track_state: str
    refusals: tuple[SightingRefusal, ...] = ()
    field: BeliefFrame | None = None


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
        hold_asset: str = DEFAULT_HOLD_ASSET,
        hold_camera: str = "quadcopter",
        ground_alt_m: float = 0.0,
        sweep: SweepParams = DEFAULT_SWEEP,
    ) -> None:
        self._geometry = geometry
        self._policy = SearchPolicy(roles, geometry, params)
        self._belief: BeliefField = belief if belief is not None else ChannelBeliefGrid(geometry)
        self._hold = hold
        self._sightings = sightings if sightings is not None else NoSightings()
        self._detection_sigma_m = float(detection_sigma_m)
        self._hold_asset = hold_asset
        self._hold_camera = hold_camera
        self._ground_alt_m = float(ground_alt_m)
        self._sweep = sweep
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
        # Also the exposure the non-detection update is credited with, so
        # that evidence and diffusion are balanced in the same units. On the
        # first tick there is no gap to measure and one reference look is the
        # honest credit: the asset has been looking, we just cannot say for
        # how long.
        exposure = self._sweep.reference_s
        if self._last_t is not None:
            elapsed = now - self._last_t
            if elapsed > 0.0:
                self._belief.diffuse(elapsed)
                exposure = elapsed
        self._last_t = now

        poses = {pose.asset_id: pose for pose in observation.poses}
        seen = self._sightings.sightings(observation)
        refusals = self._refusals()
        for sighting in seen:
            self._belief.update_detection(
                sighting.lat_deg, sighting.lon_deg, sigma_m=self._detection_sigma_m
            )
            if self._hold is not None:
                self._hold.sight(sighting)
        # Non-detections, #13. Every asset that looked and did not report is
        # evidence about the water it was looking at, and it is applied after
        # the sightings so that an asset which *did* see something is not also
        # asked to argue the vessel is not there.
        self._erode(observation, {sighting.asset_id for sighting in seen}, exposure)

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
                # Not the contact's own position: the quadcopter's camera is
                # fixed at the airframe's attitude and looks at the horizon
                # (#109), so hovering over a contact puts it straight down
                # where the camera cannot see. Stand off instead.
                point = self._hold_point(last.lat_deg, last.lon_deg, poses)
                held_lat, held_lon = point.lat_deg, point.lon_deg

        intent = self._policy.decide(
            observation, self._belief, held_lat_deg=held_lat, held_lon_deg=held_lon
        )
        return TickOutcome(
            intent=intent,
            digest=self._digest(now),
            contacts=self._contacts(now, state),
            posted=posted,
            track_state=state,
            refusals=refusals,
            field=self._field(now),
        )

    def _refusals(self) -> tuple[SightingRefusal, ...]:
        """What the source declined this tick.

        :class:`SightingSource` asks every source for this, and the shipped
        ones are held to it by a test. The ``getattr`` is tolerance for a
        hand-written double in somebody's test that predates the method, not a
        second contract: a source in ``whiteout/`` that quietly lost
        ``refusals`` would take the episode log's only record of a dark camera
        with it, so that case is caught at the gate rather than absorbed here.
        """
        refusals = getattr(self._sightings, "refusals", None)
        if not callable(refusals):
            return ()
        return tuple(refusals())

    def _erode(
        self, observation: WorldObservation, saw_something: set[str], elapsed_s: float
    ) -> None:
        """Fold every asset's non-detection into the field, #13.

        An asset contributes when it reported no sighting this tick **and**
        its pose carries the attitude a footprint needs. A pose with no pitch
        is skipped rather than assumed level: ``Pose.pitch`` is ``None``
        precisely when the transport does not know it, and treating unknown
        as level would point the camera at the horizon and erode a band of
        water nobody looked at — a confident wrong answer, from a default.

        Failures are swallowed per asset, deliberately. A camera below the
        water plane or a likelihood that leaves no mass is a reason to ignore
        *that* sweep, not to end the episode: the alternative is one bad pose
        at tick 300 taking down a run that cannot be repeated.
        """
        for pose in observation.poses:
            if pose.asset_id in saw_something:
                continue
            if pose.pitch is None or pose.roll is None:
                continue
            camera = CAMERAS.get(CLASS_CAMERAS.get(pose.cls, ""))
            if camera is None:
                continue
            try:
                self._belief.update_likelihood(
                    non_detection_likelihood(
                        camera,
                        CameraPose(
                            lat_deg=pose.lat,
                            lon_deg=pose.lon,
                            alt_m=pose.z,
                            yaw_deg=pose.heading,
                            pitch_deg=pose.pitch,
                            roll_deg=pose.roll,
                        ),
                        ground_alt_m=self._ground_alt_m,
                        elapsed_s=elapsed_s,
                        params=self._sweep,
                    )
                )
            except (BeliefError, VisionError, ProjectionError):
                continue

    def _hold_point(self, lat_deg: float, lon_deg: float, poses: dict[str, Pose]) -> GeoPoint:
        """Where the holding asset should sit to keep the contact in frame.

        Falls back to the contact's own position when the geometry cannot be
        computed — the holding asset has not reported, or its camera cannot
        see the water from where it is. Sitting on top of a contact is a poor
        place to watch from; it is still a better instruction than none, and
        the alternative would be to stop holding a contact we can see.
        """
        contact = GeoPoint(lat_deg, lon_deg)
        pose = poses.get(self._hold_asset)
        camera = CAMERAS.get(self._hold_camera)
        if pose is None or camera is None:
            return contact
        try:
            range_m = standoff_range_m(
                camera,
                pose.z,
                boresight_pitch_deg=pose.pitch if pose.pitch is not None else 0.0,
            )
            return standoff_point(contact, GeoPoint(pose.lat, pose.lon), range_m)
        except VisionError:
            return contact

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

    def _field(self, now: float) -> BeliefFrame | None:
        """The belief field, quantised for the viewer to draw (#124).

        ``None`` from a field this module has no encoder for. The encoding is
        a fact about the grid's cells — which of them are water, and where
        their corners are on the map — so it belongs to the implementation
        and not to :class:`~whiteout.belief.field.BeliefField`, which is
        written in lat/lon and metres precisely so the policy never learns
        that cells exist.

        The layout rides on the first tick's frame and no other; see
        :class:`~whiteout.types.BeliefGeometry` for what that buys.
        """
        if not isinstance(self._belief, ChannelBeliefGrid):
            return None
        return belief_frame(self._belief, now, include_geometry=self._ticks == 1)

    def _contacts(self, now: float, state: str) -> tuple[Contact, ...]:
        """The held track, as the log's contact record.

        ``TrackHold``'s states and ``CONTACT_STATES`` are different
        vocabularies on purpose — one is about whether we can still see the
        vessel, the other is the log's lifecycle — so the mapping is written
        out rather than assumed to coincide.

        The last sighting's ``sync`` rides along with its position, because
        that is whose position this is: the contact's lat/lon is the last
        fix's, so the qualification on that fix is the qualification on this
        record. ``None`` when the sighting source did not establish one — it is
        never filled in with a synchronised-looking default here.
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
                sync=last.sync,
            ),
        )
