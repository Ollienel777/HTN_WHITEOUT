"""The shadow vessel the fake fleet is looking for, and what sees it.

The `kinematic` transport is a fleet with nothing to find. It flies, the
belief field erodes, and the episode ends with the honest verdict that two of
five scored axes cannot be answered:

    detection_speed:    not detected -- no record of 400 carries a contact
    tracking_duration:  not detected -- there was never anything to hold

So the demo showed a search and never a find, while "found and held" is half
the project's own one-liner. This module is the other half: a target, and a
sensor model that decides whether an asset would have seen it.

Why it is not in the transport
-------------------------------

Because the seam forbids it, and the seam is right. ``whiteout/transport/``
is adapters only -- ``tests/test_transport.py`` walks the package and fails
any module that names a belief, policy, scoring or **ground-truth** concept.
A vessel whose real position is known is exactly ground truth, and a
``kinematic`` transport carrying one would be a fake that knows something the
real link never can.

So the vessel lives here, outside the seam, and reaches the loop the same way
the arena's cameras do: as a :class:`~whiteout.coordinate.SightingSource`.
``whiteout/cli.py`` picks one or the other. Nothing downstream can tell which
it got, which is the property that makes a rehearsal worth anything.

The detection model is the belief field's own
----------------------------------------------

An asset sees the vessel when
:func:`~whiteout.belief.negative.non_detection_likelihood` -- the same
function the field erodes with -- says it would, at the vessel's true
position. That is deliberate and it is the only honest choice available:

**If the sensor model that produces sightings disagreed with the one that
spends them, the fake would be tuning the policy against a world that does
not exist.** Belief would be erased from water a sighting could still come
out of, or held over water no camera could ever have covered, and every
number the demo quotes would be a number about the disagreement.

One consequence worth stating: **this fake cannot surprise us.** It will
never produce the detection the belief model says is impossible, so it
validates the loop's *plumbing* -- detection to contact to hold to POST --
and not the sensor model's fidelity. The arena is what tests that, and
``scripts/score_detector.py`` is what measures it.

False alarms are not modelled
------------------------------

Every sighting here is of the real vessel. ``detect.py`` has an entire
apparatus for refusing a floe shadow that looks like a hull, and none of it
is exercised by this module, so a clean run here says nothing about the false
positive rate. ``#90`` is where that lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint, StraitGeometry
from whiteout.belief.negative import (
    CLASS_CAMERAS,
    DEFAULT_SWEEP,
    SweepParams,
    non_detection_likelihood,
)
from whiteout.geo import ARENA_ORIGIN, GeoPoint, geodetic_to_local
from whiteout.tracks.maintain import Sighting
from whiteout.types import SightingRefusal, WorldObservation
from whiteout.vision.camera import CAMERAS
from whiteout.vision.projection import CameraPose

__all__ = [
    "DEFAULT_SPEED_MPS",
    "ShadowVessel",
    "ShadowSightings",
]

#: The vessel's speed, metres per second. **Read off the arena's own `.env`**,
#: which carries ``SHIP_SPEED=3.0`` alongside ``SHIP_MOVING=1`` and
#: ``SHIP_START=random``. Not a guess, and not the 2.96 m/s we once measured
#: over a short baseline -- that figure was a chord across an arc and so a
#: lower bound by construction.
DEFAULT_SPEED_MPS = 3.0

#: One-sigma error on a sighting's reported position, metres. A sighting is a
#: *measurement*, and one that landed exactly on the target every time would
#: make the track hold look better than it is and the belief field sharper
#: than it has earned. This is a standing assumption until the projection's
#: real error is measured against the arena.
DEFAULT_FIX_SIGMA_M = 60.0


#: Metres per degree of latitude at the strait, and how much shorter a degree
#: of longitude is there. Derived through :mod:`whiteout.geo`, the tree's only
#: converter, rather than spelt out: at 72 N a degree of longitude is under a
#: third of a degree of latitude, so a fix error that ignored that would be
#: three times too wide East-West.
_M_PER_DEG_LAT = geodetic_to_local(
    ARENA_ORIGIN, GeoPoint(ARENA_ORIGIN.lat_deg + 1.0, ARENA_ORIGIN.lon_deg)
).north_m
_LON_PER_LAT = (
    _M_PER_DEG_LAT
    / geodetic_to_local(
        ARENA_ORIGIN, GeoPoint(ARENA_ORIGIN.lat_deg, ARENA_ORIGIN.lon_deg + 1.0)
    ).east_m
)


@dataclass
class ShadowVessel:
    """One boat, walking the channel, with nowhere to go but water.

    The motion is a **random walk along the channel's own axis**, not across
    open water: ``ARENA.md`` §2 confines the target to a 2 km-wide strait, so
    a vessel's freedom is very nearly one-dimensional and modelling it in two
    would put it aground. It drifts across the centreline as well, reflected
    at the banks, because a vessel pinned to the centreline would make the
    across-channel half of the belief field untestable.

    Seeded, and a pure function of that seed: two runs at one seed put the
    vessel in the same water at the same tick, which is what lets the gate
    compare two episode logs byte for byte.
    """

    seed: int = 0
    geometry: StraitGeometry = DEFAULT_STRAIT
    speed_mps: float = DEFAULT_SPEED_MPS
    #: How sharply the heading wanders, radians of course change per second.
    #: Small: a boat under way holds a course, and a vessel that jittered
    #: would be found by accident rather than by searching well.
    turn_rate: float = 0.05

    def __post_init__(self) -> None:
        generator = np.random.default_rng(self.seed)
        self._rng = generator
        # `SHIP_START=random` in the arena's .env, so the fake starts random
        # too -- anywhere along the channel, anywhere across it.
        self._s_m = float(generator.uniform(0.0, self.geometry.length_m))
        half = self.geometry.half_width_at(self._s_m)
        self._w_m = float(generator.uniform(-half, half))
        self._along = 1.0 if generator.random() < 0.5 else -1.0
        self._drift = float(generator.uniform(-0.4, 0.4))
        self._t = 0.0

    @property
    def t(self) -> float:
        """The vessel's own clock, seconds since the episode began."""
        return self._t

    def position(self) -> GeoPoint:
        """Where the vessel is now, in the one frame of record."""
        lat, lon = self.geometry.to_position(ChannelPoint(s_m=self._s_m, w_m=self._w_m))
        return GeoPoint(lat, lon)

    def advance(self, dt_s: float) -> None:
        """Walk the vessel ``dt_s`` seconds along the channel."""
        if dt_s <= 0.0:
            return
        self._t += dt_s
        self._drift += float(self._rng.normal(0.0, self.turn_rate)) * dt_s
        self._drift = max(-0.9, min(0.9, self._drift))
        along = math.sqrt(max(0.0, 1.0 - self._drift * self._drift))
        self._s_m += self._along * along * self.speed_mps * dt_s
        self._w_m += self._drift * self.speed_mps * dt_s
        # Reflect at the ends rather than wrap: a vessel that teleported from
        # one end of the strait to the other would hand the belief field an
        # impossible jump and make tracking accuracy meaningless.
        if self._s_m < 0.0:
            self._s_m = -self._s_m
            self._along = 1.0
        elif self._s_m > self.geometry.length_m:
            self._s_m = 2.0 * self.geometry.length_m - self._s_m
            self._along = -1.0
        half = self.geometry.half_width_at(self._s_m)
        if abs(self._w_m) > half:
            self._w_m = math.copysign(2.0 * half - abs(self._w_m), self._w_m)
            self._drift = -self._drift


@dataclass
class ShadowSightings:
    """A :class:`~whiteout.coordinate.SightingSource` over a :class:`ShadowVessel`.

    Each tick, every asset that could see the vessel reports it, with a
    position error. "Could see" is
    :func:`~whiteout.belief.negative.non_detection_likelihood` evaluated at
    the vessel's own position: the module docstring says why it must be that
    function and not another.

    :param vessel: the target. It is advanced by this source, from the
        observation's own clock, so nothing else has to remember to.
    :param fix_sigma_m: one-sigma error on a reported position.
    :param ground_alt_m: the water plane in the poses' datum. Issue #76.
    """

    vessel: ShadowVessel
    fix_sigma_m: float = DEFAULT_FIX_SIGMA_M
    ground_alt_m: float = 0.0
    params: SweepParams = DEFAULT_SWEEP
    seed: int = 0

    def __post_init__(self) -> None:
        # A generator of its own, not the vessel's: whether a look succeeds
        # must not perturb where the vessel goes next, or the target's path
        # would depend on who was watching it.
        self._rng = np.random.default_rng(self.seed + 1)
        self._last_t: float | None = None
        self._refusals: tuple[SightingRefusal, ...] = ()

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        """Every asset that sees the vessel this tick, and where it says it is."""
        if self._last_t is not None and observation.t > self._last_t:
            self.vessel.advance(observation.t - self._last_t)
        self._last_t = observation.t

        truth = self.vessel.position()
        found: list[Sighting] = []
        for pose in observation.poses:
            if pose.pitch is None or pose.roll is None:
                continue
            camera = CAMERAS.get(CLASS_CAMERAS.get(pose.cls, ""))
            if camera is None or pose.z - self.ground_alt_m <= 0.0:
                continue
            likelihood = non_detection_likelihood(
                camera,
                CameraPose(
                    lat_deg=pose.lat,
                    lon_deg=pose.lon,
                    alt_m=pose.z,
                    yaw_deg=pose.heading,
                    pitch_deg=pose.pitch,
                    roll_deg=pose.roll,
                ),
                ground_alt_m=self.ground_alt_m,
                params=self.params,
            )
            detected = 1.0 - likelihood(truth.lat_deg, truth.lon_deg)
            if detected <= 0.0 or self._rng.random() >= detected:
                continue
            found.append(
                Sighting(
                    t=observation.t,
                    lat_deg=truth.lat_deg + self._offset(),
                    lon_deg=truth.lon_deg + self._offset() / _LON_PER_LAT,
                    asset_id=pose.asset_id,
                )
            )
        return tuple(found)

    def refusals(self) -> tuple[SightingRefusal, ...]:
        """Nothing is refused: this source has no frame to distrust.

        Not an oversight and not a stub. ``VisionSightings`` refuses a fix
        whose frame and pose were too far apart in time, and a fake that
        invented refusals would put a qualifier in the log that nothing
        measured.
        """
        return ()

    def _offset(self) -> float:
        """One axis of the fix error, in degrees of latitude."""
        return float(self._rng.normal(0.0, self.fix_sigma_m)) / _M_PER_DEG_LAT
