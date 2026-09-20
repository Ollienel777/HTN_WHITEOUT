"""The world the offline episode runs against.

``hackathon/SPEC.md`` §5. This is the stand-in that lets the whole loop —
observe, believe, decide, track, post — close with no arena, no network and no
imagery. :class:`~whiteout.transport.arena.ArenaTransport` is the real path and
this is not a model of it; what this owes the arena is only that the *shape* of
what comes out of :meth:`ArenaSim.reports` is the shape a real sensor produces,
so that nothing downstream has to be rewritten when the link comes up.

**Everything here is seeded and nothing here is wall-clock.** The same seed and
the same intents produce the same episode, byte for byte, because the gate
(``SPEC.md`` §6) runs the smoke twice and diffs the logs.

What is modelled, and what is deliberately not
----------------------------------------------

``the vessel``
    One target (``ARENA.md`` §3), moving along the channel centreline at
    :data:`VESSEL_SPEED_MPS` and turning round at the ends. It does not evade,
    it does not stop, and it does not leave the water. The belief field's
    diffusion is written against a random walk at this speed, so the two agree
    by construction — which means an episode here measures the *policy* and the
    *plumbing*, and cannot be evidence that the motion model is right.

``the fleet``
    Assets fly straight at whatever waypoint the last :class:`FleetIntent` gave
    them, at their class's speed, and stop when they arrive. No turn radius, no
    climb rate, no wind. A fixed-wing that cannot actually hold a 90° turn is
    the single biggest lie in here, and it flatters the policy: real coverage
    will be worse than an episode from this suggests.

``the sensors``
    A tower sees a disc about itself; an aircraft sees a cone along its
    heading. Detection inside that footprint is a coin weighted by range, and a
    detection's reported position is the truth plus Gaussian error that grows
    with range. There is no imagery, no ice and no fog: the detector in
    :mod:`whiteout.vision.detect` is what reads pixels, and issue #90 is why
    its measured rates are not the rates used here.

``what is not here``
    Ice, weather, terrain occlusion, other vessels, comms loss, battery limits.
    Each would make the episode harder and none of them are what the loop needs
    proving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint, StraitGeometry
from whiteout.geo import GeoPoint, LocalPoint, bearing_deg, geodetic_to_local, local_to_geodetic
from whiteout.transport.base import TransportError
from whiteout.types import (
    Detection,
    FleetIntent,
    Pose,
    SensorFootprint,
    SensorReport,
    TargetTruth,
    Truth,
    WaypointIntent,
    WorldObservation,
)

__all__ = [
    "DETECTION_SIGMA_FLOOR_M",
    "SimTransport",
    "FLEET",
    "SENSORS",
    "SPEEDS",
    "VESSEL_LENGTH_M",
    "VESSEL_SPEED_MPS",
    "ArenaSim",
    "SensorModel",
    "detection_sigma_m",
    "footprint_range_m",
]

#: The fleet this world runs, ``(asset_id, cls, altitude)``. The same roster
#: the kinematic stub uses, so an episode here and an episode there name the
#: same assets and a log from one can be compared with a log from the other.
FLEET: tuple[tuple[str, str, float], ...] = (
    ("fw-1", "fixedwing", 120.0),
    ("quad-1", "quad", 40.0),
    ("rover-1", "rover", 0.0),
    ("tower-1", "tower", 15.0),
)

#: The shadow vessel's speed along the channel, metres per second. Settled by
#: the operator on 2026-09-19 against the arena's own track samples: the deck's
#: 6.5 kn is a figure about the hull, and a chord between two fixes is a lower
#: bound on the arc it travelled, so 3.0 m/s is what the belief field diffuses
#: at and what this moves at.
VESSEL_SPEED_MPS = 3.0

#: Cruise speed by vehicle class, metres per second. ``ARENA.md`` §3. A tower
#: has no speed because it does not move; it is aimed.
SPEEDS: dict[str, float] = {
    "fixedwing": 22.0,
    "quad": 12.0,
    "rover": 2.5,
    "tower": 0.0,
}

#: Joules-equivalent per second of flight, by class, for the efficiency axis.
#: Relative numbers with no unit behind them — a quadcopter holding station
#: costs more per second than a wing covering ground, which is the only
#: ordering the scorer reads.
_ENERGY_PER_SECOND: dict[str, float] = {
    "fixedwing": 1.0,
    "quad": 1.8,
    "rover": 0.4,
    "tower": 0.05,
}

#: The stretch of channel the vessel spawns in, as fractions of its length.
#: Away from where the aircraft launch, so that "time to first fix" is a number
#: about the search and not about a lucky spawn; wide enough that the seed
#: genuinely changes the episode rather than only the detection coins.
_VESSEL_START_RANGE: tuple[float, float] = (0.55, 0.95)

#: Where the towers stand, as fractions along the channel. Spread so their
#: discs overlap near the middle and neither end is unwatched; a real
#: deployment surveys these and ``ARENA.md`` does not fix them, so a roster
#: with more towers than this cycles through the list.
_TOWER_FRACTIONS: tuple[float, ...] = (0.30, 0.70)

#: Floor on a detection's one-sigma position error, metres. A fix is never
#: better than this however close the asset is: it is the pixel quantisation
#: and the pose error that do not go away with range.
DETECTION_SIGMA_FLOOR_M = 12.0

#: Growth of that error with range: one sigma is this fraction of the ground
#: range, added in quadrature to the floor. It stands for an attitude error of
#: about half a degree, which is what an uncalibrated mount gives you.
_SIGMA_PER_RANGE = 0.009


#: How long the shadow vessel is, metres. ``ARENA.md`` §3 says "a boat" and no
#: more, so this is the one number here with no source behind it; 20 m is a
#: small coastal craft. It matters because detection is a question about how
#: many pixels the hull covers, and that is length over range.
VESSEL_LENGTH_M = 20.0

#: Hull length, in pixels, at which a detector finds the vessel half the time.
#: Not measured against arena imagery -- nothing here has seen any -- but it is
#: the right *shape* of assumption: a detector fails by running out of pixels,
#: not by running out of metres, which is why reach below is derived rather
#: than chosen.
_HALF_DETECT_PX = 6.0

#: How sharply that probability turns from "always" to "never", in pixels.
_DETECT_SOFTNESS_PX = 1.5

#: Hull length, in pixels, below which the sensor is treated as blind. Two
#: pixels is a couple of dark samples on textured water and no detector should
#: be believed there. It sets each sensor's reach.
_BLIND_BELOW_PX = 2.0


@dataclass(frozen=True, slots=True)
class SensorModel:
    """What one vehicle class can see, derived from its published camera.

    ``ARENA.md`` §4 publishes each camera's horizontal field of view and
    width, and those two give the only thing that decides whether a detector
    can work: **how many pixels the hull covers at a range**. A camera that
    spreads 60 degrees over 640 px resolves 0.094 degrees per pixel, so a 20 m
    hull at 4 km is three pixels across, and no amount of algorithm turns
    three pixels into a confident fix.

    Everything else about the sensor falls out of that. :attr:`reach_m` is not
    a choice, it is the range at which the hull drops below
    :data:`_BLIND_BELOW_PX`; :meth:`p_detect_at` is a logistic in pixel extent
    rather than in metres.

    ``p_max`` is the ceiling when the target is large in frame, and it is
    below one on purpose: a hull can be behind a floe, in glare, or at the
    edge of a rolling frame, and a sensor that never misses when the geometry
    is right would make the negative-information update meaningless.
    """

    kind: str
    hfov_deg: float
    width_px: float
    half_angle_rad: float
    p_max: float

    @property
    def px_per_rad(self) -> float:
        """Pixels per radian across the frame."""
        return self.width_px / math.radians(self.hfov_deg)

    @property
    def reach_m(self) -> float:
        """Range at which the hull falls below :data:`_BLIND_BELOW_PX`."""
        return VESSEL_LENGTH_M * self.px_per_rad / _BLIND_BELOW_PX

    def pixels_at(self, range_m: float) -> float:
        """Hull length in pixels at a ground range."""
        if range_m <= 1e-6:
            return float("inf")
        return VESSEL_LENGTH_M * self.px_per_rad / range_m

    def p_detect_at(self, range_m: float) -> float:
        """Chance of reporting the vessel this tick, at a ground range."""
        if range_m >= self.reach_m:
            return 0.0
        pixels = self.pixels_at(range_m)
        return self.p_max / (1.0 + math.exp(-(pixels - _HALF_DETECT_PX) / _DETECT_SOFTNESS_PX))


#: Sensor by vehicle class, built from the intrinsics ``ARENA.md`` §4 publishes.
#: A rover has no camera in this build and so is absent: it sees nothing and
#: files no report, rather than filing an empty one that would erode the belief
#: field on the strength of a sensor that does not exist.
SENSORS: dict[str, SensorModel] = {
    # The aircraft look where they are going, so their footprint is a cone of
    # the camera's own half-angle. Reach falls out of the optics: the wing
    # sees a 20 m hull as two pixels at 5.3 km, the quad only at 3.2 km --
    # its very wide lens buys a view of the water below and costs it range.
    "fixedwing": SensorModel("cone", 69.0, 640.0, math.radians(69.0 / 2.0), 0.85),
    "quad": SensorModel("cone", 114.6, 640.0, math.radians(114.6 / 2.0), 0.90),
    # A tower's camera pans, so across a tick it is treated as seeing all
    # round rather than as a cone that has to be aimed correctly by luck. On
    # pixels alone it reaches 6.1 km, further than this channel is long --
    # which is precisely why towers alone cannot do the job: at 3 km the hull
    # is four pixels and the coin comes up tails more often than not.
    "tower": SensorModel("circle", 60.0, 640.0, 0.0, 0.70),
}


def footprint_range_m(footprint: SensorFootprint, lat_deg: float, lon_deg: float) -> float | None:
    """Ground range from a footprint's sensor to a position, or ``None`` outside it.

    The one answer to "could this sensor have seen something there", asked by
    the sim when it decides whether the vessel is visible and by the episode
    runner when it works out what a sweep that saw nothing rules out. They
    have to agree exactly or the belief field erodes water the sensor never
    looked at -- which is a confident wrong answer, arrived at by arithmetic
    that was only nearly the same in two places.
    """
    offset = geodetic_to_local(GeoPoint(footprint.lat, footprint.lon), GeoPoint(lat_deg, lon_deg))
    range_m = math.hypot(offset.east_m, offset.north_m)
    if range_m > footprint.radius:
        return None
    if footprint.kind == "cone" and footprint.half_angle > 0.0:
        bearing = math.atan2(offset.east_m, offset.north_m)
        delta = abs((bearing - footprint.heading + math.pi) % (2.0 * math.pi) - math.pi)
        if delta > footprint.half_angle:
            return None
    return range_m


def detection_sigma_m(range_m: float) -> float:
    """One-sigma position error for a fix taken at ``range_m`` ground range."""
    return math.hypot(DETECTION_SIGMA_FLOOR_M, _SIGMA_PER_RANGE * max(0.0, range_m))


@dataclass
class _Asset:
    """One vehicle's mutable state inside the sim."""

    asset_id: str
    cls: str
    lat: float
    lon: float
    z: float
    heading: float
    speed: float
    energy: float


class ArenaSim:
    """A deterministic channel, one vessel in it, and a fleet looking for it.

    ``seed`` seeds every draw. ``tick_seconds`` is the world time one
    :meth:`step` advances. The geometry is the same
    :class:`~whiteout.belief.geometry.StraitGeometry` the belief field and the
    search policy use, so "on the water" means one thing across the run.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        tick_seconds: float = 0.5,
        geometry: StraitGeometry = DEFAULT_STRAIT,
        fleet: tuple[tuple[str, str, float], ...],
        vessel_speed_mps: float = VESSEL_SPEED_MPS,
    ) -> None:
        self._geometry = geometry
        self._tick_seconds = float(tick_seconds)
        self._rng = np.random.default_rng(seed)
        self._vessel_speed = float(vessel_speed_mps)
        self._t = 0.0

        # **The spawn is drawn, not fixed.** ``ARENA.md`` §3 says the vessel
        # spawns at random, and an episode whose target always starts in the
        # same place measures one geometry rather than the policy: the seed
        # would move the detection coins and nothing else, and two seeds could
        # produce the same log. It is drawn from the far half of the channel
        # so that "time to first fix" stays a number about the search rather
        # than about how close the vessel happened to spawn to the runway.
        low, high = _VESSEL_START_RANGE
        self._vessel_s = float(self._rng.uniform(low, high)) * geometry.length_m
        self._vessel_w = 0.0
        self._vessel_forward = bool(self._rng.random() < 0.5)

        self._assets: list[_Asset] = []
        towers = 0
        movers = 0
        for asset_id, cls, altitude in fleet:
            # Aircraft launch together from one end -- there is one operator
            # and one place to stand -- while towers are surveyed structures
            # spread along the channel to overlap. Placing the towers beside
            # the aircraft would leave the far half of the water unwatched and
            # make every episode a story about transit time.
            if cls == "tower":
                fraction = _TOWER_FRACTIONS[towers % len(_TOWER_FRACTIONS)]
                towers += 1
            else:
                fraction = 0.03 + 0.02 * movers
                movers += 1
            lat, lon = geometry.to_position(ChannelPoint(s_m=fraction * geometry.length_m, w_m=0.0))
            self._assets.append(
                _Asset(
                    asset_id=asset_id,
                    cls=cls,
                    lat=lat,
                    lon=lon,
                    z=altitude,
                    heading=0.0,
                    speed=0.0,
                    energy=0.0,
                )
            )

    @property
    def t(self) -> float:
        """World time, seconds since the episode began."""
        return self._t

    @property
    def tick_seconds(self) -> float:
        """How much world time one :meth:`step` advances."""
        return self._tick_seconds

    def vessel_position(self) -> tuple[float, float]:
        """Where the vessel truly is. Only :meth:`truth` may show this on."""
        return self._geometry.to_position(ChannelPoint(s_m=self._vessel_s, w_m=self._vessel_w))

    def poses(self) -> tuple[Pose, ...]:
        """This instant's fleet, in the seam's own type."""
        return tuple(
            Pose(
                asset_id=asset.asset_id,
                cls=asset.cls,
                t=self._t,
                lat=asset.lat,
                lon=asset.lon,
                z=asset.z,
                heading=asset.heading,
                speed=asset.speed,
                energy_used=asset.energy,
            )
            for asset in self._assets
        )

    def truth(self) -> Truth:
        """The vessel's true state. Written to the log, never observed."""
        lat, lon = self.vessel_position()
        heading = 0.0 if self._vessel_forward else math.pi
        return Truth(
            t=self._t,
            targets=(
                TargetTruth(
                    target_id="shadow-1",
                    lat=lat,
                    lon=lon,
                    z=0.0,
                    heading=heading,
                    speed=self._vessel_speed,
                    target_class="vessel",
                ),
            ),
        )

    def reports(self) -> tuple[SensorReport, ...]:
        """What every sensor saw this instant.

        One report per asset that carries a sensor, each with the footprint it
        swept. ``negative`` is true whenever that footprint was swept and the
        vessel was not reported in it — including when the vessel *was* inside
        it and the coin came up a miss, which is the whole reason the negative
        update has to be a likelihood rather than a hard cut.
        """
        vessel = GeoPoint(*self.vessel_position())
        out: list[SensorReport] = []
        for asset in self._assets:
            sensor = SENSORS.get(asset.cls)
            if sensor is None:
                continue
            footprint = SensorFootprint(
                kind=sensor.kind,
                lat=asset.lat,
                lon=asset.lon,
                radius=sensor.reach_m,
                heading=asset.heading if sensor.kind == "cone" else 0.0,
                half_angle=sensor.half_angle_rad,
            )
            detections: tuple[Detection, ...] = ()
            range_m = footprint_range_m(footprint, vessel.lat_deg, vessel.lon_deg)
            if range_m is not None and self._rng.random() < sensor.p_detect_at(range_m):
                sigma = detection_sigma_m(range_m)
                east, north = self._rng.normal(0.0, sigma, size=2)
                fix = local_to_geodetic(vessel, LocalPoint(float(east), float(north)))
                detections = (
                    Detection(
                        detection_id=f"{asset.asset_id}-{self._t:.1f}",
                        lat=fix.lat_deg,
                        lon=fix.lon_deg,
                        confidence=sensor.p_detect_at(range_m),
                        classification="vessel",
                    ),
                )
            out.append(
                SensorReport(
                    asset_id=asset.asset_id,
                    t=self._t,
                    footprint=footprint,
                    detections=detections,
                    negative=not detections,
                )
            )
        return tuple(out)

    def step(self, intents: tuple[WaypointIntent, ...]) -> None:
        """Advance the world one tick: the vessel moves, then the fleet does."""
        dt = self._tick_seconds
        self._advance_vessel(dt)
        wanted = {intent.asset_id: intent for intent in intents}
        for asset in self._assets:
            self._advance_asset(asset, wanted.get(asset.asset_id), dt)
        self._t += dt

    # -- internals ----------------------------------------------------------

    def _advance_vessel(self, dt: float) -> None:
        step = self._vessel_speed * dt
        self._vessel_s += step if self._vessel_forward else -step
        length = self._geometry.length_m
        if self._vessel_s >= length:
            self._vessel_s = length - (self._vessel_s - length)
            self._vessel_forward = False
        elif self._vessel_s <= 0.0:
            self._vessel_s = -self._vessel_s
            self._vessel_forward = True

    def _advance_asset(self, asset: _Asset, intent: WaypointIntent | None, dt: float) -> None:
        cruise = SPEEDS.get(asset.cls, 0.0)
        asset.energy += _ENERGY_PER_SECOND.get(asset.cls, 0.0) * dt
        if intent is None:
            asset.speed = 0.0
            return
        here = GeoPoint(asset.lat, asset.lon)
        target = GeoPoint(intent.target_lat, intent.target_lon)
        # A tower cannot go anywhere, so its intent is an aiming order: it
        # turns to face the waypoint and stays where it is.
        asset.heading = math.radians(bearing_deg(here, target))
        if cruise <= 0.0:
            asset.speed = 0.0
            return
        offset = geodetic_to_local(here, target)
        distance = math.hypot(offset.east_m, offset.north_m)
        reach = min(cruise * dt, distance)
        if distance <= 1e-6:
            asset.speed = 0.0
            return
        moved = local_to_geodetic(
            here,
            LocalPoint(offset.east_m * reach / distance, offset.north_m * reach / distance),
        )
        asset.lat, asset.lon = moved.lat_deg, moved.lon_deg
        asset.speed = reach / dt
        asset.z = intent.target_z if intent.target_z > 0.0 else asset.z


class SimTransport:
    """The offline world behind the :class:`~whiteout.transport.base.Transport` seam.

    **Why this is here and not in** ``whiteout/transport/`` — that package is
    guarded (``tests/test_transport.py``) against importing belief, policy,
    scoring or this module, and against so much as naming ``Truth``. The guard
    is right: an adapter to a real vehicle must not know what the coordinator
    believes, or the seam is decoration. A simulator is the opposite thing —
    it *is* the far side — so it lives out here with the world it runs, and
    the composition root picks it.

    It satisfies the four-method seam exactly, so the episode runner cannot
    tell it from the arena, and it adds :meth:`truth`, which the arena has no
    answer to. Nothing that runs against the arena may depend on that call.
    """

    def __init__(
        self,
        seed: int = 0,
        tick_seconds: float = 0.5,
        pose_age_seconds: float | None = None,
        *,
        fleet: tuple[tuple[str, str, float], ...] | None = None,
    ) -> None:
        if seed < 0:
            raise TransportError(f"sim transport: seed {seed} is negative")
        if pose_age_seconds is not None:
            # Validated *before* the widening, not after: `float()` on a
            # non-number escapes as a bare `ValueError` rather than the seam's
            # own error type, and `float(True)` is 1.0 -- an age of one second
            # nobody asked for. A negative age would stamp a fix taken after
            # the tick it arrived in.
            if isinstance(pose_age_seconds, bool) or not isinstance(pose_age_seconds, int | float):
                raise TransportError(
                    f"sim transport: pose_age_seconds "
                    f"{pose_age_seconds!r} is not a number of seconds"
                )
            pose_age_seconds = float(pose_age_seconds)
            if not math.isfinite(pose_age_seconds) or pose_age_seconds < 0.0:
                raise TransportError(
                    f"sim transport: pose_age_seconds {pose_age_seconds} "
                    f"is not a finite, non-negative number of seconds"
                )
        self._seed = int(seed)
        self._tick_seconds = float(tick_seconds)
        self._pose_age_seconds = pose_age_seconds
        self._fleet = FLEET if fleet is None else fleet
        self._world: ArenaSim | None = None
        self._closed = False
        self._truth = Truth(t=0.0, targets=())
        self._last: FleetIntent | None = None

    def connect(self) -> None:
        """Build the world. Idempotent; a closed transport is not reusable."""
        if self._world is not None:
            return
        if self._closed:
            raise TransportError(
                "sim transport: connect() after close(); a transport is "
                "single-use, so build a new one for the next episode"
            )
        self._world = ArenaSim(seed=self._seed, tick_seconds=self._tick_seconds, fleet=self._fleet)

    def observe(self) -> WorldObservation:
        """This tick's fleet and what it saw, then advance the world.

        The world steps **here**, using the intents the last :meth:`command`
        left, so that one tick is one call and a caller sees the result of its
        command on the next observe — the latency a real link has anyway.
        """
        world = self._require("observe")
        t = world.t
        measured_t = None if self._pose_age_seconds is None else t - self._pose_age_seconds
        poses = tuple(
            Pose(
                asset_id=pose.asset_id,
                cls=pose.cls,
                t=t,
                lat=pose.lat,
                lon=pose.lon,
                z=pose.z,
                heading=pose.heading,
                speed=pose.speed,
                energy_used=pose.energy_used,
                measured_t=measured_t,
            )
            for pose in world.poses()
        )
        reports = world.reports()
        self._truth = world.truth()
        world.step(() if self._last is None else self._last.intents)
        return WorldObservation(t=t, poses=poses, reports=reports)

    def command(self, intent: FleetIntent) -> None:
        """Accept one tick of intents. The next :meth:`observe` flies them."""
        self._require("command")
        self._last = intent

    def close(self) -> None:
        """Bring the world down. Idempotent, and safe to call unconnected."""
        if self._world is not None:
            self._closed = True
        self._world = None

    def truth(self) -> Truth:
        """The world's true state as of the last :meth:`observe`.

        Off the seam on purpose: ``base.TRANSPORT_METHODS`` does not list it
        and the arena cannot answer it, so the runner asks only when a
        transport happens to offer it and writes empty truth otherwise.
        """
        self._require("truth")
        return self._truth

    def _require(self, call: str) -> ArenaSim:
        if self._world is None:
            raise TransportError(f"sim transport: {call}() before connect()")
        return self._world
