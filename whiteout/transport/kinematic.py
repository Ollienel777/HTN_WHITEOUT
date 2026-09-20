"""The ``kinematic`` transport — the default, and not a degraded path.

``hackathon/SPEC.md`` §7: ``kinematic`` is the dev loop, the tuning loop, the
CI loop and the demo. It is the transport the gate forces, and it **never
opens a socket and never leaves the machine**.

It stands up the fleet ``ARENA.md`` §3 describes — two towers, one
quadcopter, one fixed-wing, no rover — under the same ``asset_id``s the arena
adapter drives, and **flies it**: an asset closes on the waypoint it was last
commanded, at its class's cruise speed, and a tower stands still because a
tower is aimed rather than sent.

That last part is the whole of this module's recent history. Until it existed,
the transport reported the fleet parked for ever, so ``run`` produced an
episode with no tasking, an empty belief digest and four zeroes from
``score`` — while the belief field, the policy, the track hold and the tracks
client were all merged, tested and unreachable. The seam was complete and
nothing was joined to it.

What is still not modelled, deliberately: turn rates, wind, stalls, sensor
footprints and detections. A fake that reimplements a flight stack badly is
worse than one that is obviously a fake, because it invites tuning against
its errors. Sightings arrive through a
:class:`~whiteout.coordinate.SightingSource`, which is a seam of its own.

Positions are the lat/lon of ``SPEC.md`` §5's one frame of record. Aircraft
start a few hundred metres about :data:`~whiteout.geo.ARENA_ORIGIN`, and that
offset — like every other metre in this module — is converted through
:mod:`whiteout.geo`, the tree's only converter. The adapter holds no frame
arithmetic of its own. The towers are literal coordinates: see
:data:`_TOWER_STATIONS` for why a transport may not ask the belief package
where the channel is.

The transport is a pure function of its seed and of what it is commanded.
Aircraft start positions are drawn once, at
:meth:`KinematicTransport.connect`, from a :class:`numpy.random.Generator`
derived from the episode seed — an explicit generator, never module-level
``np.random`` (§5 "Conventions"). The towers are not drawn at all: they are
sited on the channel, because where a tower stands is a decision and not a
throw. Nothing after ``connect`` consumes randomness, so two runs at one seed
still produce byte-identical episode logs, which is what the gate's
determinism step compares.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from whiteout.geo import ARENA_ORIGIN, GeoPoint, LocalPoint, geodetic_to_local, local_to_geodetic
from whiteout.transport.base import TransportError
from whiteout.types import FleetIntent, Pose, WaypointIntent, WorldObservation

__all__ = [
    "DEFAULT_TICK_SECONDS",
    "FLEET",
    "KinematicAsset",
    "KinematicTransport",
]

#: Seconds of world time per :meth:`KinematicTransport.observe`. The sim's
#: episode clock replaces this; it lives here so the stub's ``t`` advances at
#: some stated rate rather than counting ticks and calling them seconds.
DEFAULT_TICK_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class KinematicAsset:
    """One asset on the fake fleet, and what its class can physically do.

    The shape is structural, not nominal: ``whiteout/cli.py`` builds its
    roles from anything carrying ``asset_id`` and ``cls``, which is also what
    :class:`~whiteout.transport.arena.AssetLink` carries. The two rosters are
    therefore interchangeable to everything above the seam, which is the
    point — an episode recorded here and an episode recorded against the
    arena name the same four assets and render identically.

    :param camera: the key into :data:`whiteout.vision.camera.CAMERAS`. Held
        as a name rather than a model so that this module imports no vision
        code; it is the sensor a sighting source would read for this asset.
    :param cruise_mps: how fast the airframe closes on a waypoint.
        ``ARENA.md`` §3 makes the fixed-wing the faster of the two, which is
        the whole reason the policy gives it the distant work.
    :param mobile: false for a tower, which is **aimed and never sent**.
        Nothing here infers that from ``cls``: the policy already reasons
        about it, and two modules deciding the same fact separately is how
        they come to disagree.
    """

    asset_id: str
    cls: str
    altitude_m: float
    camera: str
    cruise_mps: float
    mobile: bool


#: The fake fleet, and it is the **real** one: two towers, one quadcopter,
#: one fixed-wing, no rover (``ARENA.md`` §3).
#:
#: The ``asset_id``s are exactly
#: :data:`whiteout.transport.arena.DEFAULT_ROSTER`'s, and deliberately not the
#: display callsigns #71's acceptance named. That ticket was written before
#: the arena adapter landed, and a demo whose fleet rows read ``WING-1`` while
#: the judged run reads ``fixed-wing`` is two episodes that cannot be compared
#: — including by us, the night before. Callsigns are a rendering concern and
#: belong to the viewer, which already maps what the log names.
#:
#: Altitudes: ``ARENA.md`` §3 starts the quadcopter over the highest point of
#: the map, which is why its figure is not a height over water and why #76's
#: datum question is not settled by this table.
FLEET: tuple[KinematicAsset, ...] = (
    KinematicAsset("quadcopter", "quad", 40.0, "quadcopter", 12.0, mobile=True),
    KinematicAsset("fixed-wing", "fixedwing", 120.0, "fixed-wing", 26.0, mobile=True),
    KinematicAsset("tower-1", "tower", 15.0, "tower", 0.0, mobile=False),
    KinematicAsset("tower-2", "tower", 15.0, "tower", 0.0, mobile=False),
)

#: Where the two masts stand, and which way each looks: ``(lat, lon, bearing)``
#: with the bearing in radians clockwise from North.
#:
#: **Literal coordinates, and not a channel parameterisation.** The obvious
#: spelling — ask ``whiteout.belief.geometry`` for a point 18% along the
#: strait, on the bank — is the one thing this module may not do: the seam
#: forbids a transport importing ``whiteout.belief`` at all, and the guard in
#: ``tests/test_transport.py`` rejected exactly that. The rule is right. Where
#: we physically put a mast is a fact about the world, like a port number; it
#: is not the coordinator's model of the world, and a transport that reached
#: for the belief package to find out where its own hardware is would have
#: inverted the dependency that makes three transports possible.
#:
#: So these were derived once, offline, from ``DEFAULT_STRAIT`` at 18% and 82%
#: of its 6 250 m length, each on the opposite bank at that station's
#: half-width, and each turned to look at the channel's midpoint. Not the
#: ends: a tower at ``s=0`` spends half its range outside the arena. Far
#: enough apart that the middle third is nobody's, which is the gap the
#: aircraft exist to fly — the choke-point argument in ``DECISION.md``,
#: placed rather than asserted. ``ARENA.md`` §3 says towers may be sited, so
#: this is a decision we are allowed to make; #68 makes it a deliberate one
#: instead of a default.
_TOWER_STATIONS: tuple[tuple[float, float, float], ...] = (
    (72.00155816096998, -94.88122329045646, 1.8586),
    (71.99210765845011, -94.76346780323156, 4.9264),
)


#: Half-width, in metres, of the box start positions are drawn from. The box
#: is East–North about :data:`~whiteout.geo.ARENA_ORIGIN` and is converted to
#: the lat/lon the records carry by :mod:`whiteout.geo`, which is the only
#: module in the tree that converts. Metres are the readable unit for "a few
#: hundred apart"; they never reach a record.
_START_SPREAD = 500.0


class KinematicTransport:
    """A :class:`~whiteout.transport.base.Transport` over the real fleet, flown.

    ``seed`` is the episode seed; ``tick_seconds`` is how much tick time one
    :meth:`observe` advances. Both are constructor arguments rather than
    environment reads, so that a test can stand two of these up side by side.

    ``pose_age_seconds`` is the third, and it is how old the fix behind each
    pose is said to be: each pose is stamped ``measured_t = t - age``, while
    ``t`` stays the tick it belongs to. ``None`` — the default — reports no
    measurement time at all, which is what this stub honestly knows, since
    its poses are drawn at ``connect`` and never move.

    It is configurable because the alternative is that nothing downstream
    ever sees a stale fix until a link to a real vehicle serves one, on the
    one run that is judged and the one where there is no debugger. A fast,
    offline, seeded transport that hands the policy a two-second-old fix on
    demand is the only place that path can be exercised cheaply.

    An age may exceed the elapsed tick time, so early ticks report a
    ``measured_t`` before the clock's zero. That is left alone rather than
    clamped: clamping would report those fixes as fresher than they are,
    which is the wrong answer in exactly the shape that is hard to notice.
    """

    def __init__(
        self,
        seed: int = 0,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        pose_age_seconds: float | None = None,
    ) -> None:
        if seed < 0:
            # The backstop behind the CLI's own check: `np.random.default_rng`
            # rejects a negative seed with an eight-frame numpy traceback, and
            # #25's episode loop will construct transports without going
            # through argparse. The seam's own error type, either way.
            raise TransportError(f"kinematic transport: seed {seed} is negative")
        if pose_age_seconds is not None:
            # Validated *before* the widening, not after: `float()` on a
            # non-number escapes as a bare `ValueError` rather than the seam's
            # own error type, and `float(True)` is 1.0 — an age of one second
            # nobody asked for.
            if isinstance(pose_age_seconds, bool) or not isinstance(pose_age_seconds, int | float):
                raise TransportError(
                    f"kinematic transport: pose_age_seconds "
                    f"{pose_age_seconds!r} is not a number of seconds"
                )
            pose_age_seconds = float(pose_age_seconds)
            if not math.isfinite(pose_age_seconds) or pose_age_seconds < 0.0:
                # A negative age would stamp a fix taken *after* the tick it
                # arrived in, and a non-finite one cannot be written to an
                # episode log at all. Both are caller mistakes; refusing here
                # names the argument instead of producing poses that fail the
                # seam's conformance suite several calls later.
                raise TransportError(
                    f"kinematic transport: pose_age_seconds {pose_age_seconds} "
                    f"is not a finite, non-negative number of seconds"
                )
        self._seed = seed
        self._tick_seconds = float(tick_seconds)
        self._pose_age_seconds = pose_age_seconds
        self._connected = False
        self._closed = False
        self._t = 0.0
        self._poses: tuple[Pose, ...] = ()
        self._last_intent: FleetIntent | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def assets(self) -> tuple[KinematicAsset, ...]:
        """The roster this transport drives.

        Its existence is what makes ``run`` against the fake a *coordinated*
        episode rather than an observed one: ``whiteout/cli.py`` builds the
        policy's roles from this and falls back to no coordinator at all when
        a transport does not publish one. Until this property existed, the
        default path — the gate's, the viewer's bundled episode, the demo's —
        logged four assets standing still with an empty belief digest and
        scored zero on every axis, while every part needed to do better was
        already merged and tested.
        """
        return FLEET

    @property
    def last_intent(self) -> FleetIntent | None:
        """The most recent :meth:`command`, or ``None``. For tests only.

        The intent is no longer inert — :meth:`observe` flies the fleet
        toward it — but it is still not state anything above the seam may
        read back. A coordinator that asked the transport what it had just
        commanded would be reading its own output as though it were a
        measurement, which is the shape of the bug that makes a fake look
        better than the real link it stands in for.
        """
        return self._last_intent

    def connect(self) -> None:
        """Place the fleet and start the clock.

        Idempotent while connected. Refuses after :meth:`close`, because the
        seam does not support reconnect (see
        :class:`~whiteout.transport.base.Transport`) and a silent reconnect
        here would rewind ``t`` to 0.0 and cost the whole episode at write
        time instead of reporting the fault here.
        """
        if self._connected:
            return
        if self._closed:
            raise TransportError(
                "kinematic transport: connect() after close(); a transport is "
                "single-use, so build a new one for the next episode"
            )
        generator = np.random.default_rng(self._seed)
        poses: list[Pose] = []
        for asset in FLEET:
            if asset.mobile:
                east, north = generator.uniform(-_START_SPREAD, _START_SPREAD, size=2)
                start = local_to_geodetic(ARENA_ORIGIN, LocalPoint(float(east), float(north)))
                lat, lon = start.lat_deg, start.lon_deg
                heading = float(generator.uniform(0.0, 2.0 * np.pi))
            else:
                lat, lon, heading = self._tower_station(len(poses))
            poses.append(
                Pose(
                    asset_id=asset.asset_id,
                    cls=asset.cls,
                    t=0.0,
                    lat=lat,
                    lon=lon,
                    z=asset.altitude_m,
                    heading=heading,
                    speed=0.0,
                    energy_used=0.0,
                )
            )
        self._poses = tuple(poses)
        self._t = 0.0
        self._connected = True

    def _tower_station(self, index: int) -> tuple[float, float, float]:
        """Where the ``index``-th tower stands, and which way it looks.

        A tower is sited on the channel's **edge** rather than its centreline
        — it is a mast on a shore, and a mast in the water is a detail a judge
        notices in the first ten seconds. The two alternate banks, so between
        them they see both sides of the strait, and each is turned to look
        down the channel toward the middle third neither covers.
        """
        return _TOWER_STATIONS[index % len(_TOWER_STATIONS)]

    def observe(self) -> WorldObservation:
        """Fly the fleet one tick toward what it was last told, and report it.

        The motion is **first-order and on purpose**: an asset closes on its
        waypoint along the straight line to it, at its class's cruise speed,
        and stops when it arrives. There is no turn rate, no bank, no stall
        and no wind. Those belong to the arena's own flight stack, which is
        the thing this transport exists to stand in for cheaply — a fake that
        reimplements ArduPilot badly is worse than one that is obviously a
        fake, because it invites us to tune against its errors.

        What it does model is the only property the policy above it can
        actually observe: **an asset takes time to get somewhere, and the
        fixed-wing gets there sooner.** That is enough for coverage to grow,
        for the belief field to be eroded where somebody has been, and for
        re-tasking to cost something — which is what makes the difference
        between an episode and a slideshow.

        A tower is never moved, whatever it is sent: ``mobile`` is false and
        the intent is recorded and ignored. Aiming one is #68's.
        """
        self._require_connected("observe")
        t = self._t
        measured_t = None if self._pose_age_seconds is None else t - self._pose_age_seconds
        targets = {
            intent.asset_id: intent
            for intent in (self._last_intent.intents if self._last_intent else ())
        }
        moved = tuple(
            self._advance(pose, asset, targets.get(pose.asset_id), t, measured_t)
            for pose, asset in zip(self._poses, FLEET, strict=True)
        )
        self._poses = moved
        self._t = t + self._tick_seconds
        return WorldObservation(t=t, poses=moved, reports=())

    def _advance(
        self,
        pose: Pose,
        asset: KinematicAsset,
        target: WaypointIntent | None,
        t: float,
        measured_t: float | None,
    ) -> Pose:
        """One asset, one tick of travel toward ``target``."""
        lat, lon, heading, speed = pose.lat, pose.lon, pose.heading, 0.0
        if asset.mobile and target is not None:
            here = geodetic_to_local(ARENA_ORIGIN, GeoPoint(pose.lat, pose.lon))
            there = geodetic_to_local(ARENA_ORIGIN, GeoPoint(target.target_lat, target.target_lon))
            east, north = there.east_m - here.east_m, there.north_m - here.north_m
            remaining = math.hypot(east, north)
            # A commanded speed of zero means "hold", not "teleport"; an
            # absent one means cruise. `min` with the remaining distance is
            # what stops an asset overshooting a near waypoint and then
            # oscillating across it, which reads on the canvas as a jitter
            # and in the log as coverage the asset never had.
            cruise = (
                asset.cruise_mps if target.speed <= 0.0 else min(target.speed, asset.cruise_mps)
            )
            step = min(cruise * self._tick_seconds, remaining)
            if remaining > 0.0 and step > 0.0:
                fraction = step / remaining
                arrived = local_to_geodetic(
                    ARENA_ORIGIN,
                    LocalPoint(here.east_m + east * fraction, here.north_m + north * fraction),
                )
                lat, lon = arrived.lat_deg, arrived.lon_deg
                heading = math.atan2(east, north) % (2.0 * math.pi)
                speed = step / self._tick_seconds
        return Pose(
            asset_id=pose.asset_id,
            cls=pose.cls,
            t=t,
            lat=lat,
            lon=lon,
            z=pose.z,
            heading=heading,
            speed=speed,
            # Energy is distance flown, not time powered: a loitering quad and
            # a parked one are not the same asset to the efficiency axis, but
            # nothing here knows a hover's cost and inventing one would be a
            # number the scorer takes seriously.
            energy_used=pose.energy_used + speed * self._tick_seconds,
            measured_t=measured_t,
        )

    def command(self, intent: FleetIntent) -> None:
        """Accept one tick of intents; the next ``observe`` flies them."""
        self._require_connected("command")
        self._last_intent = intent

    def close(self) -> None:
        """Bring the link down. Idempotent, and safe to call unconnected.

        Closing a link that was never up leaves the transport usable — that
        is the ``close()``-in-``finally`` case, where ``connect()`` may not
        have run. Closing a live one is final.
        """
        if self._connected:
            self._closed = True
        self._connected = False

    def _require_connected(self, call: str) -> None:
        if not self._connected:
            raise TransportError(f"kinematic transport: {call}() before connect()")
