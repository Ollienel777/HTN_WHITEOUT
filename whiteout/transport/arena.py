"""The arena adapter: MAVLink over UDP, one endpoint per asset.

``hackathon/ARENA.md`` §3. Q1 and Q6 are answered — Dominion Dynamics supply
the harness, it is Gazebo plus gzweb plus ArduPilot SITL in Docker, and each
asset is a stock ArduPilot vehicle reachable at its own UDP port on the arena
host:

.. code-block:: text

    quadcopter  sys 1  QUADROTOR        udp 14550
    fixed-wing  sys 2  FIXED_WING       udp 14560
    boat/rover  sys 3  (role idle)      udp 14570
    tower-1     sys 4  ANTENNA_TRACKER  udp 14580
    tower-2     sys 5  ANTENNA_TRACKER  udp 14590

Three things about this link that cost an afternoon if you learn them the hard
way
--------------------------------------------------------------------------

**It is ``udpout``, not ``udpin``.** The arena binds the port and waits. It
sends nothing at all until a ground station has said hello, so a client that
opens a socket and waits for traffic sits there forever and concludes the
asset is down. We send a GCS heartbeat first and keep sending one, and the
telemetry starts.

**Arming holds for about three seconds.** Dominion's own deck says so twice.
An arm followed by a takeoff with a round trip in between misses the window
and the vehicle sits disarmed, so :meth:`ArenaTransport.arm_and_launch`
pipelines the two.

**The towers are vehicles.** They are ArduPilot AntennaTracker instances, so
pan and tilt are servo 1 and servo 2 driven by ``MAV_CMD_DO_SET_SERVO`` over
the same link as everything else, and ``scan`` is a flight mode. They are not
a separate integration and they do not need one.

The clock, and why it is ours
------------------------------

:meth:`observe` stamps each tick from this adapter's own counter, never from
the autopilot's. ``base.py`` sets out why at length; the short version is that
a far-side clock is sampled and two polls can land on the same instant, which
would fail the seam's strict-increase rule over nothing that matters to us.

``measured_t`` is the age of the underlying fix and would have to be converted
out of the autopilot's ``time_boot_ms`` into our tick timebase. This adapter
does not establish that offset, so it reports ``None``, which the seam defines
as "the transport does not know". Reporting an unconverted ``time_boot_ms``
would pass every check and mean nothing.

What is not here
-----------------

Anything about where assets should go. This package is an adapter: poses and
reports in, waypoints out. The roster below says *where an asset answers*,
never *where it should stand* — tower siting is data loaded from a file
(``WHITEOUT_ARENA_ROSTER``) precisely so that choosing it is not a code change.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from whiteout.geo import GeoPoint, bearing_deg
from whiteout.transport.base import TransportError
from whiteout.types import FleetIntent, Pose, WorldObservation

__all__ = [
    "DEFAULT_ROSTER",
    "ENDPOINT_ENV",
    "ROSTER_ENV",
    "ArenaTransport",
    "AssetLink",
    "roster_from_env",
]

#: Arena host, for example ``10.99.4.1``. A bare host or a ``host:port`` base.
ENDPOINT_ENV = "WHITEOUT_ARENA_ENDPOINT"

#: Optional path to a JSON roster, overriding :data:`DEFAULT_ROSTER`. Tower
#: siting is the one ``.env`` lever Dominion sanctioned, so which assets we
#: drive and where they answer is data rather than a constant in this file.
ROSTER_ENV = "WHITEOUT_ARENA_ROSTER"

#: Seconds to wait for an asset's first heartbeat before calling it absent.
_HELLO_TIMEOUT_S = 8.0

#: How often to re-send our GCS heartbeat so the far side keeps talking.
_HEARTBEAT_PERIOD_S = 1.0

#: ArduPilot custom mode numbers, per vehicle family.
_MODE_GUIDED_COPTER = 4
_MODE_GUIDED_PLANE = 15
_MODE_SCAN_TRACKER = 2

#: ``SET_POSITION_TARGET_GLOBAL_INT`` type mask: use position, ignore
#: velocity, acceleration and yaw. Bits are "ignore this field".
_POSITION_ONLY_MASK = 0b0000_1111_1111_1000

#: Radians to degrees. ATTITUDE.yaw is an angle the autopilot has already
#: computed, so putting it in degrees is a unit conversion and not a frame
#: conversion — whiteout.geo owns the latter and has no business in this.
_DEG_PER_RAD = 180.0 / math.pi


@dataclass(frozen=True)
class AssetLink:
    """Where one asset answers, and what kind of vehicle it is.

    ``cls`` is the seam's vehicle-class string and decides which GUIDED mode
    number and which command shape apply. It must be one of
    :data:`whiteout.types.VEHICLE_CLASSES` — ``quad``, ``fixedwing``,
    ``rover`` or ``tower`` — because it is written straight onto a
    :class:`~whiteout.types.Pose`, and a class outside that set produces an
    episode log that cannot be read back.
    """

    asset_id: str
    cls: str
    port: int
    system_id: int


#: The roster observed on the live arena, and the default when no file is set.
DEFAULT_ROSTER: tuple[AssetLink, ...] = (
    AssetLink("quadcopter", "quad", 14550, 1),
    AssetLink("fixed-wing", "fixedwing", 14560, 2),
    AssetLink("tower-1", "tower", 14580, 4),
    AssetLink("tower-2", "tower", 14590, 5),
)


def roster_from_env(env: dict[str, str] | None = None) -> tuple[AssetLink, ...]:
    """The roster from ``WHITEOUT_ARENA_ROSTER``, or :data:`DEFAULT_ROSTER`.

    The file is a JSON list of ``{"asset_id", "cls", "port", "system_id"}``.
    A malformed file raises rather than silently falling back: a roster that
    quietly reverted to the default would drive the wrong assets and look
    like it worked.
    """
    source = os.environ if env is None else env
    path = source.get(ROSTER_ENV, "").strip()
    if not path:
        return DEFAULT_ROSTER
    try:
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise TransportError(f"{ROSTER_ENV}={path!r} could not be read: {error}") from error
    if not isinstance(rows, list) or not rows:
        raise TransportError(f"{ROSTER_ENV}={path!r} must hold a non-empty JSON list")
    try:
        return tuple(
            AssetLink(
                asset_id=str(row["asset_id"]),
                cls=str(row["cls"]),
                port=int(row["port"]),
                system_id=int(row["system_id"]),
            )
            for row in rows
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TransportError(f"{ROSTER_ENV}={path!r} has a malformed row: {error}") from error


def _host_from_env(env: dict[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    value = source.get(ENDPOINT_ENV, "").strip()
    if not value:
        raise TransportError(
            f"{ENDPOINT_ENV} is unset; set it to the arena host, for example 10.99.4.1"
        )
    return value.split("//")[-1].split(":")[0].strip("/")


class _Link:
    """One asset's MAVLink connection, and the last of each message we saw."""

    def __init__(self, asset: AssetLink, host: str) -> None:
        self.asset = asset
        self.host = host
        self.conn: Any = None
        self.last_hello = 0.0
        self.position: Any = None
        self.attitude: Any = None
        self.hud: Any = None

    def open(self) -> None:
        from pymavlink import mavutil

        # udpout, not udpin: the arena binds and waits, and sends nothing
        # until a ground station has spoken to it.
        self.conn = mavutil.mavlink_connection(
            f"udpout:{self.host}:{self.asset.port}",
            source_system=255,
            source_component=190,
            dialect="ardupilotmega",
        )
        self.hello()

    def hello(self) -> None:
        """Send the GCS heartbeat that keeps the far side talking."""
        from pymavlink import mavutil

        if self.conn is None:
            return
        self.conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )
        self.last_hello = time.monotonic()

    def pump(self, budget_s: float) -> None:
        """Drain what has arrived, keeping the newest of each message."""
        if self.conn is None:
            return
        if time.monotonic() - self.last_hello > _HEARTBEAT_PERIOD_S:
            self.hello()
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            message = self.conn.recv_match(blocking=False)
            if message is None:
                break
            kind = message.get_type()
            if kind == "GLOBAL_POSITION_INT":
                self.position = message
            elif kind == "ATTITUDE":
                self.attitude = message
            elif kind == "VFR_HUD":
                self.hud = message

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class ArenaTransport:
    """The sponsor's arena, behind the seam.

    One instance drives one episode: ``connect`` once, ``observe``/``command``
    per tick, ``close`` once. Reconnect is not supported — see
    :class:`~whiteout.transport.base.Transport`.
    """

    def __init__(
        self,
        host: str | None = None,
        roster: tuple[AssetLink, ...] | None = None,
        *,
        seed: int = 0,
        tick_s: float = 1.0,
        pump_budget_s: float = 0.05,
    ) -> None:
        # The factory offers every transport the episode seed. The arena is a
        # live simulator driven by Dominion, not something we lay out, so a
        # non-zero seed is a caller believing it controls a world it does not.
        # Refusing by name beats honouring it silently and reporting a run as
        # reproducible when nothing about it was.
        if seed:
            raise TransportError(
                f"the arena transport cannot use a seed (got {seed!r}); the "
                f"world is Dominion's and is not ours to lay out"
            )
        self._host_override = host
        self._roster = roster if roster is not None else roster_from_env()
        self._tick_s = float(tick_s)
        self._pump_budget_s = float(pump_budget_s)
        self._host = ""
        self._links: dict[str, _Link] = {}
        self._tick = 0
        self._up = False
        self._closed = False

    @property
    def connected(self) -> bool:
        """Whether the link is up."""
        return self._up

    @property
    def assets(self) -> tuple[AssetLink, ...]:
        """The roster this instance drives."""
        return self._roster

    def connect(self) -> None:
        """Open a link per asset and wait for each to say hello.

        Idempotent while up. An asset that never answers is a hard failure:
        a fleet quietly missing half its vehicles scores like one, and it is
        better to hear about it at connect than to wonder at tick 400.
        """
        if self._closed:
            raise TransportError("this transport was closed; make a new one")
        if self._up:
            return
        # Resolved here, not in the constructor: SPEC.md section 7 makes "the
        # endpoint is not set" an adapter's normal connect-time refusal, and
        # the conformance suite skips a transport that refuses to connect. A
        # constructor that raised would instead make the arena unconstructible
        # on any machine without the arena, which is every machine but one.
        host = self._host_override
        if host is None:
            host = _host_from_env()
        self._host = host
        for asset in self._roster:
            link = _Link(asset, host)
            try:
                link.open()
            except Exception as error:  # pragma: no cover - import/socket paths
                self._shutdown()
                raise TransportError(
                    f"could not open {asset.asset_id} at {self._host}:{asset.port}: {error}"
                ) from error
            self._links[asset.asset_id] = link
        missing = self._await_hello()
        if missing:
            self._shutdown()
            raise TransportError(
                f"no heartbeat from {', '.join(missing)} at {self._host} within "
                f"{_HELLO_TIMEOUT_S:.0f}s; is the arena up and the roster right?"
            )
        self._up = True

    def _await_hello(self) -> list[str]:
        deadline = time.monotonic() + _HELLO_TIMEOUT_S
        waiting = dict(self._links)
        while waiting and time.monotonic() < deadline:
            for asset_id, link in list(waiting.items()):
                link.hello()
                if link.conn is not None and link.conn.recv_match(type="HEARTBEAT"):
                    waiting.pop(asset_id)
            time.sleep(0.1)
        return sorted(waiting)

    def observe(self) -> WorldObservation:
        """One tick's poses. The clock is ours and strictly increases."""
        self._require_up()
        self._tick += 1
        t = self._tick * self._tick_s
        for link in self._links.values():
            link.pump(self._pump_budget_s)
        poses = tuple(
            pose for link in self._links.values() if (pose := self._pose(link, t)) is not None
        )
        # Silence is not a fault: an asset that has not reported yet simply
        # contributes no pose this tick.
        return WorldObservation(t=t, poses=poses, reports=())

    def _pose(self, link: _Link, t: float) -> Pose | None:
        position = link.position
        if position is None:
            return None
        heading = 0.0
        pitch: float | None = None
        roll: float | None = None
        if link.attitude is not None:
            heading = float(link.attitude.yaw) * _DEG_PER_RAD % 360.0
            # ATTITUDE carries all three angles and a camera cannot be
            # projected without them (#99). Reported as None rather than 0.0
            # when there is no ATTITUDE: zero reads as level, which is a
            # measurement we did not take.
            pitch = float(link.attitude.pitch) * _DEG_PER_RAD
            roll = float(link.attitude.roll) * _DEG_PER_RAD
        elif position.hdg not in (None, 65535):
            heading = float(position.hdg) / 100.0
        speed = 0.0
        if link.hud is not None:
            speed = float(link.hud.groundspeed)
        else:
            speed = math.hypot(float(position.vx), float(position.vy)) / 100.0
        return Pose(
            asset_id=link.asset.asset_id,
            cls=link.asset.cls,
            t=t,
            lat=float(position.lat) / 1e7,
            lon=float(position.lon) / 1e7,
            z=float(position.relative_alt) / 1000.0,
            heading=heading,
            speed=speed,
            energy_used=0.0,
            # The offset between the autopilot's boot clock and our tick
            # timebase is not established, and an unconverted stamp would
            # pass every check and mean nothing. See the module docstring.
            measured_t=None,
            pitch=pitch,
            roll=roll,
        )

    def command(self, intent: FleetIntent) -> None:
        """Send each asset where it is told to go.

        A waypoint for a tower is meaningless — it cannot move — so a tower's
        intent is read as a bearing to look along and turned into pan and
        tilt. Everything else becomes a GUIDED position target.
        """
        self._require_up()
        for waypoint in intent.intents:
            link = self._links.get(waypoint.asset_id)
            if link is None or link.conn is None:
                continue
            if link.asset.cls == "tower":
                self._aim(link, waypoint.target_lat, waypoint.target_lon)
            else:
                self._goto(link, waypoint)

    def _goto(self, link: _Link, waypoint: Any) -> None:
        from pymavlink import mavutil

        conn = link.conn
        conn.mav.set_position_target_global_int_send(
            0,
            link.asset.system_id,
            1,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            _POSITION_ONLY_MASK,
            int(waypoint.target_lat * 1e7),
            int(waypoint.target_lon * 1e7),
            float(waypoint.target_z),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _aim(self, link: _Link, lat: float, lon: float) -> None:
        """Point a tower's camera at a lat/lon, via servo 1 and servo 2."""
        if link.position is None:
            return
        here_lat = float(link.position.lat) / 1e7
        here_lon = float(link.position.lon) / 1e7
        bearing = bearing_deg(GeoPoint(here_lat, here_lon), GeoPoint(lat, lon))
        self.set_servo(link.asset.asset_id, 1, _pwm_for_bearing(bearing))

    def set_servo(self, asset_id: str, servo: int, pwm: int) -> None:
        """Raw servo command — pan is 1, tilt is 2 on an AntennaTracker."""
        self._require_up()
        link = self._links.get(asset_id)
        if link is None or link.conn is None:
            raise TransportError(f"no such asset: {asset_id!r}")
        from pymavlink import mavutil

        link.conn.mav.command_long_send(
            link.asset.system_id,
            1,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            float(servo),
            float(pwm),
            0,
            0,
            0,
            0,
            0,
        )

    def set_mode(self, asset_id: str, mode: int) -> None:
        """Set a custom flight mode by number."""
        self._require_up()
        link = self._links.get(asset_id)
        if link is None or link.conn is None:
            raise TransportError(f"no such asset: {asset_id!r}")
        from pymavlink import mavutil

        link.conn.mav.set_mode_send(
            link.asset.system_id,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            int(mode),
        )

    def scan(self, asset_id: str) -> None:
        """Put a tower into ``scan`` — the do-nothing-clever fallback."""
        self.set_mode(asset_id, _MODE_SCAN_TRACKER)

    def arm_and_launch(self, asset_id: str, altitude_m: float = 60.0) -> None:
        """GUIDED, arm, take off — with no round trip between arm and takeoff.

        Dominion's deck warns twice that arming holds for only about three
        seconds. Waiting for an ack between the arm and the takeoff misses
        that window often enough to look like a broken vehicle, so the two
        are sent back to back and the result is read from the pose stream.
        """
        self._require_up()
        link = self._links.get(asset_id)
        if link is None or link.conn is None:
            raise TransportError(f"no such asset: {asset_id!r}")
        if link.asset.cls == "tower":
            raise TransportError(f"{asset_id!r} is a tower and does not launch")
        from pymavlink import mavutil

        guided = _MODE_GUIDED_COPTER if link.asset.cls == "quad" else _MODE_GUIDED_PLANE
        self.set_mode(asset_id, guided)
        link.conn.mav.command_long_send(
            link.asset.system_id,
            1,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        link.conn.mav.command_long_send(
            link.asset.system_id,
            1,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            float(altitude_m),
        )

    def close(self) -> None:
        """Bring every link down. Safe before connect, and safe twice."""
        self._shutdown()
        self._up = False
        self._closed = True

    def _shutdown(self) -> None:
        for link in self._links.values():
            link.close()
        self._links.clear()

    def _require_up(self) -> None:
        if not self._up:
            raise TransportError("the arena link is not up; call connect() first")


def _pwm_for_bearing(bearing_deg: float) -> int:
    """Map a 0-360 bearing onto the 1100-1900 us servo band, linearly.

    A placeholder calibration, and honestly labelled as one: the mapping from
    PWM to actual pan angle is a property of the tracker's own configuration
    and has to be measured against the camera view before it means anything
    (#68's first acceptance criterion). Until it is, this is monotonic and
    covers the band, which is enough to sweep with and not enough to aim with.
    """
    return int(1100 + (bearing_deg % 360.0) / 360.0 * 800)
