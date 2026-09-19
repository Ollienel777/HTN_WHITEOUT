"""The ``kinematic`` transport — the default, and not a degraded path.

``hackathon/SPEC.md`` §7: ``kinematic`` is the dev loop, the tuning loop, the
CI loop and the demo. It is the transport the gate forces, and it **never
opens a socket and never leaves the machine**.

This is the M0 stub. It stands a fleet up — one asset of each class in
:data:`~whiteout.types.VEHICLE_CLASSES`, because the heterogeneity is what the
collaboration axis is scored on — and reports those assets parked, tick after
tick, with no sensor reports. Vehicle kinematics, sensor footprints and
detection models are ``whiteout/sim/``'s, and arrive with their own tickets;
what exists here is the far side of the seam, so that everything above it can
be written against a real ``Transport`` from now on.

The stub is a pure function of its seed. Start positions are drawn once, at
:meth:`KinematicTransport.connect`, from a :class:`numpy.random.Generator`
derived from the episode seed — an explicit generator, never module-level
``np.random`` (§5 "Conventions") — and never move again. Two runs at one seed
therefore produce byte-identical episode logs, which is what the gate's
determinism step compares.
"""

from __future__ import annotations

import math

import numpy as np

from whiteout.transport.base import TransportError
from whiteout.types import FleetIntent, Pose, WorldObservation

__all__ = ["DEFAULT_TICK_SECONDS", "FLEET", "KinematicTransport"]

#: Seconds of world time per :meth:`KinematicTransport.observe`. The sim's
#: episode clock replaces this; it lives here so the stub's ``t`` advances at
#: some stated rate rather than counting ticks and calling them seconds.
DEFAULT_TICK_SECONDS = 0.5

#: The stub fleet: ``(asset_id, cls, altitude)``, one asset per vehicle class.
FLEET: tuple[tuple[str, str, float], ...] = (
    ("fw-1", "fixedwing", 120.0),
    ("quad-1", "quad", 40.0),
    ("rover-1", "rover", 0.0),
    ("tower-1", "tower", 15.0),
)

#: Half-width, in metres, of the box start positions are drawn from.
_START_SPREAD = 500.0


class KinematicTransport:
    """A :class:`~whiteout.transport.base.Transport` over parked assets.

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
    def last_intent(self) -> FleetIntent | None:
        """The most recent :meth:`command`, or ``None``. For tests only.

        The stub has nothing to steer, so a commanded intent is retained and
        not acted on. It is read by the seam's tests to show that the call
        arrived; nothing in ``whiteout/`` may read it back as state.
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
        for asset_id, cls, altitude in FLEET:
            x, y = generator.uniform(-_START_SPREAD, _START_SPREAD, size=2)
            heading = generator.uniform(0.0, 2.0 * np.pi)
            poses.append(
                Pose(
                    asset_id=asset_id,
                    cls=cls,
                    t=0.0,
                    x=float(x),
                    y=float(y),
                    z=altitude,
                    heading=float(heading),
                    speed=0.0,
                    energy_used=0.0,
                )
            )
        self._poses = tuple(poses)
        self._t = 0.0
        self._connected = True

    def observe(self) -> WorldObservation:
        """Return this tick's parked fleet, and advance the clock."""
        self._require_connected("observe")
        t = self._t
        measured_t = None if self._pose_age_seconds is None else t - self._pose_age_seconds
        poses = tuple(
            Pose(
                asset_id=pose.asset_id,
                cls=pose.cls,
                t=t,
                x=pose.x,
                y=pose.y,
                z=pose.z,
                heading=pose.heading,
                speed=pose.speed,
                energy_used=pose.energy_used,
                measured_t=measured_t,
            )
            for pose in self._poses
        )
        self._t = t + self._tick_seconds
        return WorldObservation(t=t, poses=poses, reports=())

    def command(self, intent: FleetIntent) -> None:
        """Accept one tick of intents. Parked assets do not act on them."""
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
