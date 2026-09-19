"""The conformance suite: what any transport must do to be one.

``hackathon/SPEC.md`` §4 makes the seam the build's one structural decision,
and §10 makes it the mitigation for "the sponsor's interface is not MAVLink".
A seam is only real if a new adapter has a definition of done, and this module
is that definition: the behaviour every implementation owes its caller, in one
place, runnable against any of them.

**This is a library, not a test module.** Nothing here is named ``test_``, so
pytest does not collect it; ``tests/test_transport_conformance.py`` is the
runner, and it derives what to run from the build's factory registry. Adding
an adapter therefore changes nothing in this file — registering it is what
enrols it, and a static test asserts that this module names no implementation.

Every check takes one argument, a **factory**: a zero-argument callable
returning a fresh, unconnected transport. It is a factory and not a transport
because a transport is single-use (``whiteout/transport/base.py``) — one
instance drives one episode, and several checks are about what happens at the
ends of that life. Each check owns the whole lifecycle of everything it
builds, so no check can be broken by another's leftovers and the order they
run in cannot matter.

A check reports a breach by raising :class:`AssertionError` naming the
transport and what it did, so a failure reads as a sentence about the adapter
rather than as a traceback through the suite. That holds for a refusal too:
every call this module makes that the contract says must succeed —
``observe``, ``command``, ``close``, and a ``connect`` on a live link — is
made inside :func:`_accepts` or :func:`_closed_after`, which catch the
adapter's :class:`~whiteout.transport.base.TransportError` and re-raise it as
an ``AssertionError``. Two calls sit outside that, deliberately:

* the *first* ``connect``, which is not wrapped at all — a link this machine
  cannot bring up is not a non-conforming transport, it is an absent one, and
  :func:`unavailable` owns that case for the runner to skip by name;
* the ``close`` :func:`_closed_after` runs to tear down a block that has
  *already* failed, whose refusal is swallowed rather than reported, so that
  a broken teardown cannot replace the assertion the check came for.

Neither of those can reach the caller as a ``TransportError``, so a planted
breach against any check can be written in the usual
``pytest.raises(AssertionError)`` style.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any

from whiteout.geo import enu_to_geodetic
from whiteout.transport import TRANSPORT_METHODS, Transport, TransportError
from whiteout.types import (
    VEHICLE_CLASSES,
    FleetIntent,
    TargetTruth,
    Truth,
    WaypointIntent,
    WorldObservation,
)

__all__ = [
    "CONFORMANCE_CHECKS",
    "STATIC_CLASSES",
    "STEERABLE_CLASSES",
    "TransportFactory",
    "check_id",
    "unavailable",
]

#: A zero-argument callable returning a fresh, unconnected transport.
TransportFactory = Callable[[], Transport]

#: How many ticks the multi-tick checks drive. Enough that a clock which only
#: advances on the first call, or a fleet that goes empty after a tick or two,
#: is caught; small enough that ten checks per transport stay far inside the
#: gate's three-minute budget (``SPEC.md`` §6). The roster's *shape* is
#: deliberately not pinned — see
#: :func:`check_silence_is_not_a_fault_and_every_report_is_attributable`.
TICKS = 5

#: The observation's whole shape. Pinned, not merely searched for a forbidden
#: name: a field added to what crosses the seam is a widening of the seam, and
#: §4 makes that a deliberate edit rather than something an adapter may do.
OBSERVATION_FIELDS = frozenset({"t", "poses", "reports"})

#: Ground truth. ``SPEC.md`` §4: written by the sim, read only by the scorer
#: and the viewer, and never observed. It is the single easiest way to cheat
#: ourselves and believe the number afterwards.
TRUTH_TYPES: tuple[type, ...] = (Truth, TargetTruth)

#: Vehicle classes that cannot be sent anywhere, and so have no waypoint to
#: refuse. The seam's only outbound record is a
#: :class:`~whiteout.types.WaypointIntent` — a point to go to and a speed to
#: go at — and the fleet the build is aimed at includes two masts that are
#: bolted down: antenna trackers, whose whole command surface is a pan servo,
#: a tilt servo and a scan mode (``hackathon/`` §3 on the fleet). Addressing
#: one a waypoint is not something an adapter can honour, so the command
#: check does not ask it to.
STATIC_CLASSES = frozenset({"tower"})

#: The classes a waypoint may be addressed to. Derived from
#: :data:`~whiteout.types.VEHICLE_CLASSES` rather than listed, so a class
#: added to the build is steerable here unless it is named above.
STEERABLE_CLASSES = frozenset(VEHICLE_CLASSES) - STATIC_CLASSES


def _name(transport: object) -> str:
    return type(transport).__name__


@contextmanager
def _raises(expected: type[BaseException], what: str) -> Iterator[None]:
    """Assert the block raises ``expected``, reporting ``what`` if it does not."""
    try:
        yield
    except expected:
        return
    raise AssertionError(f"{what} did not raise {expected.__name__}")


@contextmanager
def _accepts(transport: Transport, what: str) -> Iterator[None]:
    """Assert the block is not refused, reporting a refusal as a breach.

    The mirror of :func:`_raises`, and what keeps this module's promise that
    a breach arrives as an ``AssertionError``. A ``TransportError`` from a
    call the contract says must succeed *is* the breach the surrounding check
    exists to catch, so letting it escape would report the adapter's own
    fault type from inside the suite and would defeat a guard case written in
    the usual ``pytest.raises(AssertionError)`` style.
    """
    try:
        yield
    except TransportError as refused:
        raise AssertionError(f"{_name(transport)}: {what} was refused: {refused}") from refused


@contextmanager
def _closed_after(transport: Transport) -> Iterator[None]:
    """Run the block, then close — reporting a refused close as a breach.

    ``close`` is "idempotent, and safe to call unconnected"
    (``whiteout/transport/base.py``), so refusing it is a breach like any
    other and must arrive as an ``AssertionError``. It cannot simply be run
    inside :func:`_accepts`, because this close is a teardown: when the block
    has already failed, the check's own assertion is the thing worth
    reporting, and an exception raised out of the teardown would replace it
    with a sentence about the wrong call. So a refusal here is reported only
    when the block itself succeeded, and is swallowed otherwise — either way
    the adapter's own fault type never reaches the caller.
    """
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            transport.close()
        except TransportError as refused:
            if not failed:
                raise AssertionError(
                    f"{_name(transport)}: close() at the end of the episode was refused: {refused}"
                ) from refused


@contextmanager
def _connected(factory: TransportFactory) -> Iterator[Transport]:
    """A fresh connected transport, closed however the block ends."""
    transport = factory()
    transport.connect()
    with _closed_after(transport):
        yield transport


def _assert_no_truth(value: Any, where: str) -> None:
    """Recursively assert ``value`` carries no ground truth.

    Walks dataclasses by field, sequences by index and mappings by key, so a
    truth object nested three deep inside an observation is found where a
    ``hasattr(observation, "truth")`` would report the seam clean. Both the
    type and the field name are checked: an adapter that hands ground truth
    across under a local name is the same leak as one that imports
    :class:`~whiteout.types.Truth`.
    """
    assert not isinstance(value, TRUTH_TYPES), f"{where} is ground truth ({type(value).__name__})"
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            assert "truth" not in field.name.lower(), f"{where}.{field.name} is ground truth"
            _assert_no_truth(getattr(value, field.name), f"{where}.{field.name}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_no_truth(item, f"{where}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert "truth" not in str(key).lower(), f"{where}[{key!r}] is ground truth"
            _assert_no_truth(item, f"{where}[{key!r}]")


# --------------------------------------------------------------------------
# The checks. Each takes a factory; each raises AssertionError on a breach.
# --------------------------------------------------------------------------


def check_the_protocol_surface_is_the_whole_seam(factory: TransportFactory) -> None:
    """The four methods, and a structural match against the protocol."""
    transport = factory()
    with _closed_after(transport):
        assert isinstance(transport, Transport), f"{_name(transport)} is not a Transport"
        for method in TRANSPORT_METHODS:
            attribute = getattr(transport, method, None)
            assert callable(attribute), f"{_name(transport)}.{method} is not callable"


def check_observe_and_command_refuse_before_connect(factory: TransportFactory) -> None:
    """Ordering. The link is not up, so there is nothing truthful to return.

    An adapter that connects lazily on first use looks identical until the
    link is down, at which point it reports a fault from the wrong call.
    """
    transport = factory()
    with _closed_after(transport):
        with _raises(TransportError, f"{_name(transport)}.observe() before connect()"):
            transport.observe()
        with _raises(TransportError, f"{_name(transport)}.command() before connect()"):
            transport.command(FleetIntent(t=0.0, intents=()))


def check_connect_is_idempotent_while_the_link_is_up(factory: TransportFactory) -> None:
    """A second ``connect`` on a live link is not an error and does not rewind."""
    transport = factory()
    transport.connect()
    with _closed_after(transport):
        with _accepts(transport, "observe() on a live link"):
            first = transport.observe()
        with _accepts(transport, "a second connect() on a live link"):
            transport.connect()
        with _accepts(transport, "observe() after a second connect()"):
            second = transport.observe()
        assert second.t > first.t, (
            f"{_name(transport)}: a second connect() took the clock from "
            f"t={first.t} to t={second.t}"
        )


def check_the_tick_clock_is_monotonic(factory: TransportFactory) -> None:
    """Every ``observe`` advances the tick clock, and never turns it back.

    The assertion below is a strict increase, and the log's ordering rule is
    not the reason for it: ``whiteout/log.py`` accepts a repeated ``t`` and
    refuses only one that runs backwards, so a stalled clock would pass
    there. The reason is what ``t`` *is*: **the tick is our decision cadence,
    not the world's clock.** ``observe`` is defined to serve one tick and
    advance, so a tick that stands still is a coordinator that has stopped
    deciding while its calls keep returning — a fault with no other symptom,
    because the episode goes on producing records.

    ``whiteout/transport/base.py`` states that same strict increase on
    :meth:`Transport.observe`, and this check is what holds a transport to
    it. Strictness costs an adapter nothing, because ``base.py`` also
    requires the tick to come from the adapter's own monotonic counter and
    never from the far side's clock, which may be sampled, may drift, and
    can return the same instant twice.

    A clock that goes *backwards* costs more than the tick: the log refuses a
    record whose ``t`` is less than the previous one, so a transport that
    rewound would lose the whole episode at write time rather than report the
    fault where it happened.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        with _accepts(transport, "observe() on a live link"):
            times = [transport.observe().t for _ in range(TICKS)]
    for t in times:
        assert isinstance(t, float), f"{where}: observe() returned t={t!r}, not a float"
        assert math.isfinite(t), f"{where}: observe() returned t={t!r}"
    for earlier, later in zip(times, times[1:], strict=False):
        assert later > earlier, f"{where}: observe() went from t={earlier} to t={later}"


def check_silence_is_not_a_fault_and_every_report_is_attributable(
    factory: TransportFactory,
) -> None:
    """Silence is data, not a fault, and a report names an asset that is here.

    An asset that says nothing this tick simply has no report this tick, and
    a tick that carries no reports at all is the ordinary case rather than a
    fault: the transport keeps serving, tick after tick, and every tick still
    poses at least one asset. What it does report must be attributable — a
    report names an asset the transport is posing this same tick, and is
    stamped at this same tick. ``SPEC.md`` §4 makes non-detection the input to
    the negative-information update, so a misattributed or stale report is
    evidence the belief field will erode on — the expensive kind of wrong,
    because it looks plausible.

    What is asserted about the fleet is that it is *non-empty*, not that it
    keeps its shape: a transport that returns no pose at all has stopped
    reporting a world, but an asset that drops out mid-episode is an ordinary
    thing for a link to a real vehicle to do, and pinning the roster here
    would make that non-conforming.

    **This check cannot catch invention**, and does not claim to. A transport
    that manufactures a plausible report every tick, for an asset whose
    sensor saw nothing, passes it. Nothing crosses this seam that says what
    the sensor actually saw, so a black-box check over an adapter has nothing
    to compare a report against; catching a fabricated detection needs the
    ground truth only the sim holds, and belongs to ``whiteout/sim/``'s own
    tests. What is caught here is misattribution and a stale stamp.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        for tick in range(TICKS):
            with _accepts(transport, f"observe() on tick {tick}"):
                observation = transport.observe()
            assert isinstance(observation, WorldObservation), (
                f"{where}: observe() returned {type(observation).__name__}"
            )
            assert isinstance(observation.reports, tuple), (
                f"{where} tick {tick}: reports is {type(observation.reports).__name__}"
            )
            assert observation.poses, f"{where} tick {tick}: the observation poses no asset at all"
            assets = [pose.asset_id for pose in observation.poses]
            assert len(assets) == len(set(assets)), (
                f"{where} tick {tick}: two poses for one asset in {assets}"
            )
            for pose in observation.poses:
                # An equality, not `<=`. `Pose.t` is the tick the pose
                # belongs to, so an observation is a coherent snapshot; a
                # fix taken earlier says so in `measured_t`, which the next
                # check owns. Relaxing this would leave the field, its type
                # and its value plausible while changing what every reader
                # of `pose.t` is looking at.
                assert pose.t == observation.t, (
                    f"{where} tick {tick}: pose {pose.asset_id} is stamped t={pose.t} "
                    f"in an observation at t={observation.t}"
                )
            for report in observation.reports:
                assert report.asset_id in set(assets), (
                    f"{where} tick {tick}: a report from {report.asset_id}, "
                    f"which has no pose this tick"
                )
                assert report.t == observation.t, (
                    f"{where} tick {tick}: a report from {report.asset_id} stamped "
                    f"t={report.t} in an observation at t={observation.t}"
                )


def check_a_reported_measurement_time_is_finite_and_never_after_its_tick(
    factory: TransportFactory,
) -> None:
    """A fix is taken before the tick it lands in, or the transport says nothing.

    ``Pose.t`` is the tick the pose belongs to and equals the observation's
    ``t``, which the check above asserts. When the underlying fix was taken
    earlier — the ordinary case on a link to a real vehicle — the transport
    reports when in ``measured_t``, and ``t - measured_t`` is the age a
    consumer corrects for.

    ``measured_t`` is ``None`` when the transport does not know, and ``None``
    is not a breach: it is the honest answer, and the reason the field does
    not default to ``t``. What is asserted is only about a transport that
    *does* answer. A ``measured_t`` after its tick is a fix from the future,
    which a staleness correction would read as a negative age; a non-finite
    one cannot be written to an episode log at all (``whiteout/log.py``) and
    would silently poison any arithmetic done on it — ``-inf`` in particular
    satisfies ``<= t`` and would otherwise pass here.

    **A plausible measurement time is not a true one.** Nothing crosses this
    seam to say when a fix was really taken, so a transport that stamps every
    pose one tick old passes. What is caught is the incoherent answer.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        for tick in range(TICKS):
            with _accepts(transport, f"observe() on tick {tick}"):
                observation = transport.observe()
            for pose in observation.poses:
                measured_t = pose.measured_t
                if measured_t is None:
                    continue
                assert isinstance(measured_t, float), (
                    f"{where} tick {tick}: pose {pose.asset_id} reports "
                    f"measured_t={measured_t!r}, not a float"
                )
                assert math.isfinite(measured_t), (
                    f"{where} tick {tick}: pose {pose.asset_id} reports measured_t={measured_t!r}"
                )
                assert measured_t <= pose.t, (
                    f"{where} tick {tick}: pose {pose.asset_id} was measured at "
                    f"t={measured_t}, after the tick it is stamped at, t={pose.t}"
                )


def check_command_is_accepted_for_every_tick(factory: TransportFactory) -> None:
    """Every tick's intents are accepted, including an empty one.

    An empty ``FleetIntent`` is the ordinary case for a tick the policy has
    nothing to say on, and it must not be mistaken for a fault. Every tick,
    and not just the first two: a transport that took one intent and then
    faulted would strand an episode part-way through, so the whole of
    :data:`TICKS` is commanded. Accepted means the call was not refused and
    the episode carried on — the clock still advances across a commanded
    tick.

    Each asset is asked only for what its class can do. A
    :class:`~whiteout.types.WaypointIntent` says where to go, and an asset in
    :data:`STATIC_CLASSES` cannot go anywhere, so no waypoint is addressed to
    one and refusing such a waypoint is not a breach of this check. An
    adapter free to reject what its vehicle has no channel for is the point
    of the seam; requiring it to pretend otherwise would write an assumption
    about one fleet into the contract every fleet is held to.

    **Accepted is not delivered.** ``command`` returns nothing, so nothing
    crosses this seam to say the intent reached an asset, and a transport
    that drops every intent on the floor passes this check. Proving delivery
    needs an acknowledgement the protocol does not have
    (``whiteout/transport/base.py``); until it does, that half is an
    adapter's own tests'.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        with _accepts(transport, "observe() on a live link"):
            observation = transport.observe()
        for tick in range(TICKS):
            # Tick zero sends the empty intent — the ordinary quiet tick —
            # and the rest steer every posed asset that can be steered.
            steerable = [pose for pose in observation.poses if pose.cls in STEERABLE_CLASSES]
            targets = [enu_to_geodetic(pose.lat, pose.lon, 100.0, 100.0) for pose in steerable]
            intents = tuple(
                WaypointIntent(
                    asset_id=pose.asset_id,
                    t=observation.t,
                    target_lat=target.lat_deg,
                    target_lon=target.lon_deg,
                    target_z=pose.z,
                    speed=10.0,
                    reason="sweep",
                    task_id=f"conformance-{tick}-{index}",
                )
                for index, (pose, target) in enumerate(zip(steerable, targets, strict=True))
            )
            if tick == 0:
                intents = ()
            with _accepts(transport, f"command() on tick {tick} carrying {len(intents)} intents"):
                transport.command(FleetIntent(t=observation.t, intents=intents))
            with _accepts(transport, f"observe() after commanding tick {tick}"):
                after = transport.observe()
            assert after.t > observation.t, (
                f"{where} tick {tick}: the clock did not advance across a commanded "
                f"tick (t={observation.t} then t={after.t})"
            )
            observation = after


def check_close_mid_episode_ends_the_episode(factory: TransportFactory) -> None:
    """After ``close``, the link is down: nothing more may be served.

    A transport that kept serving after ``close`` would let the episode loop's
    ``finally`` be followed by observations nobody meant to take.
    """
    transport = factory()
    transport.connect()
    where = _name(transport)
    with _accepts(transport, "observe() on a live link"):
        transport.observe()
        transport.observe()
    with _accepts(transport, "close() on a live link"):
        transport.close()
    with _raises(TransportError, f"{where}.observe() after close()"):
        transport.observe()
    with _raises(TransportError, f"{where}.command() after close()"):
        transport.command(FleetIntent(t=0.0, intents=()))
    with _accepts(transport, "a second close() after close()"):
        transport.close()


def check_connect_after_close_refuses_rather_than_rewinding(factory: TransportFactory) -> None:
    """A transport is single-use: reconnect is a new instance, never a rewind.

    Recovering a dropped link is the adapter's business inside its own
    ``connect``/``observe``, where it can re-establish it without rewinding
    the clock it has already reported. The caller cannot tell the two apart,
    so it is never the caller's.
    """
    transport = factory()
    transport.connect()
    where = _name(transport)
    with _accepts(transport, "observe() on a live link"):
        first = transport.observe()
    with _accepts(transport, "close() on a live link"):
        transport.close()
    with _raises(TransportError, f"{where}.connect() after close()"):
        transport.connect()
    assert math.isfinite(first.t), f"{where}: the episode's first tick was t={first.t!r}"
    with _raises(TransportError, f"{where}.observe() after a refused reconnect"):
        transport.observe()


def check_close_is_safe_before_connect(factory: TransportFactory) -> None:
    """``close()`` in a ``finally`` runs whether or not ``connect()`` did.

    ``whiteout.cli run`` releases a partial connect that way, so closing a
    link that was never up must neither raise nor spend the instance.
    """
    transport = factory()
    with _accepts(transport, "close() before connect()"):
        transport.close()
        transport.close()
    with _accepts(transport, "connect() after a close() on a link that was never up"):
        transport.connect()
    with _closed_after(transport):
        with _accepts(transport, "observe() after a close() on a link that was never up"):
            transport.observe()


def check_observe_never_returns_ground_truth(factory: TransportFactory) -> None:
    """The observation is exactly what the coordinator is allowed to see.

    Checked three ways, because the interesting leak is not a bare
    :class:`~whiteout.types.Truth` return: the observation's field set is
    pinned, the object graph is walked for truth types and truth-named
    fields, and its JSON shape is walked for a truth-named key — which is
    what a subclass carrying an extra field, or a dict smuggled through a
    member, would show up as.

    The JSON shape is also fed back through
    :meth:`~whiteout.types.WorldObservation.from_dict` and encoded the way
    ``whiteout/log.py`` encodes it (``allow_nan=False``), so what crosses the
    seam is a record this build can *write*. An in-process
    :class:`~whiteout.types.Pose` validates only that its floats are floats
    (``whiteout/types.py``), not that its ``cls`` is a known vehicle class
    or that those floats are finite. So an adapter mapping ``MAV_TYPE`` to
    ArduPilot's own ``plane`` / ``copter`` / ``tracker`` spellings, or one
    passing through a NaN altitude from a vehicle with no lock, would
    otherwise pass every check here and then be unable to write episode
    record one.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        for tick in range(TICKS):
            with _accepts(transport, f"observe() on tick {tick}"):
                observation = transport.observe()
            assert isinstance(observation, WorldObservation), (
                f"{where}: observe() returned {type(observation).__name__}"
            )
            present = {field.name for field in fields(observation)}
            assert present == OBSERVATION_FIELDS, (
                f"{where}: observe() returned an observation with fields "
                f"{sorted(present)}, not {sorted(OBSERVATION_FIELDS)}"
            )
            payload = observation.to_dict()
            _assert_no_truth(observation, f"{where} observation")
            _assert_no_truth(payload, f"{where} observation.to_dict()")
            try:
                WorldObservation.from_dict(payload)
                json.dumps(payload, allow_nan=False)
            except ValueError as rejected:  # RecordError, or a non-finite float
                raise AssertionError(
                    f"{where}: observe() returned an observation this build cannot "
                    f"write to an episode log: {rejected}"
                ) from rejected


#: Every check, in lifecycle order. The runner parameterises over this, so a
#: check added here is run against every transport the build registers.
CONFORMANCE_CHECKS: tuple[Callable[[TransportFactory], None], ...] = (
    check_the_protocol_surface_is_the_whole_seam,
    check_observe_and_command_refuse_before_connect,
    check_connect_is_idempotent_while_the_link_is_up,
    check_the_tick_clock_is_monotonic,
    check_silence_is_not_a_fault_and_every_report_is_attributable,
    check_a_reported_measurement_time_is_finite_and_never_after_its_tick,
    check_command_is_accepted_for_every_tick,
    check_close_mid_episode_ends_the_episode,
    check_connect_after_close_refuses_rather_than_rewinding,
    check_close_is_safe_before_connect,
    check_observe_never_returns_ground_truth,
)


def check_id(check: Callable[[TransportFactory], None]) -> str:
    """The short name a check is reported under in a test id."""
    return check.__name__.removeprefix("check_")


def unavailable(factory: TransportFactory) -> str | None:
    """Why this transport cannot be exercised here, or ``None`` if it can.

    An adapter whose link needs something this machine does not have refuses
    at ``connect`` (``SPEC.md`` §7 makes that the normal path for one of
    them), or from its constructor, which
    ``whiteout/transport/__init__.py`` documents as equally legitimate "for a
    seed or a configuration it cannot use". Both are covered: the suite has
    nothing to say about a transport it cannot start, however early it says
    so. The runner skips those *by name*, so a refusal can never be mistaken
    for a pass on the transport the gate actually runs.
    """
    try:
        transport = factory()
        with _closed_after(transport):
            transport.connect()
    except TransportError as refused:
        return str(refused)
    return None
