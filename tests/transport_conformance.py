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
rather than as a traceback through the suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any

from whiteout.transport import TRANSPORT_METHODS, Transport, TransportError
from whiteout.types import FleetIntent, TargetTruth, Truth, WaypointIntent, WorldObservation

__all__ = [
    "CONFORMANCE_CHECKS",
    "TransportFactory",
    "check_id",
    "unavailable",
]

#: A zero-argument callable returning a fresh, unconnected transport.
TransportFactory = Callable[[], Transport]

#: How many ticks the multi-tick checks drive. Enough that a clock which only
#: advances on the first call, or a fleet that changes shape after a tick or
#: two, is caught; small enough that ten checks per transport stay far inside
#: the gate's three-minute budget (``SPEC.md`` §6).
TICKS = 5

#: The observation's whole shape. Pinned, not merely searched for a forbidden
#: name: a field added to what crosses the seam is a widening of the seam, and
#: §4 makes that a deliberate edit rather than something an adapter may do.
OBSERVATION_FIELDS = frozenset({"t", "poses", "reports"})

#: Ground truth. ``SPEC.md`` §4: written by the sim, read only by the scorer
#: and the viewer, and never observed. It is the single easiest way to cheat
#: ourselves and believe the number afterwards.
TRUTH_TYPES: tuple[type, ...] = (Truth, TargetTruth)


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
def _connected(factory: TransportFactory) -> Iterator[Transport]:
    """A fresh connected transport, closed however the block ends."""
    transport = factory()
    transport.connect()
    try:
        yield transport
    finally:
        transport.close()


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
    try:
        assert isinstance(transport, Transport), f"{_name(transport)} is not a Transport"
        for method in TRANSPORT_METHODS:
            attribute = getattr(transport, method, None)
            assert callable(attribute), f"{_name(transport)}.{method} is not callable"
    finally:
        transport.close()


def check_observe_and_command_refuse_before_connect(factory: TransportFactory) -> None:
    """Ordering. The link is not up, so there is nothing truthful to return.

    An adapter that connects lazily on first use looks identical until the
    link is down, at which point it reports a fault from the wrong call.
    """
    transport = factory()
    try:
        with _raises(TransportError, f"{_name(transport)}.observe() before connect()"):
            transport.observe()
        with _raises(TransportError, f"{_name(transport)}.command() before connect()"):
            transport.command(FleetIntent(t=0.0, intents=()))
    finally:
        transport.close()


def check_connect_is_idempotent_while_the_link_is_up(factory: TransportFactory) -> None:
    """A second ``connect`` on a live link is not an error and does not rewind."""
    transport = factory()
    try:
        transport.connect()
        first = transport.observe()
        transport.connect()
        second = transport.observe()
        assert second.t > first.t, (
            f"{_name(transport)}: a second connect() took the clock from "
            f"t={first.t} to t={second.t}"
        )
    finally:
        transport.close()


def check_the_tick_clock_is_monotonic(factory: TransportFactory) -> None:
    """Every ``observe`` advances the world clock, and never turns it back.

    ``whiteout/log.py`` refuses a record whose ``t`` goes backwards, so a
    transport that repeated or rewound a tick would lose the whole episode at
    write time rather than report the fault where it happened.
    """
    with _connected(factory) as transport:
        times = [transport.observe().t for _ in range(TICKS)]
        where = _name(transport)
    for t in times:
        assert isinstance(t, float), f"{where}: observe() returned t={t!r}, not a float"
        assert math.isfinite(t), f"{where}: observe() returned t={t!r}"
    for earlier, later in zip(times, times[1:], strict=False):
        assert later > earlier, f"{where}: observe() went from t={earlier} to t={later}"


def check_a_silent_asset_is_absent_and_never_fabricated(factory: TransportFactory) -> None:
    """Silence is data, not a fault, and not something to fill in.

    An asset that says nothing this tick simply has no report this tick: the
    transport must keep ticking, must not invent one, and must not attribute
    a report to an asset it is not reporting a pose for. ``SPEC.md`` §4 makes
    non-detection the input to the negative-information update, so a
    fabricated or misattributed report is evidence the belief field will
    erode on — the expensive kind of wrong, because it looks plausible.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        for tick in range(TICKS):
            observation = transport.observe()
            assert isinstance(observation, WorldObservation), (
                f"{where}: observe() returned {type(observation).__name__}"
            )
            assert isinstance(observation.reports, tuple), (
                f"{where} tick {tick}: reports is {type(observation.reports).__name__}"
            )
            assets = [pose.asset_id for pose in observation.poses]
            assert len(assets) == len(set(assets)), (
                f"{where} tick {tick}: two poses for one asset in {assets}"
            )
            for pose in observation.poses:
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


def check_command_is_accepted_for_every_tick(factory: TransportFactory) -> None:
    """Intents go out tick after tick, including an empty one.

    An empty ``FleetIntent`` is the ordinary case for a tick the policy has
    nothing to say on, and it must not be mistaken for a fault.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        observation = transport.observe()
        transport.command(FleetIntent(t=observation.t, intents=()))
        observation = transport.observe()
        intents = tuple(
            WaypointIntent(
                asset_id=pose.asset_id,
                t=observation.t,
                target_xy=(pose.x + 100.0, pose.y + 100.0),
                target_z=pose.z,
                speed=10.0,
                reason="sweep",
                task_id=f"conformance-{index}",
            )
            for index, pose in enumerate(observation.poses)
        )
        transport.command(FleetIntent(t=observation.t, intents=intents))
        after = transport.observe()
        assert after.t > observation.t, (
            f"{where}: the clock did not advance across a commanded tick "
            f"(t={observation.t} then t={after.t})"
        )


def check_close_mid_episode_ends_the_episode(factory: TransportFactory) -> None:
    """After ``close``, the link is down: nothing more may be served.

    A transport that kept serving after ``close`` would let the episode loop's
    ``finally`` be followed by observations nobody meant to take.
    """
    transport = factory()
    transport.connect()
    transport.observe()
    transport.observe()
    transport.close()
    where = _name(transport)
    with _raises(TransportError, f"{where}.observe() after close()"):
        transport.observe()
    with _raises(TransportError, f"{where}.command() after close()"):
        transport.command(FleetIntent(t=0.0, intents=()))
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
    first = transport.observe()
    transport.close()
    where = _name(transport)
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
    transport.close()
    transport.close()
    transport.connect()
    try:
        transport.observe()
    finally:
        transport.close()


def check_observe_never_returns_ground_truth(factory: TransportFactory) -> None:
    """The observation is exactly what the coordinator is allowed to see.

    Checked three ways, because the interesting leak is not a bare
    :class:`~whiteout.types.Truth` return: the observation's field set is
    pinned, the object graph is walked for truth types and truth-named
    fields, and its JSON shape is walked for a truth-named key — which is
    what a subclass carrying an extra field, or a dict smuggled through a
    member, would show up as.
    """
    with _connected(factory) as transport:
        where = _name(transport)
        for _ in range(TICKS):
            observation = transport.observe()
            assert isinstance(observation, WorldObservation), (
                f"{where}: observe() returned {type(observation).__name__}"
            )
            present = {field.name for field in fields(observation)}
            assert present == OBSERVATION_FIELDS, (
                f"{where}: observe() returned an observation with fields "
                f"{sorted(present)}, not {sorted(OBSERVATION_FIELDS)}"
            )
            _assert_no_truth(observation, f"{where} observation")
            _assert_no_truth(observation.to_dict(), f"{where} observation.to_dict()")


#: Every check, in lifecycle order. The runner parameterises over this, so a
#: check added here is run against every transport the build registers.
CONFORMANCE_CHECKS: tuple[Callable[[TransportFactory], None], ...] = (
    check_the_protocol_surface_is_the_whole_seam,
    check_observe_and_command_refuse_before_connect,
    check_connect_is_idempotent_while_the_link_is_up,
    check_the_tick_clock_is_monotonic,
    check_a_silent_asset_is_absent_and_never_fabricated,
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
    them), and the suite has nothing to say about a transport it cannot
    start. The runner skips those *by name*, so a refusal can never be
    mistaken for a pass on the transport the gate actually runs.
    """
    transport = factory()
    try:
        transport.connect()
    except TransportError as refused:
        return str(refused)
    finally:
        transport.close()
    return None
