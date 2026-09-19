"""Issue #9: the conformance suite, run against everything this build registers.

``tests/transport_conformance.py`` is the suite — the behaviour any transport
owes its caller. This module is the runner, and it is deliberately thin: it
derives what to run from :data:`~whiteout.transport.TRANSPORT_FACTORIES`, so
the `sitl` and `arena` adapters inherit the suite by being registered and
nothing here is edited when they arrive. ``hackathon/ARENA.md`` settled what
`arena` will be — MAVLink per asset (§7, and §1 Q6) plus an HTTP tracks API
for submitting detections (§1 Q6) — which is the adapter the seam was built
for, and the one this suite exists to hand a definition of done to.

Three groups, one per acceptance criterion:

1. the suite is importable and parameterised over a transport factory;
2. `kinematic` passes it;
3. adding a transport requires no change to the suite — asserted statically,
   because a suite that had to be edited per adapter would be edited *around*
   the next adapter at hour 25.

The last group here is the suite's own guard: a conformance suite that passes
everything is worse than none, because it is believed. Each case plants one
breach in a transport that is otherwise conforming and asserts the check that
owns that breach fails on it — as an ``AssertionError``, which is what the
suite promises a breach arrives as. Every check owns at least one planted
breach: a check nothing is planted against is a check nobody has shown to be
load-bearing, and the gap is invisible while everything is green.

Five of the eighteen cases plant a *refusal* rather than a wrong answer — a
second ``connect`` on a live link, a link that drops mid-episode, a ``close``
a live link will not take, a ``command`` the link rejects, and a ``connect``
after a ``close`` — because those are the breaches that arrive as the
adapter's own ``TransportError``, and the suite's conversion of them into
``AssertionError`` is itself a claim worth a guard.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import transport_conformance as conformance
from transport_conformance import (
    CONFORMANCE_CHECKS,
    STATIC_CLASSES,
    TransportFactory,
    check_id,
    unavailable,
)

from whiteout.transport import (
    IMPLEMENTED_TRANSPORTS,
    TRANSPORT_FACTORIES,
    TRANSPORT_NAMES,
    KinematicTransport,
    Transport,
    TransportError,
    create_transport,
)
from whiteout.types import (
    FleetIntent,
    SensorFootprint,
    SensorReport,
    TargetTruth,
    WorldObservation,
)

SUITE = Path(conformance.__file__)

#: The seed every conformance run uses. Any seed; fixed so a failure is
#: reproducible from the test id alone.
SEED = 7

#: What the runner and the guard below pass a check: one argument, a factory.
Check = Callable[[TransportFactory], None]


def factory_for(name: str) -> TransportFactory:
    """A zero-argument factory for the named transport, as the suite wants."""

    def build() -> Transport:
        return create_transport(name, seed=SEED)

    build.__name__ = f"{name}_factory"
    return build


# --------------------------------------------------------------------------
# Acceptance: kinematic passes the suite — and so does every other transport
# this build registers, without this file naming any of them.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", IMPLEMENTED_TRANSPORTS)
@pytest.mark.parametrize("check", CONFORMANCE_CHECKS, ids=check_id)
def test_every_registered_transport_conforms(check: Check, name: str) -> None:
    factory = factory_for(name)
    refused = unavailable(factory)
    if refused is not None:
        pytest.skip(f"{name} cannot be connected here: {refused}")
    check(factory)


def test_the_default_transport_is_never_skipped() -> None:
    """The skip above is for an adapter that needs a link this machine lacks.

    If it ever applied to the transport the gate, the tuner and the demo all
    run, the suite would be green and running against nothing.
    """
    assert unavailable(factory_for("kinematic")) is None


def test_the_runner_covers_every_registered_transport() -> None:
    """The parameterisation is the registry, not a list kept alongside it."""
    assert set(IMPLEMENTED_TRANSPORTS) == set(TRANSPORT_FACTORIES)
    assert IMPLEMENTED_TRANSPORTS, "this build registers no transport at all"


# --------------------------------------------------------------------------
# Acceptance: the suite is importable and parameterised over a factory.
# --------------------------------------------------------------------------


def test_the_suite_is_an_importable_library_not_a_test_module() -> None:
    """Imported above; pytest must not collect it as tests of its own."""
    collected = [name for name in vars(conformance) if name.startswith("test_")]
    assert not collected, f"the suite defines collectable tests: {collected}"
    assert CONFORMANCE_CHECKS, "the suite is empty"


def test_every_check_takes_one_transport_factory_and_says_what_it_checks() -> None:
    for check in CONFORMANCE_CHECKS:
        parameters = list(inspect.signature(check).parameters)
        assert parameters == ["factory"], f"{check.__name__} takes {parameters}"
        assert check.__doc__, f"{check.__name__} has no docstring"
    names = [check.__name__ for check in CONFORMANCE_CHECKS]
    assert len(names) == len(set(names))
    assert all(name.startswith("check_") for name in names), names


def test_the_suite_covers_every_behaviour_the_ticket_names() -> None:
    """Ordering, tick monotonicity, silence, close mid-episode, no truth."""
    names = " ".join(check.__name__ for check in CONFORMANCE_CHECKS)
    for behaviour in ("before_connect", "monotonic", "silence", "close_mid_episode", "truth"):
        assert behaviour in names, f"no check covers {behaviour}"


# --------------------------------------------------------------------------
# Acceptance: adding a transport requires no change to the suite.
# --------------------------------------------------------------------------


def test_the_suite_names_no_transport_implementation() -> None:
    """A suite that knows the fake's name has stopped being a suite.

    Text, not just identifiers: a docstring that says "for the kinematic
    transport" is the first step to a check that only holds for it.
    """
    text = SUITE.read_text(encoding="utf-8").lower()
    for name in TRANSPORT_NAMES:
        assert name not in text, f"the suite names the {name!r} transport"
    for implementation in TRANSPORT_FACTORIES.values():
        # A factory need not be a class: TRANSPORT_FACTORIES is typed
        # `Callable[..., Transport]`, and a `functools.partial` carrying an
        # adapter's endpoint config has no `__name__`. An anonymous factory
        # has no name for the suite to leak, so there is nothing to check.
        registered = getattr(implementation, "__name__", "")
        assert not registered or registered.lower() not in text, (
            f"the suite names the {registered!r} implementation"
        )


def test_the_suite_imports_no_module_inside_the_transport_package() -> None:
    """The seam's package, never a module inside it.

    ``whiteout.transport`` is the interface and the registry; everything
    below it is one implementation or another, and a suite that reached for
    one would be testing it rather than the contract.
    """
    tree = ast.parse(SUITE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "whiteout.transport" in imported, "the suite does not import the seam at all"
    for module in imported:
        assert not module.startswith("whiteout.transport."), f"the suite imports {module}"


# --------------------------------------------------------------------------
# The suite's own guard. Each transport below is conforming except for one
# planted breach, and the check that owns that breach must fail on it.
# --------------------------------------------------------------------------


class _RewindingTransport(KinematicTransport):
    """Reports every tick at t=0.0 — a clock that never advances."""

    def observe(self) -> WorldObservation:
        observation = super().observe()
        return WorldObservation(t=0.0, poses=observation.poses, reports=observation.reports)


class _LazyTransport(KinematicTransport):
    """Connects lazily on first use, so ordering is never enforced."""

    def observe(self) -> WorldObservation:
        if not self.connected:
            self.connect()
        return super().observe()


class _LenientTransport(KinematicTransport):
    """Keeps serving after close(), so the episode has no end."""

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _ObservationWithTruth(WorldObservation):
    """An observation with ground truth bolted on — the leak that matters."""

    truth: object = None


class _LeakingTransport(KinematicTransport):
    """Hands the coordinator ground truth inside the observation."""

    def observe(self) -> WorldObservation:
        observation = super().observe()
        return _ObservationWithTruth(
            t=observation.t,
            poses=observation.poses,
            reports=observation.reports,
            truth=(("target-1", 0.0, 0.0),),
        )


class _DoublePosingTransport(KinematicTransport):
    """Reports one asset twice in a tick, so the fleet is not what it says."""

    def observe(self) -> WorldObservation:
        observation = super().observe()
        doubled = observation.poses + observation.poses[:1]
        return WorldObservation(t=observation.t, poses=doubled, reports=observation.reports)


class _NarrowTransport:
    """Three of the four methods. Not a subclass: the hole is the point.

    A wrapper rather than a subclass because the breach is a *missing*
    method, and the seam is a ``Protocol`` — an adapter wrapping something
    the sponsor hands us is exactly the shape that arrives with a hole in it.
    """

    def __init__(self, seed: int = 0) -> None:
        self._inner = KinematicTransport(seed=seed)

    def connect(self) -> None:
        self._inner.connect()

    def observe(self) -> WorldObservation:
        return self._inner.observe()

    def close(self) -> None:
        self._inner.close()


class _JealousTransport(KinematicTransport):
    """Refuses a second connect on a live link instead of ignoring it."""

    def connect(self) -> None:
        if self.connected:
            raise TransportError("the link is already up")
        super().connect()


class _RefusingTransport(KinematicTransport):
    """Takes the empty intent and faults on a commanded one."""

    def command(self, intent: FleetIntent) -> None:
        if intent.intents:
            raise TransportError("this link carries no intents")
        super().command(intent)


class _ReopeningTransport(KinematicTransport):
    """Reconnects after close(), rewinding a clock the caller has seen."""

    def connect(self) -> None:
        self._closed = False
        super().connect()


class _SpentTransport(KinematicTransport):
    """close() is final even on a link that was never up."""

    def close(self) -> None:
        self._closed = True
        self._connected = False


class _FlakyTransport(KinematicTransport):
    """Serves one tick, then refuses every observe: the link dropped.

    The ordinary shape of a radio link going away mid-episode, and the reason
    a refused ``observe`` has to arrive as an ``AssertionError``: it is the
    adapter's own fault type, raised from a call the contract says must
    succeed.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._served = 0

    def observe(self) -> WorldObservation:
        if self._served >= 1:
            raise TransportError("the link dropped mid-episode")
        self._served += 1
        return super().observe()


class _UnclosableTransport(KinematicTransport):
    """Refuses close() on a live link — an ordinary socket bug.

    ``close`` is documented as idempotent and safe to call unconnected, so
    this is a breach; it is planted because it is the one that fires from a
    teardown, where an unconverted refusal would also mask whatever the check
    was really asserting.
    """

    def close(self) -> None:
        if self.connected:
            raise TransportError("the link refuses to close")
        super().close()


class _EmptyFleetTransport(KinematicTransport):
    """The clock still runs, but nothing is posed: a world with no assets."""

    def observe(self) -> WorldObservation:
        observation = super().observe()
        return WorldObservation(t=observation.t, poses=(), reports=())


class _BackstampingTransport(KinematicTransport):
    """Stamps each pose at when its fix was taken, not at the tick it serves.

    The plausible mistake, and the one the equality exists to catch: every
    number is in the past, nothing is out of order, and a suite asserting
    only ``pose.t <= observation.t`` would certify it. What breaks is that
    an observation stops being one instant — and every consumer of
    ``pose.t`` keeps reading it as the tick.
    """

    def observe(self) -> WorldObservation:
        observation = super().observe()
        poses = tuple(replace(pose, t=pose.t - 1.0) for pose in observation.poses)
        return WorldObservation(t=observation.t, poses=poses, reports=observation.reports)


def _restamp(observation: WorldObservation, measured_t: float) -> WorldObservation:
    """The same observation, with every pose's ``measured_t`` overwritten."""
    poses = tuple(
        replace(pose, measured_t=measured_t) for pose in observation.poses
    )
    return WorldObservation(t=observation.t, poses=poses, reports=observation.reports)


class _PrescientTransport(KinematicTransport):
    """Reports a fix taken after the tick it arrived in — an age below zero.

    The shape an adapter lands in by subtracting the wrong way round, or by
    mixing the far side's clock into a tick of our own: every number is
    finite and plausible, and a staleness correction reads it as a fix from
    the future.
    """

    def observe(self) -> WorldObservation:
        observation = super().observe()
        return _restamp(observation, observation.t + 1.0)


class _UnlockedTransport(KinematicTransport):
    """Reports a non-finite measurement time — the vehicle with no fix.

    ``-inf`` and not ``nan``: it satisfies ``measured_t <= t``, so only the
    finiteness assertion catches it. A log carrying it cannot be written at
    all (``whiteout/log.py``), and any age computed from it is infinite.
    """

    def observe(self) -> WorldObservation:
        observation = super().observe()
        return _restamp(observation, float("-inf"))


def _footprint() -> SensorFootprint:
    """Some footprint. Nothing in the guard below depends on its shape."""
    return SensorFootprint(kind="circle", x=0.0, y=0.0, radius=100.0, heading=0.0, half_angle=0.0)


class _MisattributingTransport(KinematicTransport):
    """Reports from an asset it is not posing — a sighting from nobody."""

    def observe(self) -> WorldObservation:
        observation = super().observe()
        report = SensorReport(
            asset_id="not-in-this-fleet",
            t=observation.t,
            footprint=_footprint(),
            detections=(),
            negative=True,
        )
        return WorldObservation(t=observation.t, poses=observation.poses, reports=(report,))


@dataclass(frozen=True, slots=True)
class _ReportWithContext(SensorReport):
    """A report with room for something the seam never agreed to carry."""

    context: object = None


class _SmugglingTransport(KinematicTransport):
    """Nests ground truth inside a report, under an innocent field name.

    The leak the field-set pin and the truth-named-field walk both miss: the
    observation's own shape is untouched, nothing is called `truth`, and only
    the type walk finds it.
    """

    def observe(self) -> WorldObservation:
        observation = super().observe()
        report = _ReportWithContext(
            asset_id=observation.poses[0].asset_id,
            t=observation.t,
            footprint=_footprint(),
            detections=(),
            negative=True,
            context=TargetTruth(
                target_id="vessel-1",
                x=0.0,
                y=0.0,
                z=0.0,
                heading=0.0,
                speed=0.0,
                target_class="vessel",
            ),
        )
        return WorldObservation(t=observation.t, poses=observation.poses, reports=(report,))


#: Every planted breach, as ``(transport, the check that owns it)``. A tuple
#: and not an inline parameter list, because
#: :func:`test_every_check_owns_a_planted_breach` reads it: a check nothing is
#: planted against is one nobody has shown to be load-bearing, and that gap is
#: invisible while the suite is green.
_SILENCE_CHECK = "check_silence_is_not_a_fault_and_every_report_is_attributable"
_MEASURED_CHECK = "check_a_reported_measurement_time_is_finite_and_never_after_its_tick"

_PLANTED_BREACHES: tuple[tuple[type, str], ...] = (
    (_NarrowTransport, "check_the_protocol_surface_is_the_whole_seam"),
    (_LazyTransport, "check_observe_and_command_refuse_before_connect"),
    (_JealousTransport, "check_connect_is_idempotent_while_the_link_is_up"),
    (_RewindingTransport, "check_the_tick_clock_is_monotonic"),
    (_FlakyTransport, "check_the_tick_clock_is_monotonic"),
    (_UnclosableTransport, "check_close_mid_episode_ends_the_episode"),
    (_DoublePosingTransport, _SILENCE_CHECK),
    (_MisattributingTransport, _SILENCE_CHECK),
    (_EmptyFleetTransport, _SILENCE_CHECK),
    (_BackstampingTransport, _SILENCE_CHECK),
    (_PrescientTransport, _MEASURED_CHECK),
    (_UnlockedTransport, _MEASURED_CHECK),
    (_RefusingTransport, "check_command_is_accepted_for_every_tick"),
    (_LenientTransport, "check_close_mid_episode_ends_the_episode"),
    (_ReopeningTransport, "check_connect_after_close_refuses_rather_than_rewinding"),
    (_SpentTransport, "check_close_is_safe_before_connect"),
    (_LeakingTransport, "check_observe_never_returns_ground_truth"),
    (_SmugglingTransport, "check_observe_never_returns_ground_truth"),
)


@pytest.mark.parametrize(
    ("transport_class", "check_name"),
    _PLANTED_BREACHES,
    ids=lambda value: value.__name__ if isinstance(value, type) else value,
)
def test_the_suite_fails_a_transport_that_breaches_the_contract(
    transport_class: type, check_name: str
) -> None:
    check = getattr(conformance, check_name)
    with pytest.raises(AssertionError):
        check(lambda: transport_class(seed=SEED))


def test_every_check_owns_a_planted_breach() -> None:
    """The guard covers the whole suite, and keeps covering it.

    Without this, a check added later is green from the day it lands and
    nobody finds out whether it can fail until an adapter's behaviour turns
    on it.
    """
    covered = {check_name for _, check_name in _PLANTED_BREACHES}
    unguarded = sorted(
        check.__name__ for check in CONFORMANCE_CHECKS if check.__name__ not in covered
    )
    assert not unguarded, f"no planted breach for: {unguarded}"
    unknown = sorted(covered - {check.__name__ for check in CONFORMANCE_CHECKS})
    assert not unknown, f"a breach is planted against something the suite does not run: {unknown}"


@pytest.mark.parametrize("check", CONFORMANCE_CHECKS, ids=check_id)
def test_a_subclass_that_breaches_nothing_still_passes(check: Check) -> None:
    """The guard must fail breaches and nothing else, or it gets worked around.

    Every fake above is a subclass of the real transport, so if subclassing
    alone failed the suite the cases above would prove nothing.
    """

    class _Conforming(KinematicTransport):
        pass

    check(lambda: _Conforming(seed=SEED))


@pytest.mark.parametrize("check", CONFORMANCE_CHECKS, ids=check_id)
def test_a_transport_that_reports_a_measurement_time_still_conforms(check: Check) -> None:
    """Reporting fix age is conforming behaviour, not a breach.

    The guard above plants two incoherent measurement times; this is the
    coherent one, driven through the transport's own configurable pose age so
    that the whole suite — not just the check that owns the field — is run
    against poses carrying a ``measured_t``. Without it, every check the
    build exercises would only ever see ``None`` there, and the
    staleness-correction path would first execute against an adapter we get
    one run at.
    """
    check(lambda: KinematicTransport(seed=SEED, pose_age_seconds=2.0))


class _StaticAssetTransport(KinematicTransport):
    """Refuses a waypoint addressed to an asset that cannot move.

    ``hackathon/ARENA.md`` §3: two of the four real assets are ArduPilot
    AntennaTrackers, steered by `servo set 1/2` pan and tilt with no waypoint
    channel at all. This is the `arena` adapter's honest behaviour, not a
    breach, and the suite must not certify against it.
    """

    def command(self, intent: FleetIntent) -> None:
        self._require_connected("command")
        static = {pose.asset_id for pose in self._poses if pose.cls in STATIC_CLASSES}
        for one in intent.intents:
            if one.asset_id in static:
                raise TransportError(f"{one.asset_id} is static: no waypoint channel")
        super().command(intent)


@pytest.mark.parametrize("check", CONFORMANCE_CHECKS, ids=check_id)
def test_an_asset_with_no_waypoint_channel_is_not_a_breach(check: Check) -> None:
    """Each asset is asked only for what its class can do.

    Requiring every transport to take a waypoint for every posed asset would
    write one fleet's shape into the contract every fleet is held to, and
    would make the adapter this suite exists to hand a definition of done to
    non-conforming on its first commanded tick.
    """
    check(lambda: _StaticAssetTransport(seed=SEED))
