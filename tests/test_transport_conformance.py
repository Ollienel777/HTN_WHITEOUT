"""Issue #9: the conformance suite, run against everything this build registers.

``tests/transport_conformance.py`` is the suite — the behaviour any transport
owes its caller. This module is the runner, and it is deliberately thin: it
derives what to run from :data:`~whiteout.transport.TRANSPORT_FACTORIES`, so
the `sitl` and `arena` adapters inherit the suite by being registered and
nothing here is edited when they arrive. ``hackathon/ARENA.md`` §7 settled
what `arena` will be — MAVLink per asset plus an HTTP tracks API — which is
the adapter the seam was built for, and the one this suite exists to hand a
definition of done to.

Three groups, one per acceptance criterion:

1. the suite is importable and parameterised over a transport factory;
2. `kinematic` passes it;
3. adding a transport requires no change to the suite — asserted statically,
   because a suite that had to be edited per adapter would be edited *around*
   the next adapter at hour 25.

The last group here is the suite's own guard: a conformance suite that passes
everything is worse than none, because it is believed. Each case plants one
breach in a transport that is otherwise conforming and asserts the check that
owns that breach fails on it.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import transport_conformance as conformance
from transport_conformance import CONFORMANCE_CHECKS, TransportFactory, check_id, unavailable

from whiteout.transport import (
    IMPLEMENTED_TRANSPORTS,
    TRANSPORT_FACTORIES,
    TRANSPORT_NAMES,
    KinematicTransport,
    Transport,
    create_transport,
)
from whiteout.types import WorldObservation

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
    for behaviour in ("before_connect", "monotonic", "silent", "close_mid_episode", "truth"):
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
        assert implementation.__name__.lower() not in text


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
    """Reports every tick at t=0.0 — the clock that costs the episode."""

    def observe(self) -> WorldObservation:
        observation = super().observe()
        return WorldObservation(t=0.0, poses=observation.poses, reports=observation.reports)


class _EagerTransport(KinematicTransport):
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


class _FabricatingTransport(KinematicTransport):
    """Stamps stale poses with the current tick for an asset that is silent."""

    def observe(self) -> WorldObservation:
        observation = super().observe()
        doubled = observation.poses + observation.poses[:1]
        return WorldObservation(t=observation.t, poses=doubled, reports=observation.reports)


@pytest.mark.parametrize(
    ("transport_class", "check_name"),
    [
        (_RewindingTransport, "check_the_tick_clock_is_monotonic"),
        (_EagerTransport, "check_observe_and_command_refuse_before_connect"),
        (_LenientTransport, "check_close_mid_episode_ends_the_episode"),
        (_LeakingTransport, "check_observe_never_returns_ground_truth"),
        (_FabricatingTransport, "check_a_silent_asset_is_absent_and_never_fabricated"),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else value,
)
def test_the_suite_fails_a_transport_that_breaches_the_contract(
    transport_class: type, check_name: str
) -> None:
    check = getattr(conformance, check_name)
    with pytest.raises(AssertionError):
        check(lambda: transport_class(seed=SEED))


@pytest.mark.parametrize("check", CONFORMANCE_CHECKS, ids=check_id)
def test_a_subclass_that_breaches_nothing_still_passes(check: Check) -> None:
    """The guard must fail breaches and nothing else, or it gets worked around.

    Every fake above is a subclass of the real transport, so if subclassing
    alone failed the suite the cases above would prove nothing.
    """

    class _Conforming(KinematicTransport):
        pass

    check(lambda: _Conforming(seed=SEED))
