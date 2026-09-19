"""The seam: the one interface between the coordinator and the world.

``hackathon/SPEC.md`` §4 "The seam — structural and non-negotiable". Poses and
sensor reports in, waypoint intents out, and **nothing else crosses it**.

Three implementations sit behind it — ``kinematic`` (the dev, tuning, CI and
demo loop), ``sitl`` (ArduPilot validation) and ``arena`` (the sponsor's
harness, whose interface is not known yet). The point of the interface is that
if the sponsor's turns out not to be plain MAVLink over a socket, only the
adapter is lost: never the estimator, never the policy, never a tuned
parameter.

What that costs is a rule, and the rule is guarded by a static test
(``tests/test_transport.py``): **no belief, policy or scoring concept may
appear in ``whiteout/transport/``.** The guard reads every module in the
package with :mod:`ast` and catches what an implementer writes by accident;
it is not a sandbox, and its own docstrings state what it does not catch.
Concretely, the four methods below are
the whole protocol; the only record types named here are
:class:`~whiteout.types.WorldObservation`, :class:`~whiteout.types.FleetIntent`
and — in prose, about the tick clock — the former's own
:class:`~whiteout.types.Pose`; and this package imports nothing from
``whiteout.belief``, ``whiteout.policy``, ``whiteout.score`` or
``whiteout.sim``. A transport that knew what a contact, a belief digest, a
score or ground truth was would have to be reimplemented three times, and
would be reimplemented wrong twice.

:class:`Transport` is a :class:`typing.Protocol`, not a base class: the
``arena`` adapter may well have to wrap an object the sponsor hands us, and
structural typing lets it satisfy the seam without inheriting from it. It is
``runtime_checkable`` so that a test can assert conformance, but note that
``isinstance`` against a runtime-checkable Protocol only checks that the
method *names* exist.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from whiteout.types import FleetIntent, WorldObservation

__all__ = ["TRANSPORT_METHODS", "Transport", "TransportError"]

#: The protocol's whole surface. A test pins this against
#: :class:`Transport`, so widening the seam is a deliberate edit here and a
#: failing test until it is made.
TRANSPORT_METHODS: tuple[str, ...] = ("connect", "observe", "command", "close")


class TransportError(RuntimeError):
    """A transport could not be selected, started, or used as called.

    Raised for an unknown or unimplemented ``WHITEOUT_TRANSPORT``, for a seed
    a transport cannot use, and for a call made out of order (``observe``
    before ``connect``, or ``connect`` after ``close``). ``whiteout.cli run``
    wraps transport selection *and* the whole drive — ``connect``, ``observe``
    and ``command`` — so every one of those turns into a diagnostic and a
    non-zero exit rather than a traceback, and ``close()`` still runs on the
    way out. ``SPEC.md`` §7 makes connect-time refusal ``sitl``'s normal path
    (``WHITEOUT_SITL_ENDPOINT`` absent), so this is the path that matters.
    """


@runtime_checkable
class Transport(Protocol):
    """The one interface the coordinator talks to the world through.

    **A transport is single-use, and reconnect is not supported.** One
    instance drives one episode: ``connect`` once, ``observe``/``command``
    for as many ticks as the episode runs, ``close`` once. The tick clock
    :meth:`observe` returns strictly increases for the life of the instance,
    and the only way to get a clock back at ``t=0`` is a new instance.

    That is a decision, not an omission. The episode log's reader refuses a
    record whose ``t`` goes backwards (``whiteout/log.py``), so a transport
    that silently restarted its clock on reconnect would lose the *whole*
    episode at write time rather than report the reconnect where it
    happened. Recovering a dropped link is the adapter's business, inside
    its own ``connect``/``observe``, where it can re-establish the link
    without rewinding the clock it has already reported; it is never the
    caller's, because the caller cannot tell the two apart.
    """

    def connect(self) -> None:
        """Bring the link up.

        Idempotent *while up*: calling it again on a live link is not an
        error and does not restart the clock. Calling it after
        :meth:`close` has brought a live link down raises
        :class:`TransportError` — see the class docstring. Raises
        :class:`TransportError` if the link cannot be brought up at all,
        which for ``sitl`` is the normal case when its endpoint is unset.
        """

    def observe(self) -> WorldObservation:
        """Return the next tick's observation — everything in.

        The transport owns the clock: each call returns the observation for
        one tick and advances to the next, so ``WorldObservation.t`` is the
        transport's, not the caller's. Raises :class:`TransportError` if the
        link is not up.

        **``t`` is our decision cadence, not the world's clock, and it
        strictly increases.** Each call is defined to serve one tick and
        advance, so two calls that return the same ``t`` are a coordinator
        that has stopped deciding while its calls keep returning — a fault
        that is invisible from the outside, because the episode goes on
        producing records. That, and not the log's ordering rule, is why the
        conformance suite asserts a strict increase: ``whiteout/log.py``
        accepts a repeated ``t`` and refuses only one that runs backwards, so
        a stalled clock would cost an episode's worth of ticks and never be
        reported.

        Strictness is therefore free to an adapter, and costs nothing to
        honour: **drive the tick from our own monotonic counter, never from
        the far side's clock.** A sim's or an autopilot's clock is sampled,
        may drift, and two polls of it can land on the same instant, which
        would fail the adapter on tick one over nothing that matters to us.
        Count ticks here and stamp them; read the far side's time only where
        it is the answer to a different question.

        That other question is fix age, and it has its own field.
        ``Pose.t`` is the tick the pose belongs to and equals this
        observation's ``t``; a pose whose underlying fix was taken earlier
        says so in :attr:`~whiteout.types.Pose.measured_t`, which is
        ``None`` when the transport does not know. Carrying the fix's own
        time in ``Pose.t`` instead would make an observation an incoherent
        snapshot and would silently change what every reader of ``pose.t``
        is looking at.
        """

    def command(self, intent: FleetIntent) -> None:
        """Send one tick of waypoint intents — everything out.

        Raises :class:`TransportError` if the link is not up.
        """

    def close(self) -> None:
        """Bring the link down. Idempotent, and safe to call unconnected.

        Safe to call on a transport whose :meth:`connect` failed partway,
        which is how ``whiteout.cli run`` releases a partial connect. It is
        final: see the class docstring on reconnect.
        """
