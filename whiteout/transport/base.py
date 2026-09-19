"""The seam: the one interface between the coordinator and the world.

``hackathon/SPEC.md`` §4 "The seam — structural and non-negotiable". Poses and
sensor reports in, waypoint intents out, and **nothing else crosses it**.

Three implementations sit behind it — ``kinematic`` (the dev, tuning, CI and
demo loop), ``sitl`` (ArduPilot validation) and ``arena`` (the sponsor's
harness, whose interface is not known yet). The point of the interface is that
if the sponsor's turns out not to be plain MAVLink over a socket, only the
adapter is lost: never the estimator, never the policy, never a tuned
parameter.

What that costs is a rule, and the rule is enforced by a test
(``tests/test_transport.py``): **no belief, policy or scoring concept may
appear in ``whiteout/transport/``.** Concretely, the four methods below are
the whole protocol; the only record types named here are
:class:`~whiteout.types.WorldObservation` and
:class:`~whiteout.types.FleetIntent`; and this package imports nothing from
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

    Raised for an unknown or unimplemented ``WHITEOUT_TRANSPORT``, and for a
    call made out of order (``observe`` before ``connect``, say). The CLI
    turns it into a diagnostic and a non-zero exit rather than a traceback.
    """


@runtime_checkable
class Transport(Protocol):
    """The one interface the coordinator talks to the world through."""

    def connect(self) -> None:
        """Bring the link up. Idempotent: connecting twice is not an error."""

    def observe(self) -> WorldObservation:
        """Return the next tick's observation — everything in.

        The transport owns the world's clock: each call returns the
        observation for one tick and advances to the next, so
        ``WorldObservation.t`` is the transport's, not the caller's. Raises
        :class:`TransportError` if the link is not up.
        """

    def command(self, intent: FleetIntent) -> None:
        """Send one tick of waypoint intents — everything out.

        Raises :class:`TransportError` if the link is not up.
        """

    def close(self) -> None:
        """Bring the link down. Idempotent, and safe to call unconnected."""
