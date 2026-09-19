"""Transport selection: ``WHITEOUT_TRANSPORT`` in, a :class:`Transport` out.

``hackathon/SPEC.md`` §7 names three values — ``kinematic`` (the default and
the fake), ``sitl`` and ``arena`` — and §4 makes the interface behind them
non-negotiable. This module is the only place that turns the environment
variable into an object, so that "unset means ``kinematic``" and "an unknown
value fails loudly" are each true in exactly one implementation.

An unrecognised value is not defaulted away. A typo (``kinemtaic``) that
silently selected the fake would be discovered by a demo run that scored
suspiciously well against a fleet that was never there, so it raises
:class:`~whiteout.transport.base.TransportError` naming every valid value.
A value that is valid but not yet built (``sitl``, ``arena``) is rejected
just as loudly, and says which of the two it is: "not a transport" and "not
built yet" are different bugs with different fixes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from whiteout.transport.base import TRANSPORT_METHODS, Transport, TransportError
from whiteout.transport.kinematic import KinematicTransport

__all__ = [
    "DEFAULT_TRANSPORT",
    "IMPLEMENTED_TRANSPORTS",
    "TRANSPORT_METHODS",
    "TRANSPORT_NAMES",
    "TRANSPORT_ENV_VAR",
    "KinematicTransport",
    "Transport",
    "TransportError",
    "create_transport",
    "selected_transport_name",
]

#: The environment variable that selects the transport (``SPEC.md`` §7).
TRANSPORT_ENV_VAR = "WHITEOUT_TRANSPORT"

#: Every valid value of :data:`TRANSPORT_ENV_VAR`, in the spec's order.
TRANSPORT_NAMES: tuple[str, ...] = ("kinematic", "sitl", "arena")

#: The value an unset or empty :data:`TRANSPORT_ENV_VAR` means.
DEFAULT_TRANSPORT = "kinematic"

#: The subset of :data:`TRANSPORT_NAMES` this build can actually construct.
IMPLEMENTED_TRANSPORTS: tuple[str, ...] = ("kinematic",)


def _valid_values() -> str:
    return ", ".join(TRANSPORT_NAMES)


def selected_transport_name(env: Mapping[str, str] | None = None) -> str:
    """Return the transport ``env`` selects, or raise naming the valid values.

    ``env`` defaults to :data:`os.environ`. Unset — and set to the empty
    string, which is what an exported-but-blank shell variable looks like —
    means :data:`DEFAULT_TRANSPORT`.
    """
    source = os.environ if env is None else env
    value = source.get(TRANSPORT_ENV_VAR) or DEFAULT_TRANSPORT
    if value not in TRANSPORT_NAMES:
        raise TransportError(
            f"{TRANSPORT_ENV_VAR}={value!r} is not a known transport; "
            f"valid values are {_valid_values()}"
        )
    return value


def create_transport(
    name: str | None = None,
    *,
    seed: int = 0,
    env: Mapping[str, str] | None = None,
) -> Transport:
    """Build the selected transport.

    ``name`` overrides the environment; ``None`` reads it through
    :func:`selected_transport_name`. ``seed`` is the episode seed, which the
    fake uses to lay its fleet out deterministically.

    Raises :class:`~whiteout.transport.base.TransportError` for a value that
    is not a transport, and for one that is a transport this build does not
    implement yet.
    """
    if name is None:
        name = selected_transport_name(env)
    elif name not in TRANSPORT_NAMES:
        raise TransportError(
            f"{name!r} is not a known transport; valid values are {_valid_values()}"
        )
    if name not in IMPLEMENTED_TRANSPORTS:
        raise TransportError(
            f"the {name!r} transport is not implemented in this build; "
            f"valid values are {_valid_values()}, of which this build implements "
            f"{', '.join(IMPLEMENTED_TRANSPORTS)}"
        )
    return KinematicTransport(seed=seed)
