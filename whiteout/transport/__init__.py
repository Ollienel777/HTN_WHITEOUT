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

import inspect
import os
from collections.abc import Callable, Mapping

from whiteout.transport.base import TRANSPORT_METHODS, Transport, TransportError
from whiteout.transport.kinematic import KinematicTransport

__all__ = [
    "DEFAULT_TRANSPORT",
    "IMPLEMENTED_TRANSPORTS",
    "TRANSPORT_FACTORIES",
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

#: Every transport this build can construct, by name. This mapping is the
#: single place a new adapter is registered: :data:`IMPLEMENTED_TRANSPORTS`
#: is derived from it and :func:`create_transport` dispatches through it, so
#: there is no second line to forget. Adding a name here without a class to
#: go with it is impossible; adding a class that is never returned is too.
TRANSPORT_FACTORIES: Mapping[str, Callable[..., Transport]] = {
    "kinematic": KinematicTransport,
}

#: The subset of :data:`TRANSPORT_NAMES` this build can actually construct,
#: in the spec's order. Derived, never hand-written — see above.
IMPLEMENTED_TRANSPORTS: tuple[str, ...] = tuple(
    name for name in TRANSPORT_NAMES if name in TRANSPORT_FACTORIES
)


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
    pose_age_seconds: float | None = None,
    env: Mapping[str, str] | None = None,
) -> Transport:
    """Build the selected transport.

    ``name`` overrides the environment; ``None`` reads it through
    :func:`selected_transport_name`. ``seed`` is the episode seed, which the
    fake uses to lay its fleet out deterministically.

    ``pose_age_seconds`` asks the transport to report a fix age — every pose
    stamped ``measured_t = t - age``. ``None``, the default, is forwarded to
    nothing, so a transport that does not take the argument is unaffected and
    the episode is byte-for-byte what it was. It is threaded through here
    rather than left to direct construction because the whole point of the
    knob is that the staleness path can be driven from a real episode — a
    log, the scorer, ``replay``, the viewer — and not only from a unit test
    that builds a transport by hand. A transport that does not accept an age
    refuses it by name rather than by ``TypeError``.

    Dispatches through :data:`TRANSPORT_FACTORIES`, so the object returned
    is always the one the name asks for. Returning a fixed class here would
    be the failure this module exists to prevent: the next adapter's ticket
    registers its name, forgets the return, and ``WHITEOUT_TRANSPORT=sitl``
    quietly runs the fake.

    Raises :class:`~whiteout.transport.base.TransportError` for a value that
    is not a transport, and for one that is a transport this build does not
    implement yet. A transport may also raise it from its constructor, for
    a seed or a configuration it cannot use.
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
    factory = TRANSPORT_FACTORIES[name]
    if pose_age_seconds is None:
        return factory(seed=seed)
    # Asked of the signature, not of a caught `TypeError`: a constructor that
    # raises `TypeError` for its own reasons would otherwise be reported as a
    # transport that does not take a pose age, which is a wrong diagnosis of
    # the kind this whole contract exists to avoid.
    try:
        inspect.signature(factory).bind(seed=seed, pose_age_seconds=pose_age_seconds)
    except TypeError as exc:
        raise TransportError(f"the {name!r} transport does not take a pose age") from exc
    return factory(seed=seed, pose_age_seconds=pose_age_seconds)
