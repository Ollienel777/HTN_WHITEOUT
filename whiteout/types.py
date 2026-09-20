"""Frozen, JSON-serialisable record types — the episode log's data model.

``hackathon/SPEC.md`` §4 "The data model". Everything the scorer, the viewer,
the tuner and the ablation harness read is one of these, so the shape here is
the project's spine.

Three record types cross the transport seam:

``Pose``
    Where an asset is, and what it has spent getting there.
``SensorReport``
    What one sensor saw in one tick — including *nothing*, which is the
    ``negative`` flag and the input to the negative-information update.
``WaypointIntent``
    Where the coordinator wants one asset to go, and why.

They aggregate into ``WorldObservation`` (everything in) and ``FleetIntent``
(everything out), and one tick of an episode is an ``EpisodeRecord``.

**Every position in this file is geodetic ``lat``/``lon`` in degrees**, the
frame of record named in ``SPEC.md`` §5 and owned by :mod:`whiteout.geo`. No
record carries a local x/y: a local East–North frame exists only as a
projection for drawing and geometry, it is never written here, and
:func:`whiteout.geo.geodetic_to_local` is the one place it is computed.
Construction range-checks every pair, so the mix-up that matters — a swapped
lat/lon, which at Bellot Strait's 71.99° N would otherwise read as a
plausible position a few kilometres out — raises :class:`RecordError` rather
than reaching the episode log.

``z`` and ``target_z`` are altitudes in metres and are **not** settled by that
decision: which datum the arena reports is issue #76, and nothing in this file
or in :mod:`whiteout.geo` converts one.

**``truth`` is a top-level field of ``EpisodeRecord``, never nested inside
``WorldObservation``.** The observation is exactly what the coordinator sees;
ground truth is written by the sim and read only by the scorer and the viewer.
Nesting truth inside the observation would hand the policy ground truth by
construction, and the guard that the policy cannot read it would have nothing
left to guard.

Every type is a frozen dataclass with ``to_dict``/``from_dict``. Sequences are
tuples so that records stay hashable and immutable; ``to_dict`` returns the
JSON shape (tuples as lists), so ``d == json.loads(json.dumps(d))`` holds, and
``from_dict`` converts the arrays back to tuples. Declared ``float`` fields
are coerced to ``float`` on construction, because an ``int`` that Python and
mypy both accept there would serialise different bytes for an equal record.
``from_dict`` is strict: a missing field, an unknown field, a wrongly typed
value or a value outside a documented enum raises ``RecordError``. That
strictness is what ``whiteout.log``'s validator is built on — no dicts cross a
module boundary (``SPEC.md`` §5).
"""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

from whiteout.geo import GeoError, check_geodetic

__all__ = [
    "CONTACT_STATES",
    "DEFAULT_MAX_SKEW_S",
    "FOOTPRINT_KINDS",
    "QUANTISATION_STEPS",
    "SYNC_STATUSES",
    "VEHICLE_CLASSES",
    "BeliefDigest",
    "BeliefFrame",
    "BeliefGeometry",
    "Contact",
    "Detection",
    "EpisodeRecord",
    "FleetIntent",
    "Pose",
    "PoseSync",
    "RecordError",
    "SensorFootprint",
    "SensorReport",
    "SightingRefusal",
    "TargetTruth",
    "Truth",
    "WaypointIntent",
    "WorldObservation",
]

#: Contact lifecycle states (``SPEC.md`` §2 beat 4). The policy's lifecycle
#: machine owns the transitions; the log only records which one is current.
CONTACT_STATES: tuple[str, ...] = (
    "unconfirmed",
    "confirming",
    "tracked",
    "handed_off",
    "lost",
)

#: Vehicle classes (``Pose.cls``). The heterogeneity the collaboration axis
#: scores is keyed on these, so a typo must fail rather than mis-score.
VEHICLE_CLASSES: tuple[str, ...] = ("fixedwing", "quad", "rover", "tower")

#: Sensor footprint shapes (``SensorFootprint.kind``). ``kind`` decides
#: circle-versus-cone geometry in the coverage computation.
FOOTPRINT_KINDS: tuple[str, ...] = ("circle", "cone")

#: How a frame and the pose it was projected with stood in time
#: (``PoseSync.status``). Four states, because four are what this data model
#: can actually tell apart; :class:`PoseSync` says which fifth one it cannot.
#:
#: ``synchronised``
#:     A measurement time is known and the frame is within
#:     ``max_skew_s`` of it, so there is no skew this log can see.
#: ``telemetry_missing``
#:     The pose carries no ``measured_t``, so the skew is **unknown**. The
#:     arena adapter's state, and not a synonym for zero.
#: ``telemetry_stale``
#:     A measurement time is known and the frame is further than
#:     ``max_skew_s`` from it. The one state that is *known* bad.
#: ``attitude_missing``
#:     The pose has no ``pitch``/``roll``, so there is no attitude to be
#:     synchronised with and no projection at all.
SYNC_STATUSES: tuple[str, ...] = (
    "synchronised",
    "telemetry_missing",
    "telemetry_stale",
    "attitude_missing",
)

#: Largest frame-to-fix skew a projection is still called ``synchronised`` at,
#: seconds. The fixed-wing cruises at 22 m/s, so a quarter-second is 5.5 m of
#: position error injected into the fix before the detector's own error is
#: counted at all — which is the size of thing worth a state of its own. It is
#: a default and a parameter, never a constant: it is a statement about how
#: much error we will stand behind, and that belongs to the operator.
DEFAULT_MAX_SKEW_S = 0.25

#: What byte 255 means in a :class:`BeliefFrame`. The quantisation is over
#: ``[0, scale]`` in 255 steps, so the reconstruction error is at most
#: ``scale / 255`` — one step — and :mod:`whiteout.belief.encode` is held to
#: that by a test.
QUANTISATION_STEPS = 255


class RecordError(ValueError):
    """A JSON payload does not match the record type it was read as."""


# --------------------------------------------------------------------------
# Serialising helpers.
#
# ``to_dict`` returns the *JSON* shape, not the Python one: tuples become
# lists, so that ``d == json.loads(json.dumps(d))`` holds and a golden-file
# test or an ablation diff can compare a constructed record against a parsed
# line. ``from_dict`` converts the arrays back to tuples.
#
# Declared ``float`` fields are coerced on construction. Python accepts an
# ``int`` where a ``float`` is declared and mypy's numeric tower does too, so
# ``Pose(lat=0)`` would serialise ``"lat": 0`` while ``Pose(lat=0.0)``
# serialises ``"lat": 0.0`` — two equal records, two different byte streams,
# and the gate's determinism step compares bytes.
# --------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _to_json_dict(record: Any) -> dict[str, Any]:
    out: dict[str, Any] = {key: _json_safe(value) for key, value in asdict(record).items()}
    return out


def _as_floats(record: Any, *names: str) -> None:
    """Coerce declared-``float`` fields in place, on a frozen dataclass."""
    for name in names:
        value = getattr(record, name)
        if type(value) is not float:
            object.__setattr__(record, name, float(value))


def _check_position(record: Any, where: str, lat: str = "lat", lon: str = "lon") -> None:
    """Reject a position that is not a geodetic one, naming the record.

    The frame of record is lat/lon (``whiteout.geo``), and the mix-up this
    catches is the one that matters: at Bellot Strait a swapped pair reads
    ``lat=-94.84``, which is inside no latitude range and outside every
    plausible-looking wrong answer. Constructing the record is where it is
    caught, so a bad position cannot reach the episode log or the tracks API
    whether it was parsed from JSON or built in Python.
    """
    try:
        check_geodetic(getattr(record, lat), getattr(record, lon))
    except GeoError as exc:
        raise RecordError(f"{where}: {exc}") from exc


def _as_int_pair(record: Any, name: str) -> None:
    first, second = getattr(record, name)
    object.__setattr__(record, name, (int(first), int(second)))


# --------------------------------------------------------------------------
# Parsing helpers. Each names the field it rejected, so that the log
# validator can prefix a line number and produce an actionable message.
# --------------------------------------------------------------------------


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordError(f"{where}: expected an object, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise RecordError(f"{where}: non-string key {key!r}")
    result: Mapping[str, Any] = value
    return result


def _check_keys(payload: Mapping[str, Any], where: str, cls: Any) -> None:
    expected = tuple(field.name for field in fields(cls))
    missing = [name for name in expected if name not in payload]
    if missing:
        raise RecordError(f"{where}: missing field(s) {', '.join(missing)}")
    unknown = sorted(key for key in payload if key not in expected)
    if unknown:
        raise RecordError(f"{where}: unknown field(s) {', '.join(unknown)}")


def _to_float(value: int | float, where: str) -> float:
    """Widen to ``float``, naming the field if the value has no double.

    A JSON integer literal is unbounded, so ``float(value)`` raises
    ``OverflowError`` — an ``ArithmeticError``, which is not a
    :class:`RecordError` and would escape the log validator's handler and
    leave the caller a traceback with no line number. It is the integer
    spelling of the ``1e999`` overflow the reader already rejects by path.
    """
    try:
        return float(value)
    except OverflowError as exc:
        raise RecordError(f"{where}: {value.__class__.__name__} is too large for a float") from exc


def _float(payload: Mapping[str, Any], where: str, name: str) -> float:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RecordError(f"{where}.{name}: expected a number, got {type(value).__name__}")
    return _to_float(value, f"{where}.{name}")


def _int(payload: Mapping[str, Any], where: str, name: str) -> int:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordError(f"{where}.{name}: expected an integer, got {type(value).__name__}")
    return value


def _str(payload: Mapping[str, Any], where: str, name: str) -> str:
    value = payload[name]
    if not isinstance(value, str):
        raise RecordError(f"{where}.{name}: expected a string, got {type(value).__name__}")
    return value


def _optional_str(payload: Mapping[str, Any], where: str, name: str) -> str | None:
    if payload[name] is None:
        return None
    return _str(payload, where, name)


def _optional_float(payload: Mapping[str, Any], where: str, name: str) -> float | None:
    """``null`` or a number. ``null`` is a value here, never a missing key.

    :func:`_check_keys` still requires the key, so an omitted field is a
    rejected record and not a silent ``None`` — the difference between a
    transport that says it has no measurement time and a writer that forgot
    the field.
    """
    if payload[name] is None:
        return None
    return _float(payload, where, name)


def _bool(payload: Mapping[str, Any], where: str, name: str) -> bool:
    value = payload[name]
    if not isinstance(value, bool):
        raise RecordError(f"{where}.{name}: expected a boolean, got {type(value).__name__}")
    return value


def _sequence(payload: Mapping[str, Any], where: str, name: str) -> Sequence[Any]:
    value = payload[name]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RecordError(f"{where}.{name}: expected an array, got {type(value).__name__}")
    items: Sequence[Any] = value
    return items


def _int_pair(payload: Mapping[str, Any], where: str, name: str) -> tuple[int, int]:
    items = _sequence(payload, where, name)
    if len(items) != 2:
        raise RecordError(f"{where}.{name}: expected 2 integers, got {len(items)}")
    out: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise RecordError(
                f"{where}.{name}[{index}]: expected an integer, got {type(item).__name__}"
            )
        out.append(item)
    return (out[0], out[1])


def _one_of(value: str, allowed: tuple[str, ...], where: str, name: str) -> str:
    if value not in allowed:
        raise RecordError(f"{where}.{name}: {value!r} is not one of {', '.join(allowed)}")
    return value


def _b64_bytes(value: str, where: str) -> bytes:
    """Decode a strict base64 field, naming it if it is not one.

    ``validate=True`` so that a field with a character base64 has no meaning
    for is rejected rather than skipped over: the default discards them
    silently, which turns a corrupted payload into a short one and moves the
    complaint to a length check that cannot say what went wrong.
    """
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RecordError(f"{where}: not valid base64: {exc}") from exc


# --------------------------------------------------------------------------
# The record types.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pose:
    """One asset's state at one instant, as the transport reports it.

    ``cls`` is the vehicle class, one of :data:`VEHICLE_CLASSES`;
    ``energy_used`` is cumulative over the episode and is what the efficiency
    axis of the scorer consumes.

    **Orientation is three angles in degrees, and two of them are optional.**
    Degrees, because that is what the arena reports and what
    :class:`whiteout.vision.projection.CameraPose` takes: the adapter hands
    ``heading`` straight over as ``yaw_deg``. Said here because the unit is
    not guessable from a reader's side and this file does not hold it
    everywhere — :class:`SightingFootprint`'s ``heading`` and ``half_angle``
    are *radians*, which makes the wrong assumption the easy one. Both
    mistakes have been made: the fake transport once reported this field in
    radians, and the viewer once consumed it as radians, drawing every
    camera's footprint at a bearing hundreds of degrees off.

    ``heading`` is the yaw every transport can report. ``pitch`` and ``roll``
    are ``None`` when the transport does not know them, for the same reason
    ``measured_t`` is: there is no number that would be honest. A default of
    ``0.0`` reads as *level*, which is a measurement, and an adapter that
    never set the field would report every asset as flying straight and level
    — plausible, wrong, and invisible, which is the failure this module keeps
    refusing.

    They exist because a camera cannot be projected without them.
    :class:`whiteout.vision.projection.CameraPose` takes yaw, pitch **and**
    roll, so a ``Pose`` carrying only a heading cannot turn a detected pixel
    into a lat/lon — and a lat/lon is the only thing the tracks API scores.
    Issue #99. ``ATTITUDE`` carries all three for every asset in the arena,
    towers included, so the adapter was discarding two thirds of what it read.

    **This is the airframe's attitude, not a gimbal's.** The quadcopter
    carries ``gimbal_small_2d``, and a gimballed camera is not necessarily
    pointed where its airframe is. The arena sends no ``MOUNT_STATUS`` or
    ``GIMBAL_DEVICE_ATTITUDE_STATUS``, so the airframe attitude is all there
    is today; if a mount angle ever arrives it is a separate field or a
    separate report, never a quiet redefinition of these.

    **``t`` is the tick this pose belongs to, not when the fix was taken.**
    A pose is part of a :class:`WorldObservation`, and an observation is a
    coherent snapshot: ``pose.t == observation.t``, which the transport
    conformance suite asserts as an equality. Every consumer already reading
    ``pose.t`` reads it as the tick, and relaxing that would leave the field,
    its type and its plausible value unchanged while changing its meaning. No
    test and no reader can tell those two apart: every consumer keeps
    compiling, keeps running, and keeps reading a number that still looks
    right, so the change would land silently and stay landed. That is why the
    meaning is pinned here and the second question gets its own field.

    ``measured_t`` is where the other question goes. A link to a real vehicle
    serves a fix that was taken some time before the tick it lands in, and
    ``measured_t`` is when: it is ``<= t``, and ``t - measured_t`` is the age
    of the fix. It may be negative, for a fix taken before the episode clock's
    zero.

    **``measured_t`` is in the same clock as ``t`` — ours, the episode's tick
    timebase — and never the far side's.** An autopilot's or a sim's own
    stamp (``time_boot_ms``, a ``time_usec`` epoch) is in a different
    timebase, and putting one here unconverted is the mistake this field is
    most likely to attract: an epoch value is large enough that
    ``measured_t <= t`` fails on tick one and the adapter is told, but a
    vehicle that booted seconds ago yields a small number that lands under
    ``t``, passes every check, and reports an age that means nothing. An
    adapter that reads a far-side stamp must measure the offset between that
    clock and ours and subtract it before stamping; if it cannot, the honest
    answer is ``None``.

    **``None`` means "this transport does not report a measurement time"**,
    and it is the default. A consumer that wants fix age must handle ``None``
    explicitly, because there is no number that would be honest here: a
    default of ``t`` would make an adapter that never sets the field report
    every fix as zero seconds old — plausible, wrong, and invisible.
    """

    asset_id: str
    cls: str
    t: float
    lat: float
    lon: float
    z: float
    heading: float
    speed: float
    energy_used: float
    measured_t: float | None = None
    pitch: float | None = None
    roll: float | None = None

    def __post_init__(self) -> None:
        _as_floats(self, "t", "lat", "lon", "z", "heading", "speed", "energy_used")
        for optional in ("measured_t", "pitch", "roll"):
            if getattr(self, optional) is not None:
                _as_floats(self, optional)
        # Checked here and not only in from_dict. A class outside
        # VEHICLE_CLASSES used to construct happily and fail on the way back
        # in, so a transport could drive a whole episode, write every record,
        # and produce a log nothing could read - discovered at replay, hours
        # after the run that would have to be repeated. Refusing at
        # construction puts the error on the line that caused it.
        if self.cls not in VEHICLE_CLASSES:
            raise RecordError(f"pose.cls: {self.cls!r} is not one of {', '.join(VEHICLE_CLASSES)}")
        _check_position(self, "pose")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "pose") -> Pose:
        data = _mapping(payload, where)
        _check_keys(data, where, Pose)
        return Pose(
            asset_id=_str(data, where, "asset_id"),
            cls=_one_of(_str(data, where, "cls"), VEHICLE_CLASSES, where, "cls"),
            t=_float(data, where, "t"),
            lat=_float(data, where, "lat"),
            lon=_float(data, where, "lon"),
            z=_float(data, where, "z"),
            heading=_float(data, where, "heading"),
            speed=_float(data, where, "speed"),
            energy_used=_float(data, where, "energy_used"),
            measured_t=_optional_float(data, where, "measured_t"),
            pitch=_optional_float(data, where, "pitch"),
            roll=_optional_float(data, where, "roll"),
        )


@dataclass(frozen=True, slots=True)
class PoseSync:
    """Whether a frame and the pose it was projected with are one instant.

    A camera projection consumes a frame **and** an attitude together. Both
    are needed and neither is checked against the other today, so a pair that
    is a quarter of a second apart on the fixed-wing at 22 m/s puts 5.5 m into
    the fix with every field populated, nothing null, and nothing to look at.
    That is the failure this record exists to make visible: not a wrong number
    but an *unqualified* one.

    ``status`` is one of :data:`SYNC_STATUSES`. ``skew_s`` is
    ``frame_t - pose.measured_t`` in seconds, positive for a fix taken before
    the frame, and ``max_skew_s`` is the bound the verdict was reached against
    — carried rather than assumed, because a verdict read next to a different
    bound is not the verdict that was made.

    **``skew_s`` is ``None`` when the pose reports no measurement time**, and
    that is the whole point of the field rather than an inconvenience.
    ``Pose.measured_t`` is ``None`` for exactly one reason — the transport does
    not know — so the skew is unknown, and there is no number that would be
    honest in its place. A ``0.0`` there reads as *simultaneous*, which is a
    measurement, and would report every arena detection as perfectly
    synchronised: plausible, wrong, and invisible. Issues #80 and #101 held
    that line for ``measured_t`` and for ``pitch``/``roll``; this holds it one
    join further on.

    **``frame_t`` is in the pose's clock — ours, the episode's tick timebase —
    and never the camera's.** This is the mistake the field is most likely to
    attract, and it is ``measured_t``'s own trap one level up:
    :func:`whiteout.vision.frames.live_frames` stamps a frame with
    :func:`time.time`, and :func:`~whiteout.vision.frames.recorded_frames`
    stamps it with an index over a frame rate. Neither is our tick clock;
    subtracting one from ``measured_t`` yields an epoch-sized or an
    arbitrarily offset difference that is not a skew at all. Only a caller
    that has established the offset may convert one; a caller that has not
    passes the tick the frame was consumed in, which is what
    :class:`whiteout.vision.sightings.VisionSightings` does.

    **What ``synchronised`` does not claim.** With the tick as ``frame_t``,
    this measures the pose's age at the moment the frame was *consumed*, and
    the frame's own age inside its feed is not tracked at all
    (:mod:`whiteout.vision.sightings` says so, and a camera thread that keeps
    only the newest frame cannot say). So ``skew_s`` is a **lower bound** on
    the true frame-to-attitude skew, and ``synchronised`` means "no skew this
    log can see", not "none". It is still the state that matters: a pose
    already 0.4 s old at the tick is unsynchronised whatever the frame's age
    was, and that is a verdict nothing recorded before.

    **There is no ``attitude_unsynchronised`` state, because this data model
    cannot tell one.** A ``Pose`` carries a single ``measured_t`` for the
    position and the attitude together, even though the arena sends them as
    separate messages, so a stale attitude and a stale fix are one number
    here. Inventing a state whose inputs do not exist would produce a verdict
    that never fires, or worse, one that fires on the fix's staleness and is
    read as the attitude's. If a separate attitude stamp ever arrives it gets
    its own field and its own state — never a quiet redefinition of these.

    **Precedence, when more than one thing is wrong.** ``attitude_missing``
    wins: it is the fact that makes the projection *impossible* rather than
    merely wrong, and it is what the caller acts on first. ``skew_s`` is still
    carried there when it is known, so the log does not lose the second fact
    to the first.

    Whether a verdict *refuses* a detection or only qualifies it is not this
    record's business and is argued once, in
    :class:`whiteout.vision.detect.VesselDetection`.

    A ``PoseSync`` also belongs on :class:`Detection` — one sensor hit is
    exactly the scope of one frame-and-pose pair — the moment anything starts
    writing ``SensorReport``s. Nothing does yet, so the field is not there:
    the one thing worse than an unqualified number is a qualifier nobody
    sets.
    """

    status: str
    skew_s: float | None
    max_skew_s: float = DEFAULT_MAX_SKEW_S

    def __post_init__(self) -> None:
        _as_floats(self, "max_skew_s")
        if self.skew_s is not None:
            _as_floats(self, "skew_s")
        if self.status not in SYNC_STATUSES:
            raise RecordError(
                f"sync.status: {self.status!r} is not one of {', '.join(SYNC_STATUSES)}"
            )
        if not self.max_skew_s > 0.0:
            raise RecordError(
                f"sync.max_skew_s: expected a positive bound, got {self.max_skew_s!r}"
            )
        # The status and the number have to agree, checked at construction and
        # not only on the way in. A record saying `synchronised` beside a skew
        # of two seconds is worse than no record: every consumer here is built
        # to trust the status word, and a reader that has to re-derive it from
        # the number could not tell an unset field from a measured zero.
        if self.status == "telemetry_missing" and self.skew_s is not None:
            raise RecordError(
                f"sync: status is 'telemetry_missing' but skew_s is {self.skew_s!r}; "
                f"a known skew is not a missing measurement time"
            )
        if self.status in ("synchronised", "telemetry_stale"):
            if self.skew_s is None:
                raise RecordError(f"sync: status is {self.status!r} but skew_s is null")
            within = abs(self.skew_s) <= self.max_skew_s
            if within != (self.status == "synchronised"):
                raise RecordError(
                    f"sync: status is {self.status!r} but skew_s is {self.skew_s!r} "
                    f"against a bound of {self.max_skew_s!r}"
                )

    @classmethod
    def for_frame(
        cls, frame_t: float, pose: Pose, *, max_skew_s: float = DEFAULT_MAX_SKEW_S
    ) -> PoseSync:
        """Judge one frame against the pose it is about to be projected with.

        :param frame_t: when the frame was taken, **in ``pose``'s clock** — see
            the class docstring, which is where that rule is argued.
        :param pose: the pose the projection will use.
        :param max_skew_s: the bound; see :data:`DEFAULT_MAX_SKEW_S`.

        Pure and total: **every** pair of inputs has a verdict and none of them
        raises, which is load-bearing rather than tidy. The one caller is
        inside a control loop's tick, with no handler over it, so a verdict
        that raised would take down the tick over a number a far side sent us.

        A ``measured_t`` or a ``frame_t`` that is not finite is therefore
        ``telemetry_missing``, alongside the absent one. ``NaN`` fails every
        comparison, so it would otherwise pass the staleness test and be
        called ``synchronised`` — a non-number reported as perfect
        synchronisation, which is this module's own failure mode with the sign
        flipped. An infinity is arithmetically stale but writes a skew the log
        cannot hold (``whiteout.log`` refuses a non-finite float, correctly).
        Neither is a measurement, and "no usable measurement time" is exactly
        what ``telemetry_missing`` says.
        """
        measured_t = pose.measured_t
        skew_s = None if measured_t is None else float(frame_t) - measured_t
        if skew_s is not None and not math.isfinite(skew_s):
            skew_s = None
        if pose.pitch is None or pose.roll is None:
            status = "attitude_missing"
        elif skew_s is None:
            status = "telemetry_missing"
        elif abs(skew_s) > max_skew_s:
            # Absolute, so a *negative* skew is stale too. A fix stamped after
            # the frame it is joined to cannot have measured it, and the way
            # that happens is an unconverted far-side clock -- the mistake
            # `Pose.measured_t` names. Reading it as fresh would hide it.
            status = "telemetry_stale"
        else:
            status = "synchronised"
        return PoseSync(status=status, skew_s=skew_s, max_skew_s=max_skew_s)

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "sync") -> PoseSync:
        data = _mapping(payload, where)
        _check_keys(data, where, PoseSync)
        return PoseSync(
            status=_one_of(_str(data, where, "status"), SYNC_STATUSES, where, "status"),
            skew_s=_optional_float(data, where, "skew_s"),
            max_skew_s=_float(data, where, "max_skew_s"),
        )


@dataclass(frozen=True, slots=True)
class SightingRefusal:
    """A frame the coordinator declined to turn into a fix, and why.

    **This exists because refusing silently is a worse silence than the one
    the refusal prevents.** A sighting dropped for
    :class:`PoseSync`'s ``telemetry_stale`` or ``attitude_missing`` leaves no
    contact, no counter and no reason, and in the episode log that is
    indistinguishable from an empty sea — while the whole argument for
    recording synchronisation at all is that an unqualified fix must not look
    like a qualified one. The refused frame is the same mistake one level up:
    an asset that has stopped contributing, for a reason we know, with nothing
    to look at. So the fix is dropped and the *reason* is kept.

    ``asset_id`` is whose camera it was, ``t`` is the tick — equal to the
    record's, which :mod:`whiteout.log` checks — and ``sync`` is the verdict
    that caused it. ``sync`` is never ``None`` here: a refusal with no reason
    would be the thing this record exists to prevent.

    **It says nothing about whether the vessel was visible.** The refusal is
    taken before the detector runs, because a pose that cannot be projected
    makes the detector's answer unusable whatever it is, and paying for a
    frame we have already decided to drop buys nothing. So this records "this
    camera contributed nothing this tick, and here is why", not "we saw
    something and threw it away".
    """

    asset_id: str
    t: float
    sync: PoseSync

    def __post_init__(self) -> None:
        _as_floats(self, "t")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "refusal") -> SightingRefusal:
        data = _mapping(payload, where)
        _check_keys(data, where, SightingRefusal)
        return SightingRefusal(
            asset_id=_str(data, where, "asset_id"),
            t=_float(data, where, "t"),
            sync=PoseSync.from_dict(data["sync"], f"{where}.sync"),
        )


@dataclass(frozen=True, slots=True)
class SensorFootprint:
    """The ground area a sensor covered during a tick.

    ``kind`` is one of :data:`FOOTPRINT_KINDS`: ``circle`` (towers, nadir
    sensors) or ``cone`` (gimballed and forward-looking sensors), in which
    case ``heading`` and ``half_angle`` bound the sector. Radii and angles are
    metres and radians.
    """

    kind: str
    lat: float
    lon: float
    radius: float
    heading: float
    half_angle: float

    def __post_init__(self) -> None:
        _as_floats(self, "lat", "lon", "radius", "heading", "half_angle")
        _check_position(self, "footprint")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "footprint") -> SensorFootprint:
        data = _mapping(payload, where)
        _check_keys(data, where, SensorFootprint)
        return SensorFootprint(
            kind=_one_of(_str(data, where, "kind"), FOOTPRINT_KINDS, where, "kind"),
            lat=_float(data, where, "lat"),
            lon=_float(data, where, "lon"),
            radius=_float(data, where, "radius"),
            heading=_float(data, where, "heading"),
            half_angle=_float(data, where, "half_angle"),
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """One sensor hit. A *measurement*, not a truth: it may be a false alarm.

    ``detection_id`` identifies the hit, not the thing hit — associating hits
    with targets is the estimator's job and no ground truth is carried here.
    """

    detection_id: str
    lat: float
    lon: float
    confidence: float
    classification: str

    def __post_init__(self) -> None:
        _as_floats(self, "lat", "lon", "confidence")
        _check_position(self, "detection")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "detection") -> Detection:
        data = _mapping(payload, where)
        _check_keys(data, where, Detection)
        return Detection(
            detection_id=_str(data, where, "detection_id"),
            lat=_float(data, where, "lat"),
            lon=_float(data, where, "lon"),
            confidence=_float(data, where, "confidence"),
            classification=_str(data, where, "classification"),
        )


@dataclass(frozen=True, slots=True)
class SensorReport:
    """What one sensor saw in one tick.

    ``negative`` is true when the footprint was swept and nothing was
    detected. That is not an absence of data: it is the evidence the
    negative-information update erodes the belief field with, so an empty
    ``detections`` tuple and ``negative=True`` mean different things from an
    absent report.
    """

    asset_id: str
    t: float
    footprint: SensorFootprint
    detections: tuple[Detection, ...]
    negative: bool

    def __post_init__(self) -> None:
        _as_floats(self, "t")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "report") -> SensorReport:
        data = _mapping(payload, where)
        _check_keys(data, where, SensorReport)
        detections = tuple(
            Detection.from_dict(item, f"{where}.detections[{index}]")
            for index, item in enumerate(_sequence(data, where, "detections"))
        )
        return SensorReport(
            asset_id=_str(data, where, "asset_id"),
            t=_float(data, where, "t"),
            footprint=SensorFootprint.from_dict(data["footprint"], f"{where}.footprint"),
            detections=detections,
            negative=_bool(data, where, "negative"),
        )


@dataclass(frozen=True, slots=True)
class WaypointIntent:
    """Where the coordinator wants one asset to go, and why.

    ``reason`` is a short human-readable tag (``sweep``, ``post_cut``,
    ``confirm``, ``hold``) shown in the viewer; ``task_id`` ties the intent to
    the allocation that produced it.
    """

    asset_id: str
    t: float
    target_lat: float
    target_lon: float
    target_z: float
    speed: float
    reason: str
    task_id: str

    def __post_init__(self) -> None:
        _as_floats(self, "t", "target_lat", "target_lon", "target_z", "speed")
        _check_position(self, "intent", "target_lat", "target_lon")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "intent") -> WaypointIntent:
        data = _mapping(payload, where)
        _check_keys(data, where, WaypointIntent)
        return WaypointIntent(
            asset_id=_str(data, where, "asset_id"),
            t=_float(data, where, "t"),
            target_lat=_float(data, where, "target_lat"),
            target_lon=_float(data, where, "target_lon"),
            target_z=_float(data, where, "target_z"),
            speed=_float(data, where, "speed"),
            reason=_str(data, where, "reason"),
            task_id=_str(data, where, "task_id"),
        )


@dataclass(frozen=True, slots=True)
class WorldObservation:
    """Everything in. Exactly what the coordinator is allowed to see."""

    t: float
    poses: tuple[Pose, ...]
    reports: tuple[SensorReport, ...]

    def __post_init__(self) -> None:
        _as_floats(self, "t")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "observation") -> WorldObservation:
        data = _mapping(payload, where)
        _check_keys(data, where, WorldObservation)
        poses = tuple(
            Pose.from_dict(item, f"{where}.poses[{index}]")
            for index, item in enumerate(_sequence(data, where, "poses"))
        )
        reports = tuple(
            SensorReport.from_dict(item, f"{where}.reports[{index}]")
            for index, item in enumerate(_sequence(data, where, "reports"))
        )
        return WorldObservation(t=_float(data, where, "t"), poses=poses, reports=reports)


@dataclass(frozen=True, slots=True)
class FleetIntent:
    """Everything out. One tick of coordinator commands."""

    t: float
    intents: tuple[WaypointIntent, ...]

    def __post_init__(self) -> None:
        _as_floats(self, "t")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "intent") -> FleetIntent:
        data = _mapping(payload, where)
        _check_keys(data, where, FleetIntent)
        intents = tuple(
            WaypointIntent.from_dict(item, f"{where}.intents[{index}]")
            for index, item in enumerate(_sequence(data, where, "intents"))
        )
        return FleetIntent(t=_float(data, where, "t"), intents=intents)


@dataclass(frozen=True, slots=True)
class BeliefDigest:
    """A summary of the belief field, small enough to log every tick.

    The digest is what the viewer's header and the scorer's coverage axis
    read: ``entropy`` in nats, ``mass`` the field's total probability mass,
    ``peak_lat``/``peak_lon`` the most likely target location with ``peak_p``
    its density,
    ``covered_fraction`` the share of cells swept at least once, and
    ``grid_shape`` the field's dimensions **in cells**.

    ``grid_shape`` alone does not georeference the field, and this type does
    not try to. The per-cell probabilities and the lat/lon of the cells they
    sit in are :class:`BeliefFrame`'s, which is a sibling of this on the
    record rather than a member of it: the digest is the small thing every
    reader wants, and the frame is the large one only a renderer does.
    """

    t: float
    entropy: float
    mass: float
    peak_lat: float
    peak_lon: float
    peak_p: float
    covered_fraction: float
    grid_shape: tuple[int, int]

    def __post_init__(self) -> None:
        _as_floats(
            self, "t", "entropy", "mass", "peak_lat", "peak_lon", "peak_p", "covered_fraction"
        )
        _check_position(self, "belief_digest", "peak_lat", "peak_lon")
        _as_int_pair(self, "grid_shape")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "belief_digest") -> BeliefDigest:
        data = _mapping(payload, where)
        _check_keys(data, where, BeliefDigest)
        return BeliefDigest(
            t=_float(data, where, "t"),
            entropy=_float(data, where, "entropy"),
            mass=_float(data, where, "mass"),
            peak_lat=_float(data, where, "peak_lat"),
            peak_lon=_float(data, where, "peak_lon"),
            peak_p=_float(data, where, "peak_p"),
            covered_fraction=_float(data, where, "covered_fraction"),
            grid_shape=_int_pair(data, where, "grid_shape"),
        )


@dataclass(frozen=True, slots=True)
class BeliefGeometry:
    """Where the belief field's cells are — the part that never changes.

    A :class:`BeliefFrame` is a row of numbers; this is what turns index *i*
    into a quadrilateral on the map. It is **episode-constant**, so it ships
    once, on the first frame of the log, and is ``None`` on every frame after
    it. That is the one place this log departs from "every line is
    self-describing", and it is bought deliberately: the lattice is about
    11.8 kB on the default grid, so shipping it on all 400 ticks of an episode
    would cost 4.7 MB to restate a constant, against ``ARENA.md``'s 25 MB cap
    on committed episodes.

    ``shape`` is the field in cells, along the channel by across it.

    ``water`` is a bitfield over ``shape``, row-major, most significant bit
    first — ``numpy.packbits``' own layout — with a 1 for each cell the vessel
    could be in. Its set bits, read in that order, are exactly the cells
    :attr:`BeliefFrame.cells` carries, which is what makes a byte's position
    in that array mean a position on the map.

    ``corners`` is the **cell-corner lattice**, ``(along + 1) × (across + 1)``
    positions as interleaved big-endian ``float32`` lat/lon pairs, row-major.
    Cell ``(row, column)`` is the quadrilateral on corners ``(row, column)``,
    ``(row + 1, column)``, ``(row + 1, column + 1)`` and ``(row, column + 1)``.

    **Corners rather than centres, and float32 rather than float64**, both for
    the same reason: the consumer is a renderer. A centre needs the cell's
    local bearing to be drawn as anything but a dot, and that bearing is a
    trigonometric question about the channel — the one thing the viewer may
    not answer for itself, since :mod:`whiteout.geo` is the repository's only
    converter and the viewer cannot import it. Corners are the answer already
    computed. ``float32`` resolves about 1 m at this latitude, against cells
    100 m across.

    **Big-endian on the wire** so that the bytes, and therefore the log, do
    not depend on the machine that wrote them. The gate compares two runs byte
    for byte.
    """

    shape: tuple[int, int]
    water: str
    corners: str

    def __post_init__(self) -> None:
        _as_int_pair(self, "shape")
        along, across = self.shape
        if along < 1 or across < 1:
            raise RecordError(f"belief_field.geometry.shape must be positive, got {self.shape!r}")
        water = _b64_bytes(self.water, "belief_field.geometry.water")
        expected_water = (along * across + 7) // 8
        if len(water) != expected_water:
            raise RecordError(
                f"belief_field.geometry.water: expected {expected_water} bytes for a "
                f"{along}x{across} grid, got {len(water)}"
            )
        corners = _b64_bytes(self.corners, "belief_field.geometry.corners")
        expected_corners = (along + 1) * (across + 1) * 2 * 4
        if len(corners) != expected_corners:
            raise RecordError(
                f"belief_field.geometry.corners: expected {expected_corners} bytes for a "
                f"{along + 1}x{across + 1} lattice, got {len(corners)}"
            )

    def water_cells(self) -> int:
        """How many cells the mask marks as water."""
        return sum(byte.bit_count() for byte in _b64_bytes(self.water, "geometry.water"))

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "belief_field.geometry") -> BeliefGeometry:
        data = _mapping(payload, where)
        _check_keys(data, where, BeliefGeometry)
        return BeliefGeometry(
            shape=_int_pair(data, where, "shape"),
            water=_str(data, where, "water"),
            corners=_str(data, where, "corners"),
        )


@dataclass(frozen=True, slots=True)
class BeliefFrame:
    """The belief field's per-cell probability, quantised, for **drawing only**.

    Issue #124. The viewer reads the episode log and nothing else, so a belief
    field that exists only inside the process is a belief field nobody can
    see, and the erosion that makes the fleet's reasoning legible cannot be
    drawn.

    **Nothing reads this back into a belief field, and nothing may.** The
    quantisation below is lossy, and the field it came from is the state; this
    is a picture of it. A scorer or an estimator that reconstructed from here
    would be reasoning about a rounded copy of a number it could have had
    exactly.

    ``cells`` is one unsigned byte per water cell, base64-encoded, in the
    order :attr:`BeliefGeometry.water`'s set bits run. A byte ``b`` decodes to
    the probability ``b / 255 * scale``, so ``scale`` — the most probable
    cell's mass — is what byte 255 means, and the reconstruction is within
    ``scale / 255`` of the original. Re-scaling per frame rather than fixing
    the range at ``[0, 1]`` is what keeps the resolution useful: a field over
    850 cells peaks around 0.01 even when it is sharply concentrated, so a
    fixed range would spend 254 of its 255 codes on values that never occur
    and quantise the whole field to 0 or 1.

    ``water_cells`` restates the length ``cells`` must decode to, so that a
    frame validates on its own rather than only on the line the geometry
    happens to be on.

    ``geometry`` is present on the episode's first frame and ``None``
    afterwards; see :class:`BeliefGeometry` for why.
    """

    t: float
    water_cells: int
    scale: float
    cells: str
    geometry: BeliefGeometry | None

    def __post_init__(self) -> None:
        _as_floats(self, "t", "scale")
        if self.water_cells < 1:
            raise RecordError(
                f"belief_field.water_cells must be positive, got {self.water_cells!r}"
            )
        if self.scale < 0.0:
            raise RecordError(f"belief_field.scale must not be negative, got {self.scale!r}")
        cells = _b64_bytes(self.cells, "belief_field.cells")
        if len(cells) != self.water_cells:
            raise RecordError(
                f"belief_field.cells: expected {self.water_cells} bytes, got {len(cells)}"
            )
        if self.geometry is not None:
            masked = self.geometry.water_cells()
            if masked != self.water_cells:
                raise RecordError(
                    f"belief_field: the geometry marks {masked} water cells but the frame "
                    f"carries {self.water_cells}"
                )

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "belief_field") -> BeliefFrame:
        data = _mapping(payload, where)
        _check_keys(data, where, BeliefFrame)
        raw = data["geometry"]
        geometry = None if raw is None else BeliefGeometry.from_dict(raw, f"{where}.geometry")
        return BeliefFrame(
            t=_float(data, where, "t"),
            water_cells=_int(data, where, "water_cells"),
            scale=_float(data, where, "scale"),
            cells=_str(data, where, "cells"),
            geometry=geometry,
        )


@dataclass(frozen=True, slots=True)
class Contact:
    """The coordinator's hypothesis about a target, and its lifecycle state.

    Estimated, never true: ``lat``/``lon`` are the estimator's belief.
    ``assigned_asset_id`` is the asset currently holding the contact, or
    ``None`` when nothing is assigned.

    ``sync`` is how the **fix this position rests on** stood in time against
    the pose it was projected with (:class:`PoseSync`), and it is why this
    record and not :class:`Detection` carries it today: a contact is the only
    thing on the live vision path that reaches the episode log, so it is the
    only place the viewer and the scorer can be shown that a position they are
    about to trust was assembled from two things that were not simultaneous.
    It is the *last* sighting's synchronisation, not a summary of the track's
    — the position beside it is that sighting's too.

    ``None`` means no synchronisation was established, which is what every
    contact from a non-camera path is: the same absence as ``measured_t``'s,
    for the same reason. It is not a claim that the pair was synchronised.
    """

    contact_id: str
    t: float
    state: str
    lat: float
    lon: float
    confidence: float
    classification: str
    assigned_asset_id: str | None
    sync: PoseSync | None = None

    def __post_init__(self) -> None:
        _as_floats(self, "t", "lat", "lon", "confidence")
        _check_position(self, "contact")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "contact") -> Contact:
        data = _mapping(payload, where)
        _check_keys(data, where, Contact)
        return Contact(
            contact_id=_str(data, where, "contact_id"),
            t=_float(data, where, "t"),
            state=_one_of(_str(data, where, "state"), CONTACT_STATES, where, "state"),
            lat=_float(data, where, "lat"),
            lon=_float(data, where, "lon"),
            confidence=_float(data, where, "confidence"),
            classification=_str(data, where, "classification"),
            assigned_asset_id=_optional_str(data, where, "assigned_asset_id"),
            sync=(
                None if data["sync"] is None else PoseSync.from_dict(data["sync"], f"{where}.sync")
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetTruth:
    """One target's true state. Written by the sim; never observed."""

    target_id: str
    lat: float
    lon: float
    z: float
    heading: float
    speed: float
    target_class: str

    def __post_init__(self) -> None:
        _as_floats(self, "lat", "lon", "z", "heading", "speed")
        _check_position(self, "target")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "target") -> TargetTruth:
        data = _mapping(payload, where)
        _check_keys(data, where, TargetTruth)
        return TargetTruth(
            target_id=_str(data, where, "target_id"),
            lat=_float(data, where, "lat"),
            lon=_float(data, where, "lon"),
            z=_float(data, where, "z"),
            heading=_float(data, where, "heading"),
            speed=_float(data, where, "speed"),
            target_class=_str(data, where, "target_class"),
        )


@dataclass(frozen=True, slots=True)
class Truth:
    """Ground truth for one tick. Read only by the scorer and the viewer.

    A tuple of targets rather than a single one: the scored scenario has one
    moving target today, and a one-element tuple costs nothing against a
    field-shape migration if it ever has two.
    """

    t: float
    targets: tuple[TargetTruth, ...]

    def __post_init__(self) -> None:
        _as_floats(self, "t")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "truth") -> Truth:
        data = _mapping(payload, where)
        _check_keys(data, where, Truth)
        targets = tuple(
            TargetTruth.from_dict(item, f"{where}.targets[{index}]")
            for index, item in enumerate(_sequence(data, where, "targets"))
        )
        return Truth(t=_float(data, where, "t"), targets=targets)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One tick of an episode: one line of the episode log.

    ``schema_version`` is carried on every record rather than in a header, so
    that a single line is self-describing and a partially written log is
    rejected at the line that is wrong.

    ``truth`` is a sibling of ``observation``, not a member of it.

    ``refusals`` is what the coordinator's sighting source declined to use this
    tick and why (:class:`SightingRefusal`). It is top-level and **not** part of
    ``observation``, for the same reason ``truth`` is not: an observation is
    exactly what the transport reported, and a refusal is a decision taken
    afterwards, on this side of the seam. Empty is the normal case, and on a run
    with no camera path it is empty for the whole episode.

    ``belief_field`` is the drawable copy of the belief field
    (:class:`BeliefFrame`), and ``None`` on a run with no belief field behind
    it — a transport with no roster has nothing to coordinate and so nothing
    to believe. It is a sibling of ``belief_digest`` rather than a member of
    it, because the digest is the summary every reader wants and this is the
    raster only a renderer does.
    """

    schema_version: int
    t: float
    observation: WorldObservation
    intent: FleetIntent
    belief_digest: BeliefDigest
    contacts: tuple[Contact, ...]
    truth: Truth
    refusals: tuple[SightingRefusal, ...] = ()
    belief_field: BeliefFrame | None = None

    def __post_init__(self) -> None:
        _as_floats(self, "t")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "record") -> EpisodeRecord:
        """Parse a record of *any* version whose shape matches this build.

        **Version enforcement lives in :mod:`whiteout.log`, not here.**
        ``schema_version`` is parsed as a plain int, so a payload carrying a
        foreign version parses cleanly if its shape happens to match. Read
        logs through :func:`whiteout.log.validate_line`,
        :func:`~whiteout.log.iter_episode_log` or
        :func:`~whiteout.log.read_episode_log`, which reject a version this
        build cannot read and name the line. Keeping the check in exactly one
        place is what makes removing it a test failure.
        """
        data = _mapping(payload, where)
        _check_keys(data, where, EpisodeRecord)
        contacts = tuple(
            Contact.from_dict(item, f"{where}.contacts[{index}]")
            for index, item in enumerate(_sequence(data, where, "contacts"))
        )
        refusals = tuple(
            SightingRefusal.from_dict(item, f"{where}.refusals[{index}]")
            for index, item in enumerate(_sequence(data, where, "refusals"))
        )
        raw_field = data["belief_field"]
        belief_field = (
            None if raw_field is None else BeliefFrame.from_dict(raw_field, f"{where}.belief_field")
        )
        return EpisodeRecord(
            schema_version=_int(data, where, "schema_version"),
            t=_float(data, where, "t"),
            observation=WorldObservation.from_dict(data["observation"], f"{where}.observation"),
            intent=FleetIntent.from_dict(data["intent"], f"{where}.intent"),
            belief_digest=BeliefDigest.from_dict(data["belief_digest"], f"{where}.belief_digest"),
            contacts=contacts,
            truth=Truth.from_dict(data["truth"], f"{where}.truth"),
            refusals=refusals,
            belief_field=belief_field,
        )
