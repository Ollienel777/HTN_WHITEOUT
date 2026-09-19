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

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = [
    "CONTACT_STATES",
    "FOOTPRINT_KINDS",
    "VEHICLE_CLASSES",
    "BeliefDigest",
    "Contact",
    "Detection",
    "EpisodeRecord",
    "FleetIntent",
    "Pose",
    "RecordError",
    "SensorFootprint",
    "SensorReport",
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
# ``Pose(x=0)`` would serialise ``"x": 0`` while ``Pose(x=0.0)`` serialises
# ``"x": 0.0`` — two equal records, two different byte streams, and the
# gate's determinism step compares bytes.
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


def _as_float_pair(record: Any, name: str) -> None:
    first, second = getattr(record, name)
    object.__setattr__(record, name, (float(first), float(second)))


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


def _pair(payload: Mapping[str, Any], where: str, name: str) -> tuple[float, float]:
    items = _sequence(payload, where, name)
    if len(items) != 2:
        raise RecordError(f"{where}.{name}: expected 2 numbers, got {len(items)}")
    out: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise RecordError(
                f"{where}.{name}[{index}]: expected a number, got {type(item).__name__}"
            )
        out.append(_to_float(item, f"{where}.{name}[{index}]"))
    return (out[0], out[1])


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


# --------------------------------------------------------------------------
# The record types.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pose:
    """One asset's state at one instant, as the transport reports it.

    ``cls`` is the vehicle class, one of :data:`VEHICLE_CLASSES`;
    ``energy_used`` is cumulative over the episode and is what the efficiency
    axis of the scorer consumes.

    **``t`` is the tick this pose belongs to, not when the fix was taken.**
    A pose is part of a :class:`WorldObservation`, and an observation is a
    coherent snapshot: ``pose.t == observation.t``, which the transport
    conformance suite asserts as an equality. Every consumer already reading
    ``pose.t`` reads it as the tick, and relaxing that would leave the field,
    its type and its plausible value unchanged while changing its meaning —
    the error class ``SPEC.md`` §4 records as the one we cannot detect.

    ``measured_t`` is where the other question goes. A link to a real vehicle
    serves a fix that was taken some time before the tick it lands in, and
    ``measured_t`` is when: it is ``<= t``, and ``t - measured_t`` is the age
    of the fix. It may be negative, for a fix taken before the episode clock's
    zero.

    **``None`` means "this transport does not report a measurement time"**,
    and it is the default. A consumer that wants fix age must handle ``None``
    explicitly, because there is no number that would be honest here: a
    default of ``t`` would make an adapter that never sets the field report
    every fix as zero seconds old — plausible, wrong, and invisible.
    """

    asset_id: str
    cls: str
    t: float
    x: float
    y: float
    z: float
    heading: float
    speed: float
    energy_used: float
    measured_t: float | None = None

    def __post_init__(self) -> None:
        _as_floats(self, "t", "x", "y", "z", "heading", "speed", "energy_used")
        if self.measured_t is not None:
            _as_floats(self, "measured_t")

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
            x=_float(data, where, "x"),
            y=_float(data, where, "y"),
            z=_float(data, where, "z"),
            heading=_float(data, where, "heading"),
            speed=_float(data, where, "speed"),
            energy_used=_float(data, where, "energy_used"),
            measured_t=_optional_float(data, where, "measured_t"),
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
    x: float
    y: float
    radius: float
    heading: float
    half_angle: float

    def __post_init__(self) -> None:
        _as_floats(self, "x", "y", "radius", "heading", "half_angle")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "footprint") -> SensorFootprint:
        data = _mapping(payload, where)
        _check_keys(data, where, SensorFootprint)
        return SensorFootprint(
            kind=_one_of(_str(data, where, "kind"), FOOTPRINT_KINDS, where, "kind"),
            x=_float(data, where, "x"),
            y=_float(data, where, "y"),
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
    x: float
    y: float
    confidence: float
    classification: str

    def __post_init__(self) -> None:
        _as_floats(self, "x", "y", "confidence")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "detection") -> Detection:
        data = _mapping(payload, where)
        _check_keys(data, where, Detection)
        return Detection(
            detection_id=_str(data, where, "detection_id"),
            x=_float(data, where, "x"),
            y=_float(data, where, "y"),
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
    target_xy: tuple[float, float]
    target_z: float
    speed: float
    reason: str
    task_id: str

    def __post_init__(self) -> None:
        _as_floats(self, "t", "target_z", "speed")
        _as_float_pair(self, "target_xy")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "intent") -> WaypointIntent:
        data = _mapping(payload, where)
        _check_keys(data, where, WaypointIntent)
        return WaypointIntent(
            asset_id=_str(data, where, "asset_id"),
            t=_float(data, where, "t"),
            target_xy=_pair(data, where, "target_xy"),
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

    The field itself is a raster and does not belong in a JSONL line. The
    digest is what the viewer's header and the scorer's coverage axis read:
    ``entropy`` in nats, ``mass`` the field's total probability mass,
    ``peak_xy`` the most likely target location with ``peak_p`` its density,
    ``covered_fraction`` the share of cells swept at least once, and
    ``grid_shape`` the field's dimensions **in cells**.

    ``grid_shape`` alone does not georeference the field: mapping a cell to a
    world coordinate also needs the grid's origin and cell size (or its world
    bounds), and this log is the only artifact the viewer reads. Those fields
    are not here yet — adding them is a schema change and a
    :data:`~whiteout.log.SCHEMA_VERSION` bump, and it is a decision for the
    belief-field ticket that will populate them, not for this type.
    """

    t: float
    entropy: float
    mass: float
    peak_xy: tuple[float, float]
    peak_p: float
    covered_fraction: float
    grid_shape: tuple[int, int]

    def __post_init__(self) -> None:
        _as_floats(self, "t", "entropy", "mass", "peak_p", "covered_fraction")
        _as_float_pair(self, "peak_xy")
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
            peak_xy=_pair(data, where, "peak_xy"),
            peak_p=_float(data, where, "peak_p"),
            covered_fraction=_float(data, where, "covered_fraction"),
            grid_shape=_int_pair(data, where, "grid_shape"),
        )


@dataclass(frozen=True, slots=True)
class Contact:
    """The coordinator's hypothesis about a target, and its lifecycle state.

    Estimated, never true: ``x``/``y`` are the estimator's belief.
    ``assigned_asset_id`` is the asset currently holding the contact, or
    ``None`` when nothing is assigned.
    """

    contact_id: str
    t: float
    state: str
    x: float
    y: float
    confidence: float
    classification: str
    assigned_asset_id: str | None

    def __post_init__(self) -> None:
        _as_floats(self, "t", "x", "y", "confidence")

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
            x=_float(data, where, "x"),
            y=_float(data, where, "y"),
            confidence=_float(data, where, "confidence"),
            classification=_str(data, where, "classification"),
            assigned_asset_id=_optional_str(data, where, "assigned_asset_id"),
        )


@dataclass(frozen=True, slots=True)
class TargetTruth:
    """One target's true state. Written by the sim; never observed."""

    target_id: str
    x: float
    y: float
    z: float
    heading: float
    speed: float
    target_class: str

    def __post_init__(self) -> None:
        _as_floats(self, "x", "y", "z", "heading", "speed")

    def to_dict(self) -> dict[str, Any]:
        return _to_json_dict(self)

    @classmethod
    def from_dict(cls, payload: object, where: str = "target") -> TargetTruth:
        data = _mapping(payload, where)
        _check_keys(data, where, TargetTruth)
        return TargetTruth(
            target_id=_str(data, where, "target_id"),
            x=_float(data, where, "x"),
            y=_float(data, where, "y"),
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
    """

    schema_version: int
    t: float
    observation: WorldObservation
    intent: FleetIntent
    belief_digest: BeliefDigest
    contacts: tuple[Contact, ...]
    truth: Truth

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
        return EpisodeRecord(
            schema_version=_int(data, where, "schema_version"),
            t=_float(data, where, "t"),
            observation=WorldObservation.from_dict(data["observation"], f"{where}.observation"),
            intent=FleetIntent.from_dict(data["intent"], f"{where}.intent"),
            belief_digest=BeliefDigest.from_dict(data["belief_digest"], f"{where}.belief_digest"),
            contacts=contacts,
            truth=Truth.from_dict(data["truth"], f"{where}.truth"),
        )
