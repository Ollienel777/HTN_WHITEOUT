"""Issue #5's acceptance criteria, one test each (plus the guards around them).

The episode log is the only artifact the scorer, the viewer, the tuner and the
ablation harness read, so these tests are the contract for all of them.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

import whiteout.types
from whiteout.log import (
    SCHEMA_VERSION,
    EpisodeLogError,
    iter_episode_log,
    read_episode_log,
    validate_episode_log,
    validate_line,
    write_episode_log,
)
from whiteout.types import (
    BeliefDigest,
    Contact,
    Detection,
    EpisodeRecord,
    FleetIntent,
    Pose,
    RecordError,
    SensorFootprint,
    SensorReport,
    TargetTruth,
    Truth,
    WaypointIntent,
    WorldObservation,
)


def make_record(tick: int) -> EpisodeRecord:
    """A fully populated record: every field exercised, nothing left default."""
    t = float(tick)
    return EpisodeRecord(
        schema_version=SCHEMA_VERSION,
        t=t,
        observation=WorldObservation(
            t=t,
            poses=(
                Pose(
                    asset_id="wing-1",
                    cls="fixedwing",
                    t=t,
                    x=10.0 + tick,
                    y=-4.5,
                    z=120.0,
                    heading=0.75,
                    speed=22.0,
                    energy_used=3.5 * tick,
                    # A fix taken a quarter-second before the tick it landed
                    # in, and one from a transport that does not report a
                    # measurement time at all. Both arms are spelt out rather
                    # than left to the default, so every round-trip test here
                    # carries a populated `measured_t` and a null one.
                    measured_t=t - 0.25,
                ),
                Pose(
                    asset_id="tower-1",
                    cls="tower",
                    t=t,
                    x=0.0,
                    y=0.0,
                    z=15.0,
                    heading=0.0,
                    speed=0.0,
                    energy_used=0.0,
                    measured_t=None,
                ),
            ),
            reports=(
                SensorReport(
                    asset_id="wing-1",
                    t=t,
                    footprint=SensorFootprint(
                        kind="cone",
                        x=10.0 + tick,
                        y=-4.5,
                        radius=300.0,
                        heading=0.75,
                        half_angle=0.4,
                    ),
                    detections=(
                        Detection(
                            detection_id=f"d-{tick}-0",
                            x=101.0,
                            y=55.5,
                            confidence=0.62,
                            classification="vehicle",
                        ),
                    ),
                    negative=False,
                ),
                SensorReport(
                    asset_id="tower-1",
                    t=t,
                    footprint=SensorFootprint(
                        kind="circle",
                        x=0.0,
                        y=0.0,
                        radius=800.0,
                        heading=0.0,
                        half_angle=3.141592653589793,
                    ),
                    detections=(),
                    negative=True,
                ),
            ),
        ),
        intent=FleetIntent(
            t=t,
            intents=(
                WaypointIntent(
                    asset_id="wing-1",
                    t=t,
                    target_xy=(250.0, -125.5),
                    target_z=140.0,
                    speed=24.0,
                    reason="sweep",
                    task_id=f"task-{tick}",
                ),
            ),
        ),
        belief_digest=BeliefDigest(
            t=t,
            entropy=7.25,
            mass=1.0,
            peak_xy=(99.5, 50.25),
            peak_p=0.031,
            covered_fraction=0.42,
            grid_shape=(256, 256),
        ),
        contacts=(
            Contact(
                contact_id="c-1",
                t=t,
                state="confirming",
                x=100.0,
                y=55.0,
                confidence=0.62,
                classification="vehicle",
                assigned_asset_id="wing-1",
            ),
            Contact(
                contact_id="c-2",
                t=t,
                state="lost",
                x=-400.0,
                y=90.0,
                confidence=0.05,
                classification="unknown",
                assigned_asset_id=None,
            ),
        ),
        truth=Truth(
            t=t,
            targets=(
                TargetTruth(
                    target_id="target-0",
                    x=102.0,
                    y=54.0,
                    z=3.0,
                    heading=1.1,
                    speed=4.0,
                    target_class="snowmobile",
                ),
            ),
        ),
    )


# --- acceptance: round trip -------------------------------------------------


def test_round_trip_write_read_equality(tmp_path: Path) -> None:
    """Write N records, read them back, assert equality."""
    records = [make_record(tick) for tick in range(12)]
    log = tmp_path / "episode.jsonl"
    assert write_episode_log(log, records) == 12
    assert read_episode_log(log) == records


def test_round_trip_preserves_tuple_types(tmp_path: Path) -> None:
    """JSON arrays must come back as tuples, not lists, or equality is a lie."""
    log = tmp_path / "episode.jsonl"
    write_episode_log(log, [make_record(0)])
    back = read_episode_log(log)[0]
    assert isinstance(back.contacts, tuple)
    assert isinstance(back.observation.poses, tuple)
    assert isinstance(back.observation.reports[0].detections, tuple)
    assert isinstance(back.intent.intents[0].target_xy, tuple)
    assert isinstance(back.truth.targets, tuple)


def test_write_is_byte_identical_for_equal_records(tmp_path: Path) -> None:
    """The gate's determinism step compares bytes; the writer must not vary."""
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_episode_log(first, [make_record(tick) for tick in range(5)])
    write_episode_log(second, [make_record(tick) for tick in range(5)])
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()


def test_records_are_frozen() -> None:
    record = make_record(0)
    for value in (record, record.observation.poses[0], record.truth.targets[0]):
        assert dataclasses.is_dataclass(value)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, fields(value)[0].name, "mutated")


# --- acceptance: truth is top level ----------------------------------------


def test_truth_is_a_top_level_field_and_not_in_the_observation() -> None:
    """The policy reads WorldObservation; ground truth must not be reachable there."""
    assert "truth" in {field.name for field in fields(EpisodeRecord)}
    observation_fields = {field.name for field in fields(WorldObservation)}
    assert observation_fields == {"t", "poses", "reports"}
    encoded = json.loads(json.dumps(make_record(0).to_dict()))
    assert "truth" in encoded
    assert "truth" not in encoded["observation"]
    assert "truth" not in json.dumps(encoded["observation"])


# --- acceptance: the validator rejects, naming the line ---------------------


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


def _good_line(tick: int) -> str:
    return json.dumps(make_record(tick).to_dict(), sort_keys=True, separators=(",", ":"))


def test_validator_rejects_a_truncated_line_naming_it(tmp_path: Path) -> None:
    log = tmp_path / "truncated.jsonl"
    truncated = _good_line(1)[: len(_good_line(1)) // 2]
    _write_lines(log, [_good_line(0), truncated, _good_line(2)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 2
    assert str(caught.value).startswith("line 2:")
    assert "not valid JSON" in str(caught.value)


def test_validator_rejects_an_unknown_field_naming_the_line(tmp_path: Path) -> None:
    log = tmp_path / "unknown.jsonl"
    payload = make_record(1).to_dict()
    payload["bonus_field"] = 3
    _write_lines(log, [_good_line(0), json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 2
    assert "unknown field(s) bonus_field" in str(caught.value)


def test_validator_rejects_a_wrong_schema_version_naming_the_line(tmp_path: Path) -> None:
    log = tmp_path / "version.jsonl"
    payload = make_record(1).to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    _write_lines(log, [_good_line(0), json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 2
    assert "schema_version" in str(caught.value)
    assert str(SCHEMA_VERSION) in str(caught.value)


def test_a_log_of_the_shape_before_measured_t_is_refused_by_version(tmp_path: Path) -> None:
    """The version, not a nested missing field, is what reports an old log.

    ``measured_t`` is a required record key, so a log written before it
    existed cannot be read by this build. Without the bump that went with it,
    both shapes would self-describe as the same version, the version check
    would pass, and the reader would report
    ``record.observation.poses[0]: missing field(s) measured_t`` — which
    names a field rather than the build, and leaves a reader of an old
    artifact guessing. The version number exists to produce exactly one
    message, and this is it.
    """
    log = tmp_path / "previous_shape.jsonl"
    payload = make_record(0).to_dict()
    payload["schema_version"] = SCHEMA_VERSION - 1
    for pose in payload["observation"]["poses"]:
        del pose["measured_t"]
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    message = str(caught.value)
    assert "schema_version" in message
    assert str(SCHEMA_VERSION) in message
    assert "measured_t" not in message, "reported by version, not by the missing field"


def test_validator_rejects_a_nested_unknown_field_naming_the_path(tmp_path: Path) -> None:
    log = tmp_path / "nested.jsonl"
    payload = make_record(0).to_dict()
    payload["observation"]["poses"][0]["altitude"] = 1.0
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 1
    assert "record.observation.poses[0]" in str(caught.value)


def test_validator_rejects_a_missing_field(tmp_path: Path) -> None:
    log = tmp_path / "missing.jsonl"
    payload = make_record(0).to_dict()
    del payload["belief_digest"]
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "missing field(s) belief_digest" in str(caught.value)


def test_validator_rejects_a_wrongly_typed_value(tmp_path: Path) -> None:
    log = tmp_path / "typed.jsonl"
    payload = make_record(0).to_dict()
    payload["observation"]["poses"][0]["speed"] = "fast"
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "expected a number" in str(caught.value)


def test_validator_rejects_an_unknown_contact_state(tmp_path: Path) -> None:
    log = tmp_path / "state.jsonl"
    payload = make_record(0).to_dict()
    payload["contacts"][0]["state"] = "vibing"
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "is not one of" in str(caught.value)


def test_validator_rejects_an_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "empty.jsonl"
    log.write_text("", encoding="utf-8")
    with pytest.raises(EpisodeLogError):
        validate_episode_log(log)


def test_validator_counts_a_valid_log(tmp_path: Path) -> None:
    log = tmp_path / "ok.jsonl"
    write_episode_log(log, [make_record(tick) for tick in range(7)])
    assert validate_episode_log(log) == 7


def test_validator_rejects_a_non_object_line(tmp_path: Path) -> None:
    log = tmp_path / "scalar.jsonl"
    _write_lines(log, [_good_line(0), "42"])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 2


def test_validate_line_reports_the_number_it_was_given() -> None:
    with pytest.raises(EpisodeLogError) as caught:
        validate_line("{", 4137)
    assert caught.value.line == 4137
    assert str(caught.value).startswith("line 4137:")


# --- writer guards ----------------------------------------------------------


def test_writer_refuses_a_foreign_schema_version(tmp_path: Path) -> None:
    """A build must never write a log it cannot read back."""
    record = dataclasses.replace(make_record(0), schema_version=SCHEMA_VERSION + 1)
    with pytest.raises(EpisodeLogError):
        write_episode_log(tmp_path / "bad.jsonl", [record])


def test_from_dict_raises_record_error_on_a_non_object() -> None:
    with pytest.raises(RecordError):
        EpisodeRecord.from_dict([1, 2, 3])


# --- R1-1: non-finite floats are not JSON -----------------------------------


def _with_non_finite() -> EpisodeRecord:
    record = make_record(0)
    digest = dataclasses.replace(record.belief_digest, entropy=math.nan, peak_p=math.inf)
    return dataclasses.replace(record, belief_digest=digest)


def test_writer_refuses_a_non_finite_float_naming_the_field(tmp_path: Path) -> None:
    """NaN and Infinity are not JSON; the viewer's JSON.parse throws on them."""
    log = tmp_path / "nan.jsonl"
    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, [_with_non_finite()])
    assert "record.belief_digest.entropy" in str(caught.value)
    assert not log.exists()


def test_validator_rejects_the_nan_token_naming_the_line(tmp_path: Path) -> None:
    """A hand-made or older log carrying bare NaN is not valid JSON."""
    log = tmp_path / "nan_token.jsonl"
    payload = make_record(1).to_dict()
    payload["belief_digest"]["entropy"] = math.nan
    poisoned = json.dumps(payload, sort_keys=True)
    assert "NaN" in poisoned
    _write_lines(log, [_good_line(0), poisoned])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 2
    assert "not valid JSON" in str(caught.value)


def test_a_written_log_parses_under_strict_json(tmp_path: Path) -> None:
    """What the viewer does: JSON.parse, which has no NaN or Infinity."""
    log = tmp_path / "strict.jsonl"
    write_episode_log(log, [make_record(tick) for tick in range(3)])

    def _reject(token: str) -> float:
        raise AssertionError(f"non-JSON constant {token}")

    for line in log.read_text(encoding="utf-8").splitlines():
        json.loads(line, parse_constant=_reject)


# --- R1-2: a failed write never replaces a good log -------------------------


def test_a_failed_write_leaves_the_existing_log_untouched(tmp_path: Path) -> None:
    """A truncated log that validates clean is worse than a failed write."""
    log = tmp_path / "episode.jsonl"
    good = [make_record(tick) for tick in range(5)]
    write_episode_log(log, good)
    before = log.read_bytes()

    doomed = [make_record(tick) for tick in range(5)]
    doomed[3] = dataclasses.replace(doomed[3], schema_version=SCHEMA_VERSION + 1)
    with pytest.raises(EpisodeLogError):
        write_episode_log(log, doomed)

    assert log.read_bytes() == before
    assert validate_episode_log(log) == 5
    assert list(log.parent.iterdir()) == [log]


def test_a_failed_first_write_leaves_no_log_at_all(tmp_path: Path) -> None:
    log = tmp_path / "never.jsonl"
    doomed = [dataclasses.replace(make_record(0), schema_version=SCHEMA_VERSION + 1)]
    with pytest.raises(EpisodeLogError):
        write_episode_log(log, doomed)
    assert not log.exists()
    assert list(tmp_path.iterdir()) == []


# --- R1-3: the documented enums are enforced --------------------------------


def test_validator_rejects_an_unknown_vehicle_class(tmp_path: Path) -> None:
    log = tmp_path / "cls.jsonl"
    payload = make_record(0).to_dict()
    payload["observation"]["poses"][0]["cls"] = "fixed_wing_typo"
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "cls" in str(caught.value)
    assert "is not one of" in str(caught.value)


def test_validator_rejects_an_unknown_footprint_kind(tmp_path: Path) -> None:
    log = tmp_path / "kind.jsonl"
    payload = make_record(0).to_dict()
    payload["observation"]["reports"][0]["footprint"]["kind"] = "blob"
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "kind" in str(caught.value)
    assert "is not one of" in str(caught.value)


def test_the_enums_are_public() -> None:
    for name in ("CONTACT_STATES", "VEHICLE_CLASSES", "FOOTPRINT_KINDS"):
        assert name in whiteout.types.__all__


# --- R1-5: the clocks of one record agree, and time does not run back -------


def test_validator_rejects_a_record_whose_members_disagree_on_t(tmp_path: Path) -> None:
    log = tmp_path / "desync.jsonl"
    payload = make_record(0).to_dict()
    payload["t"] = 99.0
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "record.observation.t" in str(caught.value)


def test_validator_rejects_t_running_backwards(tmp_path: Path) -> None:
    log = tmp_path / "backwards.jsonl"
    _write_lines(log, [_good_line(0), _good_line(3), _good_line(2)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 3


def test_writer_refuses_a_record_whose_clocks_disagree(tmp_path: Path) -> None:
    log = tmp_path / "desync_write.jsonl"
    record = dataclasses.replace(make_record(0), t=99.0)
    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, [record])
    assert "clocks disagree" in str(caught.value)
    assert not log.exists()


# --- R1-7 and R1-9: one record, one byte stream, one JSON shape -------------


def test_an_int_in_a_float_field_serialises_as_a_float() -> None:
    """The gate compares bytes; an int where a float is declared breaks that."""
    integral = Pose(asset_id="a", cls="quad", t=0, x=0, y=0, z=0, heading=0, speed=0, energy_used=0)
    floating = Pose(
        asset_id="a",
        cls="quad",
        t=0.0,
        x=0.0,
        y=0.0,
        z=0.0,
        heading=0.0,
        speed=0.0,
        energy_used=0.0,
    )
    assert integral == floating
    assert json.dumps(integral.to_dict(), sort_keys=True) == json.dumps(
        floating.to_dict(), sort_keys=True
    )
    assert isinstance(integral.x, float)


def test_to_dict_is_the_json_shape() -> None:
    """A golden file or an ablation diff compares to_dict against a parsed line."""
    payload = make_record(0).to_dict()
    assert payload == json.loads(json.dumps(payload))
    assert isinstance(payload["contacts"], list)
    assert isinstance(payload["belief_digest"]["peak_xy"], list)


# --- R1-8: the reader does not hold the file open ---------------------------


def test_reading_one_tick_does_not_hold_the_file_open(tmp_path: Path) -> None:
    """On Windows a held handle blocks the next run from rewriting the log."""
    log = tmp_path / "peek.jsonl"
    write_episode_log(log, [make_record(tick) for tick in range(3)])
    records = iter_episode_log(log)
    assert next(records).t == 0.0
    os.remove(log)
    assert not log.exists()


# --- R2-1: the writer refuses anything the reader would reject --------------


def test_writer_refuses_an_unknown_enum_and_keeps_the_existing_log(
    tmp_path: Path,
) -> None:
    """The enum guard is reader-side only unless the writer runs the reader.

    ``__post_init__`` coerces but does not check, so a sim holding a typo'd
    ``cls`` reaches the writer with no warning upstream. Without the write-path
    round trip the atomic replace succeeds and hands the scorer an unreadable
    log in place of a good one.
    """
    log = tmp_path / "episode.jsonl"
    write_episode_log(log, [make_record(tick) for tick in range(5)])
    before = log.read_bytes()

    doomed = [make_record(tick) for tick in range(5)]
    observation = doomed[3].observation
    poses = list(observation.poses)
    poses[0] = dataclasses.replace(poses[0], cls="typo")
    doomed[3] = dataclasses.replace(
        doomed[3],
        observation=dataclasses.replace(observation, poses=tuple(poses)),
    )

    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, doomed)
    assert caught.value.line == 4
    assert "cls" in str(caught.value)
    assert log.read_bytes() == before
    assert validate_episode_log(log) == 5
    assert list(tmp_path.iterdir()) == [log]


def test_writer_refuses_an_unknown_footprint_kind(tmp_path: Path) -> None:
    """Any reader-side rule is mirrored, not just the one R2-1 reproduced."""
    log = tmp_path / "kind.jsonl"
    record = make_record(0)
    reports = list(record.observation.reports)
    reports[0] = dataclasses.replace(
        reports[0],
        footprint=dataclasses.replace(reports[0].footprint, kind="blob"),
    )
    doomed = dataclasses.replace(
        record,
        observation=dataclasses.replace(record.observation, reports=tuple(reports)),
    )
    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, [doomed])
    assert "kind" in str(caught.value)
    assert not log.exists()


def test_writer_refuses_a_backwards_t_and_keeps_the_existing_log(
    tmp_path: Path,
) -> None:
    """The monotonic clock lived only in the reader, so the writer could
    replace a good five-record log with one that throws on line 3."""
    log = tmp_path / "episode.jsonl"
    write_episode_log(log, [make_record(tick) for tick in range(5)])
    before = log.read_bytes()

    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, [make_record(tick) for tick in (1, 3, 2, 4)])
    assert caught.value.line == 3
    assert "before the previous record's" in str(caught.value)
    assert log.read_bytes() == before
    assert validate_episode_log(log) == 5
    assert list(tmp_path.iterdir()) == [log]


def test_writer_refuses_an_empty_record_stream_and_keeps_the_existing_log(
    tmp_path: Path,
) -> None:
    """The emptiness rule lived only in the reader.

    ``validate_episode_log`` rejects an empty log, but the write loop simply
    did not run for an empty ``records``, so the atomic replace fired anyway
    and put a 0-byte file where a good five-record episode had been, handing
    the caller ``0`` as a success value.
    """
    log = tmp_path / "episode.jsonl"
    write_episode_log(log, [make_record(tick) for tick in range(5)])
    before = log.read_bytes()

    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, [])
    assert caught.value.line == 1
    assert "empty episode log" in str(caught.value)
    assert log.read_bytes() == before
    assert validate_episode_log(log) == 5
    assert list(tmp_path.iterdir()) == [log]


def test_writer_refuses_an_empty_record_stream_with_no_existing_log(
    tmp_path: Path,
) -> None:
    """With nothing to destroy the refusal still stands, and writes nothing."""
    log = tmp_path / "fresh.jsonl"
    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, iter(()))
    assert "empty episode log" in str(caught.value)
    assert not log.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_repeated_t_is_written(tmp_path: Path) -> None:
    """The rule is non-decreasing, not strictly increasing; the reader agrees."""
    log = tmp_path / "flat.jsonl"
    assert write_episode_log(log, [make_record(1), make_record(1)]) == 2
    assert validate_episode_log(log) == 2


# --- R2-2: an overflowing literal is a non-finite float ---------------------


def test_validator_rejects_an_overflowing_number_literal(tmp_path: Path) -> None:
    """``1e999`` is valid JSON grammar, so ``parse_constant`` never sees it.

    ``json.loads`` yields ``inf``, the line survives ``JSON.parse`` so the
    viewer renders an infinite entropy, and a scorer averaging it returns
    ``nan`` for the whole episode.
    """
    log = tmp_path / "overflow.jsonl"
    payload = make_record(0).to_dict()
    payload["belief_digest"]["entropy"] = "@OVERFLOW@"
    poisoned = json.dumps(payload, sort_keys=True).replace('"@OVERFLOW@"', "1e999")
    assert math.isinf(json.loads(poisoned)["belief_digest"]["entropy"])
    _write_lines(log, [poisoned])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 1
    assert "record.belief_digest.entropy" in str(caught.value)


# --- R5-1: the integer spelling of that same overflow -----------------------


def _oversized(field_path: list[str | int]) -> str:
    """A good line with ``field_path`` set to a 401-digit integer literal."""
    payload: Any = make_record(0).to_dict()
    cursor: Any = payload
    for step in field_path[:-1]:
        cursor = cursor[step]
    cursor[field_path[-1]] = "@HUGE@"
    return json.dumps(payload, sort_keys=True).replace('"@HUGE@"', "9" * 401)


def test_validator_rejects_an_oversized_integer_literal(tmp_path: Path) -> None:
    """An unbounded JSON int has no double, and ``float()`` raises.

    ``OverflowError`` is an ``ArithmeticError``, not a ``RecordError``, so
    without this it escapes ``validate_line`` uncaught and the reader breaks
    its promise that every rejection names the 1-based line.
    """
    log = tmp_path / "huge_int.jsonl"
    poisoned = _oversized(["belief_digest", "entropy"])
    assert isinstance(json.loads(poisoned)["belief_digest"]["entropy"], int)
    _write_lines(log, [_good_line(0), poisoned])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 2
    assert "record.belief_digest.entropy" in str(caught.value)


def test_validator_rejects_an_oversized_integer_inside_a_pair(tmp_path: Path) -> None:
    """``_pair`` coerces each item the same way, so it has the same hole."""
    log = tmp_path / "huge_pair.jsonl"
    poisoned = _oversized(["intent", "intents", 0, "target_xy", 0])
    _write_lines(log, [poisoned])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert caught.value.line == 1
    assert "record.intent.intents[0].target_xy[0]" in str(caught.value)


# --- R5-2: the reader enforces the emptiness rule the writer cites ----------


def test_read_episode_log_refuses_a_zero_byte_log(tmp_path: Path) -> None:
    """The writer's refusal is justified by the reader's; both must hold.

    An episode truncated to zero by a killed run must not read back clean as
    an episode with no ticks.
    """
    log = tmp_path / "zero.jsonl"
    log.write_text("", encoding="utf-8")
    with pytest.raises(EpisodeLogError) as caught:
        read_episode_log(log)
    assert caught.value.line == 1
    assert "empty" in str(caught.value)
    with pytest.raises(EpisodeLogError):
        list(iter_episode_log(log))


def test_read_episode_log_refuses_a_log_of_only_blank_lines(tmp_path: Path) -> None:
    """Blank lines are skipped, so a file of them holds no records either."""
    log = tmp_path / "blank.jsonl"
    log.write_text("\n\n\n", encoding="utf-8", newline="\n")
    with pytest.raises(EpisodeLogError) as caught:
        read_episode_log(log)
    assert "empty" in str(caught.value)


# --- #80: the pose's measurement time -------------------------------------


def test_a_pose_reports_no_measurement_time_by_default() -> None:
    """``None``, never ``t``. A default of ``t`` would be a silent zero age.

    An adapter that forgets the field would then report every fix as taken
    at the instant of the tick it arrived in — plausible, wrong, and with no
    symptom. ``None`` makes a consumer that wants fix age say what it does
    without one.
    """
    pose = Pose(
        asset_id="a",
        cls="quad",
        t=4.0,
        x=0.0,
        y=0.0,
        z=0.0,
        heading=0.0,
        speed=0.0,
        energy_used=0.0,
    )
    assert pose.measured_t is None
    assert pose.to_dict()["measured_t"] is None


def test_a_measurement_time_round_trips_through_the_log(tmp_path: Path) -> None:
    """Both arms, through the writer and back: a float stays a float, null stays null."""
    log = tmp_path / "measured.jsonl"
    records = [make_record(tick) for tick in range(3)]
    assert write_episode_log(log, records) == 3
    back = read_episode_log(log)
    assert back == records
    poses = back[1].observation.poses
    assert poses[0].measured_t == 0.75
    assert isinstance(poses[0].measured_t, float)
    assert poses[1].measured_t is None


def test_an_int_measurement_time_is_coerced_like_every_other_float() -> None:
    """The gate compares bytes, so ``measured_t=0`` may not serialise as ``0``."""
    integral = Pose(
        asset_id="a",
        cls="quad",
        t=0,
        x=0,
        y=0,
        z=0,
        heading=0,
        speed=0,
        energy_used=0,
        measured_t=0,
    )
    assert isinstance(integral.measured_t, float)
    assert '"measured_t": 0.0' in json.dumps(integral.to_dict(), sort_keys=True)


def test_a_missing_measurement_time_is_a_rejected_record_not_a_silent_none(
    tmp_path: Path,
) -> None:
    """``null`` is a value; an absent key is a malformed line.

    The distinction is the whole point of the field: a transport saying it
    has no measurement time must not be indistinguishable from a writer that
    dropped one.
    """
    log = tmp_path / "absent.jsonl"
    payload = make_record(0).to_dict()
    del payload["observation"]["poses"][0]["measured_t"]
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "missing field(s) measured_t" in str(caught.value)
    assert "record.observation.poses[0]" in str(caught.value)


def test_a_wrongly_typed_measurement_time_is_rejected(tmp_path: Path) -> None:
    log = tmp_path / "typed.jsonl"
    payload = make_record(0).to_dict()
    payload["observation"]["poses"][0]["measured_t"] = "recently"
    _write_lines(log, [json.dumps(payload, sort_keys=True)])
    with pytest.raises(EpisodeLogError) as caught:
        validate_episode_log(log)
    assert "expected a number" in str(caught.value)
    assert "measured_t" in str(caught.value)


def test_the_writer_refuses_a_non_finite_measurement_time(tmp_path: Path) -> None:
    """A vehicle with no lock is ``None`` here, never ``NaN``."""
    log = tmp_path / "nan.jsonl"
    record = make_record(0)
    poses = (dataclasses.replace(record.observation.poses[0], measured_t=math.nan),)
    observation = dataclasses.replace(record.observation, poses=poses)
    with pytest.raises(EpisodeLogError) as caught:
        write_episode_log(log, [dataclasses.replace(record, observation=observation)])
    assert "record.observation.poses[0].measured_t" in str(caught.value)
    assert not log.exists()
