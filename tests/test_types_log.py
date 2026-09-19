"""Issue #5's acceptance criteria, one test each (plus the guards around them).

The episode log is the only artifact the scorer, the viewer, the tuner and the
ablation harness read, so these tests are the contract for all of them.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import fields
from pathlib import Path

import pytest

from whiteout.log import (
    SCHEMA_VERSION,
    EpisodeLogError,
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
