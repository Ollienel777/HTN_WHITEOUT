"""The CLI surface the gate's smoke and determinism steps depend on."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from whiteout.cli import AXES, build_parser, main
from whiteout.log import validate_episode_log

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "fixtures" / "weights" / "equal.json"
DEMO = REPO_ROOT / "fixtures" / "episodes" / "demo.jsonl"


def test_every_subcommand_parses() -> None:
    parser = build_parser()
    for argv in (
        ["run", "--seed", "7", "--ticks", "4", "--out", "x.jsonl"],
        ["replay"],
        ["score", "x.jsonl"],
        ["sweep"],
        ["ablate"],
        ["serve"],
    ):
        assert parser.parse_args(argv).command == argv[0]


def test_run_is_byte_identical_at_the_same_seed(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    for out in (first, second):
        assert main(["run", "--seed", "7", "--ticks", "40", "--out", str(out)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_run_writes_a_stale_fix_into_the_log_when_asked(tmp_path: Path) -> None:
    """The end of the path the pose age exists for.

    Built by hand, a stale fix reaches only a unit test. Through ``run`` it
    reaches an episode log, and from there the scorer, ``replay`` and the
    viewer — which is the point: the staleness-correction path is exercised
    somewhere other than the one judged arena run.
    """
    out = tmp_path / "stale.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "4", "--pose-age", "2.0", "--out", str(out)]) == 0
    validate_episode_log(out)
    for line in out.read_text().splitlines():
        record = json.loads(line)
        poses = record["observation"]["poses"]
        assert poses, "the fleet is empty, so this asserts nothing"
        for pose in poses:
            assert pose["t"] == record["t"], "the tick is still the tick"
            assert pose["measured_t"] == pytest.approx(record["t"] - 2.0)


def test_run_without_a_pose_age_reports_no_measurement_time(tmp_path: Path) -> None:
    """The default changes no episode: every pose still says it does not know."""
    out = tmp_path / "plain.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "4", "--out", str(out)]) == 0
    for line in out.read_text().splitlines():
        for pose in json.loads(line)["observation"]["poses"]:
            assert pose["measured_t"] is None


def test_run_diagnoses_an_unusable_pose_age(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A caller's mistake, reported as a diagnostic rather than a traceback."""
    out = tmp_path / "never.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "4", "--pose-age", "-1", "--out", str(out)]) == 1
    assert "pose_age_seconds" in capsys.readouterr().err
    assert not out.exists()


def test_run_differs_across_seeds(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "4", "--out", str(first)]) == 0
    assert main(["run", "--seed", "8", "--ticks", "4", "--out", str(second)]) == 0
    assert first.read_bytes() != second.read_bytes()


def _score_lines(stdout: str) -> dict[str, str]:
    """Each axis's printed value, exactly as the CLI rendered it."""
    found: dict[str, str] = {}
    for axis in AXES:
        match = re.search(rf"^{axis}:\s*(.+?)\s*$", stdout, re.MULTILINE)
        assert match is not None, f"{axis} missing from score output"
        found[axis] = match.group(1)
    return found


def test_score_prints_a_line_for_every_axis(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shape ``SPEC.md`` §6's smoke step and ``scripts/gate.py`` parse."""
    log = tmp_path / "episode.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "10", "--out", str(log)]) == 0
    capsys.readouterr()
    assert main(["score", str(log), "--weights", str(WEIGHTS)]) == 0
    printed = _score_lines(capsys.readouterr().out)
    assert math.isfinite(float(printed["coverage"])), "coverage is on every record"
    for axis, value in printed.items():
        try:
            number = float(value)
        except ValueError:
            assert value.strip(), f"{axis} printed nothing at all"
            continue
        assert math.isfinite(number), f"{axis} printed {value!r}"


def test_score_never_prints_a_zero_for_an_axis_it_did_not_measure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #130: a fake zero reads as a bad result, not as a missing one."""
    assert main(["score", str(DEMO), "--weights", str(WEIGHTS)]) == 0
    stdout = capsys.readouterr().out
    printed = _score_lines(stdout)
    assert printed["accuracy"] == "not measured (no truth in log)"
    assert float(printed["coverage"]) > 0.0
    assert float(printed["search_efficiency"]) > 0.0
    said = stdout.lower()
    assert "autonomy" in said and "collaboration" in said


def test_score_shows_where_each_axis_came_from(capsys: pytest.CaptureFixture[str]) -> None:
    """Every axis line is followed by the field it was derived from."""
    assert main(["score", str(DEMO)]) == 0
    lines = capsys.readouterr().out.splitlines()
    for index, line in enumerate(lines):
        axis = line.split(":")[0]
        if axis not in AXES:
            continue
        following = lines[index + 1]
        assert following.startswith("    "), f"{axis} is not followed by its derivation"
        assert following.strip()


def test_score_refuses_an_empty_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Zeros over no data are what this command exists to stop printing."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["score", str(empty)]) == 1
    assert "empty" in capsys.readouterr().err


def test_score_diagnoses_a_corrupt_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The scorer reads the log now, so a bad line must be a reason, not a traceback."""
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text('{"not": "a record"}\n', encoding="utf-8")
    assert main(["score", str(corrupt)]) == 1
    assert "score:" in capsys.readouterr().err


def test_score_rejects_a_missing_log(tmp_path: Path) -> None:
    assert main(["score", str(tmp_path / "nope.jsonl")]) == 1


def test_unimplemented_subcommands_refuse_loudly() -> None:
    """``serve`` is not in this list any more: it binds a port and blocks.

    Its own behaviour — ``PORT``, no reuse, what it hands out — is
    ``tests/test_serve.py``.
    """
    for name in ("replay", "sweep", "ablate"):
        assert main([name]) == 1


def test_a_bad_seed_env_does_not_break_seedless_subcommands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WHITEOUT_SEED is resolved inside ``run``, not when the parser is built.

    As an argparse default it was parsed before argv had been looked at, so a
    non-integer value tracebacked out of every subcommand, none of which but
    ``run`` takes a seed at all.
    """
    monkeypatch.setenv("WHITEOUT_SEED", "random")
    build_parser()
    assert main(["score", str(tmp_path / "nope.jsonl")]) == 1
    for name in ("replay", "sweep", "ablate"):
        assert main([name]) == 1
    # `serve` binds a port rather than returning, so the parser is as far as
    # this test follows it.
    assert build_parser().parse_args(["serve"]).command == "serve"


def test_run_diagnoses_a_bad_seed_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("WHITEOUT_SEED", "random")
    out = tmp_path / "episode.jsonl"
    assert main(["run", "--ticks", "2", "--out", str(out)]) == 1
    assert "WHITEOUT_SEED=random" in capsys.readouterr().err
    assert not out.exists()


def test_run_reads_the_seed_env_and_the_flag_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WHITEOUT_SEED", "7")
    from_env = tmp_path / "env.jsonl"
    explicit = tmp_path / "explicit.jsonl"
    overridden = tmp_path / "overridden.jsonl"
    assert main(["run", "--ticks", "4", "--out", str(from_env)]) == 0
    assert main(["run", "--seed", "7", "--ticks", "4", "--out", str(explicit)]) == 0
    assert main(["run", "--seed", "8", "--ticks", "4", "--out", str(overridden)]) == 0
    assert from_env.read_bytes() == explicit.read_bytes()
    assert from_env.read_bytes() != overridden.read_bytes()


def test_score_diagnoses_malformed_weights(tmp_path: Path) -> None:
    """The tuner writes these files, so a truncated write must not traceback."""
    log = tmp_path / "episode.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "4", "--out", str(log)]) == 0
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"coverage": 0.25,', encoding="utf-8")
    assert main(["score", str(log), "--weights", str(truncated)]) == 1
    non_numeric = tmp_path / "non_numeric.json"
    non_numeric.write_text(
        json.dumps({axis: ("x" if axis == "coverage" else 0.25) for axis in AXES}),
        encoding="utf-8",
    )
    assert main(["score", str(log), "--weights", str(non_numeric)]) == 1


def test_score_refuses_a_negative_or_non_finite_weight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R1-M4: a bad weight used to print ``total: 0.0000`` or ``total: nan``.

    Both are reachable from a plain file a sweep could write — Python's
    ``json`` accepts the bare token ``NaN`` — and the total is the one line a
    judge reads, so a fake number there is the exact output #130 abolished.
    """
    log = tmp_path / "episode.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "4", "--out", str(log)]) == 0
    capsys.readouterr()
    negative = tmp_path / "negative.json"
    negative.write_text(
        json.dumps({axis: (-4.0 if axis == "coverage" else 1.0) for axis in AXES}),
        encoding="utf-8",
    )
    assert main(["score", str(log), "--weights", str(negative)]) == 1
    assert "coverage" in capsys.readouterr().err
    not_a_number = tmp_path / "nan.json"
    not_a_number.write_text(
        "{"
        + ", ".join(f'"{axis}": ' + ("NaN" if axis == "coverage" else "1.0") for axis in AXES)
        + "}",
        encoding="utf-8",
    )
    assert main(["score", str(log), "--weights", str(not_a_number)]) == 1
    assert "coverage" in capsys.readouterr().err
