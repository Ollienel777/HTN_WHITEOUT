"""The CLI surface the gate's smoke and determinism steps depend on."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from whiteout.cli import AXES, build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "fixtures" / "weights" / "equal.json"


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


def test_run_differs_across_seeds(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "4", "--out", str(first)]) == 0
    assert main(["run", "--seed", "8", "--ticks", "4", "--out", str(second)]) == 0
    assert first.read_bytes() != second.read_bytes()


def test_score_prints_four_finite_axes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = tmp_path / "episode.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "10", "--out", str(log)]) == 0
    capsys.readouterr()
    assert main(["score", str(log), "--weights", str(WEIGHTS)]) == 0
    stdout = capsys.readouterr().out
    for axis in AXES:
        match = re.search(rf"^{axis}:\s*(\S+)$", stdout, re.MULTILINE)
        assert match is not None, f"{axis} missing from score output"
        assert math.isfinite(float(match.group(1)))


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
