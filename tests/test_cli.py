"""The CLI surface the gate's smoke and determinism steps depend on."""

from __future__ import annotations

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
    for name in ("replay", "sweep", "ablate", "serve"):
        assert main([name]) == 1
