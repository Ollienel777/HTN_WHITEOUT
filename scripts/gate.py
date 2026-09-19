#!/usr/bin/env python
"""The one gate command.

Runs the steps of ``hackathon/SPEC.md`` §6 in order, failing fast. CI runs
this and so does every implementer before opening a PR:

    python scripts/gate.py

The step commands are copied from the spec verbatim. Two flags are
load-bearing and are explained there: ``-m "not slow"`` on the test step, and
``--no-isolation`` on the build step. Do not drop either.

The gate never starts SITL, never opens a socket and never leaves the machine.
``WHITEOUT_TRANSPORT`` is forced to its default, ``kinematic``, for every step.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_LOG = REPO_ROOT / "artifacts" / "smoke.jsonl"
SMOKE_LOG_SECOND = REPO_ROOT / "artifacts" / "smoke.second.jsonl"
WEIGHTS = "fixtures/weights/equal.json"
AXES = ("coverage", "collaboration", "efficiency", "tracking_accuracy")

PY = sys.executable


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("WHITEOUT_TRANSPORT", "kinematic")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(command: Sequence[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"  $ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_env(),
        text=True,
        capture_output=capture,
        check=False,
    )


def _shell_step(command: Sequence[str]) -> str | None:
    """Run a command; return None on success or a failure reason."""
    result = _run(command)
    if result.returncode != 0:
        return f"exit code {result.returncode}"
    return None


def step_install() -> str | None:
    """`pip install -e ".[dev]"` is done by CI; the gate asserts imports resolve."""
    required = (
        "whiteout",
        "numpy",
        "scipy",
        "networkx",
        "ruff",
        "mypy",
        "pytest",
        "build",
        "setuptools",
        "wheel",
    )
    script = (
        "import importlib.util, sys\n"
        f"names = {required!r}\n"
        "print(' '.join(n for n in names if importlib.util.find_spec(n) is None))\n"
    )
    probe = _run([PY, "-c", script], capture=True)
    if probe.returncode != 0:
        return f"import probe exited {probe.returncode}: {probe.stderr.strip()}"
    missing = probe.stdout.split()
    if missing:
        return f'imports do not resolve: {", ".join(missing)} -- run: pip install -e ".[dev]"'
    return None


def step_lint() -> str | None:
    return _shell_step([PY, "-m", "ruff", "check", "whiteout", "tests", "scripts"])


def step_format() -> str | None:
    return _shell_step([PY, "-m", "ruff", "format", "--check", "whiteout", "tests", "scripts"])


def step_typecheck() -> str | None:
    return _shell_step([PY, "-m", "mypy", "whiteout"])


def step_test() -> str | None:
    return _shell_step([PY, "-m", "pytest", "-q", "-m", "not slow"])


def step_build() -> str | None:
    return _shell_step([PY, "-m", "build", "--wheel", "--no-isolation"])


def _finite_scores(stdout: str) -> list[float]:
    found: list[float] = []
    for axis in AXES:
        match = re.search(rf"^{axis}:\s*(\S+)\s*$", stdout, re.MULTILINE)
        if match is None:
            return []
        try:
            value = float(match.group(1))
        except ValueError:
            return []
        if not math.isfinite(value):
            return []
        found.append(value)
    return found


def _smoke_run(out: Path) -> str | None:
    result = _run(
        [PY, "-m", "whiteout.cli", "run", "--seed", "7", "--ticks", "400", "--out", str(out)]
    )
    if result.returncode != 0:
        return f"run exited {result.returncode}"
    if not out.is_file():
        return f"run produced no log at {out}"
    return None


def step_smoke() -> str | None:
    SMOKE_LOG.parent.mkdir(parents=True, exist_ok=True)
    failure = _smoke_run(SMOKE_LOG)
    if failure is not None:
        return failure
    scored = _run(
        [PY, "-m", "whiteout.cli", "score", str(SMOKE_LOG), "--weights", WEIGHTS],
        capture=True,
    )
    sys.stdout.write(scored.stdout)
    sys.stderr.write(scored.stderr)
    if scored.returncode != 0:
        return f"score exited {scored.returncode}"
    if len(_finite_scores(scored.stdout)) != len(AXES):
        return f"score did not print four finite values for {', '.join(AXES)}"
    return None


def step_determinism() -> str | None:
    failure = _smoke_run(SMOKE_LOG_SECOND)
    if failure is not None:
        return failure
    first = SMOKE_LOG.read_bytes()
    second = SMOKE_LOG_SECOND.read_bytes()
    if first != second:
        return (
            f"episode logs differ at the same seed: {SMOKE_LOG.name} is {len(first)} bytes, "
            f"{SMOKE_LOG_SECOND.name} is {len(second)} bytes"
        )
    print(f"  logs are byte-identical ({len(first)} bytes)")
    return None


STEPS: tuple[tuple[str, Callable[[], str | None]], ...] = (
    ("install", step_install),
    ("lint", step_lint),
    ("format", step_format),
    ("typecheck", step_typecheck),
    ("test", step_test),
    ("build", step_build),
    ("smoke", step_smoke),
    ("determinism", step_determinism),
)


def main() -> int:
    started = time.monotonic()
    for index, (name, run_step) in enumerate(STEPS, start=1):
        print(f"\n=== [{index}/{len(STEPS)}] {name} ===", flush=True)
        step_started = time.monotonic()
        failure = run_step()
        elapsed = time.monotonic() - step_started
        if failure is not None:
            print(f"GATE FAIL: {name} ({elapsed:.1f}s) -- {failure}", flush=True)
            return 1
        print(f"GATE PASS: {name} ({elapsed:.1f}s)", flush=True)
    print(f"\nGATE OK: {len(STEPS)} steps in {time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
