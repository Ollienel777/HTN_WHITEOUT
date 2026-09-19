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
import shutil
import subprocess
import sys
import tempfile
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
    env["WHITEOUT_TRANSPORT"] = "kinematic"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(
    command: Sequence[str], *, capture: bool = False, cwd: Path = REPO_ROOT
) -> subprocess.CompletedProcess[str]:
    print(f"  $ {shlex.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
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


#: Distributions `pip install -e ".[dev]"` must have put in the environment.
REQUIRED_DISTRIBUTIONS = (
    "whiteout",
    "numpy",
    "scipy",
    "networkx",
    "ruff",
    "mypy",
    "pytest",
    "build",
    "setuptools",
)


def step_install() -> str | None:
    """`pip install -e ".[dev]"` is done by CI; the gate asserts that it ran.

    The probe asks for installed *distributions*, not importable names, and it
    runs from a directory outside the repository. Both halves are needed. A
    name probe cannot fail on the entries that matter, because the other steps
    run with ``cwd`` at the repo root: ``whiteout`` resolves from the source
    tree with nothing installed, and the ``build/`` directory a gate run leaves
    behind resolves as a namespace package whether or not pypa/build is there.
    A metadata probe run from the repo root is defeated too, by the gitignored
    ``whiteout.egg-info/`` that an editable install or a build leaves behind.
    """
    script = (
        "import importlib.metadata as md\n"
        f"names = {REQUIRED_DISTRIBUTIONS!r}\n"
        "missing = []\n"
        "for name in names:\n"
        "    try:\n"
        "        md.distribution(name)\n"
        "    except md.PackageNotFoundError:\n"
        "        missing.append(name)\n"
        "print(' '.join(missing))\n"
    )
    with tempfile.TemporaryDirectory() as outside_the_repo:
        probe = _run([PY, "-c", script], capture=True, cwd=Path(outside_the_repo))
    if probe.returncode != 0:
        return f"install probe exited {probe.returncode}: {probe.stderr.strip()}"
    missing = probe.stdout.split()
    if missing:
        return f'not installed: {", ".join(missing)} -- run: pip install -e ".[dev]"'
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
    # setuptools' build_py copies only files newer than their counterpart in
    # build/lib and never prunes, so a wheel built over a stale tree ships
    # modules that have since been deleted. CI never sees it (fresh checkout),
    # which makes it exactly the machine that builds the submission artifact
    # that gets the wrong wheel.
    shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
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
