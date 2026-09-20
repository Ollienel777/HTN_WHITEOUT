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

**Which source tree the gate exercises.** ``pip install -e .`` writes one path
record per Python environment, so in an environment shared by several git
worktrees the last worktree to install wins, and every other worktree's tools
then import *that* tree's ``whiteout``. The gate's own steps mostly escape it
by accident -- ``python -m pytest`` from the repo root puts the working
directory ahead of the install -- but the spec's bare ``pytest`` does not, and
nothing downstream of the repo root does either.

So the gate resolves ``whiteout`` for its interpreter **from a directory
outside the repository**, where the working directory cannot shadow the
answer, and prints what it found. The rule, in one line:

    the gate always runs against the worktree it lives in, or it fails.

If the ambient interpreter's ``whiteout`` belongs to another tree, the gate
falls back to a per-worktree virtual environment at ``.venv/`` -- created once,
reused afterwards, ``--system-site-packages`` so the dependencies come from the
ambient environment and only the path record is worktree-local. If that still
does not resolve here, the gate fails with ``wrong source tree`` rather than
reporting a pass earned by someone else's code. ``.venv/`` is already
gitignored and excluded from every lint, typecheck and test glob.
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
#: The axes ``whiteout score`` prints, in order — kept in step with
#: ``whiteout.score.AXES`` and with ``fixtures/weights/equal.json``. Not every
#: axis is a number: one an episode log cannot answer for prints words instead
#: (``not detected``, ``not measured (no truth in log)``), which is the point
#: of #130. The smoke step below checks that every axis appears, that
#: :data:`ALWAYS_NUMERIC` are finite numbers, and that any other axis is
#: either a finite number or one of :data:`ACCEPTED_WORDS` — rather than
#: demanding five floats, or accepting any string at all.
AXES = (
    "coverage",
    "detection_speed",
    "tracking_duration",
    "search_efficiency",
    "accuracy",
)

#: The axes the smoke run must print as finite numbers. ``covered_fraction``
#: and ``energy_used`` are on every record of any run this gate makes, so an
#: axis here printing words is a regression, not an honest absence — which is
#: the failure #130 exists to catch.
ALWAYS_NUMERIC = ("coverage", "search_efficiency")

#: The exact words an axis may print in place of a number — the full set
#: ``whiteout.score`` can emit. Pinned rather than accepting any string,
#: because "not a float" is also what a stub prints: a regression that puts an
#: axis back to a constant word must fail here rather than pass as an honest
#: absence.
ACCEPTED_WORDS = (
    "not detected",
    "no energy recorded",
    "not measured (no truth in log)",
)

#: The per-worktree fallback environment. `.venv` and not some new name: it is
#: already in `.gitignore`, in `[tool.ruff] exclude`, in `[tool.mypy] exclude`
#: and in `norecursedirs`, so nothing the gate lints, typechecks, tests or
#: packages can pick it up.
VENV_DIR = REPO_ROOT / ".venv"

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
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=_env(),
            text=True,
            capture_output=capture,
            check=False,
        )
    except OSError as error:
        # An unlaunchable executable -- a half-written or hand-deleted
        # `.venv/`, say -- must reach the caller as a failed step with a
        # reason, not as a traceback out of the middle of the gate.
        print(f"  ! could not run {command[0]}: {error}", flush=True)
        return subprocess.CompletedProcess(list(command), 127, stdout="", stderr=str(error))


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


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _running_from(venv: Path) -> bool:
    """Is the interpreter executing this gate itself inside ``venv``?

    ``sys.executable`` and not ``PY``: ``PY`` is reassigned to the fallback
    environment on the way in, and that one is ours to rebuild. The question
    here is only whether deleting ``venv`` would saw off the branch we sit on.
    """
    try:
        return Path(sys.executable).resolve().is_relative_to(venv.resolve())
    except (OSError, ValueError):
        return False


def _resolved_whiteout(python: str) -> Path | None:
    """Where ``import whiteout`` lands for ``python``, asked from outside the repo.

    The probe *must* run with its working directory outside the repository.
    ``python -c`` prepends the working directory to ``sys.path``, so the same
    question asked from the repo root always answers "here" and can never see
    which tree the environment's editable install actually points at. That is
    precisely the shadowing that hides the bug this guard exists for.
    """
    script = (
        "import importlib.util\n"
        "spec = importlib.util.find_spec('whiteout')\n"
        "print(spec.origin if spec is not None and spec.origin else '')\n"
    )
    with tempfile.TemporaryDirectory() as outside_the_repo:
        probe = _run([python, "-c", script], capture=True, cwd=Path(outside_the_repo))
    if probe.returncode != 0:
        return None
    origin = probe.stdout.strip()
    return Path(origin) if origin else None


def _copy_trees() -> tuple[Path, ...]:
    """Subtrees under ``REPO_ROOT`` that are *not* this tree's source.

    ``.venv/Lib/site-packages/whiteout/`` (a non-editable install -- the wheel
    ``step_build`` just produced, or a `.venv` predating this guard) and
    ``build/lib/whiteout/`` are both under ``REPO_ROOT`` and would satisfy
    plain containment, yet both are frozen snapshots: a gate answered by one
    of them is a gate exercising code that is no longer in the tree.

    ``.claude/worktrees/`` is the other direction of the same mistake, and
    the live one: in the *main checkout* ``REPO_ROOT`` contains every loop
    worktree of this repository, and the ambient editable install spends its
    time pointing at one of them. Plain containment would take another
    worktree's working source as this one and log it as this one -- issue
    #53's exact failure, inside the guard that closes it. Inert when the gate
    runs in a worktree, which holds no worktrees of its own.

    This is the one containment predicate the whole guard rests on, so it
    excludes them rather than relying on nothing ever reaching them.
    """
    return (
        VENV_DIR,
        REPO_ROOT / "build",
        REPO_ROOT / "dist",
        REPO_ROOT / ".claude" / "worktrees",
    )


def _is_inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return False
    for copy_tree in _copy_trees():
        try:
            resolved.relative_to(copy_tree.resolve())
        except ValueError:
            continue
        return False
    return True


def _inherits_ambient(venv_dir: Path) -> bool:
    """Whether ``venv_dir`` can see the distributions the ambient probe found.

    An environment *this code creates* is ``--system-site-packages`` and does,
    so one probe covers both interpreters. One it *adopts* -- any directory
    with a ``python`` in it -- need not: a plain ``python -m venv .venv``
    carrying ``setuptools>=70.1`` takes the editable install successfully, so
    the rebuild in ``_bootstrap_worktree_env`` never fires, and the gate
    passes ``install`` into an interpreter that cannot see ruff, mypy or
    pytest. Reading ``pyvenv.cfg`` answers that without a subprocess, so the
    second probe is paid only by the environment that needs it.
    """
    try:
        text = (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "include-system-site-packages":
            return value.strip().lower() == "true"
    return False


def _build_worktree_env() -> str | None:
    """Create (once) the per-worktree `.venv/` and put this tree's source in it.

    ``--system-site-packages`` and ``--no-deps`` keep this cheap: the
    dependencies stay in the ambient environment and are simply visible, so
    the only thing installed here is the path record that has to be
    worktree-local. ``--no-build-isolation`` keeps SPEC.md §4's no-egress rule,
    exactly as ``--no-isolation`` does on the build step.

    ``--without-pip`` is load-bearing twice over. It skips ``ensurepip``, which
    is most of the cost of making a virtual environment; and ensurepip's
    bundled ``setuptools`` (65.5.0 on CPython 3.11) would otherwise sit in
    front of the ambient one in ``sys.path`` and fail the build step, which
    needs ``setuptools>=70.1`` present because it builds ``--no-isolation``.
    The ambient ``pip``, run through this interpreter, installs here anyway.
    """
    python = _venv_python(VENV_DIR)
    if not python.is_file():
        created = _run([PY, "-m", "venv", "--system-site-packages", "--without-pip", str(VENV_DIR)])
        if created.returncode != 0:
            return f"could not create {VENV_DIR.name}/ (exit code {created.returncode})"
    installed = _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--disable-pip-version-check",
            "--quiet",
        ]
    )
    if installed.returncode != 0:
        return (
            f"could not install this worktree into {VENV_DIR.name}/ "
            f"(exit code {installed.returncode})"
        )
    return None


def _bootstrap_worktree_env() -> str | None:
    """Build the per-worktree environment, rebuilding an unusable one once.

    An existing `.venv/` is adopted on the strength of ``python.is_file()``,
    which is not evidence that it works. Two shapes were observed: an
    interrupted run leaving a truncated ``python.exe`` (every later gate ends
    ``exit code 127``, for ever), and a hand-made plain ``python -m venv
    .venv`` whose bundled setuptools fails the ``--no-build-isolation``
    install with ``invalid command 'bdist_wheel'``. Both wedge the worktree
    permanently, and neither message said to delete anything. So an adopted
    environment that will not take the install is thrown away and rebuilt
    once, and a failure that survives that names the directory to remove.

    Two things the rebuild must not do. It must not delete the environment the
    gate is itself running from -- ``.venv`` is the conventional name, so a
    developer who launched the gate from their own ``<repo>/.venv`` would watch
    it be destroyed underneath them. And it must not retry against a carcass:
    ``ignore_errors=True`` leaves a locked ``python.exe`` behind, and
    ``_build_worktree_env`` treats that one file as evidence that a usable
    environment exists, so the retry would reinstall into a gutted tree and
    report a second, more confusing exit code. A partial deletion ends the
    attempt and says what survived.
    """
    adopted = _venv_python(VENV_DIR).is_file()
    failure = _build_worktree_env()
    if failure is None:
        return None
    if adopted:
        if _running_from(VENV_DIR):
            return (
                f"{failure} -- this gate is running from {sys.executable}, inside "
                f"{VENV_DIR}, so it will not delete it; rerun from an interpreter "
                f"outside that directory to have it rebuilt"
            )
        print(f"  ! {failure}; rebuilding {VENV_DIR.name}/ from scratch", flush=True)
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        survivor = _venv_python(VENV_DIR)
        if survivor.exists():
            return (
                f"{failure}, and {VENV_DIR} could not be emptied: {survivor} survived, "
                f"most likely locked by a running process -- close it, delete "
                f"{VENV_DIR} and rerun"
            )
        failure = _build_worktree_env()
        if failure is None:
            return None
    return f"{failure} -- delete {VENV_DIR} and rerun"


def _select_interpreter() -> str | None:
    """Point the gate's steps at an interpreter whose `whiteout` is this tree.

    Returns None on success, or the failure reason. On success ``PY`` is the
    interpreter every later step runs, and the tree it imports has been
    printed, so a gate log records which source was exercised.
    """
    global PY

    origin = _resolved_whiteout(PY)
    if origin is None or _is_inside(origin, REPO_ROOT):
        # Either this worktree owns the ambient install, or nothing is
        # installed at all -- which the distribution probe below reports with
        # the right instruction. Neither case is somebody else's source.
        print(f"  whiteout source: {origin if origin else '<not installed>'}")
        return None

    print(f"  whiteout source: {origin} -- NOT this worktree; falling back to {VENV_DIR.name}/")
    candidate = str(_venv_python(VENV_DIR))

    # The fallback environment is built once and reused. Re-installing on
    # every run would put a multi-second pip invocation in front of every
    # gate, which every future PR would pay for.
    cached = _resolved_whiteout(candidate) if _venv_python(VENV_DIR).is_file() else None
    if cached is not None and _is_inside(cached, REPO_ROOT):
        PY = candidate
        print(f"  whiteout source: {cached} (via {VENV_DIR.name}/)")
        return None

    failure = _bootstrap_worktree_env()
    if failure is not None:
        return f"wrong source tree: {origin} is outside {REPO_ROOT}, and {failure}"

    origin = _resolved_whiteout(candidate)
    if origin is None or not _is_inside(origin, REPO_ROOT):
        # State the observed fact and list the candidates; do not assert a
        # cause. The editable install is only one of them -- with PYTHONPATH
        # set to another tree no worktree "owns" anything, and rebuilding
        # `.venv/` cannot help, because PYTHONPATH precedes its site-packages
        # too.
        return (
            f"wrong source tree: even through {VENV_DIR.name}/, whiteout resolves to "
            f"{origin if origin else '<nowhere>'}, outside this worktree ({REPO_ROOT}). "
            f"Check PYTHONPATH ({os.environ.get('PYTHONPATH') or 'unset'}), then delete "
            f"{VENV_DIR} and rerun to rebuild it."
        )

    PY = candidate
    print(f"  whiteout source: {origin} (via {VENV_DIR.name}/)")
    return None


def _missing_distributions(python: str) -> tuple[list[str], str | None]:
    """Which of ``REQUIRED_DISTRIBUTIONS`` ``python`` cannot see, and why not."""
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
        probe = _run([python, "-c", script], capture=True, cwd=Path(outside_the_repo))
    if probe.returncode != 0:
        return [], f"install probe exited {probe.returncode}: {probe.stderr.strip()}"
    return probe.stdout.split(), None


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

    The ambient environment is probed *first*, then the interpreter is
    settled, so that no later step can be answered by another worktree's
    source. The order matters for the diagnosis: selecting first can trigger
    a fallback bootstrap, and that bootstrap needs the ambient ``pip`` and
    ``setuptools>=70.1`` -- the very things this probe reports. Probing
    second turned a missing-toolchain problem into ``wrong source tree``,
    sending the reader after the wrong bug.
    """
    missing, failure = _missing_distributions(PY)
    if failure is not None:
        return failure
    if missing:
        return f'not installed: {", ".join(missing)} -- run: pip install -e ".[dev]"'

    failure = _select_interpreter()
    if failure is not None:
        return failure

    # Probing the ambient interpreter covers a `.venv/` this code created,
    # which is `--system-site-packages`: every distribution found above is
    # visible there, and the one entry that is not inherited -- `whiteout`
    # itself -- `_select_interpreter` has just checked directly, by where it
    # imports from. An *adopted* `.venv/` gets the second probe, because it
    # inherits nothing and can otherwise pass this step into an interpreter
    # missing most of the toolchain.
    if PY != str(_venv_python(VENV_DIR)) or _inherits_ambient(VENV_DIR):
        return None
    missing, failure = _missing_distributions(PY)
    if failure is not None:
        return failure
    if missing:
        return (
            f"not installed in {VENV_DIR.name}/: {', '.join(missing)} -- "
            f"{VENV_DIR.name}/ was adopted, not created here, and does not inherit the "
            f"ambient environment. Delete {VENV_DIR} and rerun to rebuild it."
        )
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
    # `python -m build` writes into dist/ and never prunes either, so once the
    # version moves off 0.1.0 the same machine ends up holding two wheels and
    # "the wheel the submission links" stops being a single file.
    shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
    shutil.rmtree(REPO_ROOT / "dist", ignore_errors=True)
    return _shell_step([PY, "-m", "build", "--wheel", "--no-isolation"])


def _axis_lines(stdout: str) -> dict[str, str] | None:
    """Each axis's printed value, or None if an axis is missing.

    The value is returned as it was printed: a number for an axis the log
    answered for, and words for one it did not.
    """
    found: dict[str, str] = {}
    for axis in AXES:
        match = re.search(rf"^{axis}:\s*(.+?)\s*$", stdout, re.MULTILINE)
        if match is None:
            return None
        found[axis] = match.group(1)
    return found


def _score_failure(stdout: str) -> str | None:
    """Why the smoke step's score output is unacceptable, or None."""
    axes = _axis_lines(stdout)
    if axes is None:
        return f"score did not print a line for each of {', '.join(AXES)}"
    for axis, printed in axes.items():
        try:
            value = float(printed)
        except ValueError:
            if axis in ALWAYS_NUMERIC:
                return f"score printed a non-numeric {axis}: {printed!r}"
            if printed not in ACCEPTED_WORDS:
                return (
                    f"score printed {printed!r} for {axis}, which is neither a number "
                    f"nor one of {', '.join(repr(word) for word in ACCEPTED_WORDS)}"
                )
            continue  # words this scorer can honestly emit for an unanswerable axis
        if not math.isfinite(value):
            return f"score printed a non-finite {axis}: {printed!r}"
    return None


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
    return _score_failure(scored.stdout)


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
