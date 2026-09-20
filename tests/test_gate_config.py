"""The gate, the workflow and the tool config are themselves acceptance criteria.

Each test here fails if one of issue #4's criteria regresses.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
GATE = REPO_ROOT / "scripts" / "gate.py"

# hackathon/SPEC.md §7.
ENV_NAMES = [
    "WHITEOUT_TRANSPORT",
    "WHITEOUT_SEED",
    "WHITEOUT_DEM",
    "WHITEOUT_SITL_ENDPOINT",
    "WHITEOUT_ARENA_ENDPOINT",
    "WHITEOUT_ARTIFACTS",
    "PORT",
]

# hackathon/SPEC.md §6, in order.
GATE_STEPS = [
    "install",
    "lint",
    "format",
    "typecheck",
    "test",
    "build",
    "smoke",
    "determinism",
]


def _load_gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("whiteout_gate", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ruff_excludes_claude() -> None:
    assert ".claude" in PYPROJECT["tool"]["ruff"]["exclude"]


def test_mypy_excludes_claude() -> None:
    exclude = PYPROJECT["tool"]["mypy"]["exclude"]
    assert re.search(exclude, ".claude/worktrees/a/whiteout/x.py") is not None


def test_pytest_excludes_claude() -> None:
    assert ".claude" in PYPROJECT["tool"]["pytest"]["ini_options"]["norecursedirs"]


def test_slow_marker_is_registered() -> None:
    markers = PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.split(":")[0].strip() == "slow" for marker in markers)


def test_dev_extra_carries_the_gate_tools() -> None:
    dev = " ".join(PYPROJECT["project"]["optional-dependencies"]["dev"])
    for tool in ("ruff", "mypy", "pytest", "build"):
        assert tool in dev


def test_gate_runs_the_spec_steps_in_order() -> None:
    gate = _load_gate()
    assert [name for name, _ in gate.STEPS] == GATE_STEPS


def test_gate_forces_fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """SPEC.md §6: the gate runs in fake mode always, not merely by default."""
    gate = _load_gate()
    monkeypatch.setenv("WHITEOUT_TRANSPORT", "sitl")
    assert gate._env()["WHITEOUT_TRANSPORT"] == "kinematic"


def test_gate_test_step_deselects_slow() -> None:
    source = GATE.read_text(encoding="utf-8")
    assert '"-m", "not slow"' in source


def test_gate_build_step_is_not_isolated() -> None:
    source = GATE.read_text(encoding="utf-8")
    assert '"--wheel", "--no-isolation"' in source


def _recorder(
    gate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    returncode: object = 0,
    stdout: str = "",
) -> list[tuple[list[str], Path | None]]:
    """Replace ``gate._run`` with a recorder over (command, cwd).

    ``returncode`` may be an int or a callable taking the command, so a test
    can fail one invocation and pass another.
    """
    calls: list[tuple[list[str], Path | None]] = []

    def fake_run(
        command: list[str], *, capture: bool = False, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((list(command), cwd))
        code = returncode(command) if callable(returncode) else returncode
        return subprocess.CompletedProcess(list(command), int(code), stdout=stdout, stderr="")

    monkeypatch.setattr(gate, "_run", fake_run)
    return calls


def test_gate_resolves_whiteout_from_outside_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #53. Asked from the repo root, the answer is always "here".

    ``python -c`` prepends the working directory to ``sys.path``, so a probe
    run at the repo root cannot see which tree the environment's editable
    install points at -- which is the whole failure being guarded against.
    So assert the working directory the probe actually runs in.
    """
    gate = _load_gate()
    calls = _recorder(gate, monkeypatch)

    gate._resolved_whiteout("python")

    assert len(calls) == 1
    _command, cwd = calls[0]
    assert cwd is not None
    assert cwd.resolve() != REPO_ROOT
    with pytest.raises(ValueError):
        cwd.resolve().relative_to(REPO_ROOT)


def test_gate_fails_when_whiteout_resolves_outside_this_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #53: a foreign source tree is a named failure, never a pass."""
    gate = _load_gate()
    foreign = Path(REPO_ROOT.anchor) / "elsewhere" / "other-worktree" / "whiteout" / "__init__.py"

    monkeypatch.setattr(gate, "_resolved_whiteout", lambda _python: foreign)
    monkeypatch.setattr(gate, "_bootstrap_worktree_env", lambda: None)

    failure = gate._select_interpreter()
    assert failure is not None
    assert "wrong source tree" in failure
    assert str(foreign) in failure

    # And the whole step fails, rather than falling through to the rest.
    monkeypatch.setattr(gate, "_missing_distributions", lambda _python: ([], None))
    assert gate.step_install() == failure


def test_gate_accepts_this_worktrees_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #53: the guard must not fire on the tree the gate lives in."""
    gate = _load_gate()
    here = REPO_ROOT / "whiteout" / "__init__.py"
    monkeypatch.setattr(gate, "_resolved_whiteout", lambda _python: here)
    monkeypatch.setattr(
        gate,
        "_bootstrap_worktree_env",
        lambda: pytest.fail("bootstrapped a venv for a correctly resolving environment"),
    )
    assert gate._select_interpreter() is None


def test_gate_falls_back_to_a_worktree_local_venv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #53: two worktrees run concurrently because each gets its own env."""
    gate = _load_gate()
    assert gate.VENV_DIR == REPO_ROOT / ".venv"

    venv = tmp_path / "absent"
    foreign = Path(REPO_ROOT.anchor) / "elsewhere" / "other-worktree" / "whiteout" / "__init__.py"
    here = REPO_ROOT / "whiteout" / "__init__.py"
    answers = iter([foreign, here])
    bootstrapped: list[bool] = []

    monkeypatch.setattr(gate, "VENV_DIR", venv)
    monkeypatch.setattr(gate, "_resolved_whiteout", lambda _python: next(answers))
    monkeypatch.setattr(gate, "_bootstrap_worktree_env", lambda: bootstrapped.append(True))

    assert gate._select_interpreter() is None
    assert bootstrapped == [True]
    assert gate.PY == str(gate._venv_python(venv))


def test_gate_reuses_an_existing_worktree_venv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #53: the fallback env is built once, not on every gate run.

    A pip invocation per run is several seconds that every future PR would
    pay, so a `.venv/` that already resolves here is used as it stands.
    """
    gate = _load_gate()
    venv = tmp_path / "venv"
    python = gate._venv_python(venv)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")

    foreign = Path(REPO_ROOT.anchor) / "elsewhere" / "other-worktree" / "whiteout" / "__init__.py"
    here = REPO_ROOT / "whiteout" / "__init__.py"
    answers = iter([foreign, here])

    monkeypatch.setattr(gate, "VENV_DIR", venv)
    monkeypatch.setattr(gate, "_resolved_whiteout", lambda _python: next(answers))
    monkeypatch.setattr(
        gate,
        "_bootstrap_worktree_env",
        lambda: pytest.fail("reinstalled into a venv that already resolved to this worktree"),
    )

    assert gate._select_interpreter() is None
    assert gate.PY == str(python)


def test_gate_reports_an_unlaunchable_interpreter(tmp_path: Path) -> None:
    """Issue #53: a `.venv/` that cannot be launched is a reason, not a traceback."""
    gate = _load_gate()
    absent = tmp_path / "not-a-python.exe"
    result = gate._run([str(absent)], capture=True)
    assert result.returncode == 127


def test_gate_venv_cannot_pollute_the_other_steps() -> None:
    """Issue #53: `.venv` is already ignored and excluded everywhere."""
    assert ".venv" in PYPROJECT["tool"]["ruff"]["exclude"]
    assert ".venv" in PYPROJECT["tool"]["pytest"]["ini_options"]["norecursedirs"]
    assert re.search(PYPROJECT["tool"]["mypy"]["exclude"], ".venv/lib/whiteout/x.py") is not None
    assert ".venv/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()


def test_gate_venv_install_stays_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SPEC.md §4: no egress. The fallback install must not fetch a backend.

    Asserted on the commands the bootstrap actually issues, because these
    flags are the ones the setuptools failure mode turns on -- a flag
    surviving only in a comment would satisfy a grep of the source.
    """
    gate = _load_gate()
    monkeypatch.setattr(gate, "VENV_DIR", tmp_path / "venv")
    calls = _recorder(gate, monkeypatch)

    assert gate._bootstrap_worktree_env() is None

    creation = next(command for command, _ in calls if "venv" in command)
    assert "--system-site-packages" in creation
    # ensurepip's bundled setuptools is older than the build step's floor.
    assert "--without-pip" in creation
    install = next(command for command, _ in calls if "pip" in command)
    assert "--no-build-isolation" in install
    assert "--no-deps" in install


def test_gate_rebuilds_a_venv_it_cannot_install_into(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #53: an unusable `.venv/` must not wedge the worktree for ever.

    A truncated `python.exe` or a hand-made plain `python -m venv .venv` is
    adopted on the strength of one file existing, and then fails the install
    identically on every later run. It is thrown away and rebuilt once.
    """
    gate = _load_gate()
    venv = tmp_path / "venv"
    python = gate._venv_python(venv)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    unusable = venv / "hand-made.marker"
    unusable.write_text("adopted, but broken", encoding="utf-8")
    monkeypatch.setattr(gate, "VENV_DIR", venv)

    def outcome(command: list[str]) -> int:
        if "venv" in command:  # recreate what `python -m venv` would leave.
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_bytes(b"")
            return 0
        return 1 if unusable.exists() else 0

    calls = _recorder(gate, monkeypatch, returncode=outcome)

    assert gate._bootstrap_worktree_env() is None
    assert not unusable.exists(), "the unusable environment was reused, not rebuilt"
    assert [command for command, _ in calls if "venv" in command], "never rebuilt"
    assert len([command for command, _ in calls if "pip" in command]) == 2


def test_gate_names_the_venv_when_the_rebuild_also_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #53: a failure an implementer cannot act on is the expensive one."""
    gate = _load_gate()
    venv = tmp_path / "venv"
    monkeypatch.setattr(gate, "VENV_DIR", venv)
    _recorder(gate, monkeypatch, returncode=lambda command: 0 if "venv" in command else 1)

    failure = gate._bootstrap_worktree_env()

    assert failure is not None
    assert str(venv) in failure
    assert "delete" in failure


def test_gate_will_not_delete_the_venv_it_is_running_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R4-1: `.venv` is the conventional name, so it may be the gate's own.

    A developer debugging by hand launches the gate from `<repo>/.venv`. The
    rebuild would then delete the interpreter mid-run. It refuses, and says so.
    """
    gate = _load_gate()
    venv = tmp_path / ".venv"
    python = gate._venv_python(venv)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    keep = venv / "keep.marker"
    keep.write_text("the developer's own environment", encoding="utf-8")
    monkeypatch.setattr(gate, "VENV_DIR", venv)
    monkeypatch.setattr(gate.sys, "executable", str(python))
    calls = _recorder(gate, monkeypatch, returncode=lambda command: 2)

    failure = gate._bootstrap_worktree_env()

    assert keep.exists(), "the gate deleted the environment it is running from"
    assert failure is not None
    assert "running from" in failure
    assert str(venv) in failure
    assert len([command for command, _ in calls if "pip" in command]) == 1


def test_gate_stops_when_the_rebuild_cannot_empty_the_venv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R4-1: a surviving `python.exe` is not evidence of a usable environment.

    `rmtree(ignore_errors=True)` leaves a locked interpreter behind, and
    `_build_worktree_env` builds only `if not python.is_file()`. Retrying then
    installs into a gutted tree and reports a second, unrelated exit code.
    """
    gate = _load_gate()
    venv = tmp_path / ".venv"
    python = gate._venv_python(venv)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    monkeypatch.setattr(gate, "VENV_DIR", venv)
    calls = _recorder(gate, monkeypatch, returncode=lambda command: 2)
    # Stand in for the lock Windows puts on a running interpreter.
    monkeypatch.setattr(gate.shutil, "rmtree", lambda path, ignore_errors=False: None)

    failure = gate._bootstrap_worktree_env()

    assert failure is not None
    assert str(python) in failure, "the survivor is not named"
    assert str(venv) in failure and "delete" in failure
    assert len([command for command, _ in calls if "pip" in command]) == 1, (
        "reinstalled into an environment the rebuild failed to delete"
    )


def test_gate_rejects_a_frozen_copy_of_the_source_inside_this_worktree() -> None:
    """Issue #53: containment alone accepts the gate's own build outputs.

    A non-editable `whiteout` in `.venv/`, or the tree `step_build` leaves in
    `build/lib/`, is under REPO_ROOT but is a snapshot -- passing a gate on
    one is passing on code that is no longer in the worktree.
    """
    gate = _load_gate()
    assert gate._is_inside(REPO_ROOT / "whiteout" / "__init__.py", REPO_ROOT)
    for copy_tree in (
        gate.VENV_DIR / "Lib" / "site-packages",
        gate.VENV_DIR / "lib" / "python3.11" / "site-packages",
        REPO_ROOT / "build" / "lib",
        REPO_ROOT / "dist",
    ):
        frozen = copy_tree / "whiteout" / "__init__.py"
        assert not gate._is_inside(frozen, REPO_ROOT), f"{frozen} accepted as this worktree"


def test_gate_rejects_another_worktree_under_this_repo_root() -> None:
    """Issue #53: in the main checkout, every loop worktree is *inside* REPO_ROOT.

    The ambient editable install points at one of them, so plain containment
    takes another worktree's live source as this one -- the failure this
    guard exists to catch, reached from inside it.
    """
    gate = _load_gate()
    foreign = REPO_ROOT / ".claude" / "worktrees" / "agent-other" / "whiteout" / "__init__.py"
    assert not gate._is_inside(foreign, REPO_ROOT), f"{foreign} accepted as this worktree"


def test_gate_probes_an_adopted_venv_that_inherits_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #53: an adopted `.venv/` need not be `--system-site-packages`.

    A plain `python -m venv .venv` carrying `setuptools>=70.1` takes the
    editable install, so the rebuild never fires; the ambient probe then
    vouches for an interpreter that cannot see ruff, mypy or pytest, and
    `install` PASSes into `No module named ruff` for ever. The diagnosis has
    to name `.venv/`.
    """
    gate = _load_gate()
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
    monkeypatch.setattr(gate, "VENV_DIR", venv)

    probed: list[str] = []

    def probe(python: str) -> tuple[list[str], None]:
        probed.append(python)
        return ([] if len(probed) == 1 else ["ruff", "mypy", "pytest"], None)

    def select() -> None:
        gate.PY = str(gate._venv_python(venv))
        return None

    monkeypatch.setattr(gate, "_missing_distributions", probe)
    monkeypatch.setattr(gate, "_select_interpreter", select)

    failure = gate.step_install()

    assert failure is not None, "passed install into an interpreter missing the toolchain"
    assert "ruff" in failure
    assert venv.name in failure

    # A `.venv/` this code created inherits the ambient environment, so it
    # must not cost a second subprocess.
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
    probed.clear()
    assert gate.step_install() is None
    assert len(probed) == 1


def test_gate_reports_a_missing_toolchain_before_choosing_an_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #53: the fallback bootstrap needs the very tools this probe checks.

    Selecting first reported a missing/too-old setuptools as `wrong source
    tree`, which points at the wrong bug.
    """
    gate = _load_gate()
    monkeypatch.setattr(gate, "_missing_distributions", lambda _python: (["setuptools"], None))
    monkeypatch.setattr(
        gate,
        "_select_interpreter",
        lambda: pytest.fail("selected an interpreter before probing the toolchain"),
    )

    failure = gate.step_install()

    assert failure is not None
    assert "setuptools" in failure
    assert "wrong source tree" not in failure


def test_workflow_runs_the_gate_script() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")
    assert "python scripts/gate.py" in body
    assert 'pip install -e ".[dev]"' in body
    assert "actions/setup-python@v5" in body
    assert "'3.11'" in body


def test_workflow_only_cancels_in_progress_pull_request_runs() -> None:
    """On push the concurrency group is refs/heads/main, shared by every merge."""
    body = WORKFLOW.read_text(encoding="utf-8")
    assert "cancel-in-progress: true" not in body
    assert "github.event_name == 'pull_request'" in body


def test_gate_build_step_prunes_both_output_trees() -> None:
    source = GATE.read_text(encoding="utf-8")
    for tree in ("build", "dist"):
        assert f'rmtree(REPO_ROOT / "{tree}", ignore_errors=True)' in source


def test_workflow_has_no_path_filter() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")
    for line in body.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("paths:")
        assert not stripped.startswith("paths-ignore:")


def test_workflow_triggers_on_every_pull_request_and_main_push() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^\s*pull_request:\s*$", body, re.MULTILINE) is not None
    assert re.search(r"^\s*push:\s*$", body, re.MULTILINE) is not None
    assert re.search(r"^\s*branches:\s*\[main\]\s*$", body, re.MULTILINE) is not None


def test_env_example_has_the_spec_names_and_no_values() -> None:
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assignments = [line for line in lines if line and not line.startswith("#")]
    names = []
    for line in assignments:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*=", line), f"{line!r} carries a value"
        names.append(line[:-1])
    assert names == ENV_NAMES


def test_backlog_draft_is_deleted() -> None:
    """The criterion is deletion from the repository, so ask the index.

    Checking the working tree instead would turn this red for an untracked
    local copy -- and `docs/build/PLAN.md` describes a planning lap that
    recreates the file -- which has nothing to do with issue #4.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "--", "hackathon/backlog-draft.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == ""


def test_the_smoke_score_check_refuses_words_where_a_number_is_owed() -> None:
    """R1-M2: accepting any non-numeric string proved almost nothing.

    A regression that puts an axis back to a constant word — a stub, or
    energy plumbing that stops reaching the scorer — must fail the gate, not
    pass it as an honest absence. ``coverage`` and ``search_efficiency`` are
    on every record of any run the gate makes, so words there are a
    regression by construction.
    """
    gate = _load_gate()

    def printed(**axes: str) -> str:
        return "\n".join(f"{axis}: {axes[axis]}" for axis in gate.AXES)

    real = printed(
        coverage="1.0000",
        detection_speed="not detected",
        tracking_duration="not detected",
        search_efficiency="0.9833",
        accuracy="not measured (no truth in log)",
    )
    assert gate._score_failure(real) is None

    stubbed = printed(
        coverage="0.0000",
        detection_speed="not detected",
        tracking_duration="not detected",
        search_efficiency="no energy recorded",
        accuracy="not measured (no truth in log)",
    )
    assert gate._score_failure(stubbed) is not None

    invented = printed(
        coverage="1.0000",
        detection_speed="not detected",
        tracking_duration="pending",
        search_efficiency="0.9833",
        accuracy="not measured (no truth in log)",
    )
    failure = gate._score_failure(invented)
    assert failure is not None and "pending" in failure

    assert gate._score_failure(real.replace("1.0000", "nan")) is not None
