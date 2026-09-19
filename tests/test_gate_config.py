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


def test_gate_resolves_whiteout_from_outside_the_repo() -> None:
    """Issue #53. Asked from the repo root, the answer is always "here".

    ``python -c`` prepends the working directory to ``sys.path``, so a probe
    run at the repo root cannot see which tree the environment's editable
    install points at -- which is the whole failure being guarded against.
    """
    source = GATE.read_text(encoding="utf-8")
    body = source.split("def _resolved_whiteout(")[1].split("\ndef ")[0]
    assert "TemporaryDirectory" in body
    assert "cwd=Path(outside_the_repo)" in body


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

    # And the whole step fails, rather than falling through to the probe.
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


def test_gate_venv_cannot_pollute_the_other_steps() -> None:
    """Issue #53: `.venv` is already ignored and excluded everywhere."""
    assert ".venv" in PYPROJECT["tool"]["ruff"]["exclude"]
    assert ".venv" in PYPROJECT["tool"]["pytest"]["ini_options"]["norecursedirs"]
    assert re.search(PYPROJECT["tool"]["mypy"]["exclude"], ".venv/lib/whiteout/x.py") is not None
    assert ".venv/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").split()


def test_gate_venv_install_stays_offline() -> None:
    """SPEC.md §4: no egress. The fallback install must not fetch a backend."""
    source = GATE.read_text(encoding="utf-8")
    assert '"--no-build-isolation"' in source
    assert '"--no-deps"' in source
    assert '"--system-site-packages"' in source
    # ensurepip's bundled setuptools is older than the build step's floor.
    assert '"--without-pip"' in source


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
