"""The gate, the workflow and the tool config are themselves acceptance criteria.

Each test here fails if one of issue #4's criteria regresses.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType

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


def test_gate_test_step_deselects_slow() -> None:
    source = GATE.read_text(encoding="utf-8")
    assert '"-m", "not slow"' in source


def test_gate_build_step_is_not_isolated() -> None:
    source = GATE.read_text(encoding="utf-8")
    assert '"--wheel", "--no-isolation"' in source


def test_workflow_runs_the_gate_script() -> None:
    body = WORKFLOW.read_text(encoding="utf-8")
    assert "python scripts/gate.py" in body
    assert 'pip install -e ".[dev]"' in body
    assert "actions/setup-python@v5" in body
    assert "'3.11'" in body


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
    assert not (REPO_ROOT / "hackathon" / "backlog-draft.md").exists()
