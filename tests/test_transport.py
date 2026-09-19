"""Issue #6's acceptance criteria, one test each (plus the guards around them).

The seam is ``SPEC.md`` §4's non-negotiable structural decision, so the first
group of tests is about what may *not* cross it. They are deliberately
mechanical: a reviewer cannot be relied on to notice a belief concept added to
an adapter at hour 25, and by then there are three adapters.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from whiteout.log import SCHEMA_VERSION, read_episode_log, validate_episode_log
from whiteout.transport import (
    DEFAULT_TRANSPORT,
    IMPLEMENTED_TRANSPORTS,
    TRANSPORT_ENV_VAR,
    TRANSPORT_METHODS,
    TRANSPORT_NAMES,
    KinematicTransport,
    Transport,
    TransportError,
    create_transport,
    selected_transport_name,
)
from whiteout.types import VEHICLE_CLASSES, FleetIntent, WaypointIntent

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_PACKAGE = REPO_ROOT / "whiteout" / "transport"

#: Packages the transport may never import. An adapter that reached into any
#: of them would have to be reimplemented three times.
FORBIDDEN_IMPORTS = (
    "whiteout.belief",
    "whiteout.policy",
    "whiteout.score",
    "whiteout.sim",
    "whiteout.tune",
)

#: Record types that are belief, policy, scoring or ground-truth concepts.
#: ``WorldObservation``, ``FleetIntent`` and their members are the seam;
#: these are the far side of it.
FORBIDDEN_TYPES = ("BeliefDigest", "Contact", "Truth", "TargetTruth", "EpisodeRecord")

#: Vocabulary that has no business naming anything inside an adapter. Matched
#: against identifiers only — the module docstrings explain what the seam
#: keeps out, and are allowed to use the words to do it.
FORBIDDEN_WORDS = ("belief", "contact", "truth", "score", "policy", "entropy", "allocat")


def _transport_sources() -> list[Path]:
    sources = sorted(TRANSPORT_PACKAGE.glob("*.py"))
    assert sources, "no modules found in whiteout/transport"
    return sources


def _identifiers(tree: ast.AST) -> set[str]:
    """Every name the module *uses*, ignoring strings, docstrings and comments."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name)
    return names


# --------------------------------------------------------------------------
# Acceptance: the protocol carries no belief, policy or scoring concept.
# --------------------------------------------------------------------------


def test_the_protocol_is_exactly_four_methods() -> None:
    surface = {name for name in vars(Transport) if not name.startswith("_")}
    assert surface == set(TRANSPORT_METHODS)
    assert TRANSPORT_METHODS == ("connect", "observe", "command", "close")


def test_observe_and_command_move_only_the_seam_types() -> None:
    annotations = Transport.observe.__annotations__, Transport.command.__annotations__
    assert annotations[0]["return"] == "WorldObservation"
    assert annotations[1]["intent"] == "FleetIntent"
    assert annotations[1]["return"] == "None"


def test_the_transport_package_imports_no_belief_policy_or_scoring_module() -> None:
    for source in _transport_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
        for module in modules:
            for forbidden in FORBIDDEN_IMPORTS:
                assert module != forbidden and not module.startswith(f"{forbidden}."), (
                    f"{source.name} imports {module}"
                )


def test_the_transport_package_names_no_belief_policy_or_scoring_type() -> None:
    for source in _transport_sources():
        used = _identifiers(ast.parse(source.read_text(encoding="utf-8")))
        leaked = sorted(used & set(FORBIDDEN_TYPES))
        assert not leaked, f"{source.name} names {', '.join(leaked)}"
        for name in used:
            lowered = name.lower()
            for word in FORBIDDEN_WORDS:
                assert word not in lowered, f"{source.name} names {name!r}"


def test_the_kinematic_transport_satisfies_the_protocol() -> None:
    assert isinstance(KinematicTransport(), Transport)


# --------------------------------------------------------------------------
# Acceptance: WHITEOUT_TRANSPORT unset selects kinematic.
# --------------------------------------------------------------------------


def test_an_unset_transport_env_selects_kinematic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRANSPORT_ENV_VAR, raising=False)
    assert selected_transport_name() == "kinematic"
    assert DEFAULT_TRANSPORT == "kinematic"
    assert isinstance(create_transport(), KinematicTransport)


def test_an_empty_transport_env_selects_kinematic(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-blank shell variable is indistinguishable from unset."""
    monkeypatch.setenv(TRANSPORT_ENV_VAR, "")
    assert selected_transport_name() == "kinematic"


def test_the_transport_env_is_read_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRANSPORT_ENV_VAR, "kinematic")
    assert selected_transport_name() == "kinematic"
    monkeypatch.setenv(TRANSPORT_ENV_VAR, "sitl")
    assert selected_transport_name() == "sitl"


# --------------------------------------------------------------------------
# Acceptance: an unknown value fails loudly, naming the valid values.
# --------------------------------------------------------------------------


def test_an_unknown_transport_env_fails_naming_every_valid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRANSPORT_ENV_VAR, "kinemtaic")
    with pytest.raises(TransportError) as raised:
        selected_transport_name()
    message = str(raised.value)
    assert "kinemtaic" in message
    for name in TRANSPORT_NAMES:
        assert name in message, f"{name} missing from {message!r}"
    assert TRANSPORT_NAMES == ("kinematic", "sitl", "arena")


def test_an_unknown_transport_name_fails_naming_every_valid_value() -> None:
    with pytest.raises(TransportError) as raised:
        create_transport("mavlink")
    message = str(raised.value)
    assert "mavlink" in message
    for name in TRANSPORT_NAMES:
        assert name in message


def test_a_valid_but_unbuilt_transport_fails_and_says_which_it_is() -> None:
    for name in TRANSPORT_NAMES:
        if name in IMPLEMENTED_TRANSPORTS:
            continue
        with pytest.raises(TransportError) as raised:
            create_transport(name)
        message = str(raised.value)
        assert "not implemented" in message
        assert name in message
        for valid in TRANSPORT_NAMES:
            assert valid in message


def test_the_cli_refuses_an_unknown_transport_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from whiteout.cli import main

    monkeypatch.setenv(TRANSPORT_ENV_VAR, "nope")
    out = tmp_path / "episode.jsonl"
    assert main(["run", "--ticks", "4", "--out", str(out)]) == 1
    stderr = capsys.readouterr().err
    for name in TRANSPORT_NAMES:
        assert name in stderr
    assert not out.exists()


# --------------------------------------------------------------------------
# Acceptance: `python -m whiteout.cli run --ticks 10` produces a valid log.
# --------------------------------------------------------------------------


def test_the_run_command_line_produces_a_valid_episode_log(tmp_path: Path) -> None:
    out = tmp_path / "episode.jsonl"
    completed = subprocess.run(
        [sys.executable, "-m", "whiteout.cli", "run", "--ticks", "10", "--out", str(out)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert validate_episode_log(out) == 10


def test_the_written_log_carries_the_fleet_and_this_schema_version(tmp_path: Path) -> None:
    from whiteout.cli import main

    out = tmp_path / "episode.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "10", "--out", str(out)]) == 0
    records = read_episode_log(out)
    assert len(records) == 10
    for record in records:
        assert record.schema_version == SCHEMA_VERSION
        assert {pose.cls for pose in record.observation.poses} == set(VEHICLE_CLASSES)
    assert [record.t for record in records] == sorted(record.t for record in records)


def test_a_zero_tick_run_refuses_rather_than_writing_an_unreadable_log(tmp_path: Path) -> None:
    """``write_episode_log`` rejects an empty log, so ``run`` must not pretend."""
    from whiteout.cli import main

    out = tmp_path / "episode.jsonl"
    assert main(["run", "--ticks", "0", "--out", str(out)]) == 1
    assert not out.exists()


# --------------------------------------------------------------------------
# The kinematic stub itself.
# --------------------------------------------------------------------------


def test_the_stub_reports_one_asset_of_every_vehicle_class() -> None:
    transport = KinematicTransport(seed=3)
    transport.connect()
    observation = transport.observe()
    assert {pose.cls for pose in observation.poses} == set(VEHICLE_CLASSES)
    assert len({pose.asset_id for pose in observation.poses}) == len(observation.poses)
    transport.close()


def test_the_stub_reports_static_poses_and_an_advancing_clock() -> None:
    transport = KinematicTransport(seed=3, tick_seconds=0.5)
    transport.connect()
    first = transport.observe()
    second = transport.observe()
    assert first.t == 0.0
    assert second.t == 0.5
    assert [(pose.x, pose.y) for pose in first.poses] == [(pose.x, pose.y) for pose in second.poses]
    assert all(pose.t == first.t for pose in first.poses)
    assert all(pose.t == second.t for pose in second.poses)


def test_the_stub_is_a_pure_function_of_its_seed() -> None:
    def first_observation(seed: int) -> tuple[tuple[float, float], ...]:
        transport = KinematicTransport(seed=seed)
        transport.connect()
        observation = transport.observe()
        transport.close()
        return tuple((pose.x, pose.y) for pose in observation.poses)

    assert first_observation(7) == first_observation(7)
    assert first_observation(7) != first_observation(8)


def test_the_stub_accepts_intents_without_acting_on_them() -> None:
    transport = KinematicTransport(seed=1)
    transport.connect()
    before = transport.observe()
    intent = FleetIntent(
        t=before.t,
        intents=(
            WaypointIntent(
                asset_id=before.poses[0].asset_id,
                t=before.t,
                target_xy=(10_000.0, 10_000.0),
                target_z=100.0,
                speed=20.0,
                reason="sweep",
                task_id="t-1",
            ),
        ),
    )
    transport.command(intent)
    after = transport.observe()
    assert transport.last_intent is intent
    assert [(pose.x, pose.y) for pose in after.poses] == [(pose.x, pose.y) for pose in before.poses]


def test_calls_before_connect_fail_loudly() -> None:
    transport = KinematicTransport()
    with pytest.raises(TransportError):
        transport.observe()
    with pytest.raises(TransportError):
        transport.command(FleetIntent(t=0.0, intents=()))


def test_connect_and_close_are_idempotent() -> None:
    transport = KinematicTransport(seed=2)
    transport.close()
    transport.connect()
    first = transport.observe()
    transport.connect()
    second = transport.observe()
    assert second.t > first.t, "a second connect must not restart the clock"
    transport.close()
    transport.close()
    with pytest.raises(TransportError):
        transport.observe()


def test_the_kinematic_transport_opens_no_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """§4: no network egress on any code path. The fake is the CI and demo path."""
    import socket

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the kinematic transport opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    transport = create_transport("kinematic", seed=5)
    transport.connect()
    transport.command(FleetIntent(t=transport.observe().t, intents=()))
    transport.close()
