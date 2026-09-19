"""Issue #6's acceptance criteria, one test each (plus the guards around them).

The seam is ``SPEC.md`` §4's non-negotiable structural decision, so the first
group of tests is about what may *not* cross it. They are deliberately
mechanical: a reviewer cannot be relied on to notice a belief concept added to
an adapter at hour 25, and by then there are three adapters.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from whiteout.log import SCHEMA_VERSION, read_episode_log, validate_episode_log
from whiteout.transport import (
    DEFAULT_TRANSPORT,
    IMPLEMENTED_TRANSPORTS,
    TRANSPORT_ENV_VAR,
    TRANSPORT_FACTORIES,
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
#: of them would have to be reimplemented three times. ``whiteout.log`` and
#: ``whiteout.cli`` are neither belief nor policy, but a transport that
#: imported the episode-log writer would have coupled the adapter to the
#: artifact format, which is the same shape of mistake.
FORBIDDEN_IMPORTS = (
    "whiteout.belief",
    "whiteout.cli",
    "whiteout.log",
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
#: keeps out, and are allowed to use the words to do it. The four scored
#: axes — coverage, collaboration, efficiency, tracking accuracy — are in
#: here because they are the scoring vocabulary the seam must not carry:
#: ``def coverage_fraction`` in an adapter is a scoring concept whatever it
#: is called locally.
FORBIDDEN_WORDS = (
    "allocat",
    "belief",
    "collaborat",
    "contact",
    "coverage",
    "covered",
    "detection",
    "digest",
    "efficien",
    "entropy",
    "estimat",
    "policy",
    "reward",
    "score",
    "tracking",
    "truth",
)


def _transport_sources() -> list[Path]:
    """Every module in the package, **recursively**.

    ``rglob``, not ``glob``: an adapter is as likely to be a subpackage
    (``transport/arena/adapter.py``) as a module, and a non-recursive scan
    disables every rule below for it while still reporting green.
    """
    sources = sorted(TRANSPORT_PACKAGE.rglob("*.py"))
    assert sources, "no modules found in whiteout/transport"
    return sources


def _package_parts(source: Path) -> list[str]:
    """The dotted package a module lives in, as parts, for relative imports.

    Anchored on :data:`TRANSPORT_PACKAGE` rather than the repository root so
    that the guard can be pointed at a mutated copy of the package in a
    scratch directory — which is how these guards are themselves tested.
    """
    inside = source.relative_to(TRANSPORT_PACKAGE).with_suffix("").parts
    return ["whiteout", "transport", *inside[:-1]]


def _imported_modules(source: Path, tree: ast.AST) -> list[str]:
    """Every dotted name the source imports, with relative imports resolved.

    For ``import X`` that is ``X``. For ``from X import Y`` it is both ``X``
    and ``X.Y``, because ``Y`` may itself be a module: ``from whiteout
    import log`` imports ``whiteout.log``, and recording only ``whiteout``
    would match nothing in :data:`FORBIDDEN_IMPORTS`. The name a module is
    bound to locally is irrelevant — ``from whiteout import log as _m``
    records ``whiteout.log`` all the same.

    ``from ..belief import prior_for`` parses to ``module='belief',
    level=2``, which matches nothing in :data:`FORBIDDEN_IMPORTS` until it
    is resolved against the importing module's own package.

    What it does not resolve: a star import (``from whiteout import *``)
    records only ``whiteout``, since the names it binds are not in the
    source. Ruff's ``F403`` is a gate step and rejects those outright.
    """
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is None:
                    continue
                base_name = node.module
            else:
                package = _package_parts(source)
                keep = len(package) - (node.level - 1)
                base = package[: max(keep, 0)]
                base_name = ".".join([*base, node.module] if node.module else base)
            modules.append(base_name)
            modules.extend(f"{base_name}.{alias.name}" for alias in node.names if alias.name != "*")
    return modules


def _string_annotations(tree: ast.AST) -> set[str]:
    """String annotations (``-> "BeliefDigest"``), which are never walked."""
    found: set[str] = set()
    holders: list[ast.expr | None] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            holders.append(node.annotation)
        elif isinstance(node, ast.arg):
            holders.append(node.annotation)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            holders.append(node.returns)
    for holder in holders:
        if isinstance(holder, ast.Constant) and isinstance(holder.value, str):
            found.add(holder.value)
    return found


def _identifiers(tree: ast.AST) -> set[str]:
    """Every name the module *uses*, ignoring strings, docstrings and comments.

    What this catches: names bound or read anywhere in the module,
    attributes, class/function/argument names, keyword-argument names,
    **both halves of an aliased import** (``BeliefDigest as _Digest``
    contributes both, so aliasing the concept away does not hide it), and
    string annotations.

    What it does not catch, and is not trying to: a concept reached
    dynamically (``getattr(o, "belief_digest")``,
    ``importlib.import_module("whiteout.belief")``), one spelled out of
    string fragments, or one whose name shares no root with
    :data:`FORBIDDEN_WORDS`. This is a guard against the mistake an
    implementer makes at hour 25, not a sandbox against one who means it —
    a static check can always be defeated by an author who wants to.
    """
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
        elif isinstance(node, ast.keyword) and node.arg is not None:
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            # Both, not `asname or name`: the alias is the hiding place.
            names.add(node.name)
            if node.asname is not None:
                names.add(node.asname)
    return names | _string_annotations(tree)


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
        where = source.relative_to(TRANSPORT_PACKAGE).as_posix()
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for module in _imported_modules(source, tree):
            for forbidden in FORBIDDEN_IMPORTS:
                assert module != forbidden and not module.startswith(f"{forbidden}."), (
                    f"{where} imports {module}"
                )


def test_the_transport_package_names_no_belief_policy_or_scoring_type() -> None:
    for source in _transport_sources():
        where = source.relative_to(TRANSPORT_PACKAGE).as_posix()
        used = _identifiers(ast.parse(source.read_text(encoding="utf-8")))
        leaked = sorted(used & set(FORBIDDEN_TYPES))
        assert not leaked, f"{where} names {', '.join(leaked)}"
        for name in used:
            lowered = name.lower()
            for word in FORBIDDEN_WORDS:
                assert word not in lowered, f"{where} names {name!r}"


def test_the_kinematic_transport_satisfies_the_protocol() -> None:
    assert isinstance(KinematicTransport(), Transport)


# --------------------------------------------------------------------------
# The guards' own guards: a no-leak test that passes everything is worse
# than none, because it is believed. Each case below copies the package to a
# scratch directory, plants one leak in the copy, and asserts the real guard
# functions above fail on it. Nothing under `whiteout/` is touched.
# --------------------------------------------------------------------------

_GUARDS = (
    test_the_transport_package_imports_no_belief_policy_or_scoring_module,
    test_the_transport_package_names_no_belief_policy_or_scoring_type,
)


def _guard_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planted: dict[str, str],
) -> list[str]:
    """Run both guards against a copy of the package with ``planted`` applied.

    Keys are paths relative to the package; a value is appended to an
    existing module and written as a new one otherwise. Returns the
    assertion messages the guards raised, so a caller can assert both that
    something was caught and what it was.
    """
    destination = tmp_path / "transport"
    shutil.copytree(TRANSPORT_PACKAGE, destination)
    for relative, text in planted.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.write_text(target.read_text(encoding="utf-8") + text, encoding="utf-8")
        else:
            target.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "TRANSPORT_PACKAGE", destination)
    failures: list[str] = []
    for guard in _GUARDS:
        try:
            guard()
        except AssertionError as raised:
            failures.append(str(raised))
    return failures


def test_the_no_leak_guard_catches_an_aliased_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nobody writes ``def belief_digest_for``; people alias imports daily."""
    failures = _guard_failures(
        tmp_path,
        monkeypatch,
        {
            "kinematic.py": (
                "\n\nfrom whiteout.types import BeliefDigest as _Digest\n\n\n"
                "def latest(t: float) -> _Digest:\n    return _Digest\n"
            )
        },
    )
    assert failures, "an aliased BeliefDigest import went through the guard"
    assert any("BeliefDigest" in failure for failure in failures)


def test_the_no_leak_guard_scans_subpackages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-recursive scan disables every rule for an adapter subpackage."""
    failures = _guard_failures(
        tmp_path,
        monkeypatch,
        {
            "arena/__init__.py": "",
            "arena/adapter.py": "from whiteout.policy import allocate\n\n\nx = allocate\n",
        },
    )
    assert failures, "a leak in a subpackage went through the guard"
    assert any("arena/adapter.py" in failure for failure in failures)


def test_the_no_leak_guard_resolves_relative_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``from ..belief import x`` parses as ``belief``, matching nothing raw."""
    failures = _guard_failures(
        tmp_path, monkeypatch, {"kinematic.py": "\n\nfrom ..belief import prior_for\n"}
    )
    assert failures, "a relative import of whiteout.belief went through the guard"
    assert any("whiteout.belief" in failure for failure in failures)


def test_the_no_leak_guard_catches_the_four_scored_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coverage, collaboration, efficiency and tracking accuracy are scoring."""
    for name in (
        "coverage_fraction",
        "collaboration_bonus",
        "efficiency_ratio",
        "tracking_error",
    ):
        failures = _guard_failures(
            tmp_path / name, monkeypatch, {"kinematic.py": f"\n\ndef {name}(x):\n    return x\n"}
        )
        assert failures, f"{name} went through the guard"


def test_the_no_leak_guard_bans_the_episode_log_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a belief concept, but the same shape of coupling."""
    failures = _guard_failures(
        tmp_path, monkeypatch, {"kinematic.py": "\n\nfrom whiteout.log import write_episode_log\n"}
    )
    assert failures, "an import of whiteout.log went through the guard"
    assert any("whiteout.log" in failure for failure in failures)


@pytest.mark.parametrize(
    ("where", "planted", "expected"),
    [
        ("kinematic.py", "\n\nfrom whiteout import log\n\n\n_writer = log\n", "whiteout.log"),
        (
            "kinematic.py",
            "\n\nfrom whiteout import sim as _world\n\n\n_w = _world\n",
            "whiteout.sim",
        ),
        ("kinematic.py", "\n\nfrom .. import tune\n\n\n_t = tune\n", "whiteout.tune"),
    ],
    ids=["from-package-import-module", "aliased", "relative"],
)
def test_the_no_leak_guard_catches_a_module_imported_from_its_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, planted: str, expected: str
) -> None:
    """``from whiteout import log`` imports ``whiteout.log``, not ``whiteout``.

    The most ordinary spelling in Python, and the one that defeats
    :data:`FORBIDDEN_IMPORTS` if only the left-hand module is recorded.
    ``sim``, ``log``, ``cli`` and ``tune`` have no identifier in
    :data:`FORBIDDEN_WORDS`, so the import rule is the only thing standing
    between them and an adapter.
    """
    failures = _guard_failures(tmp_path, monkeypatch, {where: planted})
    assert failures, f"an import of {expected} went through the guard"
    assert any(expected in failure for failure in failures)


def test_the_no_leak_guard_passes_a_plausible_clean_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must kill leaks and nothing else, or it gets worked around."""
    failures = _guard_failures(
        tmp_path,
        monkeypatch,
        {
            "sitl.py": (
                '"""A plausible sitl adapter that respects the seam."""\n\n'
                "from whiteout.transport.base import TransportError\n"
                "from whiteout.types import FleetIntent, WorldObservation\n\n\n"
                "class SitlTransport:\n"
                "    def connect(self) -> None:\n"
                "        raise TransportError('WHITEOUT_SITL_ENDPOINT is unset')\n\n"
                "    def observe(self) -> WorldObservation:\n"
                "        raise TransportError('not connected')\n\n"
                "    def command(self, intent: FleetIntent) -> None:\n"
                "        raise TransportError('not connected')\n\n"
                "    def close(self) -> None:\n"
                "        return None\n"
            )
        },
    )
    assert not failures, f"the guard flagged a clean adapter: {failures}"


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


def test_every_implemented_transport_constructs_the_class_it_names() -> None:
    """Validating the name and then returning a fixed class is the trap.

    Without this, the ``sitl`` ticket registers its name, forgets the
    ``return``, and ``WHITEOUT_TRANSPORT=sitl`` silently runs the fake.
    """
    assert IMPLEMENTED_TRANSPORTS, "this build implements no transport at all"
    for name in IMPLEMENTED_TRANSPORTS:
        assert isinstance(create_transport(name), TRANSPORT_FACTORIES[name])


def test_create_transport_dispatches_on_the_name_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With one implementation, a hard-coded ``return`` looks correct.

    So this registers a second one and checks the name is honoured — which
    is the moment the ``sitl`` ticket would otherwise have discovered by
    demoing the fake.
    """
    import whiteout.transport as transport_module

    class _Stub:
        def __init__(self, seed: int = 0) -> None:
            self.seed = seed

    monkeypatch.setattr(
        transport_module, "TRANSPORT_FACTORIES", {**TRANSPORT_FACTORIES, "sitl": _Stub}
    )
    monkeypatch.setattr(transport_module, "IMPLEMENTED_TRANSPORTS", ("kinematic", "sitl"))
    built = transport_module.create_transport("sitl", seed=3)
    assert isinstance(built, _Stub), f"asked for sitl, got {type(built).__name__}"
    assert built.seed == 3
    assert isinstance(transport_module.create_transport("kinematic"), KinematicTransport)


def test_the_implemented_transports_are_derived_from_the_factory_table() -> None:
    """The two cannot drift, because one is computed from the other."""
    assert set(IMPLEMENTED_TRANSPORTS) == set(TRANSPORT_FACTORIES)
    assert set(TRANSPORT_FACTORIES) <= set(TRANSPORT_NAMES)
    assert IMPLEMENTED_TRANSPORTS == tuple(
        name for name in TRANSPORT_NAMES if name in TRANSPORT_FACTORIES
    )


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
    assert [(pose.lat, pose.lon) for pose in first.poses] == [
        (pose.lat, pose.lon) for pose in second.poses
    ]
    assert all(pose.t == first.t for pose in first.poses)
    assert all(pose.t == second.t for pose in second.poses)


def test_the_stub_reports_no_measurement_time_unless_it_is_asked_to() -> None:
    """The honest default: parked poses drawn at connect know nothing of fixes."""
    transport = KinematicTransport(seed=3)
    transport.connect()
    for _tick in range(3):
        assert all(pose.measured_t is None for pose in transport.observe().poses)
    transport.close()


def test_a_configured_pose_age_stamps_every_pose_behind_its_tick() -> None:
    """The staleness-correction path, exercisable offline and at speed.

    Without this the first stale fix anything downstream sees arrives from a
    link to a real vehicle, on the one run that is judged. ``t`` stays the
    tick; the age travels in ``measured_t``.
    """
    transport = KinematicTransport(seed=3, tick_seconds=0.5, pose_age_seconds=2.0)
    transport.connect()
    for tick in range(4):
        observation = transport.observe()
        assert observation.t == 0.5 * tick
        for pose in observation.poses:
            assert pose.t == observation.t, "the tick is still the tick"
            assert pose.measured_t == pytest.approx(observation.t - 2.0)
    transport.close()


def test_a_pose_age_of_zero_is_not_the_same_as_no_pose_age() -> None:
    """``0.0`` is a claim that the fix is this tick's; ``None`` is no claim."""
    transport = KinematicTransport(seed=3, pose_age_seconds=0.0)
    transport.connect()
    observation = transport.observe()
    transport.close()
    assert all(pose.measured_t == pose.t for pose in observation.poses)
    assert all(pose.measured_t is not None for pose in observation.poses)


def test_an_unusable_pose_age_is_refused_by_the_transport() -> None:
    """A negative age is a fix from the future; a non-finite one cannot be logged.

    Refused at construction, naming the argument, rather than producing
    poses that breach the seam's conformance suite several calls later.
    """
    for age in (-0.5, float("nan"), float("inf")):
        with pytest.raises(TransportError) as raised:
            KinematicTransport(seed=3, pose_age_seconds=age)
        assert "pose_age_seconds" in str(raised.value)


def test_a_pose_age_that_is_not_a_number_is_refused_as_a_transport_error() -> None:
    """Validated before the widening, so the seam's own error type comes out.

    ``float()`` first would let a non-numeric escape as a bare ``ValueError``
    past every caller that handles ``TransportError``, and would silently
    accept ``True`` as an age of one second and ``"2.0"`` as two.
    """
    for age in ("soon", "2.0", True, None.__class__, [2.0]):
        with pytest.raises(TransportError) as raised:
            KinematicTransport(seed=3, pose_age_seconds=age)  # type: ignore[arg-type]
        assert "pose_age_seconds" in str(raised.value)


def test_create_transport_carries_a_pose_age_through_to_the_transport() -> None:
    """The knob has to be reachable from the factory, or no episode can use it.

    Built by hand it only ever reaches a unit test; the staleness path is
    meant to be drivable through a real episode — a log, the scorer,
    ``replay``, the viewer.
    """
    transport = create_transport("kinematic", seed=3, pose_age_seconds=2.0)
    transport.connect()
    observation = transport.observe()
    transport.close()
    assert observation.poses, "the fleet is empty, so this asserts nothing"
    for pose in observation.poses:
        assert pose.t == observation.t, "the tick is still the tick"
        assert pose.measured_t == pytest.approx(observation.t - 2.0)


def test_create_transport_without_a_pose_age_reports_no_measurement_time() -> None:
    """The default forwards nothing, so an episode is what it always was."""
    transport = create_transport("kinematic", seed=3)
    transport.connect()
    observation = transport.observe()
    transport.close()
    assert all(pose.measured_t is None for pose in observation.poses)


def test_create_transport_refuses_an_unusable_pose_age_by_name() -> None:
    transport_error = pytest.raises(TransportError)
    with transport_error as raised:
        create_transport("kinematic", seed=3, pose_age_seconds=-1.0)
    assert "pose_age_seconds" in str(raised.value)


def test_a_transport_that_does_not_take_a_pose_age_refuses_it_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a ``TypeError`` from inside the factory table.

    The next adapter may not model fix age at all. Asking it for one is a
    caller mistake, and it is reported as the seam's own error naming the
    transport rather than as an arity error from ``__init__``.
    """

    import whiteout.transport as transport_module

    class _Ageless:
        def __init__(self, seed: int = 0) -> None:
            self.seed = seed

    monkeypatch.setattr(
        transport_module, "TRANSPORT_FACTORIES", {**TRANSPORT_FACTORIES, "kinematic": _Ageless}
    )
    with pytest.raises(TransportError) as raised:
        transport_module.create_transport("kinematic", seed=3, pose_age_seconds=2.0)
    assert "does not take a pose age" in str(raised.value)
    assert "kinematic" in str(raised.value)


def test_a_constructors_own_type_error_is_not_reported_as_a_missing_pose_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is read off the signature, not off a caught ``TypeError``.

    A transport that accepts a pose age and then raises ``TypeError`` for its
    own reasons must surface that, not a confident and wrong claim that it
    does not take the argument.
    """
    import whiteout.transport as transport_module

    class _Broken:
        def __init__(self, seed: int = 0, pose_age_seconds: float | None = None) -> None:
            raise TypeError("something else entirely")

    monkeypatch.setattr(
        transport_module, "TRANSPORT_FACTORIES", {**TRANSPORT_FACTORIES, "kinematic": _Broken}
    )
    with pytest.raises(TypeError) as raised:
        transport_module.create_transport("kinematic", seed=3, pose_age_seconds=2.0)
    assert "something else entirely" in str(raised.value)
    assert "does not take a pose age" not in str(raised.value)


def test_the_stub_is_a_pure_function_of_its_seed() -> None:
    def first_observation(seed: int) -> tuple[tuple[float, float], ...]:
        transport = KinematicTransport(seed=seed)
        transport.connect()
        observation = transport.observe()
        transport.close()
        return tuple((pose.lat, pose.lon) for pose in observation.poses)

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
                target_lat=72.05,
                target_lon=-94.60,
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
    assert [(pose.lat, pose.lon) for pose in after.poses] == [
        (pose.lat, pose.lon) for pose in before.poses
    ]


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


def test_connect_after_close_refuses_rather_than_rewinding_the_clock() -> None:
    """A transport is single-use. Reconnect would restart the world clock.

    ``write_episode_log`` refuses a record whose ``t`` goes backwards, so a
    silent reconnect costs the whole episode at write time instead of
    reporting the fault where it happened.
    """
    transport = KinematicTransport(seed=2)
    transport.connect()
    first = transport.observe()
    transport.close()
    with pytest.raises(TransportError) as raised:
        transport.connect()
    assert "close" in str(raised.value)
    assert first.t == 0.0
    with pytest.raises(TransportError):
        transport.observe()


def test_a_negative_seed_is_refused_by_the_transport_not_by_numpy() -> None:
    """``np.random.default_rng(-1)`` is an eight-frame numpy traceback."""
    with pytest.raises(TransportError) as raised:
        KinematicTransport(seed=-1)
    assert "-1" in str(raised.value)
    with pytest.raises(TransportError):
        create_transport("kinematic", seed=-1)


def test_the_cli_refuses_a_negative_seed_with_a_one_line_diagnostic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from whiteout.cli import main

    out = tmp_path / "episode.jsonl"
    assert main(["run", "--seed", "-1", "--ticks", "3", "--out", str(out)]) == 1
    stderr = capsys.readouterr().err
    # The CLI's own wording, not the transport backstop's: a bad `--seed` is
    # the caller's mistake and should not be reported as a transport fault.
    assert "episode seeds are non-negative" in stderr
    assert "transport" not in stderr
    assert "Traceback" not in stderr
    assert not out.exists()


def test_the_cli_refuses_a_negative_seed_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from whiteout.cli import main

    monkeypatch.setenv("WHITEOUT_SEED", "-7")
    out = tmp_path / "episode.jsonl"
    assert main(["run", "--ticks", "3", "--out", str(out)]) == 1
    assert "episode seeds are non-negative" in capsys.readouterr().err
    assert not out.exists()


# --------------------------------------------------------------------------
# The CLI turns a transport fault into a diagnostic, and always closes.
# `base.TransportError` promises this of `connect`, `observe` and `command`,
# and `SPEC.md` §7 makes connect-time refusal sitl's normal path.
# --------------------------------------------------------------------------


class _RefusingTransport:
    """A transport that fails at a chosen call, and records its ``close``."""

    def __init__(self, fail_at: str) -> None:
        self._fail_at = fail_at
        self.closes = 0
        self._t = 0.0

    def connect(self) -> None:
        if self._fail_at == "connect":
            raise TransportError("endpoint is unset")

    def observe(self) -> object:
        if self._fail_at == "observe":
            raise TransportError("the link dropped")
        t, self._t = self._t, self._t + 0.5
        return _observation_at(t)

    def command(self, intent: object) -> None:
        if self._fail_at == "command":
            raise TransportError("the link dropped")

    def close(self) -> None:
        self.closes += 1


def _observation_at(t: float) -> object:
    """One real observation at ``t``, so the record built from it validates."""
    transport = KinematicTransport(seed=0)
    transport.connect()
    observation = transport.observe()
    transport.close()
    return type(observation)(t=t, poses=observation.poses, reports=())


@pytest.mark.parametrize("fail_at", ["connect", "observe", "command"])
def test_the_cli_diagnoses_a_transport_fault_and_still_closes(
    fail_at: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import whiteout.cli as cli

    refusing = _RefusingTransport(fail_at)
    monkeypatch.setattr(cli, "create_transport", lambda *a, **k: refusing)
    out = tmp_path / "episode.jsonl"
    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 1
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    assert stderr.startswith("run: ")
    assert refusing.closes == 1, "a transport that failed mid-drive was not closed"
    assert not out.exists()


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
