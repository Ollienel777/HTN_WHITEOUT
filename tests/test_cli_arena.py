"""``run`` against the arena: the camera path and the track poster, wired.

Issue #132. Everything below ``cmd_run`` — the detector, the motion gate, the
sightings source, the hold and the tracks client — was merged and tested and
unreachable from the one command that runs the product. These tests drive
``run`` itself, against an arena-shaped transport and stub feeds, and assert
what the wiring is supposed to produce: a contact on an episode record, a fix
at the tracks endpoint, feeds that are closed however the run ends — and a
kinematic run that is untouched by all of it.

Nothing here opens a socket to anything but loopback: the transport is a fake,
the feeds are stubs, and the tracks endpoint is the in-process stub server.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import whiteout.cli as cli
from whiteout.geo import GeoPoint, LocalPoint, local_to_geodetic
from whiteout.log import validate_episode_log
from whiteout.tracks.stub import StubTracksServer, open_stub_tracks_server
from whiteout.types import Pose, WorldObservation
from whiteout.vision.camera import CAMERAS, VisionError
from whiteout.vision.imagery import LumaFrame

QUAD = CAMERAS["quadcopter"]

#: Somewhere in the strait. The vessel starts here and heads east at 3 m/s,
#: which is the speed the arena's own plugin moves it at, so the motion gate
#: releases it as the real thing rather than as ice.
ORIGIN = GeoPoint(71.9965, -94.8448)
VESSEL_SPEED_MPS = 3.0


class _StubFeed:
    """A camera that always has a frame, and counts its own open and close."""

    def __init__(self, asset_id: str) -> None:
        self.asset_id = asset_id
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        self.opened += 1

    def close(self) -> None:
        self.closed += 1

    def latest(self) -> LumaFrame:
        return LumaFrame(
            asset_id=self.asset_id,
            camera=QUAD,
            seq=0,
            t=0.0,
            luma=np.full((QUAD.height, QUAD.width), 60, dtype=np.uint8),
        )


class _StubPoster:
    """A poster that counts what it was handed and whether it was closed."""

    def __init__(self, client: object = None) -> None:
        self.client = client
        self.fixes: list[object] = []
        self.closed = 0

    def submit(self, fix: object) -> None:
        self.fixes.append(fix)

    def close(self, timeout_s: float = 5.0) -> None:
        self.closed += 1


class _Detection:
    """What the detector would have returned, at a point that moves."""

    def __init__(self, t: float) -> None:
        self._t = t

    def to_ground(self) -> GeoPoint:
        return local_to_geodetic(ORIGIN, LocalPoint(east_m=VESSEL_SPEED_MPS * self._t, north_m=0.0))


class _ArenaShapedTransport:
    """A transport that rosters assets the way the arena does, over no network."""

    def __init__(
        self,
        fail_at: str | None = None,
        fail_close: bool = False,
        roster: tuple[tuple[str, str], ...] = (("quadcopter", "quad"), ("tower-1", "tower")),
        refuses: frozenset[str] = frozenset(),
    ) -> None:
        self._fail_close = fail_close
        self.assets = tuple(
            type("AssetLink", (), {"asset_id": asset_id, "cls": cls})() for asset_id, cls in roster
        )
        self.closes = 0
        self._t = 0.0
        self._fail_at = fail_at
        self._refuses = refuses
        #: Every (asset_id, altitude) this transport was asked to launch.
        self.launched: list[tuple[str, float]] = []

    def arm_and_launch(self, asset_id: str, altitude_m: float = 60.0) -> None:
        from whiteout.transport import TransportError

        if asset_id in self._refuses:
            raise TransportError(f"{asset_id!r} would not arm")
        self.launched.append((asset_id, altitude_m))

    def connect(self) -> None:
        return None

    def observe(self) -> WorldObservation:
        if self._fail_at == "observe":
            from whiteout.transport import TransportError

            raise TransportError("the link dropped")
        self._t += 1.0
        poses = tuple(
            Pose(
                asset_id=asset.asset_id,
                cls=asset.cls,
                t=self._t,
                lat=ORIGIN.lat_deg,
                lon=ORIGIN.lon_deg,
                z=120.0,
                heading=90.0,
                speed=0.0,
                energy_used=0.0,
                measured_t=None,
                pitch=-20.0,
                roll=0.0,
            )
            for asset in self.assets
        )
        return WorldObservation(t=self._t, poses=poses, reports=())

    def command(self, intent: object) -> None:
        return None

    def close(self) -> None:
        self.closes += 1
        if self._fail_close:
            from whiteout.transport import TransportError

            raise TransportError("the link would not close")


@pytest.fixture
def arena(monkeypatch: pytest.MonkeyPatch) -> _ArenaShapedTransport:
    """Select the arena, and make it a fake one that needs no network."""
    transport = _ArenaShapedTransport()
    monkeypatch.setenv("WHITEOUT_ARENA_ENDPOINT", "10.99.4.1")
    monkeypatch.delenv("WHITEOUT_TRACKS_ENDPOINT", raising=False)
    monkeypatch.setattr(cli, "selected_transport_name", lambda *a, **k: "arena")
    monkeypatch.setattr(cli, "create_transport", lambda *a, **k: transport)
    return transport


@pytest.fixture
def feeds(monkeypatch: pytest.MonkeyPatch) -> tuple[_StubFeed, ...]:
    """Stub cameras in place of the arena's four MJPEG streams."""
    stubs = (_StubFeed("quadcopter"), _StubFeed("tower-1"))
    monkeypatch.setattr(cli, "arena_feeds", lambda host, *a, **k: stubs)
    return stubs


@pytest.fixture
def seeing(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every camera sees the vessel, and the vessel moves.

    The detector itself has its own tests over synthetic imagery; what is
    under test here is the wiring, so the pixel-to-vessel step is replaced by
    a detection whose ground point walks east at the vessel's speed.
    """
    ground_alts: list[float] = []
    from whiteout.vision import sightings as module

    def _detect(*_args: object, **kwargs: object) -> _Detection:
        ground_alts.append(float(kwargs["ground_alt_m"]))  # type: ignore[arg-type]
        return _Detection(float(kwargs["t"]))  # type: ignore[arg-type]

    monkeypatch.setattr(module, "detect_vessel", _detect)
    return ground_alts


@pytest.fixture
def tracks(monkeypatch: pytest.MonkeyPatch) -> StubTracksServer:
    """A tracks endpoint on loopback, standing in for the sponsor's."""
    with open_stub_tracks_server() as server:
        monkeypatch.setenv("WHITEOUT_TRACKS_ENDPOINT", server.endpoint)
        yield server


def _contacts(log: Path) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        found.extend(json.loads(line)["contacts"])
    return found


# -- the loop closes ---------------------------------------------------------


def test_an_arena_run_sights_the_vessel_and_submits_a_track(
    tmp_path: Path,
    arena: _ArenaShapedTransport,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
    tracks: StubTracksServer,
) -> None:
    """The whole point of the ticket: a camera in, a contact and a fix out."""
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "20", "--out", str(out)]) == 0
    validate_episode_log(out)
    contacts = _contacts(out)
    assert contacts, "an arena run saw the vessel every tick and logged no contact"
    assert [row["name"] for row in tracks.tracks()] == [cli.DEFAULT_TRACK_NAME]
    assert tracks.tracks()[0]["fixes"] >= 1


def test_the_cameras_are_opened_before_the_loop_and_closed_after_it(
    tmp_path: Path,
    arena: _ArenaShapedTransport,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 0
    assert all(feed.opened == 1 for feed in feeds), "a camera was never opened"
    assert all(feed.closed == 1 for feed in feeds), "a camera thread was left draining"


def test_a_run_that_fails_partway_still_closes_its_cameras(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """Four sockets and a thread each; a failed run must not leak them."""
    transport = _ArenaShapedTransport(fail_at="observe")
    monkeypatch.setenv("WHITEOUT_ARENA_ENDPOINT", "10.99.4.1")
    monkeypatch.delenv("WHITEOUT_TRACKS_ENDPOINT", raising=False)
    monkeypatch.setattr(cli, "selected_transport_name", lambda *a, **k: "arena")
    monkeypatch.setattr(cli, "create_transport", lambda *a, **k: transport)
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 1
    assert transport.closes == 1
    assert all(feed.closed == 1 for feed in feeds), "a camera thread outlived a failed run"


# -- arming the fleet (issue #138) -------------------------------------------

#: The arena's real roster: two aircraft and two towers.
_FULL_ROSTER = (
    ("quadcopter", "quad"),
    ("fixed-wing", "fixedwing"),
    ("tower-1", "tower"),
    ("tower-2", "tower"),
)


def _fleet(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> _ArenaShapedTransport:
    """An arena run over the full four-asset roster."""
    transport = _ArenaShapedTransport(roster=_FULL_ROSTER, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setenv("WHITEOUT_ARENA_ENDPOINT", "10.99.4.1")
    monkeypatch.delenv("WHITEOUT_TRACKS_ENDPOINT", raising=False)
    monkeypatch.setattr(cli, "selected_transport_name", lambda *a, **k: "arena")
    monkeypatch.setattr(cli, "create_transport", lambda *a, **k: transport)
    return transport


def test_arm_launches_the_aircraft_and_never_a_tower(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """``arm_and_launch`` refuses a tower, so the caller must not ask one.

    Asserting on what was *asked* rather than on what succeeded is the point:
    a caller that asked and swallowed the refusal would pass a test written
    the other way round, and would print a fault it had caused itself.
    """
    transport = _fleet(monkeypatch)
    out = tmp_path / "arena.jsonl"

    assert cli.main(["run", "--arm", "--ticks", "3", "--out", str(out)]) == 0

    assert [asset for asset, _ in transport.launched] == ["quadcopter", "fixed-wing"]


def test_the_fleet_is_not_armed_unless_it_is_asked_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """Off by default. This moves real vehicles in a simulator others share."""
    transport = _fleet(monkeypatch)
    out = tmp_path / "arena.jsonl"

    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 0

    assert transport.launched == []


def test_a_dry_run_arms_nothing_however_it_was_asked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """``--dry-run`` is the stronger word, and it wins over ``--arm``.

    ``docs/arena.md`` sells the dry run as the cheapest honest check, one that
    sends nothing. A dry run that armed four vehicles would be the opposite.
    """
    transport = _fleet(monkeypatch)
    out = tmp_path / "arena.jsonl"

    assert cli.main(["run", "--arm", "--dry-run", "--ticks", "3", "--out", str(out)]) == 0

    assert transport.launched == []


def test_one_asset_that_will_not_arm_does_not_end_the_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """An arena run is live and cannot be repeated; three assets beat none."""
    transport = _fleet(monkeypatch, refuses=frozenset({"quadcopter"}))
    out = tmp_path / "arena.jsonl"

    assert cli.main(["run", "--arm", "--ticks", "3", "--out", str(out)]) == 0

    assert [asset for asset, _ in transport.launched] == ["fixed-wing"]
    validate_episode_log(out)


def test_the_launch_altitude_is_the_one_the_policy_flies_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """A default that disagreed would have the first waypoint undo the climb."""
    from whiteout.policy import DEFAULT_SEARCH_PARAMS

    transport = _fleet(monkeypatch)
    out = tmp_path / "arena.jsonl"

    assert cli.main(["run", "--arm", "--ticks", "3", "--out", str(out)]) == 0

    assert {altitude for _, altitude in transport.launched} == {DEFAULT_SEARCH_PARAMS.search_alt_m}


def test_the_launch_altitude_is_a_parameter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    transport = _fleet(monkeypatch)
    out = tmp_path / "arena.jsonl"

    assert cli.main(["run", "--arm", "--launch-alt", "45", "--ticks", "3", "--out", str(out)]) == 0

    assert {altitude for _, altitude in transport.launched} == {45.0}


def test_a_kinematic_run_never_arms_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate runs the smoke episode on this path and compares bytes.

    ``--arm`` is accepted there rather than refused, so that a script with the
    flag in it does not fail on the fake; it simply has nothing to launch.
    """
    transport = _ArenaShapedTransport(roster=_FULL_ROSTER)
    monkeypatch.setattr(cli, "selected_transport_name", lambda *a, **k: "kinematic")
    monkeypatch.setattr(cli, "create_transport", lambda *a, **k: transport)
    out = tmp_path / "kinematic.jsonl"

    assert cli.main(["run", "--arm", "--ticks", "3", "--out", str(out)]) == 0

    assert transport.launched == []


def test_a_transport_that_will_not_close_still_closes_the_cameras_and_the_poster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """The failure the guarantee was written for: `transport.close()` raising.

    Run in sequence, that skipped every line under it, leaving four camera
    threads draining sockets and the poster's worker alive. Each shutdown now
    has a `finally` of its own.
    """
    transport = _ArenaShapedTransport(fail_close=True)
    posters: list[_StubPoster] = []

    def _poster(client: object) -> _StubPoster:
        built = _StubPoster(client)
        posters.append(built)
        return built

    monkeypatch.setenv("WHITEOUT_ARENA_ENDPOINT", "10.99.4.1")
    monkeypatch.setenv("WHITEOUT_TRACKS_ENDPOINT", "http://127.0.0.1:1/api/tracks")
    monkeypatch.setattr(cli, "selected_transport_name", lambda *a, **k: "arena")
    monkeypatch.setattr(cli, "create_transport", lambda *a, **k: transport)
    monkeypatch.setattr(cli, "TracksClient", lambda *a, **k: object())
    monkeypatch.setattr(cli, "TrackPoster", _poster)
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 1
    assert transport.closes == 1
    assert all(feed.closed == 1 for feed in feeds), "a camera thread outlived a close failure"
    assert posters and posters[0].closed == 1, "the poster's worker outlived a close failure"


def test_a_close_that_raises_something_else_is_diagnosed_not_traced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arena: _ArenaShapedTransport,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-`TransportError` from a teardown escaped as a traceback."""

    def _boom() -> None:
        raise OSError("the socket is already gone")

    monkeypatch.setattr(arena, "close", _boom)
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 1
    assert "the socket is already gone" in capsys.readouterr().err
    assert all(feed.closed == 1 for feed in feeds), "a camera thread outlived a close failure"


def test_a_camera_that_will_not_open_does_not_end_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arena: _ArenaShapedTransport,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An arena episode is live and cannot be repeated; half a fleet beats none."""

    def _refuse(host: str) -> tuple[object, ...]:
        raise VisionError("that camera is not there")

    monkeypatch.setattr(cli, "arena_feeds", _refuse)
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 0
    assert "no camera sightings" in capsys.readouterr().err
    validate_episode_log(out)


# -- what is never posted ----------------------------------------------------


def test_without_a_tracks_endpoint_an_arena_run_still_tracks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arena: _ArenaShapedTransport,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """The endpoint gates submission, and only submission.

    It used to gate the `TrackHold` too, so a rehearsal with it unset produced
    a log in which the cameras saw the vessel every tick and the episode log,
    the viewer and `whiteout score` all showed an empty sea.
    """

    def _refuse(*_a: object, **_k: object) -> object:
        raise AssertionError("a tracks client was built with no endpoint configured")

    monkeypatch.setattr(cli, "TracksClient", _refuse)
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "20", "--out", str(out)]) == 0
    assert _contacts(out), "an arena run with no endpoint recorded no contacts at all"


def test_a_dry_run_submits_nothing_even_with_an_endpoint_exported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arena: _ArenaShapedTransport,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
    tracks: StubTracksServer,
) -> None:
    """`docs/arena.md` §3 sells the dry run as the check that "sends nothing".

    §4 is where the operator is told to export `WHITEOUT_TRACKS_ENDPOINT`, so
    re-running §3 in that shell used to create and then keep updating a scored
    track from a run nobody intended to count.
    """

    def _refuse(*_a: object, **_k: object) -> object:
        raise AssertionError("a tracks client was built for a dry run")

    monkeypatch.setattr(cli, "TracksClient", _refuse)
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "20", "--dry-run", "--out", str(out)]) == 0
    assert tracks.tracks() == [], "a dry run submitted to the judged endpoint"
    assert _contacts(out), "a dry run stopped tracking as well as submitting"


def test_the_kinematic_run_builds_no_camera_and_no_poster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's determinism step runs on this path and compares bytes."""

    def _refuse(*_a: object, **_k: object) -> object:
        raise AssertionError("the default transport reached the arena-only wiring")

    monkeypatch.setattr(cli, "arena_feeds", _refuse)
    monkeypatch.setattr(cli, "TracksClient", _refuse)
    monkeypatch.setattr(cli, "host_from_env", _refuse)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    for out in (first, second):
        assert cli.main(["run", "--seed", "7", "--ticks", "20", "--out", str(out)]) == 0
    assert first.read_bytes() == second.read_bytes()


# -- one datum, threaded ------------------------------------------------------


def test_the_water_plane_is_one_number_for_belief_and_for_vision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arena: _ArenaShapedTransport,
    feeds: tuple[_StubFeed, ...],
    seeing: list[float],
) -> None:
    """#76: two defaults that happen to agree are still two numbers."""
    seen: list[float] = []
    real = cli.Coordinator

    def _capture(*args: object, **kwargs: object) -> object:
        seen.append(float(kwargs["ground_alt_m"]))  # type: ignore[arg-type]
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "Coordinator", _capture)
    out = tmp_path / "arena.jsonl"
    assert cli.main(["run", "--ticks", "3", "--out", str(out)]) == 0
    assert seen == [cli.GROUND_ALT_M], "the coordinator was given its own water plane"
    assert seeing and set(seeing) == {cli.GROUND_ALT_M}, "vision was given a different one"
