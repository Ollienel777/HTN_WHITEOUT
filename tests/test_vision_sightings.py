"""Camera frames becoming sightings, without a camera.

Issue #105. The live path needs four MJPEG sockets and a WireGuard network;
what is tested here is the join — which assets reach the detector, which are
skipped and why, and that a stalled camera cannot stall the fleet.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from whiteout.tracks.maintain import Sighting
from whiteout.types import Pose, WorldObservation
from whiteout.vision.camera import CAMERAS, VisionError
from whiteout.vision.imagery import LumaFrame
from whiteout.vision.projection import CameraPose
from whiteout.vision.sightings import (
    ARENA_CAMERA_PORTS,
    CameraFeed,
    VisionSightings,
    arena_feeds,
)

TOWER = CAMERAS["tower"]


def _pose(
    asset_id: str,
    *,
    alt: float = 120.0,
    attitude: bool = True,
    measured_t: float | None = None,
) -> Pose:
    return Pose(
        asset_id=asset_id,
        cls="tower",
        t=1.0,
        lat=71.9950,
        lon=-94.8700,
        z=alt,
        heading=90.0,
        speed=0.0,
        energy_used=0.0,
        pitch=-10.0 if attitude else None,
        roll=0.0 if attitude else None,
        measured_t=measured_t,
    )


def _frame(seq: int = 0) -> LumaFrame:
    return LumaFrame(
        asset_id="tower-1",
        camera=TOWER,
        seq=seq,
        t=0.0,
        luma=np.full((TOWER.height, TOWER.width), 60, dtype=np.uint8),
    )


class _StubFeed:
    """A feed that hands back whatever it was given.

    ``draining`` defaults to ``True`` — a live camera — because that is what
    almost every case here is about. The dead-stream case is what
    :func:`test_a_feed_that_has_stopped_draining_is_no_longer_a_sensor`
    builds, and it is deliberately *not* the same thing as having no frame:
    a stopped feed keeps its last one.
    """

    def __init__(self, asset_id: str, frame: LumaFrame | None, *, draining: bool = True) -> None:
        self.asset_id = asset_id
        self._frame = frame
        self.draining = draining

    def latest(self) -> LumaFrame | None:
        return self._frame

    def open(self) -> None: ...

    def close(self) -> None: ...


def _observation(poses: tuple[Pose, ...]) -> WorldObservation:
    return WorldObservation(t=1.0, poses=poses, reports=())


# -- the roster --------------------------------------------------------------


def test_the_published_camera_ports_are_the_arena_s() -> None:
    assert ARENA_CAMERA_PORTS == {
        "quadcopter": 8600,
        "fixed-wing": 8610,
        "tower-1": 8630,
        "tower-2": 8640,
    }


def test_feeds_are_built_from_a_configured_host_never_a_constant() -> None:
    """The arena's address is handed out per team (#63)."""
    feeds = arena_feeds("10.0.0.9")
    assert {f.asset_id for f in feeds} == set(ARENA_CAMERA_PORTS)
    assert all("10.0.0.9" in f._source for f in feeds)


def test_an_asset_with_no_camera_is_skipped_not_invented() -> None:
    assert arena_feeds("10.0.0.9", assets=("rover",)) == ()


# -- what is skipped, and why ------------------------------------------------


def test_an_asset_with_no_pose_contributes_nothing() -> None:
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.sightings(_observation(())) == ()


def test_a_feed_stops_draining_when_its_stream_ends_however_it_ends() -> None:
    """``draining`` covers a clean end of stream, which sets no ``error``.

    ``CameraFeed._run`` records an error only for ``VisionError``/``OSError``.
    An MJPEG server that simply closes the connection raises nothing, so a
    feed that had gone quiet for the most ordinary reason of all would have
    looked healthy to any check written on ``error`` alone.
    """
    from whiteout.vision import sightings as module

    class _Ends:
        def __init__(self, *_: object) -> None: ...

        def frames(self):
            return iter(())

    feed = module.CameraFeed("tower-1", "http://nowhere/stream", CAMERAS["tower"])
    assert not feed.draining, "a feed is not draining before it is opened"
    module_frames = module.MjpegFrames
    try:
        module.MjpegFrames = _Ends  # type: ignore[misc]
        feed.open()
        feed._thread.join(2.0)  # noqa: SLF001
    finally:
        module.MjpegFrames = module_frames  # type: ignore[misc]
    assert feed.error is None, "a clean end of stream is not an error"
    assert not feed.draining, "but it is still the end of the stream"


def test_an_asset_with_no_frame_yet_contributes_nothing() -> None:
    """A camera that has not produced a frame is silence, not a fault."""
    source = VisionSightings(feeds=(_StubFeed("tower-1", None),))
    assert source.sightings(_observation((_pose("tower-1"),))) == ()


def test_without_pitch_and_roll_there_is_no_sighting() -> None:
    """Projection takes three angles, and guessing "level" is a measurement.

    A `Pose` whose pitch and roll are `None` says the transport does not know
    them; inventing zeros would put the projected point somewhere confidently
    wrong, and accuracy is scored.
    """
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    poses = (_pose("tower-1", attitude=False),)
    assert source.sightings(_observation(poses)) == ()


def test_a_measured_skew_past_the_bound_is_no_sighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`telemetry_stale`: the pose is *known* to belong to another instant.

    The tick is at t=1.0, the fix was measured at 0.4, and the bound is a
    quarter-second. At the fixed-wing's 22 m/s that is 13 m of position error
    in a lat/lon the tracks API scores, so this one is refused rather than
    labelled — the only sync state that is.
    """
    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self) -> GeoPoint:
            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    poses = (_pose("tower-1", measured_t=0.4),)
    assert source.sightings(_observation(poses)) == ()


def test_a_skew_that_is_merely_unknown_is_carried_and_labelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`telemetry_missing` is every arena pose today, and must not blind us.

    The arena adapter reports `measured_t=None` because it has not established
    the `time_boot_ms` offset, which is the honest answer. Refusing on it would
    post nothing at all in the arena we are flying in, so the sighting stands
    and says what it does not know.
    """
    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self) -> GeoPoint:
            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    (found,) = source.sightings(_observation((_pose("tower-1"),)))
    assert found.sync is not None
    assert found.sync.status == "telemetry_missing"
    assert found.sync.skew_s is None, "an unknown skew is never a zero one"


def test_a_fresh_fix_is_carried_as_synchronised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self) -> GeoPoint:
            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    (found,) = source.sightings(_observation((_pose("tower-1", measured_t=0.9),)))
    assert found.sync is not None
    assert found.sync.status == "synchronised"
    assert found.sync.skew_s == pytest.approx(0.1)
    assert found.sync.max_skew_s == pytest.approx(0.25)


def test_the_skew_is_measured_against_the_tick_not_the_frame_s_own_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame's `t` is a wall clock or an index over a frame rate.

    Neither is the clock `measured_t` is in, so subtracting one from the other
    would yield a number that looks like a skew and is not one. The frame here
    is stamped at an epoch-sized time and must not move the verdict.
    """
    from dataclasses import replace

    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self) -> GeoPoint:
            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    frame = replace(_frame(), t=1_780_000_000.0)
    source = VisionSightings(feeds=(_StubFeed("tower-1", frame),))
    (found,) = source.sightings(_observation((_pose("tower-1", measured_t=0.9),)))
    assert found.sync is not None
    assert found.sync.status == "synchronised"


def test_the_detector_is_told_how_the_pair_stood_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detector cannot compute it: a `CameraPose` has no clock in it."""
    from whiteout.vision import sightings as module

    seen: list[object] = []

    def _capture(*_a: object, **kwargs: object) -> None:
        seen.append(kwargs.get("sync"))
        return None

    monkeypatch.setattr(module, "detect_vessel", _capture)
    VisionSightings(feeds=(_StubFeed("tower-1", _frame()),)).sightings(
        _observation((_pose("tower-1", measured_t=0.9),))
    )
    (sync,) = seen
    assert getattr(sync, "status", None) == "synchronised"


def test_the_skew_bound_is_the_callers_to_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It bounds the error this source will post, so it is not a constant."""
    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self) -> GeoPoint:
            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    feeds = (_StubFeed("tower-1", _frame()),)
    poses = (_pose("tower-1", measured_t=0.4),)
    assert VisionSightings(feeds=feeds).sightings(_observation(poses)) == ()
    (found,) = VisionSightings(feeds=feeds, max_skew_s=1.0).sightings(_observation(poses))
    assert found.sync is not None
    assert found.sync.status == "synchronised"
    assert found.sync.max_skew_s == pytest.approx(1.0)


# -- a refusal is recorded, never silent -------------------------------------


def test_a_refused_fix_is_recorded_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping a fix silently is a worse silence than an unqualified one.

    With no record of it, a camera refused every tick is indistinguishable in
    the log from an empty sea — no contact, no counter, no reason — and the two
    sync states that say *do not trust this* could never reach the log at all.
    """
    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self) -> GeoPoint:
            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.refusals() == (), "nothing is refused before the first tick"
    assert source.sightings(_observation((_pose("tower-1", measured_t=0.4),))) == ()
    (refusal,) = source.refusals()
    assert refusal.asset_id == "tower-1"
    assert refusal.t == 1.0
    assert refusal.sync.status == "telemetry_stale"
    assert refusal.sync.skew_s == pytest.approx(0.6)


def test_a_pose_with_no_attitude_is_recorded_as_a_refusal() -> None:
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.sightings(_observation((_pose("tower-1", attitude=False),))) == ()
    (refusal,) = source.refusals()
    assert refusal.sync.status == "attitude_missing"


def test_a_frame_with_no_vessel_is_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The join worked and the answer was "nothing"; counting it buries the real ones."""
    from whiteout.vision import sightings as module

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: None)
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.sightings(_observation((_pose("tower-1", measured_t=0.9),))) == ()
    assert source.refusals() == ()


def test_a_missing_pose_or_frame_is_an_absence_not_a_refusal() -> None:
    source = VisionSightings(feeds=(_StubFeed("tower-1", None),))
    assert source.sightings(_observation((_pose("tower-1"),))) == ()
    assert source.refusals() == ()
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.sightings(_observation(())) == ()
    assert source.refusals() == ()


def test_the_refusals_of_a_tick_replace_the_last_tick_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They describe the most recent call, so a fixed tick cannot read as still broken."""
    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self) -> GeoPoint:
            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    source.sightings(_observation((_pose("tower-1", measured_t=0.4),)))
    assert len(source.refusals()) == 1
    source.sightings(_observation((_pose("tower-1", measured_t=0.9),)))
    assert source.refusals() == ()


def test_a_non_positive_skew_bound_is_refused_at_construction() -> None:
    """Otherwise it fails inside `PoseSync`, once per frame, at tick time."""
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(VisionError):
            VisionSightings(feeds=(), max_skew_s=bad)


def test_a_camera_at_the_water_plane_is_skipped_rather_than_raising() -> None:
    """The detector refuses it; the run must not end because of it.

    This is the failure that hid behind ``relative_alt``: every stationary
    asset reported about zero height, so the towers were discarded silently.
    """
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.sightings(_observation((_pose("tower-1", alt=0.0),))) == ()


# -- what a detection becomes ------------------------------------------------


def test_a_detection_becomes_a_sighting_with_a_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self):
            from whiteout.geo import GeoPoint

            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    (found,) = source.sightings(_observation((_pose("tower-1"),)))
    assert isinstance(found, Sighting)
    assert found.asset_id == "tower-1"
    assert found.lat_deg == pytest.approx(71.9965)
    assert found.t == 1.0, "stamped with the tick, not the frame"


def test_no_detection_is_no_sighting(monkeypatch: pytest.MonkeyPatch) -> None:
    from whiteout.vision import sightings as module

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: None)
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.sightings(_observation((_pose("tower-1"),))) == ()


def test_a_feed_that_has_stopped_draining_is_no_longer_a_sensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead camera goes quiet rather than repeating its last frame.

    ``CameraFeed`` keeps its newest frame for ever, which is right for a feed
    between frames and wrong for one that will never get another. ``_sight``
    re-projects whatever it is handed through the pose of *this* tick, so a
    dead feed on a moving airframe would report a vessel every tick whose
    ground point walks at the aircraft's own speed — inside ``MotionGate``'s
    believable band, defended by ``TrackHold`` against the real vessel, and on
    a judged run POSTed to the scored endpoint.

    The frame is deliberately still there, and a detection is deliberately
    still available: the only thing this asserts is that a stopped feed is
    not asked.
    """
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self):
            from whiteout.geo import GeoPoint

            return GeoPoint(71.9965, -94.8448)

    monkeypatch.setattr(module, "detect_vessel", lambda *a, **k: _Detection())
    observation = _observation((_pose("tower-1"),))

    live = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert len(live.sightings(observation)) == 1, "a draining feed still sights"

    dead = VisionSightings(feeds=(_StubFeed("tower-1", _frame(), draining=False),))
    assert dead.sightings(observation) == ()
    assert dead.refusals() == (), "an absence of input is not a fix thrown away"


def test_a_malformed_frame_does_not_end_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whiteout.vision import sightings as module

    def _boom(*_a: object, **_k: object) -> None:
        raise VisionError("frame is the wrong shape")

    monkeypatch.setattr(module, "detect_vessel", _boom)
    source = VisionSightings(feeds=(_StubFeed("tower-1", _frame()),))
    assert source.sightings(_observation((_pose("tower-1"),))) == ()


def test_the_pose_handed_to_the_detector_carries_all_three_angles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from whiteout.vision import sightings as module

    seen: list[CameraPose] = []

    def _capture(_luma, _camera, pose, **_k):
        seen.append(pose)
        return None

    monkeypatch.setattr(module, "detect_vessel", _capture)
    VisionSightings(feeds=(_StubFeed("tower-1", _frame()),)).sightings(
        _observation((_pose("tower-1"),))
    )
    (camera_pose,) = seen
    assert camera_pose.yaw_deg == pytest.approx(90.0)
    assert camera_pose.pitch_deg == pytest.approx(-10.0)
    assert camera_pose.roll_deg == pytest.approx(0.0)


# -- a stalled camera cannot stall the fleet ---------------------------------


def test_latest_never_blocks_on_a_camera_that_has_stopped() -> None:
    """The control loop must run at tick rate whatever a socket is doing.

    Four cameras read inside the tick would put the fleet at the mercy of
    four sockets — the same failure the tracks poster refuses to have.
    """
    feed = CameraFeed("tower-1", "http://127.0.0.1:1/stream", TOWER)
    feed.open()
    started = time.monotonic()
    for _ in range(50):
        feed.latest()
    elapsed = time.monotonic() - started
    feed.close()
    assert elapsed < 0.2


def test_a_feed_that_dies_records_why_rather_than_raising() -> None:
    """One camera failing must not take the fleet down with it."""
    feed = CameraFeed("tower-1", "http://127.0.0.1:1/stream", TOWER)
    feed.open()
    deadline = time.monotonic() + 5.0
    while feed.error is None and time.monotonic() < deadline:
        time.sleep(0.05)
    feed.close()
    assert feed.error is not None
    assert feed.frames_seen == 0


def test_a_feed_keeps_only_the_newest_frame() -> None:
    feed = CameraFeed("tower-1", "unused", TOWER)
    for seq in range(5):
        with feed._lock:
            feed._latest = _frame(seq)
            feed._frames += 1
    latest = feed.latest()
    assert latest is not None
    assert latest.seq == 4, "a backlog is stale by definition"
    assert feed.frames_seen == 5


def test_sightings_are_independent_across_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One camera seeing nothing must not suppress another that does."""
    from whiteout.geo import GeoPoint
    from whiteout.vision import sightings as module

    class _Detection:
        def to_ground(self):
            return GeoPoint(71.9965, -94.8448)

    calls = {"n": 0}

    def _every_other(*_a: object, **_k: object):
        calls["n"] += 1
        return _Detection() if calls["n"] % 2 == 0 else None

    monkeypatch.setattr(module, "detect_vessel", _every_other)
    feeds = (_StubFeed("tower-1", _frame()), _StubFeed("tower-2", _frame()))
    poses = (_pose("tower-1"), _pose("tower-2"))
    found = VisionSightings(feeds=feeds).sightings(_observation(poses))
    assert [f.asset_id for f in found] == ["tower-2"]


def test_threads_do_not_outlive_a_closed_source() -> None:
    source = VisionSightings(feeds=arena_feeds("127.0.0.1"))
    before = threading.active_count()
    source.open()
    source.close()
    assert threading.active_count() <= before + 1
