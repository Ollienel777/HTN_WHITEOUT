"""Turning camera frames into sightings the run loop can use.

The join between :mod:`whiteout.vision` and :mod:`whiteout.coordinate`. Each
tick it takes the newest frame from each camera, pairs it with that asset's
pose from the observation, runs the detector, projects the pixel to the
water, and hands back a :class:`~whiteout.tracks.maintain.Sighting`.

Why the frames are pulled from a thread and not from the tick
--------------------------------------------------------------

An MJPEG stream is a socket that yields frames at the camera's rate, not a
thing you can ask for "the current frame". Reading one inside the tick would
put the control loop at the mercy of four sockets: a camera that stalls would
stall the fleet, which is the failure the tracks poster already refuses to
have (#64).

So each camera is drained by its own thread that keeps **only the newest
frame**, and the tick takes whatever is there. A camera that has produced
nothing yet contributes nothing, exactly as an asset with no pose does. The
age of the frame it does have is not tracked, and that is a real limitation
rather than an oversight — see below.

The pose is joined by asset, not by time
-----------------------------------------

The frame carries no telemetry, so its pose comes from the tick's own
observation. At 3 m/s and a tick of about a second that is close enough for
the vessel; it is **not** close enough for a fast-moving camera, and the
fixed-wing is the asset it is worst for. A frame a quarter-second stale on a
25 m/s aircraft is a 6 m position error and, far worse, an attitude error if
it is turning — which walks the projected point across the water. Joining
properly needs the frame's own timestamp against `Pose.measured_t`, and the
arena adapter reports `measured_t` as ``None`` because the clock offset has
never been established. That is the honest state of it.

What it refuses to do
----------------------

- **No pitch or roll, no sighting.** Projection takes three angles; a `Pose`
  with `None` for two of them cannot be turned into a `CameraPose`, and
  guessing "level" would put the point somewhere confidently wrong.
- **No detection, no sighting.** The detector already returns ``None`` rather
  than a guess, and every candidate it emits has been through the projection,
  so a sighting from here always has a lat/lon behind it.
- **Nothing is posted from here.** This produces sightings; the hold decides
  what reaches the tracks API, and it posts a fix only when it is new (#67).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from whiteout.tracks.maintain import Sighting
from whiteout.types import Pose, WorldObservation
from whiteout.vision.camera import CAMERAS, CameraModel, VisionError
from whiteout.vision.detect import DEFAULT_PARAMS, DetectorParams, detect_vessel
from whiteout.vision.imagery import LumaFrame, MjpegFrames
from whiteout.vision.projection import CameraPose

__all__ = [
    "ARENA_CAMERA_PORTS",
    "CameraFeed",
    "VisionSightings",
    "arena_feeds",
]

#: ``ARENA.md`` §4. The camera port each asset streams on.
ARENA_CAMERA_PORTS: dict[str, int] = {
    "quadcopter": 8600,
    "fixed-wing": 8610,
    "tower-1": 8630,
    "tower-2": 8640,
}

#: Which published camera each asset carries. Both towers share one.
ARENA_CAMERA_NAMES: dict[str, str] = {
    "quadcopter": "quadcopter",
    "fixed-wing": "fixed-wing",
    "tower-1": "tower",
    "tower-2": "tower",
}


class CameraFeed:
    """One camera, drained by a thread, holding only its newest frame.

    Start it with :meth:`open` and stop it with :meth:`close`; it is also a
    context manager. :meth:`latest` never blocks and returns ``None`` until
    the first frame has arrived.
    """

    def __init__(self, asset_id: str, source: str, camera: CameraModel) -> None:
        self._asset_id = asset_id
        self._source = source
        self._camera = camera
        self._lock = threading.Lock()
        self._latest: LumaFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames = 0
        self._error: str | None = None

    @property
    def asset_id(self) -> str:
        """Which asset this camera belongs to."""
        return self._asset_id

    @property
    def frames_seen(self) -> int:
        """How many frames have arrived. Zero means the feed never started."""
        with self._lock:
            return self._frames

    @property
    def error(self) -> str | None:
        """Why the feed stopped, if it did.

        A stream that dies is recorded rather than raised: one camera failing
        must not take the fleet down, so the run continues with three sensors
        and this says which one it lost.
        """
        with self._lock:
            return self._error

    def open(self) -> None:
        """Start draining the stream."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{self._asset_id}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        source = MjpegFrames(self._source, self._asset_id, self._camera)
        try:
            for frame in source.frames():
                if self._stop.is_set():
                    return
                with self._lock:
                    self._latest = frame
                    self._frames += 1
        except (VisionError, OSError) as error:
            with self._lock:
                self._error = str(error)

    def latest(self) -> LumaFrame | None:
        """The newest frame, or ``None`` if none has arrived. Never blocks."""
        with self._lock:
            return self._latest

    def close(self) -> None:
        """Stop draining. The socket closes with the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    def __enter__(self) -> CameraFeed:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def arena_feeds(host: str, assets: tuple[str, ...] | None = None) -> tuple[CameraFeed, ...]:
    """Feeds for the arena's cameras, by asset id.

    ``host`` is the arena address and is configuration, never a constant
    (#63). An asset with no camera in :data:`ARENA_CAMERA_PORTS` is skipped.
    """
    wanted = assets if assets is not None else tuple(ARENA_CAMERA_PORTS)
    feeds = []
    for asset_id in wanted:
        port = ARENA_CAMERA_PORTS.get(asset_id)
        if port is None:
            continue
        camera = CAMERAS[ARENA_CAMERA_NAMES[asset_id]]
        feeds.append(CameraFeed(asset_id, f"http://{host}:{port}/stream", camera))
    return tuple(feeds)


@dataclass
class VisionSightings:
    """A :class:`~whiteout.coordinate.SightingSource` backed by real cameras.

    :param feeds: one per asset, already opened or opened by :meth:`open`.
    :param ground_alt_m: the water plane's altitude in the poses' datum.
        Issue #76 has not settled which datum the arena reports, so this is a
        parameter with the detector's own meaning and default.
    :param params: detector thresholds.
    """

    feeds: tuple[CameraFeed, ...]
    ground_alt_m: float = 0.0
    params: DetectorParams = DEFAULT_PARAMS

    def open(self) -> None:
        """Start every feed."""
        for feed in self.feeds:
            feed.open()

    def close(self) -> None:
        """Stop every feed."""
        for feed in self.feeds:
            feed.close()

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        """Every asset that can see the vessel right now, and where it says it is."""
        poses = {pose.asset_id: pose for pose in observation.poses}
        found: list[Sighting] = []
        for feed in self.feeds:
            pose = poses.get(feed.asset_id)
            frame = feed.latest()
            if pose is None or frame is None:
                continue
            sighting = self._sight(feed.asset_id, pose, frame, observation.t)
            if sighting is not None:
                found.append(sighting)
        return tuple(found)

    def _sight(self, asset_id: str, pose: Pose, frame: LumaFrame, t: float) -> Sighting | None:
        if pose.pitch is None or pose.roll is None:
            # Two of the three angles are missing. Guessing "level" would put
            # the projected point somewhere confidently wrong, and ARENA.md
            # §5 scores accuracy.
            return None
        camera_pose = CameraPose(
            lat_deg=pose.lat,
            lon_deg=pose.lon,
            alt_m=pose.z,
            yaw_deg=pose.heading,
            pitch_deg=pose.pitch,
            roll_deg=pose.roll,
        )
        try:
            detection = detect_vessel(
                frame.luma,
                frame.camera,
                camera_pose,
                asset_id=asset_id,
                seq=frame.seq,
                t=t,
                ground_alt_m=self.ground_alt_m,
                params=self.params,
            )
        except VisionError:
            # A malformed frame, or a camera below the water plane. Neither is
            # a reason to stop the run; the next frame may be fine.
            return None
        if detection is None:
            return None
        ground = detection.to_ground()
        return Sighting(
            t=t,
            lat_deg=ground.lat_deg,
            lon_deg=ground.lon_deg,
            asset_id=asset_id,
        )
