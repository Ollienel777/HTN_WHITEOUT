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

**What the join can now say about itself.** Every sighting from here carries a
:class:`~whiteout.types.PoseSync`, so the pair's skew is *recorded* rather than
left to this docstring: how old the pose was at the tick the frame was consumed
in, against ``max_skew_s``. It cannot close the gap above — the frame's own age
inside its feed is still untracked, so the recorded skew is a lower bound, and
``PoseSync`` says so where the verdict is defined rather than only here. What it
does end is the silence: a fix assembled out of two instants is no longer
indistinguishable, in the log or in the viewer, from one that was not.

What it refuses to do
----------------------

- **No pitch or roll, no sighting.** Projection takes three angles; a `Pose`
  with `None` for two of them cannot be turned into a `CameraPose`, and
  guessing "level" would put the point somewhere confidently wrong. This is
  ``attitude_missing``, and it was already the rule before the status had a
  name.
- **A measured skew past the bound, no sighting.** ``telemetry_stale``: a
  lat/lon this one posts is scored, and past ``max_skew_s`` the pose is known
  to have moved further under the frame than the operator said they would
  stand behind. A skew that is merely *unknown* — ``telemetry_missing``, which
  is every arena pose today — is **carried and labelled instead**, because
  refusing it would post nothing at all in the arena we are flying in and
  would call that caution.

**Every refusal is recorded, because a silent refusal is the worse silence.**
Dropping a fix and saying nothing leaves no contact, no counter and no reason,
which in the episode log is indistinguishable from an empty sea — and the whole
argument for recording synchronisation is that an unqualified fix must not look
like a qualified one. :meth:`VisionSightings.refusals` hands the tick's
refusals to the coordinator as
:class:`~whiteout.types.SightingRefusal`\\ s, they are written on the episode
record, and the viewer says how many and why. Without that, the two states this
module refuses could never reach the log at all, and the states that say *do not
trust this* would be exactly the ones nobody could see.
- **No detection, no sighting.** The detector already returns ``None`` rather
  than a guess, and every candidate it emits has been through the projection,
  so a sighting from here always has a lat/lon behind it.
- **Nothing is posted from here.** This produces sightings; the hold decides
  what reaches the tracks API, and it posts a fix only when it is new (#67).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from whiteout.tracks.maintain import Sighting
from whiteout.types import (
    DEFAULT_MAX_SKEW_S,
    Pose,
    PoseSync,
    SightingRefusal,
    WorldObservation,
)
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
    :param max_skew_s: how far the pose joined to a frame may have been
        measured from the tick that consumed it before the sighting is refused
        (:data:`~whiteout.types.DEFAULT_MAX_SKEW_S`). It bounds the position
        error this source is willing to post, so it is the operator's number
        and a parameter rather than a constant. It is checked here, at
        construction: a non-positive bound would otherwise fail inside
        `PoseSync` once per frame, at tick time, in a loop with no handler over
        it -- the error landing four layers from the line that caused it.

        The default is set for the **fastest** thing in the fleet, the
        fixed-wing at 22 m/s, and is therefore conservative for the towers,
        which do not move at all and for which a stale fix costs nothing. One
        number wearing an airframe's argument is a known limitation, not a
        finding of this design: the bound a rover or a tower deserves is a
        different number, and a per-class bound is the follow-up.
    """

    feeds: tuple[CameraFeed, ...]
    ground_alt_m: float = 0.0
    params: DetectorParams = DEFAULT_PARAMS
    max_skew_s: float = DEFAULT_MAX_SKEW_S

    #: The refusals of the most recent `sightings` call. Captured there rather
    #: than recomputed on demand: a feed keeps only its newest frame, so asking
    #: again a moment later can pair a *different* frame with the same pose and
    #: report a refusal that never happened.
    _refused: tuple[SightingRefusal, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_skew_s) and self.max_skew_s > 0.0):
            raise VisionError(f"max_skew_s must be positive and finite, got {self.max_skew_s!r}")

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
        refused: list[SightingRefusal] = []
        for feed in self.feeds:
            pose = poses.get(feed.asset_id)
            frame = feed.latest()
            if pose is None or frame is None:
                # Not a refusal. An asset with no pose and a camera with no
                # frame yet are absences of input, not fixes thrown away, and
                # counting them would bury the ones that were.
                continue
            sighting = self._sight(feed.asset_id, pose, frame, observation.t)
            if sighting is not None:
                found.append(sighting)
                continue
            reason = self._refusal(feed.asset_id, pose, observation.t)
            if reason is not None:
                refused.append(reason)
        self._refused = tuple(refused)
        return tuple(found)

    def refusals(self) -> tuple[SightingRefusal, ...]:
        """What the last :meth:`sightings` call declined to use, and why.

        The coordinator writes these onto the episode record, which is the only
        artifact anything downstream reads: without them, the two states this
        source refuses on could never appear in a log, and a camera that had
        gone dark for a reason we knew would look exactly like an empty sea.

        Empty when nothing was refused, and empty before the first tick.
        """
        return self._refused

    def _refusal(self, asset_id: str, pose: Pose, t: float) -> SightingRefusal | None:
        """The record for a frame `_sight` dropped, or `None` if it dropped it for
        a reason that is not about synchronisation.

        A frame with no vessel in it, or one the detector could not read, is not
        a refusal: the join worked and the answer was "nothing". Only the two
        sync verdicts `_sight` acts on are recorded, which is why this asks
        `PoseSync` again rather than trusting that a `None` sighting means skew.
        """
        sync = PoseSync.for_frame(t, pose, max_skew_s=self.max_skew_s)
        if sync.status in ("attitude_missing", "telemetry_stale"):
            return SightingRefusal(asset_id=asset_id, t=t, sync=sync)
        return None

    def _sight(self, asset_id: str, pose: Pose, frame: LumaFrame, t: float) -> Sighting | None:
        # `t` and not `frame.t`: the frame's own stamp is a wall clock (live) or
        # an index over a frame rate (a recording), and neither is the clock
        # `measured_t` is in. Subtracting one from the other would produce a
        # number that looks like a skew and is not one -- the mistake
        # `Pose.measured_t` names one level down. The tick that consumed the
        # frame is in the right clock, and `PoseSync` says what that costs.
        sync = PoseSync.for_frame(t, pose, max_skew_s=self.max_skew_s)
        if pose.pitch is None or pose.roll is None:
            # `sync.status == "attitude_missing"`, spelt on the fields it is
            # derived from so that the two angles narrow for the `CameraPose`
            # below. Two of the three angles are missing; guessing "level"
            # would put the projected point somewhere confidently wrong, and
            # ARENA.md §5 scores accuracy.
            return None
        if sync.status == "telemetry_stale":
            # Known bad, not merely unknown: the pose was measured further from
            # this frame than `max_skew_s` allows, so the lat/lon would be off
            # by whatever the asset covered in between. `telemetry_missing`
            # deliberately does *not* land here -- see the module docstring.
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
                sync=sync,
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
            sync=sync,
        )
