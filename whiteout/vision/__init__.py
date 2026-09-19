"""Vision: frames in, a coordinate out.

``hackathon/ARENA.md`` §4. Detection in this arena is a computer-vision
problem — every asset carries a camera, and finding the boat means finding it
in a video frame — and a detection is worth nothing until it is a lat/lon.
This package is the half of that which is **fully determined by published
quantities**: the camera intrinsics, the pose, and the fact that the target is
confined to water.

| module | what it holds |
|---|---|
| :mod:`~whiteout.vision.camera` | the published fields of view, as a pinhole model |
| :mod:`~whiteout.vision.projection` | pixel → ENU ray → water plane → lat/lon |
| :mod:`~whiteout.vision.tower` | AntennaTracker servo PWM → camera orientation |
| :mod:`~whiteout.vision.frames` | MJPEG from a live camera or a recording of one |
| :mod:`~whiteout.vision.imagery` | frames as *pixels*: the real source, and the fake |
| :mod:`~whiteout.vision.detect` | a pixel, a confidence and the pose it was seen from |
| :mod:`~whiteout.vision.scene` | a synthetic channel, to measure the detector against |

The detector (issue #66) is here now, and so is the thing it is measured
against. **No arena imagery exists in this repository** — there is no route to
the arena from the machine this was built on — so
:mod:`~whiteout.vision.scene` renders the channel issue #66 describes and
``scripts/score_detector.py`` scores against that. Every rate in the pull
request is a rate on those frames. Measuring on real ones is a human action
and is outstanding; :mod:`~whiteout.vision.imagery` is shaped so that it is
running the script at a directory.
"""

from __future__ import annotations

from whiteout.vision.camera import (
    CAMERAS,
    FIXED_WING_CAMERA,
    QUADCOPTER_CAMERA,
    TOWER_CAMERA,
    CameraModel,
    VisionError,
)
from whiteout.vision.detect import (
    DEFAULT_PARAMS,
    DetectorParams,
    VesselDetection,
    detect_vessel,
)
from whiteout.vision.frames import (
    DEFAULT_FPS,
    Frame,
    FrameError,
    live_frames,
    read_mjpeg,
    recorded_frames,
)
from whiteout.vision.imagery import (
    DEFAULT_FIXTURE_DIR,
    FixtureFrames,
    FrameSource,
    Label,
    LumaFrame,
    MjpegFrames,
    decode_jpeg_luma,
    frame_source_from_env,
    read_pgm,
    write_fixtures,
    write_pgm,
)
from whiteout.vision.projection import (
    EARTH_MEAN_RADIUS_M,
    MAX_FLAT_PLANE_RANGE_ERROR,
    WGS84_A,
    WGS84_E2,
    WGS84_F,
    CameraPose,
    GeoPoint,
    ProjectionError,
    Vec3,
    camera_basis,
    enu_to_geodetic,
    horizon_range_m,
    max_flat_plane_range_m,
    pixel_ray_enu,
    project_pixel_to_ground,
)
from whiteout.vision.scene import (
    CLEAR,
    FOGGY,
    HEAVY_FOG,
    RenderedScene,
    SceneParams,
    at_fog,
    render_scene,
)
from whiteout.vision.tower import (
    TOWER_PAN_TILT,
    PanTiltCalibration,
    ServoRange,
    pan_tilt_from_pwm,
    tower_camera_pose,
)

__all__ = [
    "CAMERAS",
    "CLEAR",
    "DEFAULT_FIXTURE_DIR",
    "DEFAULT_PARAMS",
    "DEFAULT_FPS",
    "EARTH_MEAN_RADIUS_M",
    "FIXED_WING_CAMERA",
    "FOGGY",
    "HEAVY_FOG",
    "MAX_FLAT_PLANE_RANGE_ERROR",
    "QUADCOPTER_CAMERA",
    "TOWER_CAMERA",
    "TOWER_PAN_TILT",
    "WGS84_A",
    "WGS84_E2",
    "WGS84_F",
    "CameraModel",
    "CameraPose",
    "DetectorParams",
    "FixtureFrames",
    "Frame",
    "FrameError",
    "FrameSource",
    "GeoPoint",
    "Label",
    "LumaFrame",
    "MjpegFrames",
    "PanTiltCalibration",
    "ProjectionError",
    "RenderedScene",
    "SceneParams",
    "ServoRange",
    "Vec3",
    "VesselDetection",
    "VisionError",
    "at_fog",
    "camera_basis",
    "decode_jpeg_luma",
    "detect_vessel",
    "enu_to_geodetic",
    "frame_source_from_env",
    "horizon_range_m",
    "live_frames",
    "max_flat_plane_range_m",
    "pan_tilt_from_pwm",
    "pixel_ray_enu",
    "project_pixel_to_ground",
    "read_mjpeg",
    "read_pgm",
    "recorded_frames",
    "render_scene",
    "tower_camera_pose",
    "write_fixtures",
    "write_pgm",
]
