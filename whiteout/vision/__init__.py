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

The detector itself is not here: that needs the arena's imagery, and it has
its own ticket. What is here is everything around it, which needs none.
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
from whiteout.vision.frames import (
    DEFAULT_FPS,
    Frame,
    FrameError,
    live_frames,
    read_mjpeg,
    recorded_frames,
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
from whiteout.vision.tower import (
    TOWER_PAN_TILT,
    PanTiltCalibration,
    ServoRange,
    pan_tilt_from_pwm,
    tower_camera_pose,
)

__all__ = [
    "CAMERAS",
    "DEFAULT_FPS",
    "EARTH_MEAN_RADIUS_M",
    "FIXED_WING_CAMERA",
    "MAX_FLAT_PLANE_RANGE_ERROR",
    "QUADCOPTER_CAMERA",
    "TOWER_CAMERA",
    "TOWER_PAN_TILT",
    "WGS84_A",
    "WGS84_E2",
    "WGS84_F",
    "CameraModel",
    "CameraPose",
    "Frame",
    "FrameError",
    "GeoPoint",
    "PanTiltCalibration",
    "ProjectionError",
    "ServoRange",
    "Vec3",
    "VisionError",
    "camera_basis",
    "enu_to_geodetic",
    "horizon_range_m",
    "live_frames",
    "max_flat_plane_range_m",
    "pan_tilt_from_pwm",
    "pixel_ray_enu",
    "project_pixel_to_ground",
    "read_mjpeg",
    "recorded_frames",
    "tower_camera_pose",
]
