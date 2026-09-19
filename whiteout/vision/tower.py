"""Tower servo PWM to camera orientation.

``hackathon/ARENA.md`` §3. ``tower-1`` and ``tower-2`` are stock ArduPilot
**AntennaTrackers**, and they are steered by raw servo output:

.. code-block:: text

    servo set 1 <pwm>    pan  / yaw
    servo set 2 <pwm>    tilt / pitch

That is the whole control surface, and it means the only record of where a
tower camera was pointing when a frame arrived is the pair of PWM values.
This module turns that pair into the yaw and pitch that
:class:`~whiteout.vision.projection.CameraPose` wants, so a tower detection
projects by exactly the same path as an aircraft's.

The map is linear across the servo's pulse range:

.. math::

    \\text{angle} = \\text{angle}_{\\min}
    + \\frac{\\text{pwm} - \\text{pwm}_{\\min}}{\\text{pwm}_{\\max} - \\text{pwm}_{\\min}}
    \\,(\\text{angle}_{\\max} - \\text{angle}_{\\min})

clamped to the pulse range at both ends, which is what the servo itself does.

**The default range is ArduPilot's AntennaTracker default, and it is an
assumption until someone reads the live parameters.** ``YAW_RANGE`` defaults
to 360° and ``PITCH_MIN``/``PITCH_MAX`` to −90°/+90°, over a 1000–2000 µs
pulse; that is what :data:`TOWER_PAN_TILT` encodes, and it puts the servo trim
at 1500 µs on 0° for both axes. If the arena's trackers are configured
differently — and ``ARENA.md`` §8 does not say either way — the fix is to read
those three parameters off the tracker over MAVLink and pass a
:class:`PanTiltCalibration` built from them. **Nothing else in the projection
chain changes**, which is why the calibration is a parameter and not a
constant folded into the arithmetic.

:data:`TOWER_PAN_TILT` also assumes the tracker's pan zero points at true
North. A tracker mounted with a different reference is handled by
``boresight_yaw_deg`` on :func:`tower_camera_pose` rather than by bending the
calibration, so that the two corrections stay separable: one is the servo's
geometry, the other is where the mast was bolted down.
"""

from __future__ import annotations

from dataclasses import dataclass

from whiteout.vision.camera import VisionError
from whiteout.vision.projection import CameraPose

__all__ = [
    "TOWER_PAN_TILT",
    "PanTiltCalibration",
    "ServoRange",
    "pan_tilt_from_pwm",
    "tower_camera_pose",
]


@dataclass(frozen=True, slots=True)
class ServoRange:
    """The linear map from one servo's pulse width to one axis' angle.

    :param pwm_min: pulse width, microseconds, at ``angle_min_deg``.
    :param pwm_max: pulse width, microseconds, at ``angle_max_deg``.
    :param angle_min_deg: the angle commanded by ``pwm_min``.
    :param angle_max_deg: the angle commanded by ``pwm_max``.
    """

    pwm_min: int
    pwm_max: int
    angle_min_deg: float
    angle_max_deg: float

    def __post_init__(self) -> None:
        if self.pwm_max <= self.pwm_min:
            raise VisionError(
                f"servo pulse range must increase, got {self.pwm_min!r}..{self.pwm_max!r}"
            )

    def angle(self, pwm: float) -> float:
        """Return the angle in degrees for ``pwm``, clamping outside the range.

        A pulse outside ``[pwm_min, pwm_max]`` is clamped rather than
        rejected: the servo saturates at its travel limits, so clamping is
        what the hardware does, and a tower scan that overshoots its limit
        should report the limit rather than raise while the fleet is
        searching.
        """
        clamped = min(max(pwm, float(self.pwm_min)), float(self.pwm_max))
        fraction = (clamped - self.pwm_min) / (self.pwm_max - self.pwm_min)
        return self.angle_min_deg + fraction * (self.angle_max_deg - self.angle_min_deg)


@dataclass(frozen=True, slots=True)
class PanTiltCalibration:
    """Both axes of one tracker: servo 1 is pan, servo 2 is tilt."""

    pan: ServoRange
    tilt: ServoRange


#: ArduPilot AntennaTracker defaults — ``YAW_RANGE`` 360°, ``PITCH_MIN``
#: −90°, ``PITCH_MAX`` +90°, over a 1000–2000 µs pulse. See the module
#: docstring: this is an assumption until the live parameters are read.
TOWER_PAN_TILT = PanTiltCalibration(
    pan=ServoRange(1000, 2000, -180.0, 180.0),
    tilt=ServoRange(1000, 2000, -90.0, 90.0),
)


def pan_tilt_from_pwm(
    pan_pwm: float,
    tilt_pwm: float,
    calibration: PanTiltCalibration = TOWER_PAN_TILT,
) -> tuple[float, float]:
    """Return ``(pan_deg, tilt_deg)`` for a pair of servo pulse widths.

    ``pan_deg`` is relative to the tracker's own zero, not to North;
    :func:`tower_camera_pose` adds the mounting offset. ``tilt_deg`` is an
    elevation, so it is already
    :class:`~whiteout.vision.projection.CameraPose`'s pitch: negative points
    the camera down at the water.
    """
    return calibration.pan.angle(pan_pwm), calibration.tilt.angle(tilt_pwm)


def tower_camera_pose(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    pan_pwm: float,
    tilt_pwm: float,
    *,
    calibration: PanTiltCalibration = TOWER_PAN_TILT,
    boresight_yaw_deg: float = 0.0,
) -> CameraPose:
    """Build the pose of a tower camera from its two servo pulse widths.

    :param lat_deg: the tower's latitude — one of the two values
        ``ARENA.md`` §3 sanctions editing in the sim's ``.env``.
    :param lon_deg: the tower's longitude.
    :param alt_m: the camera's height above the water plane.
    :param pan_pwm: ``servo set 1`` pulse width, microseconds.
    :param tilt_pwm: ``servo set 2`` pulse width, microseconds.
    :param calibration: the tracker's servo travel; see :data:`TOWER_PAN_TILT`.
    :param boresight_yaw_deg: bearing of the tracker's pan zero, degrees
        clockwise from true North.

    Roll is zero: the tracker is a two-axis mount on a fixed mast, so the
    camera's horizontal axis stays horizontal. The yaw is wrapped into
    ``[0, 360)`` so that a pose printed in a diagnostic reads as a bearing.
    """
    pan_deg, tilt_deg = pan_tilt_from_pwm(pan_pwm, tilt_pwm, calibration)
    return CameraPose(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        yaw_deg=(boresight_yaw_deg + pan_deg) % 360.0,
        pitch_deg=tilt_deg,
        roll_deg=0.0,
    )
