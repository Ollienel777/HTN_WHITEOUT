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
assumption until someone reads the live parameters.** Four defaults in
ArduPilot master combine to give it, and each was read out of the source
rather than recalled:

* ``AntennaTracker/config.h`` — ``YAW_RANGE_DEFAULT 360``,
  ``PITCH_MIN_DEFAULT -90``, ``PITCH_MAX_DEFAULT 90``.
* ``AntennaTracker/servos.cpp`` — ``init_servos`` calls
  ``SRV_Channels::set_angle(k_tracker_yaw, g.yaw_range * 100 / 2)`` and
  ``set_angle(k_tracker_pitch, (-g.pitch_min + g.pitch_max) * 100 / 2)``, so
  each axis is an *angle* channel spanning ±180° and ±90° respectively.
* ``libraries/SRV_Channel/SRV_Channel.cpp`` — ``SERVOn_MIN`` defaults to
  **1100**, ``SERVOn_MAX`` to **1900**, ``SERVOn_TRIM`` to **1500**.
* ``SRV_Channel::pwm_from_angle`` interpolates the positive half between
  ``TRIM`` and ``MAX`` and the negative half between ``MIN`` and ``TRIM``.
  With the default 1100/1500/1900 those two halves have equal width, so the
  map is a single straight line from 1100 to 1900.

So the stock travel is **1100–1900 µs**, not 1000–2000, and that is what
:data:`TOWER_PAN_TILT` encodes. Trim still lands on 0° for both axes.
Reading it as 1000–2000 would put the full ±180° across a 20% wider pulse
span and under-read every angle by a ninth — 36° of pan at the travel limit,
and −18° instead of −22.5° at a tilt pulse of 1400 µs.

**It remains an assumption about the arena**, because ``ARENA.md`` §8 does
not say whether these trackers are on ArduPilot's defaults, and because
``servo set N <pwm>`` is ``MAV_CMD_DO_SET_SERVO``, which writes the raw pulse
to the output and bypasses ``pwm_from_angle`` altogether — so what the pulse
means physically is the sim's servo model, which these parameters describe
but do not control. The fix either way is to read ``YAW_RANGE``,
``PITCH_MIN``, ``PITCH_MAX``, ``SERVO1_MIN``/``MAX`` and ``SERVO2_MIN``/``MAX``
off the tracker over MAVLink and pass a :class:`PanTiltCalibration` built from
them. **Nothing else in the projection chain changes**, which is why the
calibration is a parameter and not a constant folded into the arithmetic.

:data:`TOWER_PAN_TILT` also assumes the tracker's pan zero points at true
North. A tracker mounted with a different reference is handled by
``boresight_yaw_deg`` on :func:`tower_camera_pose` rather than by bending the
calibration, so that the two corrections stay separable: one is the servo's
geometry, the other is where the mast was bolted down.
"""

from __future__ import annotations

import math
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
        for field_name, value in (
            ("angle_min_deg", self.angle_min_deg),
            ("angle_max_deg", self.angle_max_deg),
        ):
            if not math.isfinite(value):
                raise VisionError(f"{field_name} must be finite, got {value!r}")

    def angle(self, pwm: float) -> float:
        """Return the angle in degrees for ``pwm``, clamping outside the range.

        A pulse outside ``[pwm_min, pwm_max]`` is clamped rather than
        rejected: the servo saturates at its travel limits, so clamping is
        what the hardware does, and a tower scan that overshoots its limit
        should report the limit rather than raise while the fleet is
        searching.

        A pulse that is **not finite** is refused instead of clamped.
        ``min(max(nan, lo), hi)`` is ``nan`` in Python, so a dropped MAVLink
        field would otherwise sail through the clamp, through the projection
        and into a ``POST /api/tracks`` body — where a NaN is not even valid
        JSON. Refusing here means the detection is discarded at the point the
        bad value entered.
        """
        if not math.isfinite(pwm):
            raise VisionError(f"servo pulse width must be finite, got {pwm!r}")
        clamped = min(max(pwm, float(self.pwm_min)), float(self.pwm_max))
        fraction = (clamped - self.pwm_min) / (self.pwm_max - self.pwm_min)
        return self.angle_min_deg + fraction * (self.angle_max_deg - self.angle_min_deg)


@dataclass(frozen=True, slots=True)
class PanTiltCalibration:
    """Both axes of one tracker: servo 1 is pan, servo 2 is tilt."""

    pan: ServoRange
    tilt: ServoRange


#: ArduPilot AntennaTracker defaults — ``YAW_RANGE`` 360°, ``PITCH_MIN``
#: −90°, ``PITCH_MAX`` +90°, over the **1100–1900 µs** pulse that
#: ``SERVOn_MIN``/``SERVOn_MAX`` default to. The module docstring cites each
#: source file; this is still an assumption about *this arena* until the live
#: parameters are read off the trackers.
TOWER_PAN_TILT = PanTiltCalibration(
    pan=ServoRange(1100, 1900, -180.0, 180.0),
    tilt=ServoRange(1100, 1900, -90.0, 90.0),
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
