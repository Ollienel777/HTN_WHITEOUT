"""AntennaTracker servo PWM to a camera orientation.

The towers are steered by ``servo set 1 <pwm>`` and ``servo set 2 <pwm>``
(``ARENA.md`` §3), so the PWM pair is the only record of where a tower camera
pointed. The arithmetic is one linear map per axis and every expected value
below is worked in its own docstring.

The numbers assume the default AntennaTracker travel — pan ±180°, tilt ±90°,
across 1000–2000 µs — which is what :data:`~whiteout.vision.tower.TOWER_PAN_TILT`
encodes and what the module docstring flags as an assumption until the live
parameters are read. The last test here is the one that matters if that
assumption turns out wrong: a different calibration is a different argument
and nothing else moves.
"""

from __future__ import annotations

import pytest

from whiteout.vision.camera import VisionError
from whiteout.vision.tower import (
    TOWER_PAN_TILT,
    PanTiltCalibration,
    ServoRange,
    pan_tilt_from_pwm,
    tower_camera_pose,
)


def test_servo_trim_is_the_centre_of_travel() -> None:
    """1500 µs is dead centre of 1000–2000, so both axes read zero.

    ``pan  = -180 + (1500 - 1000)/1000 × (180 - -180) = -180 + 0.5 × 360 = 0``
    ``tilt = -90  + (1500 - 1000)/1000 × (90 - -90)  = -90  + 0.5 × 180 = 0``
    """
    pan_deg, tilt_deg = pan_tilt_from_pwm(1500, 1500)
    assert pan_deg == pytest.approx(0.0, abs=1e-12)
    assert tilt_deg == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("pwm", "expected_pan", "expected_tilt"),
    [
        # fraction 0.00 → pan -180 + 0 = -180 ; tilt -90 + 0 = -90
        (1000, -180.0, -90.0),
        # fraction 0.25 → pan -180 + 90  = -90 ; tilt -90 + 45 = -45
        (1250, -90.0, -45.0),
        # fraction 0.60 → pan -180 + 216 = +36 ; tilt -90 + 108 = +18
        (1600, 36.0, 18.0),
        # fraction 0.75 → pan -180 + 270 = +90 ; tilt -90 + 135 = +45
        (1750, 90.0, 45.0),
        # fraction 1.00 → pan -180 + 360 = +180 ; tilt -90 + 180 = +90
        (2000, 180.0, 90.0),
    ],
)
def test_pwm_maps_linearly_across_the_travel(
    pwm: int, expected_pan: float, expected_tilt: float
) -> None:
    """``angle_min + (pwm - pwm_min)/(pwm_max - pwm_min) × (angle_max - angle_min)``.

    Each case's fraction and arithmetic is in the comment beside it. The
    endpoints are included deliberately: an off-by-one in the denominator, or
    a map built on the servo's 1100–1900 µs "usable" range instead of its
    1000–2000 µs full range, fails at 1000 and 2000 first.
    """
    pan_deg, tilt_deg = pan_tilt_from_pwm(pwm, pwm)
    assert pan_deg == pytest.approx(expected_pan, abs=1e-12)
    assert tilt_deg == pytest.approx(expected_tilt, abs=1e-12)


def test_a_pulse_outside_the_travel_is_clamped_not_extrapolated() -> None:
    """The servo saturates at its limits, so the model does too.

    Extrapolating 900 µs to −216° would hand the projection a bearing the
    tracker cannot reach, and the resulting fix would be confidently wrong
    rather than merely imprecise.
    """
    assert pan_tilt_from_pwm(900, 2400) == (
        pytest.approx(-180.0, abs=1e-12),
        pytest.approx(90.0, abs=1e-12),
    )


def test_tower_pose_is_yaw_pitch_and_no_roll() -> None:
    """Pan becomes yaw, tilt becomes pitch, and a mast does not roll.

    At 1600 µs pan the angle is +36° (above), and with the tracker's zero on
    true North that is a bearing of 36°. At 1250 µs tilt the angle is −45°,
    which is already :class:`~whiteout.vision.projection.CameraPose`'s pitch
    convention: negative points down at the water.
    """
    pose = tower_camera_pose(71.99, -94.84, 42.0, pan_pwm=1600, tilt_pwm=1250)
    assert (pose.lat_deg, pose.lon_deg, pose.alt_m) == (71.99, -94.84, 42.0)
    assert pose.yaw_deg == pytest.approx(36.0, abs=1e-12)
    assert pose.pitch_deg == pytest.approx(-45.0, abs=1e-12)
    assert pose.roll_deg == 0.0


def test_the_mount_bearing_offsets_the_yaw_and_wraps() -> None:
    """``boresight_yaw_deg`` is where the tracker's pan zero points.

    Pan +90° off a zero at 300° is ``300 + 90 = 390``, which wraps to 30° so
    that the pose reads as a bearing. Pan −180° off the same zero is
    ``300 - 180 = 120``.
    """
    east_of_north = tower_camera_pose(
        72.0, -95.0, 30.0, pan_pwm=1750, tilt_pwm=1500, boresight_yaw_deg=300.0
    )
    assert east_of_north.yaw_deg == pytest.approx(30.0, abs=1e-12)

    behind = tower_camera_pose(
        72.0, -95.0, 30.0, pan_pwm=1000, tilt_pwm=1500, boresight_yaw_deg=300.0
    )
    assert behind.yaw_deg == pytest.approx(120.0, abs=1e-12)


def test_a_different_calibration_is_the_only_thing_that_has_to_change() -> None:
    """A tracker configured ``YAW_RANGE 180`` and ``PITCH_MIN -45`` still works.

    ``pan  = -90 + (1750 - 1000)/1000 × 180 = -90 + 135 = +45``
    ``tilt = -45 + (1750 - 1000)/1000 × 90  = -45 + 67.5 = +22.5``

    This is the escape hatch the module docstring names: if the arena's
    trackers are not on ArduPilot's defaults, the correction is one argument
    and the projection chain is untouched.
    """
    narrow = PanTiltCalibration(
        pan=ServoRange(1000, 2000, -90.0, 90.0),
        tilt=ServoRange(1000, 2000, -45.0, 45.0),
    )
    pan_deg, tilt_deg = pan_tilt_from_pwm(1750, 1750, narrow)
    assert pan_deg == pytest.approx(45.0, abs=1e-12)
    assert tilt_deg == pytest.approx(22.5, abs=1e-12)

    pose = tower_camera_pose(72.0, -95.0, 30.0, 1750, 1750, calibration=narrow)
    assert pose.yaw_deg == pytest.approx(45.0, abs=1e-12)
    assert pose.pitch_deg == pytest.approx(22.5, abs=1e-12)


def test_the_default_calibration_is_ardupilots_default_travel() -> None:
    """Pinned, so that changing it is a deliberate edit and a failing test."""
    assert TOWER_PAN_TILT.pan == ServoRange(1000, 2000, -180.0, 180.0)
    assert TOWER_PAN_TILT.tilt == ServoRange(1000, 2000, -90.0, 90.0)


def test_a_reversed_pulse_range_is_refused() -> None:
    """A range that does not increase has no linear map."""
    with pytest.raises(VisionError, match="must increase"):
        ServoRange(2000, 1000, -90.0, 90.0)
