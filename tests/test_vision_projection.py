"""Pixel-to-world projection, checked against arithmetic done by hand.

Issue #65's acceptance is explicit that the known-pose cases are *checked by
hand and the hand-working is in the test*, and this file takes that literally.
Every expected number below was derived with a calculator from the formulas in
``whiteout/vision/projection.py``'s docstrings, step by step, and the steps are
written out in the test that uses them. **None of them came from running the
code.** A geometry test whose expected value is whatever the implementation
printed asserts only that the implementation is deterministic.

The constants every case shares, from WGS-84:

.. code-block:: text

    a  = 6378137.0
    f  = 1 / 298.257223563              = 0.003352810664747...
    e² = f (2 - f)                      = 0.006694379990141...

and the two radii of curvature, evaluated at the *camera's* latitude:

.. code-block:: text

    N(φ) = a / sqrt(1 - e² sin²φ)
    M(φ) = a (1 - e²) / (1 - e² sin²φ)^(3/2)

At φ = 72° exactly, used by two of the cases below:

.. code-block:: text

    sin 72°      = 0.9510565163      sin²    = 0.9045084972
    e² sin²φ     = 0.0060551236      1 - …   = 0.9939448764
    sqrt(…)      = 0.9969678412      (…)^1.5 = 0.9909310818
    N            = 6378137 / 0.9969678412  = 6397535.343 m
    M            = 6335439.33 / 0.9909310818 = 6393420.763 m
    cos 72°      = 0.3090169944
    N cos φ      = 1976947.143 m

``a (1 - e²) = 6378137 × 0.99330562 = 6335439.33``.

Tolerances. Positions are asserted to **1e-8 degrees**, about a millimetre,
which is inside the precision of the longhand above and some eight orders of
magnitude tighter than any plausible error in the geometry: a wrong axis
convention, a dropped roll or a confused focal length all move the answer by
metres or kilometres, never by millimetres.
"""

from __future__ import annotations

import math

import pytest

from whiteout.vision.camera import (
    CAMERAS,
    FIXED_WING_CAMERA,
    QUADCOPTER_CAMERA,
    TOWER_CAMERA,
    CameraModel,
    VisionError,
)
from whiteout.vision.projection import (
    EARTH_MEAN_RADIUS_M,
    MAX_FLAT_PLANE_RANGE_ERROR,
    WGS84_E2,
    WGS84_F,
    CameraPose,
    ProjectionError,
    camera_basis,
    enu_to_geodetic,
    horizon_range_m,
    max_flat_plane_range_m,
    pixel_ray_enu,
    project_pixel_to_ground,
)
from whiteout.vision.tower import tower_camera_pose

#: A millimetre of latitude, near enough. See the module docstring.
DEGREE_TOL = 1e-8

#: A tenth of a millimetre, for the metric intermediates.
METRE_TOL = 1e-4


# --------------------------------------------------------------------------
# The intrinsics, and the identity every hand-worked case leans on
# --------------------------------------------------------------------------


def test_wgs84_constants_are_the_published_ellipsoid() -> None:
    """``e² = f (2 - f)`` with ``f = 1/298.257223563``, by hand.

    ``1 / 298.257223563 = 0.0033528106647474805``
    ``2 - f            = 1.9966471893352525``
    ``f (2 - f)        = 0.0066943799901413``
    """
    assert WGS84_F == pytest.approx(0.0033528106647474805, abs=1e-18)
    assert WGS84_E2 == pytest.approx(0.0066943799901413, abs=1e-15)


@pytest.mark.parametrize("camera", [QUADCOPTER_CAMERA, FIXED_WING_CAMERA, TOWER_CAMERA])
def test_frame_edges_sit_at_exactly_half_the_published_field_of_view(
    camera: CameraModel,
) -> None:
    """The edge of the frame is HFOV/2 off the boresight, and VFOV/2.

    This is the definition of ``fx = (W/2) / tan(HFOV/2)`` read backwards:
    at ``px = W`` the normalised coordinate is ``(W - W/2) / fx =
    (W/2) / fx = tan(HFOV/2)``, whose arctangent is half the published
    horizontal field of view. It pins the intrinsics without restating them,
    and it is the identity the off-centre cases below use to pick pixels
    whose angle is exact.

    It also pins ``fx`` and ``fy`` as *separate* numbers: for every asset the
    published pair is not quite consistent with square pixels, so a single
    averaged focal length fails the vertical half of this test.
    """
    right_x, right_y = camera.normalised(float(camera.width), camera.cy)
    assert math.degrees(math.atan(right_x)) == pytest.approx(camera.hfov_deg / 2.0, abs=1e-12)
    assert right_y == pytest.approx(0.0, abs=1e-15)

    bottom_x, bottom_y = camera.normalised(camera.cx, float(camera.height))
    assert math.degrees(math.atan(bottom_y)) == pytest.approx(camera.vfov_deg / 2.0, abs=1e-12)
    assert bottom_x == pytest.approx(0.0, abs=1e-15)


def test_camera_basis_is_right_handed_and_orthonormal() -> None:
    """``right × down = forward``, and all three are unit vectors.

    Checked at an attitude with all three angles non-zero, so a sign error in
    the roll rotation cannot hide behind a zero.
    """
    pose = CameraPose(71.99, -94.84, 300.0, yaw_deg=37.0, pitch_deg=-22.0, roll_deg=15.0)
    right, down, forward = camera_basis(pose)
    for axis in (right, down, forward):
        assert math.sqrt(sum(component * component for component in axis)) == pytest.approx(
            1.0, abs=1e-12
        )
    cross = (
        right[1] * down[2] - right[2] * down[1],
        right[2] * down[0] - right[0] * down[2],
        right[0] * down[1] - right[1] * down[0],
    )
    for got, want in zip(cross, forward, strict=True):
        assert got == pytest.approx(want, abs=1e-12)


def test_camera_basis_at_a_nadir_camera_heading_north() -> None:
    """Yaw 0, pitch -90: forward is Down, right is East, down-in-image is South.

    ``forward = (cos(-90°)·sin 0, cos(-90°)·cos 0, sin(-90°)) = (0, 0, -1)``.
    ``right = (cos 0, -sin 0, 0) = (1, 0, 0)``, East.
    ``down = forward × right = (0,0,-1) × (1,0,0) = (0·0 - (-1)·0, (-1)·1 -
    0·0, 0·0 - 0·1) = (0, -1, 0)``, South.

    The top of a nadir frame therefore points the way the vehicle faces,
    which is the convention a downward camera is read with.
    """
    right, down, forward = camera_basis(CameraPose(72.0, -95.0, 100.0, 0.0, -90.0))
    assert forward == pytest.approx((0.0, 0.0, -1.0), abs=1e-12)
    assert right == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert down == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)


# --------------------------------------------------------------------------
# Acceptance: nadir + centre pixel returns the asset's own position
# --------------------------------------------------------------------------


@pytest.mark.parametrize("asset", sorted(CAMERAS))
@pytest.mark.parametrize("yaw_deg", [0.0, 37.0, 180.0, 355.5])
@pytest.mark.parametrize("roll_deg", [0.0, 12.5])
def test_nadir_centre_pixel_returns_the_asset_position(
    asset: str, yaw_deg: float, roll_deg: float
) -> None:
    """Straight down, dead centre, is the asset's own lat/lon.

    By hand: the centre pixel normalises to ``(0, 0)``, so the ray is the
    boresight alone, and at pitch −90° the boresight is ``(0, 0, −1)``. The
    intersection parameter is ``t = −h / −1 = h``, giving ``east = north =
    0``, and an ENU offset of zero leaves the latitude and longitude
    untouched **whatever** the radii of curvature are. The answer is the
    camera's own position exactly, for every asset, every heading and every
    roll — and this asserts it for all three cameras and eight attitudes.

    The residual is not zero only because ``cos(-90°)`` is 6.1e-17 rather
    than 0 in binary floating point, which at 120 m altitude displaces the
    answer by about 7e-15 m.
    """
    camera = CAMERAS[asset]
    pose = CameraPose(
        lat_deg=71.9965,
        lon_deg=-94.8448,
        alt_m=120.0,
        yaw_deg=yaw_deg,
        pitch_deg=-90.0,
        roll_deg=roll_deg,
    )
    fix = project_pixel_to_ground(camera, pose, camera.cx, camera.cy)
    assert fix.lat_deg == pytest.approx(pose.lat_deg, abs=1e-12)
    assert fix.lon_deg == pytest.approx(pose.lon_deg, abs=1e-12)


# --------------------------------------------------------------------------
# Acceptance: known poses, worked by hand
# --------------------------------------------------------------------------


def test_tower_right_edge_pixel_at_thirty_degrees_depression() -> None:
    """Case A — tower, right edge of frame, hand-worked end to end.

    **Setup.** Tower camera (60.0° × 36.1°, 640×360) at 72.0000° N,
    95.0000° W, 100.0 m above the water. Yaw 90° (due East), pitch −30°,
    roll 0. Pixel ``(640.0, 180.0)``: the exact middle of the frame's right
    edge.

    **1. Pixel → camera frame.**
    ``fx = 320 / tan 30° = 320 / 0.5773502692 = 554.2562584``.
    ``x_c = (640 - 320) / 554.2562584 = 320 / 554.2562584 = 0.5773502692``,
    which is ``tan 30°`` — the right edge is HFOV/2 off axis, as it must be.
    ``y_c = (180 - 180) / fy = 0``.

    **2. Camera frame → ENU.** ψ = 90°, θ = −30°:
    ``forward = (cos(-30)·sin 90, cos(-30)·cos 90, sin(-30))
    = (0.8660254038, 0, -0.5)``;
    ``right = (cos 90, -sin 90, 0) = (0, -1, 0)`` — facing East, the camera's
    right is South. Roll is zero, so the axes stand.

    ``v = x_c·right + y_c·down + forward``
    ``  = 0.5773502692·(0, -1, 0) + (0.8660254038, 0, -0.5)``
    ``  = (0.8660254038, -0.5773502692, -0.5)``.

    Note ``right`` is horizontal, so it adds nothing to the Up component:
    this ray descends at exactly the boresight's rate.

    **3. Ray → water plane.** ``t = -h / v_U = -100 / -0.5 = 200``.
    ``east  = 200 × 0.8660254038 = 173.2050808 m``
    ``north = 200 × -0.5773502692 = -115.4700538 m``
    Cross-check: the centre ray lands at ``100 / tan 30° = 173.2050808 m``
    due East, so East is unchanged and the whole excursion is 115.47 m South.

    **4. ENU → geodetic** at φ = 72°, with M and N from the module
    docstring:
    ``Δφ = -115.4700538 / 6393420.763 = -1.806076254e-5 rad``
    ``   = -1.806076254e-5 × 57.295779513 = -0.0010348055°``
    ``Δλ = 173.2050808 / 1976947.143 = 8.761239853e-5 rad``
    ``   = 8.761239853e-5 × 57.295779513 = +0.0050198207°``

    ``lat = 72.0 - 0.0010348055 = 71.9989651945``
    ``lon = -95.0 + 0.0050198207 = -94.9949801793``
    """
    pose = CameraPose(72.0, -95.0, 100.0, yaw_deg=90.0, pitch_deg=-30.0, roll_deg=0.0)

    ray = pixel_ray_enu(TOWER_CAMERA, pose, 640.0, 180.0)
    assert ray == pytest.approx((0.8660254038, -0.5773502692, -0.5), abs=1e-9)

    # Step 3 in isolation, so a failure says which step broke.
    scale = -(pose.alt_m - 0.0) / ray[2]
    assert scale == pytest.approx(200.0, abs=1e-9)
    assert scale * ray[0] == pytest.approx(173.2050808, abs=METRE_TOL)
    assert scale * ray[1] == pytest.approx(-115.4700538, abs=METRE_TOL)

    fix = project_pixel_to_ground(TOWER_CAMERA, pose, 640.0, 180.0)
    assert fix.lat_deg == pytest.approx(71.9989651945, abs=DEGREE_TOL)
    assert fix.lon_deg == pytest.approx(-94.9949801793, abs=DEGREE_TOL)


def test_fixed_wing_bottom_edge_pixel_with_ninety_degrees_of_roll() -> None:
    """Case B — fixed-wing, banked 90°, hand-worked end to end.

    Roll is the axis easiest to drop silently, so this case is built so that
    **ignoring roll moves the answer by 2.2 km**: the pixel is offset only
    vertically, and the 90° bank turns that vertical offset into a horizontal
    one.

    **Setup.** Fixed-wing camera (69.0° × 42.6°, 640×360) at 71.9500° N,
    94.9000° W, 500.0 m above the water. Yaw 0° (North), pitch −10°, roll
    +90°. Pixel ``(320.0, 360.0)``: the middle of the frame's bottom edge.

    **1. Pixel → camera frame.** ``x_c = 0``.
    ``fy = 180 / tan 21.3° = 180 / 0.3898837079 = 461.6761263``, so
    ``y_c = (360 - 180) / 461.6761263 = 0.3898837079 = tan 21.3°`` — the
    bottom edge is VFOV/2 below the axis.

    **2. Camera frame → ENU.** ψ = 0, θ = −10°:
    ``forward = (0, cos(-10), sin(-10)) = (0, 0.9848077530, -0.1736481777)``
    ``right₀  = (1, 0, 0)``, East;  ``down₀ = forward × right₀ =
    (0.9848077530·0 - (-0.1736481777)·0, (-0.1736481777)·1 - 0·0,
    0·0 - 0.9848077530·1) = (0, -0.1736481777, -0.9848077530)``.

    Now roll +90° about ``forward``. A right-handed quarter turn about the
    forward axis sends ``right → down₀`` and ``down₀ → -right₀``, because
    ``right``, ``down`` and ``forward`` are a right-handed orthonormal
    triple: ``forward × down₀ = -right₀``. So
    ``down = -right₀ = (-1, 0, 0)``: due **West**.

    ``v = y_c·down + forward``
    ``  = 0.3898837079·(-1, 0, 0) + (0, 0.9848077530, -0.1736481777)``
    ``  = (-0.3898837079, 0.9848077530, -0.1736481777)``.

    With roll ignored the same pixel would give
    ``v = (0, 0.9848077530 - 0.3898837079·0.1736481777, …)`` — a point
    2.2 km away, which is the margin this case is asserting on.

    **3. Ray → water plane.** ``t = -500 / -0.1736481777 = 2879.3852416``.
    ``east  = 2879.3852416 × -0.3898837079 = -1122.6253946 m``
    ``north = 2879.3852416 × 0.9848077530 = 2835.6409098 m``

    **4. ENU → geodetic** at φ = 71.95°:
    ``sin φ = 0.9507865, sin²φ = 0.9039950, e² sin²φ = 0.0060517``
    ``1 - e² sin²φ = 0.9939483; sqrt = 0.9969696; ^1.5 = 0.9909362``
    ``N = 6378137 / 0.9969696 = 6397524.279``
    ``M = 6335439.33 / 0.9909362 = 6393387.592``
    ``cos φ = 0.3098468; N cos φ = 1982252.618``

    ``Δφ = 2835.6409098 / 6393387.592 = 4.4352713875e-4 rad``
    ``   = 4.4352713875e-4 × 57.295779513 = +0.0254122332°``
    ``Δλ = -1122.6253946 / 1982252.618 = -5.6633820763e-4 rad``
    ``   = -5.6633820763e-4 × 57.295779513 = -0.0324487891°``

    ``lat = 71.9500 + 0.0254122332 = 71.9754122332``
    ``lon = -94.9000 - 0.0324487891 = -94.9324487891``
    """
    pose = CameraPose(71.95, -94.90, 500.0, yaw_deg=0.0, pitch_deg=-10.0, roll_deg=90.0)

    ray = pixel_ray_enu(FIXED_WING_CAMERA, pose, 320.0, 360.0)
    assert ray == pytest.approx((-0.3898837079, 0.9848077530, -0.1736481777), abs=1e-9)

    scale = -pose.alt_m / ray[2]
    assert scale == pytest.approx(2879.3852416, abs=1e-6)
    assert scale * ray[0] == pytest.approx(-1122.6253946, abs=METRE_TOL)
    assert scale * ray[1] == pytest.approx(2835.6409098, abs=METRE_TOL)

    fix = project_pixel_to_ground(FIXED_WING_CAMERA, pose, 320.0, 360.0)
    assert fix.lat_deg == pytest.approx(71.9754122332, abs=DEGREE_TOL)
    assert fix.lon_deg == pytest.approx(-94.9324487891, abs=DEGREE_TOL)

    # And the margin the case was built for: the same pixel with the bank
    # ignored lands kilometres away, so roll cannot be quietly dropped.
    level = CameraPose(71.95, -94.90, 500.0, yaw_deg=0.0, pitch_deg=-10.0, roll_deg=0.0)
    level_fix = project_pixel_to_ground(FIXED_WING_CAMERA, level, 320.0, 360.0)
    assert abs(level_fix.lat_deg - fix.lat_deg) > 1e-3


def test_tower_pose_from_servo_pwm_projects_its_centre_pixel() -> None:
    """Case C — a tower pose built from raw servo PWM, hand-worked.

    **Setup.** Tower at 72.0000° N, 95.0000° W, camera 60.0 m above the
    water, pan servo at 1700 µs and tilt servo at 1420 µs, on the default
    AntennaTracker travel (pan ±180° and tilt ±90° across the 1100–1900 µs
    that ``SERVOn_MIN``/``SERVOn_MAX`` default to). Centre pixel
    ``(320.0, 180.0)``.

    **1. PWM → angles**, linearly across the pulse range:
    ``pan  = -180 + (1700 - 1100)/800 × 360 = -180 + 0.750 × 360 = +90°``
    ``tilt = -90  + (1420 - 1100)/800 × 180 = -90  + 0.400 × 180 = -18°``
    So the camera looks due East, 18° below the horizon.

    **2 and 3. Centre pixel → water plane.** The centre pixel normalises to
    ``(0, 0)``, so the ray is the boresight:
    ``v = (cos(-18)·sin 90, cos(-18)·cos 90, sin(-18))
    = (0.9510565163, 0, -0.3090169944)``.
    ``t = -60 / -0.3090169944 = 194.1640786``
    ``east  = 194.1640786 × 0.9510565163 = 184.6610122 m``
    ``north = 0``
    Cross-check: the ground range is ``60 / tan 18° = 60 / 0.3249197 =
    184.6610 m``, which is what a boresight at 18° depression from 60 m must
    give.

    **4. ENU → geodetic** at φ = 72°, N cos φ = 1976947.143 m:
    ``Δφ = 0``
    ``Δλ = 184.6610122 / 1976947.143 = 9.340715703e-5 rad``
    ``   = 9.340715703e-5 × 57.295779513 = +0.0053518359°``

    ``lat = 72.0`` unchanged — a due-East look does not change latitude
    ``lon = -95.0 + 0.0053518359 = -94.9946481641``
    """
    pose = tower_camera_pose(72.0, -95.0, 60.0, pan_pwm=1700, tilt_pwm=1420)
    assert pose.yaw_deg == pytest.approx(90.0, abs=1e-12)
    assert pose.pitch_deg == pytest.approx(-18.0, abs=1e-12)
    assert pose.roll_deg == 0.0

    ray = pixel_ray_enu(TOWER_CAMERA, pose, 320.0, 180.0)
    assert ray == pytest.approx((0.9510565163, 0.0, -0.3090169944), abs=1e-9)

    fix = project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 180.0)
    assert fix.lat_deg == pytest.approx(72.0, abs=DEGREE_TOL)
    assert fix.lon_deg == pytest.approx(-94.9946481641, abs=DEGREE_TOL)


def test_enu_to_geodetic_matches_the_radii_worked_by_hand() -> None:
    """The tangent-plane step alone, at φ = 72°, from the module docstring.

    ``M = 6393420.763 m`` and ``N cos φ = 1976947.143 m``, so:

    ``1000 / 6393420.763``: ``6393420.763 × 1.564e-4 = 999.9310074``,
    remainder ``0.0689926``, ``0.0689926 / 6393420.763 = 1.07911e-8``, so
    ``1.564107911e-4`` rad; ``× 57.295779513 = 8.9616782e-3°``.

    ``1000 / 1976947.143``: ``1976947.143 × 5.058e-4 = 999.9398649``,
    remainder ``0.0601351``, ``0.0601351 / 1976947.143 = 3.04185e-8``, so
    ``5.058304185e-4`` rad; ``× 57.295779513 = 2.89819481e-2°``.
    """
    north = enu_to_geodetic(72.0, -95.0, 0.0, 1000.0)
    assert north.lat_deg == pytest.approx(72.0 + 0.0089616782, abs=DEGREE_TOL)
    assert north.lon_deg == -95.0

    east = enu_to_geodetic(72.0, -95.0, 1000.0, 0.0)
    assert east.lat_deg == 72.0
    assert east.lon_deg == pytest.approx(-95.0 + 0.0289819481, abs=DEGREE_TOL)


# --------------------------------------------------------------------------
# The rays that have no answer
# --------------------------------------------------------------------------


def test_a_pixel_above_the_horizon_is_refused() -> None:
    """A level camera's top-edge pixel looks at sky, and there is no fix.

    Pitch 0 puts the boresight on the horizon, so every pixel above the
    frame's middle row rises above it and the ray never reaches the water.
    Returning some very distant point would be worse than refusing: it would
    be submitted to the tracks API as a detection.
    """
    pose = CameraPose(72.0, -95.0, 100.0, yaw_deg=0.0, pitch_deg=0.0)
    with pytest.raises(ProjectionError, match="horizon"):
        project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 0.0)
    with pytest.raises(ProjectionError, match="horizon"):
        project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 180.0)


def test_a_camera_at_or_below_the_water_plane_is_refused() -> None:
    """No altitude above the plane, no intersection to compute."""
    pose = CameraPose(72.0, -95.0, 0.0, yaw_deg=0.0, pitch_deg=-45.0)
    with pytest.raises(ProjectionError, match="not above the plane"):
        project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 180.0)


def test_ground_altitude_offsets_the_plane() -> None:
    """``ground_alt_m`` moves the plane, not the camera.

    A camera 160 m above a plane at 60 m is geometrically the camera of case
    C, which is 60 m above a plane at 0: same height above the water, same
    answer. If ``ground_alt_m`` were ignored, the height would be 160 m and
    the fix would land 2.7× further East.
    """
    high = CameraPose(72.0, -95.0, 160.0, yaw_deg=90.0, pitch_deg=-18.0)
    fix = project_pixel_to_ground(TOWER_CAMERA, high, 320.0, 180.0, ground_alt_m=100.0)
    assert fix.lat_deg == pytest.approx(72.0, abs=DEGREE_TOL)
    assert fix.lon_deg == pytest.approx(-94.9946481641, abs=DEGREE_TOL)


def test_the_horizon_and_the_default_range_bound_are_worked_by_hand() -> None:
    """``d_horizon = sqrt(2 R h)`` and the bound is ``sqrt(0.1)`` of it.

    ``R = (2a + b)/3`` with ``b = a(1 - f) = 6356752.314...``, so
    ``R = (12756274 + 6356752.314)/3 = 6371008.771 m``.

    ``h = 60``:   ``2 R h = 764521052.6``, ``sqrt = 27650.0 m``
    ``h = 120``:  ``2 R h = 1529042105``, ``sqrt = 39103.0 m``
    ``h = 500``:  ``2 R h = 6371008771``, ``sqrt = 79818.6 m``

    and ``sqrt(0.1) = 0.31622777``, so the default bounds are 8743.7 m,
    12365.4 m and 25240.9 m. Those are the ranges at which the flat water
    plane's own curvature error reaches a tenth of the range, because that
    error is ``(d / d_horizon)²`` — see the module docstring.
    """
    assert EARTH_MEAN_RADIUS_M == pytest.approx(6371008.7714, abs=1e-3)
    assert MAX_FLAT_PLANE_RANGE_ERROR == 0.1

    assert horizon_range_m(60.0) == pytest.approx(27650.0, abs=0.1)
    assert horizon_range_m(120.0) == pytest.approx(39103.0, abs=0.1)
    assert horizon_range_m(500.0) == pytest.approx(79818.6, abs=0.1)

    assert max_flat_plane_range_m(60.0) == pytest.approx(8743.7, abs=0.1)
    assert max_flat_plane_range_m(120.0) == pytest.approx(12365.4, abs=0.1)
    assert max_flat_plane_range_m(500.0) == pytest.approx(25240.9, abs=0.1)

    # The identity the bound is defined by: the error is (d/d_horizon)², so
    # the tolerance and the ratio of the two ranges are a square apart.
    assert (max_flat_plane_range_m(60.0) / horizon_range_m(60.0)) ** 2 == pytest.approx(
        MAX_FLAT_PLANE_RANGE_ERROR, abs=1e-12
    )


@pytest.mark.parametrize(
    ("py", "unbounded_range_m"),
    [
        # 0.1° of pitch per pixel row on a 36.1°/360 px camera, so these rows
        # sit 17.95°, 17.85° and 17.75° below the horizon: still descending,
        # and still nonsense.
        (1.0, 78445.0),
        (2.0, 24957.0),
        (3.0, 14833.0),
    ],
)
def test_a_ray_that_descends_but_grazes_the_horizon_is_refused(
    py: float, unbounded_range_m: float
) -> None:
    """Descending is not enough: ``-h / ray[2]`` is unbounded as the ray flattens.

    This is case C's attitude — a tower camera 60 m up, pitched 18° down —
    scanning the top of an ordinary frame. Two pixel rows from the top it
    projects 25.0 km, the whole length of the strait, and one row from the
    top 78.4 km, well past the 27.6 km horizon a 60 m camera actually has.
    Nothing about those points looks wrong: they are ordinary
    :class:`GeoPoint`\\ s, and they would be submitted to ``POST /api/tracks``
    as detections and read by the *accuracy* criterion.

    The ranges in the parameters are what the unbounded intersection gives,
    computed from the ray here rather than taken from the implementation's
    output, so this test pins the size of the problem and not just its sign.
    """
    pose = CameraPose(72.0, -95.0, 60.0, yaw_deg=90.0, pitch_deg=-18.0)

    ray = pixel_ray_enu(TOWER_CAMERA, pose, 320.0, py)
    assert ray[2] < 0.0, "the ray descends, which is exactly why the old guard passed it"
    scale = -60.0 / ray[2]
    assert math.hypot(scale * ray[0], scale * ray[1]) == pytest.approx(unbounded_range_m, rel=1e-4)

    with pytest.raises(ProjectionError, match="grazes the horizon"):
        project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, py)


def test_the_same_frame_still_projects_everywhere_the_plane_is_trustworthy() -> None:
    """The bound must not cost the frame: row 5 downward still projects.

    Same pose. Row 5 is 8.2 km, inside the 8.74 km bound, and every row below
    it is nearer. A bound that also rejected the useful part of the frame
    would trade one silent failure for another.
    """
    pose = CameraPose(72.0, -95.0, 60.0, yaw_deg=90.0, pitch_deg=-18.0)
    for py in (5.0, 10.0, 20.0, 40.0, 180.0, 359.0, 360.0):
        fix = project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, py)
        assert fix.lat_deg == pytest.approx(72.0, abs=DEGREE_TOL)
        assert -95.0 < fix.lon_deg < -94.7


def test_a_caller_may_bound_the_range_more_tightly_than_the_default() -> None:
    """Bellot Strait is 2 km across, and this module cannot know that.

    ``max_range_m`` is how a caller that does know says so. The default is a
    *model* bound — where this arithmetic stops meaning anything — not an
    arena one.
    """
    pose = CameraPose(72.0, -95.0, 60.0, yaw_deg=90.0, pitch_deg=-18.0)
    assert project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 10.0) is not None

    with pytest.raises(ProjectionError, match="grazes the horizon"):
        project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 10.0, max_range_m=3000.0)

    near = project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 180.0, max_range_m=3000.0)
    assert near.lon_deg == pytest.approx(-94.9946481641, abs=DEGREE_TOL)

    with pytest.raises(ProjectionError, match="max_range_m"):
        project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 180.0, max_range_m=0.0)


def test_a_pixel_outside_the_frame_is_refused() -> None:
    """A detector that reports a pixel off the sensor has a bug, not a fix.

    On a 640-wide tower camera, ``px = 100000`` used to project 35 km
    off-axis without a word. The frame is ``[0, W] × [0, H]``, corners
    included, because a pixel on the edge is a real pixel.
    """
    pose = CameraPose(72.0, -95.0, 100.0, yaw_deg=0.0, pitch_deg=-45.0)
    for px, py in ((100000.0, 180.0), (-1.0, 180.0), (320.0, -0.5), (320.0, 361.0)):
        with pytest.raises(ProjectionError, match="outside"):
            project_pixel_to_ground(TOWER_CAMERA, pose, px, py)

    # The edges themselves are inside.
    for px, py in ((0.0, 0.1), (640.0, 360.0), (320.0, 360.0)):
        project_pixel_to_ground(TOWER_CAMERA, pose, px, py)


def test_a_non_finite_pose_is_refused_at_construction() -> None:
    """A dropped MAVLink field is a NaN, and NaN is not valid JSON.

    Unchecked, ``yaw_deg=nan`` produces ``GeoPoint(lat_deg=nan,
    lon_deg=nan)`` with no error raised anywhere in the chain — the one
    failure mode that reaches ``POST /api/tracks`` and breaks the request
    body itself.
    """
    for field, value in (
        ("lat_deg", math.nan),
        ("alt_m", math.inf),
        ("yaw_deg", math.nan),
        ("pitch_deg", math.nan),
        ("roll_deg", math.nan),
    ):
        kwargs = {
            "lat_deg": 72.0,
            "lon_deg": -95.0,
            "alt_m": 100.0,
            "yaw_deg": 0.0,
            "pitch_deg": -45.0,
            field: value,
        }
        with pytest.raises(ProjectionError, match="must be finite"):
            CameraPose(**kwargs)  # type: ignore[arg-type]

    pose = CameraPose(72.0, -95.0, 100.0, yaw_deg=0.0, pitch_deg=-45.0)
    with pytest.raises(ProjectionError, match="ground_alt_m must be finite"):
        project_pixel_to_ground(TOWER_CAMERA, pose, 320.0, 180.0, ground_alt_m=math.nan)
    with pytest.raises(ProjectionError, match="must be finite"):
        enu_to_geodetic(72.0, -95.0, math.nan, 0.0)


def test_the_pole_guard_fires_at_the_pole() -> None:
    """``cos(radians(90))`` is ``6.12e-17``, not ``0``.

    A guard written as ``east_radius <= 0.0`` therefore never fires, and
    ``enu_to_geodetic(90.0, 0.0, 1000.0, 0.0)`` returned
    ``lon_deg=146214141690153.44``. Unreachable at Bellot Strait, but it is
    an exported function and a guard that cannot fire is worse than none.
    """
    with pytest.raises(ProjectionError, match="East radius"):
        enu_to_geodetic(90.0, 0.0, 1000.0, 0.0)
    with pytest.raises(ProjectionError, match="East radius"):
        enu_to_geodetic(-90.0, 0.0, 1000.0, 0.0)

    # Bellot Strait is nowhere near it, and must be unaffected.
    assert enu_to_geodetic(71.99, -94.84, 1000.0, 0.0).lon_deg == pytest.approx(
        -94.811034, abs=1e-6
    )


def test_a_malformed_camera_is_refused() -> None:
    """The intrinsics are validated where they are built, not where used."""
    with pytest.raises(VisionError, match="hfov_deg"):
        CameraModel("broken", 0.0, 36.1, 640, 360)
    with pytest.raises(VisionError, match="vfov_deg"):
        CameraModel("broken", 60.0, 180.0, 640, 360)
    with pytest.raises(VisionError, match="width"):
        CameraModel("broken", 60.0, 36.1, 0, 360)
    with pytest.raises(VisionError, match="must be finite"):
        TOWER_CAMERA.normalised(math.nan, 180.0)
