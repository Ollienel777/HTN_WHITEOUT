"""Pixel to lat/lon: intersect the camera ray with the water plane.

``hackathon/ARENA.md`` §4 and §7b.3. This is the step between "there is a
vessel in this frame" and the ``lat``/``lon`` the tracks API takes, and it is
what the *accuracy* criterion (``ARENA.md`` §5) is measured on.

The chain is four steps, and each one is a separate function so that a test
can check them one at a time:

1. **Pixel → camera-frame direction.** :meth:`CameraModel.normalised
   <whiteout.vision.camera.CameraModel.normalised>`, the pinhole model.
2. **Camera frame → ENU.** :func:`camera_basis` builds the camera's right,
   down and forward axes as vectors in a local East–North–Up frame, from the
   yaw/pitch/roll of the boresight. :func:`pixel_ray_enu` combines 1 and 2.
3. **Ray → water plane.** The target is a boat: it is confined to water, so
   the plane is ``up = 0`` and there is no terrain raycast to do. One
   division.
4. **ENU offset → lat/lon.** :func:`enu_to_geodetic`, a tangent-plane step
   using the WGS-84 radii of curvature at the camera's latitude.

**Why the water plane makes this well posed.** A single camera cannot
triangulate: a pixel is a ray, not a point, and a ray has a one-parameter
family of world points on it. What pins the answer down is knowing the
target's altitude. Against terrain that would mean a raycast against a
heightfield and an answer that degrades with the heightfield. Against a boat
in a strait it is a plane at sea level, the intersection is exact, and the
error budget collapses to the pose and the intrinsics. ``ARENA.md``'s note on
the ticket says this in one line: "the target is confined to water, which
makes the ray intersection well-posed".

Frames of reference
-------------------

**World: local ENU**, tangent to the ellipsoid at the camera — ``x`` East,
``y`` North, ``z`` Up, metres, origin at the camera.

**Camera: the computer-vision convention** — ``x`` right across the image,
``y`` down the image, ``z`` forward along the boresight.

**Orientation: yaw, pitch and roll of the boresight**, degrees, matching the
ArduPilot ``ATTITUDE`` message so that a body-fixed forward-looking camera
takes the vehicle's attitude unchanged:

``yaw``
    Bearing of the boresight's horizontal projection, clockwise from **true**
    North. 0 is North, 90 is East.
``pitch``
    Elevation of the boresight above the horizon. **Negative is downward**,
    so a nadir camera is ``-90``.
``roll``
    Rotation about the boresight, positive right-wing-down — the
    right-handed rotation about the forward axis, which is the one that tips
    the image's right edge toward the ground.

A gimballed camera is the same type with the gimbal's angles substituted for
the airframe's; a tower's come from :mod:`whiteout.vision.tower`.

Accuracy of the tangent-plane step
----------------------------------

:func:`enu_to_geodetic` divides the North offset by the meridional radius of
curvature :math:`M` and the East offset by :math:`N \\cos\\varphi`, both
evaluated at the camera's latitude. That is a first-order expansion of the
geodesic, and its error grows as the square of the range: it is under a
centimetre at 3 km and under a decimetre at 10 km at Bellot Strait's latitude.
The strait is 25 km × 2 km and the fleet's useful slant ranges are a few
kilometres, so the approximation is two orders of magnitude inside the pose
error and is not worth replacing with a full geodesic solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from whiteout.vision.camera import CameraModel, VisionError

__all__ = [
    "WGS84_A",
    "WGS84_E2",
    "WGS84_F",
    "CameraPose",
    "GeoPoint",
    "ProjectionError",
    "Vec3",
    "camera_basis",
    "enu_to_geodetic",
    "pixel_ray_enu",
    "project_pixel_to_ground",
]

#: A vector in the local ENU frame: ``(east, north, up)``, metres or unitless.
Vec3 = tuple[float, float, float]

#: WGS-84 semi-major axis, metres.
WGS84_A = 6378137.0

#: WGS-84 flattening.
WGS84_F = 1.0 / 298.257223563

#: WGS-84 first eccentricity squared, ``f (2 - f)``.
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


class ProjectionError(VisionError):
    """A pixel has no well-defined point on the water plane.

    Raised when the ray does not descend — the pixel is on or above the
    horizon — and when the camera is not above the plane it is being
    projected onto. Both are real conditions in the arena rather than
    programming errors: a fixed-wing at 500 m with a 42.6° vertical field of
    view sees the horizon in the top of almost every frame, and a tower
    scanning at ``mode scan`` sweeps through it. The caller's response is to
    discard the detection, not to crash, so this is a catchable error with a
    message that names the cause.
    """


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """A geodetic position on the water plane: what the tracks API takes.

    ``ARENA.md`` §5's ``POST /api/tracks`` body is ``{"name": …, "lat": …,
    "lon": …}``, degrees, so this type is deliberately two floats and not a
    third: the altitude is the water plane and carrying it would invite
    someone to submit it.
    """

    lat_deg: float
    lon_deg: float


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Where a camera is and where it points, at the instant of a frame.

    :param lat_deg: geodetic latitude of the camera, degrees North.
    :param lon_deg: geodetic longitude of the camera, degrees East.
    :param alt_m: camera altitude, metres above the water plane's datum.
    :param yaw_deg: boresight bearing, degrees clockwise from true North.
    :param pitch_deg: boresight elevation, degrees; negative is downward.
    :param roll_deg: rotation about the boresight, degrees, right-wing-down
        positive.

    The angles are the module docstring's conventions, which are ArduPilot's.
    For the quadcopter and the fixed-wing they come from ``GLOBAL_POSITION_INT``
    and ``ATTITUDE``; for a tower they come from
    :func:`whiteout.vision.tower.tower_camera_pose`.
    """

    lat_deg: float
    lon_deg: float
    alt_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotate_about(v: Vec3, axis: Vec3, angle_deg: float) -> Vec3:
    """Rodrigues' rotation of ``v`` about a unit ``axis``, right-handed.

    ``v cos θ + (axis × v) sin θ + axis (axis · v)(1 - cos θ)``.
    """
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cross = _cross(axis, v)
    dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2]
    return (
        v[0] * cos_t + cross[0] * sin_t + axis[0] * dot * (1.0 - cos_t),
        v[1] * cos_t + cross[1] * sin_t + axis[1] * dot * (1.0 - cos_t),
        v[2] * cos_t + cross[2] * sin_t + axis[2] * dot * (1.0 - cos_t),
    )


def camera_basis(pose: CameraPose) -> tuple[Vec3, Vec3, Vec3]:
    """Return the camera's ``(right, down, forward)`` axes as ENU vectors.

    All three are unit vectors and they form a right-handed triple with
    ``right × down = forward``, which is what makes the camera frame the
    computer-vision one.

    The construction, before roll:

    .. code-block:: text

        forward = (cos θ sin ψ,  cos θ cos ψ,  sin θ)
        right   = (cos ψ,       -sin ψ,        0    )
        down    = forward × right

    with ψ the yaw and θ the pitch. ``right`` is horizontal by construction —
    it is the yaw bearing turned 90° clockwise — so facing North the camera's
    right is East, and facing East it is South. ``down`` then falls out of the
    right-handedness, and for a nadir camera heading North it comes to South:
    the top of a nadir frame points the way the vehicle is facing, which is
    the usual convention for a downward camera.

    Roll is applied last, as a right-handed rotation of ``right`` and ``down``
    about ``forward``. A right-handed rotation about an axis pointing away
    from the viewer turns clockwise as the camera sees it, so positive roll
    tips the image's right edge downward: right-wing-down, ArduPilot's sign.
    """
    psi = math.radians(pose.yaw_deg)
    theta = math.radians(pose.pitch_deg)
    forward: Vec3 = (
        math.cos(theta) * math.sin(psi),
        math.cos(theta) * math.cos(psi),
        math.sin(theta),
    )
    right: Vec3 = (math.cos(psi), -math.sin(psi), 0.0)
    down: Vec3 = _cross(forward, right)
    if pose.roll_deg:
        right = _rotate_about(right, forward, pose.roll_deg)
        down = _rotate_about(down, forward, pose.roll_deg)
    return right, down, forward


def pixel_ray_enu(camera: CameraModel, pose: CameraPose, px: float, py: float) -> Vec3:
    """Return the ENU direction of the ray through pixel ``(px, py)``.

    ``x_c · right + y_c · down + forward``, where ``(x_c, y_c)`` is
    :meth:`CameraModel.normalised <whiteout.vision.camera.CameraModel.normalised>`.

    **The result is a direction, not a unit vector.** Its component along the
    boresight is exactly 1, because the camera-frame direction is taken at
    ``z = 1``. That is deliberate: it keeps the arithmetic short enough to
    check by hand, which ``tests/test_vision_projection.py`` does, and a
    ray–plane intersection is invariant to the scale of the direction, so
    nothing downstream needs it normalised.
    """
    x_c, y_c = camera.normalised(px, py)
    right, down, forward = camera_basis(pose)
    return (
        x_c * right[0] + y_c * down[0] + forward[0],
        x_c * right[1] + y_c * down[1] + forward[1],
        x_c * right[2] + y_c * down[2] + forward[2],
    )


def enu_to_geodetic(lat_deg: float, lon_deg: float, east_m: float, north_m: float) -> GeoPoint:
    """Offset a geodetic position by a local ENU displacement.

    .. math::

        \\Delta\\varphi = \\frac{\\text{north}}{M(\\varphi)}
        \\qquad
        \\Delta\\lambda = \\frac{\\text{east}}{N(\\varphi)\\cos\\varphi}

    in radians, with the WGS-84 radii of curvature at the origin latitude:

    .. math::

        N = \\frac{a}{\\sqrt{1 - e^2 \\sin^2\\varphi}}
        \\qquad
        M = \\frac{a(1 - e^2)}{(1 - e^2 \\sin^2\\varphi)^{3/2}}

    See the module docstring for the error this carries: under a centimetre
    at 3 km.
    """
    phi = math.radians(lat_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    w = 1.0 - WGS84_E2 * sin_phi * sin_phi
    prime_vertical = WGS84_A / math.sqrt(w)
    meridional = WGS84_A * (1.0 - WGS84_E2) / (w**1.5)
    east_radius = prime_vertical * cos_phi
    if east_radius <= 0.0:
        raise ProjectionError(
            f"longitude is undefined at latitude {lat_deg!r}: the East radius vanishes at the pole"
        )
    return GeoPoint(
        lat_deg + math.degrees(north_m / meridional),
        lon_deg + math.degrees(east_m / east_radius),
    )


def project_pixel_to_ground(
    camera: CameraModel,
    pose: CameraPose,
    px: float,
    py: float,
    *,
    ground_alt_m: float = 0.0,
) -> GeoPoint:
    """Project a pixel onto the water plane and return its lat/lon.

    :param camera: the asset's published intrinsics, from
        :mod:`whiteout.vision.camera`.
    :param pose: where the camera was, and where it pointed, for this frame.
    :param px: pixel column, ``[0, width]``, origin at the frame's left edge.
    :param py: pixel row, ``[0, height]``, origin at the frame's top edge.
    :param ground_alt_m: altitude of the plane being projected onto, in the
        same datum as ``pose.alt_m``. It is the water's, so the default of 0
        is right whenever ``alt_m`` is height above the water. It is a
        parameter because the arena reports altitude in more than one datum
        and the caller, not this function, knows which one it has.
    :raises ProjectionError: if the camera is not above the plane, or if the
        ray does not descend to it.

    The intersection: with the camera at height :math:`h = \\text{alt} -
    \\text{ground}` above the plane and the ray direction
    :math:`v = (v_E, v_N, v_U)`, the ray is :math:`(0, 0, h) + t v` and meets
    ``up = 0`` at

    .. math::

        t = -\\frac{h}{v_U}
        \\qquad
        \\text{east} = t\\, v_E
        \\qquad
        \\text{north} = t\\, v_N

    which needs :math:`v_U < 0` — the pixel must look below the horizon.
    """
    height_above_plane = pose.alt_m - ground_alt_m
    if height_above_plane <= 0.0:
        raise ProjectionError(
            f"camera is not above the plane: altitude {pose.alt_m!r} m against a plane at "
            f"{ground_alt_m!r} m"
        )
    ray = pixel_ray_enu(camera, pose, px, py)
    if ray[2] >= 0.0:
        raise ProjectionError(
            f"pixel ({px!r}, {py!r}) does not descend to the plane: it is on or above the "
            f"horizon for a camera at yaw {pose.yaw_deg!r}, pitch {pose.pitch_deg!r}, "
            f"roll {pose.roll_deg!r}"
        )
    scale = -height_above_plane / ray[2]
    return enu_to_geodetic(pose.lat_deg, pose.lon_deg, scale * ray[0], scale * ray[1])
