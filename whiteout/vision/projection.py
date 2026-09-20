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

Grid North, true North, and why ``convergence_deg`` is not a bias here
-----------------------------------------------------------------------

``GET :8090/api/site`` publishes ``convergence_deg`` beside ``bounds3413``: the
angle between **grid North** — the northing axis of EPSG:3413, the polar
stereographic grid the arena tiles its terrain in — and **true North**. At this
site it is about **49.8°**, because for the polar aspect of a stereographic
projection the convergence is exactly :math:`\\lambda - \\lambda_0`, and
EPSG:3413's central meridian is 45° W against a site at 94.82° W.
:mod:`whiteout.site` records the arena's answer and checks it against that
identity.

**It corrects nothing in this module, and nothing anywhere else in the
build.** This is written down rather than left implied because 49.8° is not a
small angle, and the reflex on first meeting it — somewhere a heading must
need it subtracted — is wrong here for a reason that takes a paragraph and
would otherwise be re-derived by every reader in turn.

**The grid is a frame we never enter.** Positions arrive geodetic
(``GLOBAL_POSITION_INT``, degE7) and leave geodetic (``POST /api/tracks``).
The one frame in between is :mod:`whiteout.geo`'s local East–North tangent
plane, and its North axis is true North by construction: ``enu_to_geodetic``
divides the North offset by the meridional radius of curvature to get a change
of *latitude*, not a change of northing. No easting, no northing and no scale
factor of EPSG:3413 is read or written anywhere in ``whiteout/``. A rotation
between two frames is a correction only for a quantity expressed in one of
them, and none of ours is. What the grid is for is *rendering*: it is how the
arena places its terrain mesh and imagery, and it is what a dashboard drawing
that imagery has to work in. We draw in lat/lon.

**The two heading-shaped inputs are true-North referenced, and they are the
only two.** ``pose.heading`` becomes :attr:`CameraPose.yaw_deg` in
``whiteout/vision/sightings.py``; it is ArduPilot's ``ATTITUDE.yaw``, an AHRS
heading. ArduPilot has no notion of a map projection, and the simulator's
world frame is an East–North–Up frame about a geodetic origin, so that yaw is
referenced to true North and not to any grid. The other is a tower's pan zero
(:func:`whiteout.vision.tower.tower_camera_pose`'s ``boresight_yaw_deg``),
which is our own convention in a roster file, compared against
:func:`whiteout.geo.bearing_deg`, and true North for the same reason that
function is.

**And the frame is always built where the angle is measured**, which is what
makes the argument airtight rather than merely probable.
:func:`camera_basis` builds its axes at the camera, and
:func:`project_pixel_to_ground` resolves the resulting offset with
:func:`enu_to_geodetic` about *the camera's own* lat/lon. Origin and
measurement point coincide, so the plane's North is true North exactly there —
there is no residual convergence to carry, not even the tangent plane's own.

**What it would cost if this were wrong**, since a bias nobody can price is a
bias nobody weighs. A yaw error :math:`\\gamma` puts a fix at ground range
:math:`d` off by :math:`2 d \\sin(\\gamma/2)`, which at 49.8° is 0.843 of the
range:

.. code-block:: text

    ground range   position error at a 49.8 deg yaw bias
      302 m  (the #111 stand-off)      254 m
     1000 m                            842 m
     3000 m                           2527 m

The channel's modelled water is 535 m to 772 m of half-width
(:data:`whiteout.belief.geometry.DEFAULT_STRAIT`), so at a kilometre the bias
alone would exceed it: every projected fix would land on rock or outside the
6.5 km site, and :meth:`ChannelBeliefGrid.update_detection
<whiteout.belief.grid.ChannelBeliefGrid.update_detection>` would be folding
belief onto water the camera was not looking at. This is therefore not a
subtle error that could be sitting there unnoticed — it is the sort that
breaks a run in the first minute. #110 flew the detector live and reported its
false alarms landing "one to three kilometres away, on the channel's ice":
still in the channel, which is where a camera with a true-North yaw puts them
and is not where a 49.8° rotation would.

**The one-minute check, for whoever next has the arena.** Fly any asset a
straight leg and compare its ``ATTITUDE.yaw`` against the course over ground
that :func:`whiteout.geo.bearing_deg` gives between two consecutive
``GLOBAL_POSITION_INT`` fixes on that leg. Agreement to a few degrees is a
true-North yaw; a constant offset near 49.8° would mean the arena hands out a
grid-referenced heading and every paragraph above needs re-deriving. Nothing
in the repository asserts this, because it needs a live arena and a moving
asset, and an assertion nobody can run is worse than a procedure somebody can.

**The convergence that does exist here, and where it is zero.** A local
tangent plane has a meridian convergence of its own: true North at a point
:math:`\\Delta\\lambda` East of the origin is turned from the plane's North
axis by :math:`\\Delta\\lambda \\sin\\varphi`. Over the whole 6.5 km site about
:data:`whiteout.geo.ARENA_ORIGIN` that reaches **0.106°** at the far corner,
6.0 m of lateral error at 3.25 km. It is not corrected anywhere and does not
need to be, for two separate reasons. In this module it is identically zero,
per the paragraph above. In :mod:`whiteout.belief.geometry`, which does use one
fixed origin across the whole site, no heading is consumed at all: the ribbon
is a geometric construction inside a single plane, both of its conversions use
that same plane, and :mod:`whiteout.belief.grid` diffuses isotropically from a
speed. A heading measured at one point and used in another's frame is the
thing to watch for, and there is not one.

Accuracy, and the two approximations that cost the most
-------------------------------------------------------

Both of the following were **measured**, not estimated, and the numbers are
the measurement. Neither is corrected for, and the reason in both cases is
that something else in the chain is larger.

**1. The tangent plane is not the ellipsoid.** :func:`enu_to_geodetic`
divides the North offset by the meridional radius of curvature :math:`M` and
the East offset by :math:`N \\cos\\varphi`, both evaluated at the camera's
latitude. That is a first-order expansion, and the term it drops is not
symmetric: a purely **East** displacement on the tangent plane also changes
latitude, by about

.. math::

    \\Delta(\\text{north}) \\approx -\\frac{E^2 \\tan\\varphi}{2N}

and :math:`\\Delta\\varphi = \\text{north}/M` sets that to zero. Measured
against an exact ECEF round trip (geodetic → ECEF, displace along the true
ENU basis, invert with Bowring iteration) at φ = 71.99°, where
:math:`\\tan\\varphi = 3.076`:

.. code-block:: text

    displacement    position error of enu_to_geodetic
    E    200 m          0.010 m
    E    500 m          0.060 m
    E   1000 m          0.240 m
    E   2000 m          0.962 m
    E   3000 m          2.164 m
    E   5000 m          6.010 m
    E  10000 m         24.039 m
    N   1000 m          0.0005 m
    N   3000 m          0.0044 m
    N  10000 m          0.055 m

So it is **sub-millimetre along the meridian and a couple of metres at 3 km
East**, growing as the square of the East offset. East is the direction that
matters here — Bellot Strait's long axis runs roughly East–West — so these
are the figures to quote, not the North ones.

**2. The water plane is flat and the sea is not.** The sea surface falls
:math:`d^2/2R` below the tangent plane at ground range :math:`d`, so the ray
meets the flat plane *beyond* where it meets the water. With the camera at
height :math:`h` the ray descends at about :math:`h/d`, so the range is long
by :math:`d^3/(2Rh)` — a fractional error of

.. math::

    \\frac{\\Delta d}{d} \\approx \\frac{d^2}{2Rh}
    = \\left(\\frac{d}{d_{\\text{horizon}}}\\right)^2
    \\qquad d_{\\text{horizon}} = \\sqrt{2Rh}

This is a systematic bias, always outward:

.. code-block:: text

    tower  h= 60 m  d=2000 m   +10.5 m    tower  h= 60 m  d=5000 m  +163.5 m
    quad   h=120 m  d=5000 m   +81.8 m    f-wing h=500 m  d=5000 m   +19.6 m

It is the **larger of the two at every range for the low-mounted assets**, and
it is why :func:`project_pixel_to_ground` bounds the range it will return at
all: see :data:`MAX_FLAT_PLANE_RANGE_ERROR` and :func:`max_flat_plane_range_m`.

**Why neither is corrected.** The altitude datum is a bigger number than
both. ``pose.alt_m`` comes from ``GLOBAL_POSITION_INT``, which is above the
ellipsoid, and the geoid separation in the Canadian Arctic is tens of metres;
that scales the range *linearly*, so on a 120 m quadcopter it is a
double-digit percentage. Correcting either approximation above while the
datum is unresolved would move the answer by less than the thing it is
measured against. Both corrections are small — one extra term in
:func:`enu_to_geodetic`, and a two-iteration fixpoint on the plane height —
and the only real cost of applying them is re-deriving the hand-worked cases
in ``tests/test_vision_projection.py``, which are pinned tighter than the
corrections are large.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from whiteout.geo import WGS84_A, WGS84_E2, WGS84_F, GeoError, GeoPoint, enu_to_geodetic
from whiteout.vision.camera import CameraModel, VisionError

__all__ = [
    "EARTH_MEAN_RADIUS_M",
    "MAX_FLAT_PLANE_RANGE_ERROR",
    "WGS84_A",
    "WGS84_E2",
    "WGS84_F",
    "CameraPose",
    "GeoPoint",
    "ProjectionError",
    "Vec3",
    "camera_basis",
    "enu_to_geodetic",
    "horizon_range_m",
    "max_flat_plane_range_m",
    "pixel_ray_enu",
    "project_pixel_to_ground",
]

#: A vector in the local ENU frame: ``(east, north, up)``, metres or unitless.
Vec3 = tuple[float, float, float]

#: WGS-84 mean radius ``R1 = (2a + b) / 3``, metres. Used only for the two
#: curvature quantities in the module docstring — the drop of the sea below
#: the tangent plane and the horizon that follows from it — where the
#: difference between a mean sphere and the ellipsoid is far below the error
#: being bounded.
EARTH_MEAN_RADIUS_M = (2.0 * WGS84_A + WGS84_A * (1.0 - WGS84_F)) / 3.0

#: How much of the range the flat water plane's own curvature error may reach
#: before :func:`project_pixel_to_ground` refuses the fix. See the module
#: docstring: that error is ``(d / d_horizon)²`` of the range, so 0.10 puts
#: the default bound at ``d_horizon / sqrt(10)``.
MAX_FLAT_PLANE_RANGE_ERROR = 0.10


class ProjectionError(VisionError):
    """A pixel has no well-defined point on the water plane.

    Raised when the ray does not descend — the pixel is on or above the
    horizon — when it descends so shallowly that the intersection is past the
    range the flat water plane can be trusted to, when the camera is not
    above the plane it is being projected onto, and when the pixel is outside
    the frame. It also wraps a :class:`whiteout.geo.GeoError` from the final
    conversion, so that this stays the one type a caller of
    :func:`project_pixel_to_ground` has to catch. The rest are real
    conditions in the arena rather than programming
    errors: a fixed-wing at 500 m with a 42.6° vertical field of view sees
    the horizon in the top of almost every frame, and a tower scanning at
    ``mode scan`` sweeps through it. The caller's response is to discard the
    detection, not to crash, so this is a catchable error with a message that
    names the cause.

    **The shallow-ray case is the one that matters.** ``ray[2] < 0`` alone is
    not enough: a ray one pixel below the horizon still descends, and
    ``-h / ray[2]`` is unbounded, so it returns an ordinary-looking
    :class:`GeoPoint` tens of kilometres away. Refusing is the whole point —
    that point would be submitted to the tracks API as a detection.
    """


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

    def __post_init__(self) -> None:
        """Refuse a pose with a non-finite field.

        A dropped or unset MAVLink field arrives as a NaN, and every step
        after this one propagates it silently: ``camera_basis`` returns NaN
        axes, the intersection returns a NaN scale, and
        :class:`GeoPoint` ends up with ``lat_deg=nan``, which is not valid
        JSON in a ``POST /api/tracks`` body. Catching it at construction is
        the one place the caller still knows which message the value came
        from.
        """
        for field_name, value in (
            ("lat_deg", self.lat_deg),
            ("lon_deg", self.lon_deg),
            ("alt_m", self.alt_m),
            ("yaw_deg", self.yaw_deg),
            ("pitch_deg", self.pitch_deg),
            ("roll_deg", self.roll_deg),
        ):
            if not math.isfinite(value):
                raise ProjectionError(f"pose {field_name} must be finite, got {value!r}")


def horizon_range_m(height_m: float) -> float:
    """Ground range to the geometric horizon from ``height_m``, ``sqrt(2 R h)``.

    The hard limit of the model: past it the ray does not meet the real sea
    at all, whatever the flat plane says. 27.6 km from a 60 m tower, 39.1 km
    from a 120 m quadcopter, 79.8 km from a fixed-wing at 500 m. Refraction
    would extend it by roughly 8%; that is ignored, which makes this bound
    the conservative one.
    """
    if not (math.isfinite(height_m) and height_m > 0.0):
        raise ProjectionError(f"horizon needs a positive finite height, got {height_m!r}")
    return math.sqrt(2.0 * EARTH_MEAN_RADIUS_M * height_m)


def max_flat_plane_range_m(height_m: float, tolerance: float = MAX_FLAT_PLANE_RANGE_ERROR) -> float:
    """Ground range at which the flat plane's error reaches ``tolerance`` of the range.

    ``sqrt(tolerance) × horizon_range_m(height_m)``, from the module
    docstring's ``Δd/d ≈ (d / d_horizon)²``. At the default tolerance of 0.10
    that is 8.7 km from a 60 m tower, 12.4 km from a 120 m quadcopter and
    25.2 km from a fixed-wing at 500 m.

    This is the bound :func:`project_pixel_to_ground` applies when the caller
    names none, and it is deliberately a *model* bound rather than an arena
    one: it says where this module's own arithmetic stops meaning anything,
    not where the boat can be. A caller that knows the arena should pass a
    tighter ``max_range_m`` — Bellot Strait is 2 km across, so a
    cross-channel fix beyond about 3 km is impossible for reasons this
    module cannot see.
    """
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise ProjectionError(f"tolerance must be positive and finite, got {tolerance!r}")
    return math.sqrt(tolerance) * horizon_range_m(height_m)


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


def project_pixel_to_ground(
    camera: CameraModel,
    pose: CameraPose,
    px: float,
    py: float,
    *,
    ground_alt_m: float = 0.0,
    max_range_m: float | None = None,
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
    :param max_range_m: the furthest ground range this function will return a
        fix at. ``None``, the default, is
        :func:`max_flat_plane_range_m` of the camera's height above the
        plane — the range at which the flat plane's own curvature error
        reaches :data:`MAX_FLAT_PLANE_RANGE_ERROR` of the range itself.
    :raises ProjectionError: if the pixel is outside the frame, if the camera
        is not above the plane, if the ray does not descend to it, or if the
        intersection is beyond ``max_range_m``.

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

    **Descending is not sufficient, which is why there is a range bound.**
    :math:`t = -h/v_U` is unbounded as :math:`v_U \\to 0^-`, so a ray one
    pixel below the horizon still satisfies :math:`v_U < 0` and returns a
    perfectly ordinary :class:`GeoPoint`. A tower camera at 60 m, pitched
    18° down — 0.1° of pitch per pixel row — projects row 2 of its frame to
    25.0 km and row 1 to 78.4 km. The strait is 25 km end to end. Without a
    bound, one hot pixel of ice glare near the top of a routine frame becomes
    a track submission on the far side of the arena, and the *accuracy*
    criterion reads it.

    **Why this bound.** Two candidates are defensible. The geometric horizon,
    :func:`horizon_range_m`, is the hard one: past it the ray does not meet
    the sea at all. But at the horizon the flat plane's range error is
    already 100% of the range — the error is exactly
    :math:`(d/d_\\text{horizon})^2` — so a fix taken there is not wrong by a
    little, it is meaningless. The default is therefore the *accuracy* bound
    rather than the geometric one: the range at which that error reaches
    10%, which is :math:`d_\\text{horizon}/\\sqrt{10}`, or 8.7 km from a 60 m
    tower. It rejects both rows above. An arena-aware caller should pass
    something tighter still; Bellot Strait is 2 km across.
    """
    if not (0.0 <= px <= camera.width and 0.0 <= py <= camera.height):
        raise ProjectionError(
            f"pixel ({px!r}, {py!r}) is outside {camera.name}'s "
            f"{camera.width}×{camera.height} frame"
        )
    if not math.isfinite(ground_alt_m):
        raise ProjectionError(f"ground_alt_m must be finite, got {ground_alt_m!r}")
    height_above_plane = pose.alt_m - ground_alt_m
    if height_above_plane <= 0.0:
        raise ProjectionError(
            f"camera is not above the plane: altitude {pose.alt_m!r} m against a plane at "
            f"{ground_alt_m!r} m"
        )
    limit = (
        max_flat_plane_range_m(height_above_plane) if max_range_m is None else float(max_range_m)
    )
    if not (math.isfinite(limit) and limit > 0.0):
        raise ProjectionError(f"max_range_m must be positive and finite, got {max_range_m!r}")
    ray = pixel_ray_enu(camera, pose, px, py)
    if ray[2] >= 0.0:
        raise ProjectionError(
            f"pixel ({px!r}, {py!r}) does not descend to the plane: it is on or above the "
            f"horizon for a camera at yaw {pose.yaw_deg!r}, pitch {pose.pitch_deg!r}, "
            f"roll {pose.roll_deg!r}"
        )
    scale = -height_above_plane / ray[2]
    east_m = scale * ray[0]
    north_m = scale * ray[1]
    ground_range_m = math.hypot(east_m, north_m)
    if ground_range_m > limit:
        raise ProjectionError(
            f"pixel ({px!r}, {py!r}) grazes the horizon: it projects to {ground_range_m:.0f} m, "
            f"past the {limit:.0f} m this camera's {height_above_plane!r} m height supports "
            f"(geometric horizon {horizon_range_m(height_above_plane):.0f} m)"
        )
    try:
        return enu_to_geodetic(pose.lat_deg, pose.lon_deg, east_m, north_m)
    except GeoError as exc:
        # The converter is whiteout.geo's, so it raises whiteout.geo's error;
        # this function documents one catchable failure and callers discard
        # the detection on it. A pose whose lat/lon is out of range or at a
        # pole reaches here because CameraPose checks only finiteness.
        raise ProjectionError(f"pixel ({px!r}, {py!r}) has no geodetic fix: {exc}") from exc
