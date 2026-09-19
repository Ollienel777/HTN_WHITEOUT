"""The water of Bellot Strait, as a centreline and a half-width.

``hackathon/ARENA.md`` §2: the arena is **25 km long and 2 km wide**, a narrow
water channel flanked by rocky shores, and the target is a boat. So the
searchable set is not an area with a hole in it — it is a *ribbon*, and this
module is the ribbon.

Two coordinates describe a point on it:

``s``
    Arc length along the centreline from the western end, metres, in
    ``[0, length_m]``.
``w``
    Signed offset across the centreline, metres, **positive to the left of the
    direction of travel** (north-ish here, since the strait runs roughly
    East–West).

A point is **on water** when ``abs(w) <= half_width_at(s)``. That one
inequality is the shoreline, and it is why the belief grid can enforce "mass
stays on water" with a mask rather than a terrain model: ``ARENA.md`` §7 lists
the terrain generator as dead work, and there is no heightfield to consult.

Frames, and the conversion this module owns
-------------------------------------------

**Positions cross this module's boundary as lat/lon**, because that is what
the arena speaks in both directions: ``GLOBAL_POSITION_INT`` per asset,
``whiteout/vision/`` for a detected pixel, and ``POST /api/tracks`` for the
answer that is scored (``ARENA.md`` §5). ``(s, w)`` is an internal
parameterisation, never a wire format.

The geodetic conversion here is a **local tangent plane** about
:data:`REFERENCE_LAT_DEG` / :data:`REFERENCE_LON_DEG`: North offsets divided by
the meridional radius of curvature ``M``, East offsets by ``N cos φ``, both
evaluated at the reference latitude. Over a 25 km × 2 km box that is good to
well under a metre along the meridian and a couple of metres at the East ends,
which is far inside the metres-to-tens-of-metres error of a camera fix.

**This conversion is deliberately a private helper, and it is duplicated.**
``whiteout/vision/projection.py`` (PR #72) carries the same tangent-plane step
in the opposite direction, and issue #73 is deciding the repository's single
frame and the one module that converts. When #73 lands, :func:`enu_from_geodetic`
and :func:`geodetic_from_enu` should collapse into it and this module should
import them. Nothing else here changes: the ribbon is defined by a polyline of
lat/lon vertices whichever way the conversion is spelled.

Where the default polyline comes from
--------------------------------------

:data:`DEFAULT_STRAIT` is **a parameterisation, not a survey**. Its numbers are
exactly the facts in ``ARENA.md`` §2 and §5 — 25 km long, about 2 km wide, at
roughly 71.99 N, −94.84 W, long axis running East–West, with a narrows in the
middle — laid onto four vertices. It reproduces the one independent datum
available: the tracks-API example position ``71.9965, -94.8448`` (``ARENA.md``
§5) falls about 90 m off this centreline, comfortably inside the water.

It is data, and replacing it is a one-line change: every function here takes a
:class:`StraitGeometry` built from any polyline, so a surveyed shoreline read
off the sim (``ARENA.md`` §3: "click anywhere to get x, y, z and lat/lon")
drops straight in.

**The one constraint on a polyline** is that its bends must be gentle relative
to its half-width: ``(s, w)`` stops being one-to-one once the half-width
exceeds the radius of curvature, because the inside of a tight bend folds over
itself. The default turns about 3.6° over 25 km, so its radius of curvature is
hundreds of kilometres against a half-width of one, and the fold is nowhere
near.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_STRAIT",
    "REFERENCE_LAT_DEG",
    "REFERENCE_LON_DEG",
    "WGS84_A",
    "WGS84_E2",
    "WGS84_F",
    "ChannelPoint",
    "ChannelVertex",
    "GeometryError",
    "StraitGeometry",
    "enu_from_geodetic",
    "geodetic_from_enu",
]

#: WGS-84 semi-major axis, metres.
WGS84_A = 6378137.0

#: WGS-84 flattening.
WGS84_F = 1.0 / 298.257223563

#: WGS-84 first eccentricity squared, ``f (2 - f)``.
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

#: Origin of the local tangent plane: the centre of the arena (``ARENA.md`` §2).
REFERENCE_LAT_DEG = 71.99
REFERENCE_LON_DEG = -94.84


class GeometryError(ValueError):
    """A polyline or a position does not describe a point on the strait."""


def _radii_of_curvature(lat_deg: float) -> tuple[float, float]:
    """``(M, N)`` at ``lat_deg``: meridional and prime-vertical radii, metres."""
    sin_lat = math.sin(math.radians(lat_deg))
    denominator = 1.0 - WGS84_E2 * sin_lat * sin_lat
    prime_vertical = WGS84_A / math.sqrt(denominator)
    meridional = WGS84_A * (1.0 - WGS84_E2) / (denominator * math.sqrt(denominator))
    return meridional, prime_vertical


def enu_from_geodetic(
    lat_deg: float,
    lon_deg: float,
    ref_lat_deg: float = REFERENCE_LAT_DEG,
    ref_lon_deg: float = REFERENCE_LON_DEG,
) -> tuple[float, float]:
    """``(east, north)`` metres of a geodetic position about a reference point.

    First-order tangent plane: ``north = M Δφ`` and ``east = N cos φ Δλ``, with
    ``M`` and ``N`` evaluated at ``ref_lat_deg``. See the module docstring for
    why this lives here and where it should end up (issue #73).
    """
    meridional, prime_vertical = _radii_of_curvature(ref_lat_deg)
    delta_lat = math.radians(lat_deg - ref_lat_deg)
    delta_lon = math.radians(_wrap_longitude(lon_deg - ref_lon_deg))
    east = prime_vertical * math.cos(math.radians(ref_lat_deg)) * delta_lon
    north = meridional * delta_lat
    return east, north


def geodetic_from_enu(
    east_m: float,
    north_m: float,
    ref_lat_deg: float = REFERENCE_LAT_DEG,
    ref_lon_deg: float = REFERENCE_LON_DEG,
) -> tuple[float, float]:
    """``(lat_deg, lon_deg)`` of a local offset — the inverse of :func:`enu_from_geodetic`."""
    meridional, prime_vertical = _radii_of_curvature(ref_lat_deg)
    east_radius = prime_vertical * math.cos(math.radians(ref_lat_deg))
    lat_deg = ref_lat_deg + math.degrees(north_m / meridional)
    lon_deg = ref_lon_deg + math.degrees(east_m / east_radius)
    return lat_deg, _wrap_longitude(lon_deg)


def _wrap_longitude(lon_deg: float) -> float:
    """Fold a longitude difference into ``(-180, 180]``.

    The arena is 25 km wide and nowhere near the antimeridian, so this never
    fires in the run. It is here because a caller that passes a longitude in
    ``[0, 360)`` would otherwise be handed a position 20 000 km away with no
    complaint, and silently wrong is the failure mode this whole module is
    built to avoid.
    """
    wrapped = math.fmod(lon_deg + 180.0, 360.0)
    if wrapped <= 0.0:
        wrapped += 360.0
    return wrapped - 180.0


@dataclass(frozen=True, slots=True)
class ChannelVertex:
    """One vertex of the centreline, with the channel's half-width there."""

    lat_deg: float
    lon_deg: float
    half_width_m: float

    def __post_init__(self) -> None:
        for name in ("lat_deg", "lon_deg", "half_width_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise GeometryError(f"vertex {name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.half_width_m <= 0.0:
            raise GeometryError(f"vertex half_width_m must be positive, got {self.half_width_m}")


@dataclass(frozen=True, slots=True)
class ChannelPoint:
    """A position in channel coordinates: ``s`` along, ``w`` across, metres."""

    s_m: float
    w_m: float


@dataclass(frozen=True)
class StraitGeometry:
    """A water channel: a centreline polyline in lat/lon, plus a half-width.

    Construct it from at least two :class:`ChannelVertex`; everything else is
    derived once, on the local tangent plane, and cached on the instance.
    """

    vertices: tuple[ChannelVertex, ...]
    _east: tuple[float, ...] = field(init=False, repr=False)
    _north: tuple[float, ...] = field(init=False, repr=False)
    _cumulative: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.vertices) < 2:
            raise GeometryError("a centreline needs at least two vertices")
        east: list[float] = []
        north: list[float] = []
        for vertex in self.vertices:
            point = enu_from_geodetic(vertex.lat_deg, vertex.lon_deg)
            east.append(point[0])
            north.append(point[1])
        cumulative = [0.0]
        for index in range(1, len(east)):
            step = math.hypot(east[index] - east[index - 1], north[index] - north[index - 1])
            if step <= 0.0:
                raise GeometryError(f"centreline vertices {index - 1} and {index} coincide")
            cumulative.append(cumulative[-1] + step)
        object.__setattr__(self, "_east", tuple(east))
        object.__setattr__(self, "_north", tuple(north))
        object.__setattr__(self, "_cumulative", tuple(cumulative))

    # -- shape -------------------------------------------------------------

    @property
    def length_m(self) -> float:
        """Centreline length, metres."""
        return self._cumulative[-1]

    @property
    def max_half_width_m(self) -> float:
        """The widest half-width anywhere, metres — the grid's across extent."""
        return max(vertex.half_width_m for vertex in self.vertices)

    def half_width_at(self, s_m: float) -> float:
        """Half-width at arc length ``s_m``, linearly interpolated between vertices.

        Clamped at both ends, so a caller past the mouth of the strait gets the
        end vertex's width rather than an extrapolated or negative one.
        """
        if s_m <= 0.0:
            return self.vertices[0].half_width_m
        if s_m >= self.length_m:
            return self.vertices[-1].half_width_m
        index = self._segment_index(s_m)
        start = self._cumulative[index]
        span = self._cumulative[index + 1] - start
        fraction = (s_m - start) / span
        low = self.vertices[index].half_width_m
        high = self.vertices[index + 1].half_width_m
        return low + fraction * (high - low)

    def _segment_index(self, s_m: float) -> int:
        """Index of the segment containing ``s_m`` (clamped to a real segment)."""
        last = len(self._cumulative) - 2
        for index in range(last + 1):
            if s_m < self._cumulative[index + 1]:
                return index
        return last

    # -- the two conversions ----------------------------------------------

    def to_channel(self, lat_deg: float, lon_deg: float) -> ChannelPoint:
        """Project a geodetic position onto the centreline.

        Takes the nearest point of the polyline (each segment clamped to its
        own extent, so the answer is never off the end of a segment) and
        returns its arc length, with ``w`` the signed perpendicular offset from
        that segment's *line*: positive to the left of the direction of travel.
        """
        east, north = enu_from_geodetic(lat_deg, lon_deg)
        best_distance = math.inf
        best_s = 0.0
        best_w = 0.0
        for index in range(len(self._east) - 1):
            ax = self._east[index]
            ay = self._north[index]
            dx = self._east[index + 1] - ax
            dy = self._north[index + 1] - ay
            span = math.hypot(dx, dy)
            ux = dx / span
            uy = dy / span
            along = (east - ax) * ux + (north - ay) * uy
            clamped = min(max(along, 0.0), span)
            closest_x = ax + clamped * ux
            closest_y = ay + clamped * uy
            distance = math.hypot(east - closest_x, north - closest_y)
            if distance < best_distance:
                best_distance = distance
                best_s = self._cumulative[index] + clamped
                # Left-positive: the 2-D cross product of the unit direction
                # with the offset vector.
                best_w = ux * (north - ay) - uy * (east - ax)
        return ChannelPoint(s_m=best_s, w_m=best_w)

    def to_geodetic(self, point: ChannelPoint) -> tuple[float, float]:
        """``(lat_deg, lon_deg)`` of a channel coordinate — inverse of :meth:`to_channel`.

        Exact inverse for any point whose nearest centreline point is interior
        to a segment; at a convex bend the two parameterisations differ by the
        wedge the bend opens, which is bounded by the turn angle times ``w``
        and is metres for the default geometry.
        """
        s_m = min(max(point.s_m, 0.0), self.length_m)
        index = self._segment_index(s_m)
        ax = self._east[index]
        ay = self._north[index]
        dx = self._east[index + 1] - ax
        dy = self._north[index + 1] - ay
        span = math.hypot(dx, dy)
        ux = dx / span
        uy = dy / span
        along = s_m - self._cumulative[index]
        # Left normal of (ux, uy) is (-uy, ux).
        east = ax + along * ux - point.w_m * uy
        north = ay + along * uy + point.w_m * ux
        return geodetic_from_enu(east, north)

    def is_water(self, lat_deg: float, lon_deg: float) -> bool:
        """Is this position inside the channel — on water the vessel can occupy?"""
        point = self.to_channel(lat_deg, lon_deg)
        if point.s_m <= 0.0 or point.s_m >= self.length_m:
            return False
        return abs(point.w_m) <= self.half_width_at(point.s_m)


#: Bellot Strait, as a four-vertex ribbon. A parameterisation of ``ARENA.md``
#: §2's "25 km × 2 km … near Fort Ross", **not a survey** — see the module
#: docstring. The half-widths taper from 1 km at each mouth to 520 m at the
#: narrows, which is what makes the shoreline a real constraint on diffusion
#: rather than a box the grid never touches.
DEFAULT_STRAIT = StraitGeometry(
    vertices=(
        ChannelVertex(lat_deg=71.9860, lon_deg=-95.2023, half_width_m=1000.0),
        ChannelVertex(lat_deg=71.9930, lon_deg=-95.0000, half_width_m=900.0),
        ChannelVertex(lat_deg=71.9975, lon_deg=-94.7400, half_width_m=520.0),
        ChannelVertex(lat_deg=72.0000, lon_deg=-94.4777, half_width_m=1000.0),
    )
)
