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

Frames, and the conversion this module does *not* own
------------------------------------------------------

**Positions cross this module's boundary as lat/lon**, because that is what
the arena speaks in both directions: ``GLOBAL_POSITION_INT`` per asset,
``whiteout/vision/`` for a detected pixel, and ``POST /api/tracks`` for the
answer that is scored (``ARENA.md`` §5). ``(s, w)`` is an internal
parameterisation, never a wire format.

The geodetic conversion is :mod:`whiteout.geo`'s, and this module imports it:
:func:`~whiteout.geo.geodetic_to_local` and
:func:`~whiteout.geo.local_to_geodetic` about :data:`~whiteout.geo.ARENA_ORIGIN`,
which is the same 71.99 N, −94.84 W this module used to spell for itself. That
is a local tangent plane — North offsets divided by the meridional radius of
curvature ``M``, East offsets by ``N cos φ``, both evaluated at the origin's
latitude — and ``whiteout/geo.py``'s docstring carries the measured error
table for it.

Until #73 landed this module carried a **private copy** of that step, with its
own ellipsoid constants. The copy has been deleted (#85): a second converter
is a second answer to the question ``ARENA.md`` §5 scores, and
``tests/test_geo.py``'s three AST guards now forbid one anywhere but
``whiteout/geo.py``. Nothing about the ribbon changed with it — it is a
polyline of lat/lon vertices whichever way the conversion is spelled.

**The two planes were already the same plane**, which is why no figure stated
in this module or in ``grid.py`` moved. Same constants, same origin, and the
two spellings of the meridional radius — ``A(1 - e²)/(w √w)`` here against
``A(1 - e²)/w^1.5`` there — evaluate to the same float at 71.99 N, bit for
bit. Every quantity below was re-measured across the migration; the largest
change anywhere was **4.9 × 10⁻¹⁰ m**, half a nanometre, and it comes from a
deleted helper rather than from the geodesy. The private copy folded a
longitude difference through ``fmod(Δλ + 180, 360) - 180``, which is not quite
the identity in the last bit, and it moved two of the four default vertices by
that much. ``whiteout.geo`` does not fold, because it does not need to.

**The one behavioural difference** is at the edge of the geodetic range rather
than inside the arena. That fold also meant the private copy accepted a
longitude in ``[0, 360)``, and it accepted any float as a latitude;
:class:`whiteout.geo.GeoPoint` instead *rejects* a position outside
``[-90, 90]`` / ``[-180, 180]`` with :class:`~whiteout.geo.GeoError`. Nothing
in this arena is near that edge — not even the 200 km-beyond-the-mouth
positions ``tests/test_belief_grid.py`` builds, which reach 101° W — so the
change is a refusal where there was silence, on inputs no caller here has.

Where the default polyline comes from
--------------------------------------

:data:`DEFAULT_STRAIT` is **a parameterisation, not a survey**. Its numbers are
exactly the facts in ``ARENA.md`` §2 and §5 — a 25 km × 2 km extent at roughly
71.99 N, −94.84 W, long axis running East–West, with a narrows in the middle —
laid onto four vertices. That 2 km is the **extent**, and ARENA describes the
channel as "flanked by rocky shores and ridges", so the half-widths here model
the *water* inside it; see :data:`DEFAULT_STRAIT`. It reproduces the one
independent datum
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

from whiteout.geo import ARENA_ORIGIN, GeoPoint, LocalPoint, geodetic_to_local, local_to_geodetic

__all__ = [
    "DEFAULT_STRAIT",
    "ChannelPoint",
    "ChannelVertex",
    "GeometryError",
    "StraitGeometry",
]


class GeometryError(ValueError):
    """A polyline or a position does not describe a point on the strait."""


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
            offset = geodetic_to_local(ARENA_ORIGIN, GeoPoint(vertex.lat_deg, vertex.lon_deg))
            east.append(offset.east_m)
            north.append(offset.north_m)
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
        offset = geodetic_to_local(ARENA_ORIGIN, GeoPoint(lat_deg, lon_deg))
        east = offset.east_m
        north = offset.north_m
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

    def to_position(self, point: ChannelPoint) -> tuple[float, float]:
        """``(lat_deg, lon_deg)`` of a channel coordinate — inverse of :meth:`to_channel`.

        Exact inverse for any point whose nearest centreline point is interior
        to a segment; at a convex bend the two parameterisations differ by the
        wedge the bend opens, which is bounded by the turn angle times ``w``
        and is metres for the default geometry.

        Named ``to_position`` rather than ``to_geodetic`` because
        ``tests/test_geo.py`` reserves every name that reads as a frame
        conversion for ``whiteout/geo.py``, and this method is not one: the
        conversion it ends with is :func:`whiteout.geo.local_to_geodetic`'s.
        The pair is spelled ``to_channel`` / ``to_position`` — into the
        ribbon's coordinates and back out to the frame of record.
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
        place = local_to_geodetic(ARENA_ORIGIN, LocalPoint(east, north))
        return place.lat_deg, place.lon_deg

    def is_water(self, lat_deg: float, lon_deg: float) -> bool:
        """Is this position inside the channel — on water the vessel can occupy?"""
        point = self.to_channel(lat_deg, lon_deg)
        if point.s_m <= 0.0 or point.s_m >= self.length_m:
            return False
        return abs(point.w_m) <= self.half_width_at(point.s_m)


#: Bellot Strait, as a four-vertex ribbon. A parameterisation of ``ARENA.md``
#: §2's "25 km × 2 km … near Fort Ross", **not a survey** — see the module
#: docstring.
#:
#: **The 2 km in ARENA is the arena's extent, not the channel's water.** The
#: same sentence that gives "Extent: 25 km × 2 km" describes "a narrow water
#: channel with ice floes, **flanked by rocky shores and ridges**", and those
#: flanking shores are inside that 2 km. So the published figure is a bounding
#: box and the modelled quantity here is the water width: these half-widths
#: taper from 1 km at each mouth to 520 m at the narrows, a mean water width
#: of 1.59 km and 1.04 km at the choke — leaving about 410 m of rocky shore
#: across the box, which is exactly where a boat cannot go.
#:
#: What it costs is a discretisation, not unreachable water. ``_water`` is
#: boolean, so a position outside the ribbon is not merely unlikely — belief
#: there is identically zero, ``probability_at`` returns 0 and ``peak()`` can
#: never report it — and the outermost *water cell centre* is inboard of the
#: shore by up to half a cell. At the narrowest row (``s`` ≈ 16 035 m,
#: half-width 520.9 m) that outermost centre sits at ``|w|`` = 450 m, so a
#: fix reported at ``|w|`` = 600 m is pulled 150 m toward the centreline in
#: the position submitted to ``POST /api/tracks``, and one at the very edge of
#: the arena's extent, ``|w|`` = 1000 m, is pulled 550 m.
#:
#: Clicking the real shoreline off the sim (``ARENA.md`` §3) stays the right
#: way to replace these four vertices, and every function takes any polyline.
#:
#: The taper is *not* here to make the shoreline bite in the tests, whatever
#: an earlier version of this comment said: every shoreline test builds its
#: own ``choked_channel()`` with a matched control, and only
#: ``test_the_default_mask_has_land_in_it`` touches this geometry's mask.
DEFAULT_STRAIT = StraitGeometry(
    vertices=(
        ChannelVertex(lat_deg=71.9860, lon_deg=-95.2023, half_width_m=1000.0),
        ChannelVertex(lat_deg=71.9930, lon_deg=-95.0000, half_width_m=900.0),
        ChannelVertex(lat_deg=71.9975, lon_deg=-94.7400, half_width_m=520.0),
        ChannelVertex(lat_deg=72.0000, lon_deg=-94.4777, half_width_m=1000.0),
    )
)
