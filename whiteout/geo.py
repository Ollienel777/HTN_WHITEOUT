"""The one coordinate frame, and the one place a conversion happens.

``hackathon/SPEC.md`` §5 "Conventions" names the decision; this module is it.

**The frame of record is geodetic WGS-84 latitude and longitude, in degrees.**
Every record in the episode log, every field that crosses the transport seam
and every body posted to the tracks API is lat/lon. There is no second frame
of record and no "internal" one.

**A local metric frame exists only as a projection, for drawing and for
geometry.** It is East–North in metres about a stated origin, it is never
written to the episode log, and it is never posted anywhere. Anything that
wants metres — a viewer's canvas, a range computation, a footprint radius —
asks for the projection here rather than inventing one.

Why the decision is worth a module
----------------------------------

``ARENA.md`` §5 measures *accuracy* on the ``lat``/``lon`` that arrives at
``POST /api/tracks``. At Bellot Strait's 71.99° N a degree of longitude spans
about 0.31 of a degree of latitude, so two callers that disagree about the
frame do not produce an obviously broken answer: they produce a plausible one,
a few kilometres out, which survives to the judged run. Three consumers need
this conversion at once — the detection emitter, the tracks poster and the
viewer — and the cheapest way for them to agree is for there to be nothing to
disagree about.

Two guards make the rule mechanical rather than aspirational:

* :class:`~whiteout.types.RecordError` on a record whose ``lat``/``lon`` is
  out of geodetic range. A swapped pair at the strait is ``lat=-94.84``, which
  is not a latitude, so the classic mix-up fails loudly at construction.
* ``tests/test_geo.py`` walks ``whiteout/`` with :mod:`ast` and fails if any
  module other than this one defines the ellipsoid constants or its own
  geodetic conversion.

The tangent-plane approximation
-------------------------------

:func:`enu_to_geodetic` divides the North offset by the meridional radius of
curvature :math:`M` and the East offset by :math:`N \\cos\\varphi`, both
evaluated at the *origin's* latitude:

.. math::

    N = \\frac{a}{\\sqrt{1 - e^2 \\sin^2\\varphi}}
    \\qquad
    M = \\frac{a(1 - e^2)}{(1 - e^2 \\sin^2\\varphi)^{3/2}}

    \\Delta\\varphi = \\frac{\\text{north}}{M(\\varphi)}
    \\qquad
    \\Delta\\lambda = \\frac{\\text{east}}{N(\\varphi)\\cos\\varphi}

:func:`geodetic_to_local` is the algebraic inverse of that, with the radii
taken at the same origin latitude, so the pair round-trips to floating-point
noise rather than to the approximation's error. The approximation itself is a
first-order expansion of the geodesic whose error grows as the square of the
range: under a centimetre at 3 km and under a decimetre at 10 km at this
latitude. The strait is 25 km × 2 km, so a corner-to-corner projection is
still well inside the pose error, and a full geodesic solver would buy
nothing.

**Altitude is not this module's.** Nothing here reads or returns one. The
arena reports altitude in more than one datum and settling which is issue #76,
which needs a measurement against the live arena; any caller that needs a
height passes it explicitly and states its datum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ARENA_ORIGIN",
    "WGS84_A",
    "WGS84_E2",
    "WGS84_F",
    "GeoError",
    "GeoPoint",
    "LocalPoint",
    "check_geodetic",
    "enu_to_geodetic",
    "geodetic_to_local",
    "local_to_geodetic",
    "radii_of_curvature",
]

#: WGS-84 semi-major axis, metres.
WGS84_A = 6378137.0

#: WGS-84 flattening.
WGS84_F = 1.0 / 298.257223563

#: WGS-84 first eccentricity squared, ``f (2 - f)``.
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

#: Below this East radius, in metres per radian, a position is at a pole and
#: has no usable local frame. One metre per radian is about six metres of
#: East around the *whole* parallel, so nothing in an arena is near it —
#: while a bare ``> 0`` test is not enough, because ``cos(radians(90.0))`` is
#: 6.1e-17 rather than zero and would hand the caller a frame in which a
#: metre East is ninety thousand degrees.
_POLE_EAST_RADIUS_M = 1.0


class GeoError(ValueError):
    """A position is not a geodetic position, or has no local projection.

    Raised for a latitude outside ``[-90, 90]``, a longitude outside
    ``[-180, 180]``, a non-finite coordinate, and for an origin at a pole,
    where the East radius vanishes and longitude carries no metres.
    """


def check_geodetic(lat_deg: float, lon_deg: float) -> None:
    """Raise :class:`GeoError` unless this is a geodetic position in degrees.

    Separate from :class:`GeoPoint` so that record types carrying flat
    ``lat``/``lon`` floats — which is most of ``whiteout.types`` — validate
    against the same rule rather than a second copy of it.
    """
    if not math.isfinite(lat_deg) or not -90.0 <= lat_deg <= 90.0:
        raise GeoError(f"latitude {lat_deg!r} is not a latitude in degrees, [-90, 90]")
    if not math.isfinite(lon_deg) or not -180.0 <= lon_deg <= 180.0:
        raise GeoError(f"longitude {lon_deg!r} is not a longitude in degrees, [-180, 180]")


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """A geodetic position: the frame of record, and what the API takes.

    ``ARENA.md`` §5's ``POST /api/tracks`` body is ``{"name": …, "lat": …,
    "lon": …}`` in degrees, so this type is deliberately two floats and not a
    third — the altitude datum is unsettled (#76) and carrying one here would
    invite someone to submit it.

    Construction validates the range, which is the cheap half of catching a
    frame mix-up: at Bellot Strait a swapped pair reads ``lat=-94.84``, and
    that is not a latitude.
    """

    lat_deg: float
    lon_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lat_deg", float(self.lat_deg))
        object.__setattr__(self, "lon_deg", float(self.lon_deg))
        check_geodetic(self.lat_deg, self.lon_deg)


@dataclass(frozen=True, slots=True)
class LocalPoint:
    """An East–North offset in metres from a stated origin.

    **Not a position.** It means nothing without the origin it was taken
    about, it is never written to the episode log and it is never posted. It
    exists so that a canvas and a range computation have metres to work in.
    """

    east_m: float
    north_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "east_m", float(self.east_m))
        object.__setattr__(self, "north_m", float(self.north_m))


#: The arena's reference position: Bellot Strait, ``ARENA.md`` §2 and §7b.5.
#: It is the origin the viewer draws about and the point the stub fleet is
#: placed around. It is a *convention for local projections*, not a frame:
#: nothing derives a posted lat/lon from it.
ARENA_ORIGIN = GeoPoint(71.99, -94.84)


def radii_of_curvature(lat_deg: float) -> tuple[float, float]:
    """Return ``(meridional, east)`` radii in metres at ``lat_deg``.

    ``meridional`` is :math:`M`, the metres per radian of latitude.
    ``east`` is :math:`N \\cos\\varphi`, the metres per radian of longitude —
    the factor that collapses toward the pole and that makes a frame mix-up at
    71.99° N look plausible instead of absurd.

    Both directions of the conversion read their scale from here, so the
    forward and inverse cannot drift apart.
    """
    check_geodetic(lat_deg, 0.0)
    phi = math.radians(lat_deg)
    sin_phi = math.sin(phi)
    w = 1.0 - WGS84_E2 * sin_phi * sin_phi
    prime_vertical = WGS84_A / math.sqrt(w)
    meridional = WGS84_A * (1.0 - WGS84_E2) / (w**1.5)
    east = prime_vertical * math.cos(phi)
    if east <= _POLE_EAST_RADIUS_M:
        raise GeoError(
            f"no local frame at latitude {lat_deg!r}: the East radius vanishes at the pole"
        )
    return meridional, east


def enu_to_geodetic(lat_deg: float, lon_deg: float, east_m: float, north_m: float) -> GeoPoint:
    """Offset a geodetic position by a local East–North displacement.

    The positional form, taken by the camera projection, which has the
    camera's own lat/lon to hand and no ``GeoPoint`` for it. See
    :func:`local_to_geodetic` for the same step over the module's own types.
    """
    meridional, east = radii_of_curvature(lat_deg)
    return GeoPoint(
        lat_deg + math.degrees(north_m / meridional),
        lon_deg + math.degrees(east_m / east),
    )


def local_to_geodetic(origin: GeoPoint, point: LocalPoint) -> GeoPoint:
    """Place a local East–North offset about ``origin``, as a lat/lon."""
    return enu_to_geodetic(origin.lat_deg, origin.lon_deg, point.east_m, point.north_m)


def geodetic_to_local(origin: GeoPoint, point: GeoPoint) -> LocalPoint:
    """Project ``point`` into the local East–North frame about ``origin``.

    The algebraic inverse of :func:`local_to_geodetic`, with the radii taken
    at ``origin``'s latitude in both directions, so the two round-trip to
    floating-point noise.
    """
    meridional, east = radii_of_curvature(origin.lat_deg)
    return LocalPoint(
        math.radians(point.lon_deg - origin.lon_deg) * east,
        math.radians(point.lat_deg - origin.lat_deg) * meridional,
    )
