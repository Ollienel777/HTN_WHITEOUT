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
* ``tests/test_geo.py`` walks **every Python file in the repository** with
  :mod:`ast` and fails if any of them spells an ellipsoid-scale constant,
  defines a name that reads as a frame conversion, or calls trigonometry at
  all. Two files are exempt and no others: this module, and the guard itself,
  which needs both to name the rule. The viewer's JavaScript copy
  (``viz/viewer.js``) is outside :mod:`ast`'s reach, so it is pinned
  numerically instead — ``tests/test_viz_shell.py`` runs its projection under
  ``node`` and compares the metres it returns to
  :func:`geodetic_to_local`'s.

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
noise rather than to the approximation's error. **That round trip is not a
statement about accuracy**: both directions share the same model, so the
model's error cancels and cannot show up there.

How wrong the model is, and where that matters
----------------------------------------------

The two axes are scaled independently and the cross term is dropped, so an
East displacement produces no change in latitude — but on the ellipsoid,
moving East at 71.99° N curves poleward. The error is therefore **not
symmetric**: it is essentially zero along the meridian and quadratic along the
parallel, which is the strait's long axis and the axis ``ARENA.md`` §5 scores
*accuracy* on. A mixed offset is worse than either alone, because the dropped
cross term couples them. Measured against an exact ENU → ECEF → geodetic round trip and
against a Vincenty direct solution, which agree with each other to under a
centimetre (``tests/test_geo.py`` pins these):

===================  ==============
offset from origin   position error
===================  ==============
3 km North           0.004 m
10 km North          0.055 m
500 m East           0.060 m
1 km East            0.240 m
1 km E, 1 km N       0.538 m
2 km East            0.962 m
3 km East            2.164 m
10 km East           24.039 m
12.5 km E, 1 km N    38.033 m
===================  ==============

**The verdict: this is fine for a nearby origin and is not fine for a
strait-wide one.** The error passes 0.1 m at about 650 m East, 1 m at about
2 km East, and reaches 38 m at the far corner of a 25 km × 2 km arena — which
is not inside the pose error, and which would be posted as a confident wrong
answer on the scored axis. So the rule for callers is:

**Project about an origin near the point.** A camera converting its own
detection (#66) has its lat/lon to hand and works at detection range, where
the error is centimetres to a couple of metres; that is what
:func:`enu_to_geodetic`'s positional signature is for. Taking
:data:`ARENA_ORIGIN` and offsetting 10 km down the channel is the case this
table forbids.

If a strait-wide origin is ever needed, the fix is to carry the two
second-order terms rather than to swap in a geodesic solver:

.. math::

    \\Delta\\varphi = \\frac{\\text{north}}{M}
        - \\frac{\\text{east}^2 \\tan\\varphi}{2 M N}
    \\qquad
    \\Delta\\lambda = \\frac{\\text{east}}{N\\cos\\varphi}
        \\left(1 + \\frac{\\text{north}\\tan\\varphi}{M}\\right)

Measured: the first term alone takes 10 km East from 24.039 m to 0.085 m, but
still leaves 5.843 m at the corner — the cross term is what the corner needs,
and the pair together bring it to 0.169 m. That change is deliberately **not**
made here. It is not a one-liner: :func:`geodetic_to_local` would have to
solve for ``north`` and ``east`` rather than divide, and the exact round trip
that :func:`geodetic_to_local` and its test rely on has to move with it. #66
and #67 are being written against this signature now, and the table above
tells them what they are getting; widening it is a separate change.

**Altitude is not this module's.** Nothing here reads or returns one. The
arena reports altitude in more than one datum and settling which is issue #76,
which needs a measurement against the live arena; any caller that needs a
height passes it explicitly and states its datum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "bearing_deg",
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
#: has no usable local frame. A bare ``> 0`` test is not enough, because
#: ``cos(radians(90.0))`` is 6.1e-17 rather than zero and would hand the
#: caller a frame in which one metre East is a hundred billion degrees. But 1.0
#: m/rad is barely better — it is one metre East to 57°, and it only bites
#: within about a millimetre of the pole. A thousand metres per radian is
#: 0.06° of longitude per metre East, still absurd, and it draws the line
#: about 0.01° (roughly a kilometre) from the pole, which is what the guard
#: is meant to mean. Nothing in an arena at 72° N is within nine degrees of
#: it.
_POLE_EAST_RADIUS_M = 1.0e3


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

    The displacements are checked here rather than left to :class:`GeoPoint`.
    A dropped MAVLink field arrives as a NaN, and a NaN ``east_m`` left to
    fall through reaches the range check as ``longitude nan is not a
    longitude`` — which names the output, not the input that was bad, and
    sends the reader to a ``lon_deg`` that was fine.
    """
    for name, value in (("east_m", east_m), ("north_m", north_m)):
        if not math.isfinite(value):
            raise GeoError(f"{name} must be finite, got {value!r}")
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


def bearing_deg(origin: GeoPoint, point: GeoPoint) -> float:
    """Bearing from ``origin`` to ``point``, degrees clockwise from north.

    Lives here because it is a frame question, and this module is the one
    converter in the build. It is defined on top of :func:`geodetic_to_local`
    rather than on a great-circle formula of its own, so it cannot disagree
    with the projection every other range and offset in the build is taken
    in — which is the whole point of there being one converter.

    Over a 6.5 km site the difference between this and a great-circle initial
    bearing is far below the pointing accuracy of anything that consumes it.
    Two coincident points have no bearing between them, and ``atan2(0, 0)``
    is 0.0 rather than an error; callers that care must check the range.

    **The bearing is from true North, and the arena's ``convergence_deg`` is
    not subtracted from it** (#112). That field, published by ``GET
    :8090/api/site`` and recorded by :mod:`whiteout.site`, is the ~49.8° turn
    between true North and the EPSG:3413 grid the arena renders its terrain
    in — a frame nothing under ``whiteout/`` enters, though
    ``scripts/truth_probe.py`` does and must rotate (#121). The reasoning is
    written out once, in :mod:`whiteout.vision.projection` under "Grid North,
    true North, and what ``convergence_deg`` does and does not reach"; the
    half of it that belongs here is that **what this function returns is
    built, not received**. It comes from two lat/lon pairs through
    :func:`geodetic_to_local`, and the angle is taken at the origin of its own
    frame, where the tangent plane's North is true North exactly — so there is
    no convergence of any kind left in the answer, whatever the arena's own
    conventions are. ``whiteout/tracks/maintain.py``, which takes a course
    about the earlier of the two fixes it runs between, needs nothing more
    than that.

    **``whiteout/transport/arena.py`` is the one caller that needs more**, and
    it is not settled. It turns this bearing into a tower's pan, so the answer
    is only as good as the tower's own yaw reference — and whether ArduPilot
    hands out a true-North or a grid-North yaw in this arena is the open
    question that section ends on. If it turned out to be grid North, this
    function would still be right and that *caller* would need the rotation.
    """
    offset = geodetic_to_local(origin, point)
    return math.degrees(math.atan2(offset.east_m, offset.north_m)) % 360.0
