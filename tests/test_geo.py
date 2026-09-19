"""Issue #73's acceptance criteria, one test each.

The decision under test is ``SPEC.md`` §5's: **the frame of record is geodetic
lat/lon, a local East–North frame exists only as a projection for drawing, and
``whiteout/geo.py`` is the one module that converts.**

Two of these are static guards rather than behaviour tests, and deliberately
so. The failure this ticket exists to prevent is not a crash — it is a second
converter appearing in a second module at hour 20, disagreeing with the first
by a few kilometres, and posting a *plausible* lat/lon to the scored API. A
reviewer will not catch that; :mod:`ast` will.

Everything numeric here is evaluated at Bellot Strait's latitude, ~71.99 N.
At the equator the two radii of curvature are within half a percent of each
other, so a projection that confuses East with North, or that drops
:math:`\\cos\\varphi` entirely, round-trips almost perfectly and every test
passes. At 71.99 N it does not.
"""

from __future__ import annotations

import ast
import math
from dataclasses import fields
from pathlib import Path

import pytest

from whiteout import types as record_types
from whiteout.geo import (
    ARENA_ORIGIN,
    WGS84_A,
    WGS84_F,
    GeoError,
    GeoPoint,
    LocalPoint,
    enu_to_geodetic,
    geodetic_to_local,
    local_to_geodetic,
    radii_of_curvature,
)
from whiteout.types import Contact, Detection, Pose, RecordError

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "whiteout"
THE_ONE_CONVERTER = PACKAGE / "geo.py"

#: The strait, ``ARENA.md`` §2: 25 km along the channel and 2 km across it.
#: Offsets are taken to the corners, which is the worst case the arena has.
STRAIT_HALF_LENGTH_M = 12_500.0
STRAIT_HALF_WIDTH_M = 1_000.0


# --- A1: the frame of record is lat/lon, and the log schema matches ---------


def test_no_record_type_carries_a_local_coordinate() -> None:
    """``SPEC.md`` §5: nothing in the episode log is x/y.

    The names are the check. A field called ``x`` in a log that the scorer,
    the viewer and the tracks poster all read is an invitation for one of them
    to treat metres as degrees.
    """
    banned = {"x", "y", "xy", "target_xy", "peak_xy", "east", "north"}
    for name in record_types.__all__:
        member = getattr(record_types, name)
        if not hasattr(member, "__dataclass_fields__"):
            continue
        found = {field.name for field in fields(member)} & banned
        assert not found, f"{name} carries {', '.join(sorted(found))}"


def test_every_position_in_the_log_is_a_geodetic_pair() -> None:
    """A ``lat`` without a ``lon`` is half a position, and vice versa."""
    for name in record_types.__all__:
        member = getattr(record_types, name)
        if not hasattr(member, "__dataclass_fields__"):
            continue
        present = {field.name for field in fields(member)}
        for latitude in sorted(field for field in present if field.endswith("lat")):
            assert f"{latitude[:-3]}lon" in present, f"{name}.{latitude} has no longitude"


def test_a_swapped_lat_lon_at_the_strait_is_refused_at_construction() -> None:
    """The mix-up this ticket exists to prevent, caught where it is made.

    Bellot Strait is 71.99 N, -94.84 E. Swapped, the latitude is -94.84, which
    is not a latitude — so the one frame error that would otherwise post a
    confident wrong answer to the scored API cannot be represented at all.
    """
    for build in (
        lambda lat, lon: Pose("a", "quad", 0.0, lat, lon, 0.0, 0.0, 0.0, 0.0),
        lambda lat, lon: Contact("c", 0.0, "tracked", lat, lon, 0.5, "vessel", None),
        lambda lat, lon: Detection("d", lat, lon, 0.5, "vessel"),
    ):
        build(ARENA_ORIGIN.lat_deg, ARENA_ORIGIN.lon_deg)
        with pytest.raises(RecordError) as caught:
            build(ARENA_ORIGIN.lon_deg, ARENA_ORIGIN.lat_deg)
        assert "-94.84" in str(caught.value)


# --- A2: exactly one module converts ---------------------------------------


def _sources() -> list[Path]:
    sources = sorted(PACKAGE.rglob("*.py"))
    assert THE_ONE_CONVERTER in sources, "whiteout/geo.py is not in the tree"
    return [source for source in sources if source != THE_ONE_CONVERTER]


#: Numbers that only a second ellipsoid model would need. The semi-major axis
#: and the inverse flattening are how every copy of this arithmetic starts.
_ELLIPSOID_LITERALS = (WGS84_A, 1.0 / WGS84_F, 6371000.0, 6378137.0)

#: Substrings that name a frame conversion. A helper called
#: ``_to_latlon`` in the detector is exactly the duplicate this forbids.
_CONVERSION_WORDS = ("geodetic", "latlon", "lat_lon", "to_enu", "enu_to", "wgs84")


def test_only_whiteout_geo_defines_the_ellipsoid() -> None:
    """A second copy of the constants is a second answer to the same question."""
    for source in _sources():
        where = source.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, int | float):
                continue
            if isinstance(node.value, bool):
                continue
            for literal in _ELLIPSOID_LITERALS:
                assert not math.isclose(float(node.value), literal, rel_tol=1e-9), (
                    f"{where} line {node.lineno} spells out {node.value!r}: "
                    f"the ellipsoid lives in whiteout/geo.py and nowhere else"
                )


def test_only_whiteout_geo_defines_a_frame_conversion() -> None:
    """Importing the one converter is fine; defining a second one is not."""
    for source in _sources():
        where = source.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(source.read_text(encoding="utf-8"))
        defined: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
        for name in sorted(defined):
            lowered = name.lower()
            for word in _CONVERSION_WORDS:
                assert word not in lowered, (
                    f"{where} defines {name!r}: whiteout/geo.py is the one converter, "
                    f"and every other caller imports it"
                )


def test_the_transport_takes_its_positions_from_the_one_converter() -> None:
    """The stub fleet is placed in metres and reported in lat/lon.

    Not a style point: the adapter is the first thing above the seam that has
    to name a position, and it is where a private ``x + 0.0001`` frame would
    otherwise be born.
    """
    source = (PACKAGE / "transport" / "kinematic.py").read_text(encoding="utf-8")
    assert "from whiteout.geo import" in source


# --- A3: the round trip, at the strait's latitude ---------------------------


def test_geodetic_to_local_and_back_round_trips_across_the_strait() -> None:
    """Every corner of the arena, out and back, to floating-point noise.

    The tolerance is 1e-6 m, which is not a claim about the tangent-plane
    approximation — that is a *model* error of a few centimetres at this
    range, and it cancels exactly here because both directions read their
    radii from the same origin latitude. What this measures is that the
    inverse is the inverse.
    """
    for east in (-STRAIT_HALF_LENGTH_M, -1.0, 0.0, 1.0, STRAIT_HALF_LENGTH_M):
        for north in (-STRAIT_HALF_WIDTH_M, 0.0, STRAIT_HALF_WIDTH_M):
            offset = LocalPoint(east, north)
            there = local_to_geodetic(ARENA_ORIGIN, offset)
            back = geodetic_to_local(ARENA_ORIGIN, there)
            assert back.east_m == pytest.approx(east, abs=1e-6)
            assert back.north_m == pytest.approx(north, abs=1e-6)


def test_a_geodetic_position_round_trips_through_the_local_frame() -> None:
    """The other direction: lat/lon in, lat/lon out, at 71.99 N."""
    for lat, lon in ((71.9965, -94.8448), (71.9975, -94.8450), (71.9820, -94.9000)):
        point = GeoPoint(lat, lon)
        back = local_to_geodetic(ARENA_ORIGIN, geodetic_to_local(ARENA_ORIGIN, point))
        assert back.lat_deg == pytest.approx(lat, abs=1e-12)
        assert back.lon_deg == pytest.approx(lon, abs=1e-12)


def test_the_longitude_convergence_at_the_strait_is_real() -> None:
    """~0.31, the number ``ARENA.md`` and #73 both quote.

    This is the test that would pass at the equator and must not. A degree of
    longitude at Bellot Strait is a third of a degree of latitude, so a
    projection that drops the cosine, or that hands East to the latitude
    slot, lands about 2 km out at the far end of the channel — a plausible
    position, in the water, wrong.
    """
    meridional, east = radii_of_curvature(ARENA_ORIGIN.lat_deg)
    assert east / meridional == pytest.approx(0.31, abs=0.005)

    equatorial_meridional, equatorial_east = radii_of_curvature(0.0)
    assert equatorial_east / equatorial_meridional == pytest.approx(1.0, abs=0.01)


def test_a_thousand_metres_east_is_not_a_thousand_metres_north() -> None:
    """The same offset in the two axes moves a different number of degrees."""
    east_only = local_to_geodetic(ARENA_ORIGIN, LocalPoint(1000.0, 0.0))
    north_only = local_to_geodetic(ARENA_ORIGIN, LocalPoint(0.0, 1000.0))
    moved_lon = abs(east_only.lon_deg - ARENA_ORIGIN.lon_deg)
    moved_lat = abs(north_only.lat_deg - ARENA_ORIGIN.lat_deg)
    assert moved_lon / moved_lat == pytest.approx(1.0 / 0.31, rel=0.02)


def test_enu_to_geodetic_is_the_positional_form_of_local_to_geodetic() -> None:
    """The camera projection calls the positional one; they must not diverge."""
    positional = enu_to_geodetic(ARENA_ORIGIN.lat_deg, ARENA_ORIGIN.lon_deg, 750.0, -250.0)
    typed = local_to_geodetic(ARENA_ORIGIN, LocalPoint(750.0, -250.0))
    assert positional == typed


# --- A4: nothing leaves in any frame but lat/lon ----------------------------


def test_the_type_that_leaves_for_the_tracks_api_carries_only_lat_lon() -> None:
    """``ARENA.md`` §5's body is ``{"name", "lat", "lon"}`` and nothing else.

    :class:`~whiteout.geo.GeoPoint` is what the camera projection returns and
    what the tracks poster (#67) takes, so a metre, a pixel or an altitude
    cannot reach the scored API without someone widening this type first.
    """
    assert [field.name for field in fields(GeoPoint)] == ["lat_deg", "lon_deg"]


def test_a_local_offset_is_not_a_position() -> None:
    """:class:`~whiteout.geo.LocalPoint` has no lat/lon to post by mistake."""
    assert [field.name for field in fields(LocalPoint)] == ["east_m", "north_m"]


def test_an_impossible_position_is_refused_rather_than_clamped() -> None:
    for lat, lon in ((91.0, 0.0), (-90.5, 0.0), (0.0, 181.0), (float("nan"), 0.0)):
        with pytest.raises(GeoError):
            GeoPoint(lat, lon)


def test_a_pole_has_no_local_frame() -> None:
    """The East radius vanishes, so metres East stop meaning degrees."""
    with pytest.raises(GeoError):
        geodetic_to_local(GeoPoint(90.0, 0.0), GeoPoint(89.0, 10.0))
