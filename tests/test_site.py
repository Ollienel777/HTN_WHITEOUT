"""The site record: what it must contain, and what it must refuse.

Issue #112. Two jobs here.

The first is to keep ``whiteout/belief/geometry.py``'s site constants tied to
``fixtures/arena/site.json``. They are the same three numbers in two places,
and PR #102 is what a site constant with no live source costs — so the record
is the source and :func:`test_the_geometry_constants_are_the_recorded_ones`
is what makes it one.

The second is the checks in :mod:`whiteout.site`, which all run on a record
the *committed* file must never be in: two centres that disagree, a
convergence that is not the projection's, bounds that are not bounds. Those
cannot be exercised against the committed file, so they are exercised against
records built here.

**No four-digit-and-up metre constants below.** ``tests/test_geo.py``'s
ellipsoid guard walks this file and fails any literal in
``[1e5, 1e8]``, which is where real EPSG:3413 coordinates live: the site sits
about 1.6e6 m from the grid's origin on each axis. The synthetic bounds below
are therefore small numbers in the right *shape* — the parser checks ordering
and arity, and has no opinion about magnitude — and the real ones stay in the
JSON, which the guard does not walk.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from whiteout.belief.geometry import SITE_CENTRE_LAT, SITE_CENTRE_LON, SITE_EXTENT_M
from whiteout.site import (
    CONVERGENCE_TOLERANCE_DEG,
    GRID_CENTRAL_MERIDIAN_DEG,
    SITE_FIXTURE,
    SiteError,
    expected_convergence_deg,
    load_site,
    parse_site,
)

#: The site centre the committed record holds, spelled once here.
CENTRE_LAT = 71.991960
CENTRE_LON = -94.822428
EXTENT_M = 6500.0

#: A bounds3413 in the right shape and the wrong place: see the module
#: docstring on why a realistic one cannot be written in a test file.
BOUNDS: list[float] = [-2.0, -1.0, 2.0, 1.0]


def a_record(site: dict[str, Any] | None = None, **env: Any) -> dict[str, Any]:
    """A loaded record, with the ``/api/site`` half optional and overridable."""
    response = {"SITE_LAT": CENTRE_LAT, "SITE_LON": CENTRE_LON, "SITE_EXTENT": EXTENT_M}
    response.update(env)
    return {
        "schema_version": 1,
        "records": {
            "env": {
                "provenance": {"request": "GET /api/env", "host": None, "at": None, "note": "t"},
                "response": response,
            },
            "site": {
                "provenance": {"request": "GET /api/site", "host": None, "at": None, "note": "t"},
                "response": site,
            },
        },
    }


def a_site_response(**overrides: Any) -> dict[str, Any]:
    """A plausible ``/api/site`` body, with the convergence the identity gives."""
    response: dict[str, Any] = {
        "centre": {"lat": CENTRE_LAT, "lon": CENTRE_LON},
        "bounds3413": list(BOUNDS),
        "convergence_deg": expected_convergence_deg(CENTRE_LON),
    }
    response.update(overrides)
    return response


# --- the committed record --------------------------------------------------


def test_the_committed_record_loads() -> None:
    """The file in the tree is readable and passes every check in the module."""
    site = load_site()
    assert site.extent_m == EXTENT_M
    assert site.half_extent_m == EXTENT_M / 2.0
    assert site.env.request.endswith("/api/env")


def test_the_geometry_constants_are_the_recorded_ones() -> None:
    """``geometry.py``'s three site constants say what the record says.

    This is the point of the record. The constants stay literals in
    ``geometry.py`` -- it is imported everywhere and must not read a file at
    import time, and the fixture is not in the wheel -- so this is what stops
    the two drifting apart, which is the failure PR #102 spent twenty hours
    inside.
    """
    site = load_site()
    assert SITE_CENTRE_LAT == site.centre_lat_deg
    assert SITE_CENTRE_LON == site.centre_lon_deg
    assert SITE_EXTENT_M == site.extent_m


def test_loading_the_record_opens_no_socket() -> None:
    """``load_site`` is what CI runs, and CI cannot reach the arena.

    The fetch lives in ``scripts/arena_site.py`` and nothing imports it. This
    pins the other half of that split: reading the record must not be a way to
    reach the network by accident.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_site opened a socket")

    original = socket.socket
    socket.socket = refuse  # type: ignore[assignment,misc]
    try:
        load_site()
    finally:
        socket.socket = original  # type: ignore[misc]


def test_a_missing_record_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(SiteError) as caught:
        load_site(tmp_path / "nothing.json")
    assert "scripts/arena_site.py" in str(caught.value)


def test_a_record_that_is_not_json_names_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "site.json"
    broken.write_text("{ not json", encoding="utf-8")
    with pytest.raises(SiteError, match="is not JSON"):
        load_site(broken)


# --- the /api/site half, recorded and not ----------------------------------


def test_an_unrecorded_site_endpoint_is_not_an_error() -> None:
    """A ``null`` response means "never called", and that is the honest state.

    It is also the state the committed file ships in. The centre still comes
    from ``/api/env``; the two 3413 fields report as ``None`` rather than as
    a plausible number nobody measured.
    """
    site = parse_site(a_record(site=None))
    assert site.convergence_deg is None
    assert site.bounds3413 is None
    assert site.site is None
    assert (site.centre_lat_deg, site.centre_lon_deg) == (CENTRE_LAT, CENTRE_LON)


def test_a_recorded_site_endpoint_supplies_the_grid_fields() -> None:
    site = parse_site(a_record(site=a_site_response()))
    assert site.bounds3413 == tuple(BOUNDS)
    assert site.convergence_deg == pytest.approx(expected_convergence_deg(CENTRE_LON))
    assert site.site is not None and site.site.request.endswith("/api/site")


def test_the_site_endpoints_centre_wins_where_the_two_overlap() -> None:
    """``/api/site`` is the site's own description, so it is preferred.

    Preferred, not merged: the check below is what makes that safe.
    """
    nudged = CENTRE_LAT + 0.0001
    site = parse_site(a_record(site=a_site_response(centre={"lat": nudged, "lon": CENTRE_LON})))
    assert site.centre_lat_deg == nudged


def test_two_centres_that_disagree_are_refused() -> None:
    """One arena, one site, one centre. A kilometre apart is a bad record."""
    far = {"lat": CENTRE_LAT + 0.02, "lon": CENTRE_LON}
    with pytest.raises(SiteError, match="do not pick one"):
        parse_site(a_record(site=a_site_response(centre=far)))


# --- the convergence check -------------------------------------------------


def test_the_expected_convergence_at_the_site_is_the_number_the_docstrings_quote() -> None:
    """``lambda - lambda0`` at the site, which every write-up of this rounds to 49.8 deg."""
    assert expected_convergence_deg(CENTRE_LON) == pytest.approx(-49.822428, abs=1e-6)
    assert expected_convergence_deg(GRID_CENTRAL_MERIDIAN_DEG) == pytest.approx(0.0)


def test_either_sign_convention_is_accepted() -> None:
    """A convergence may be published grid-to-true or true-to-grid.

    Which one the arena means is the arena's business, and nothing of ours
    consumes the value, so the check is on magnitude. Reading a sign as a
    correction to apply is exactly what ``projection.py`` explains not to do.
    """
    for value in (expected_convergence_deg(CENTRE_LON), -expected_convergence_deg(CENTRE_LON)):
        site = parse_site(a_record(site=a_site_response(convergence_deg=value)))
        assert site.convergence_deg == value


def test_a_convergence_the_identity_does_not_give_is_refused() -> None:
    """The one check worth making on this field.

    If the arena's grid North is not EPSG:3413's, the reasoning in
    ``projection.py`` rests on a false premise, and a quietly recorded number
    is how that would go unnoticed.
    """
    wrong = expected_convergence_deg(CENTRE_LON) + 2.0 * CONVERGENCE_TOLERANCE_DEG
    with pytest.raises(SiteError, match="re-derived"):
        parse_site(a_record(site=a_site_response(convergence_deg=wrong)))


def test_the_tolerance_is_slack_and_not_a_second_projection() -> None:
    """Inside the tolerance passes; the tolerance absorbs rounding, not a grid."""
    near = expected_convergence_deg(CENTRE_LON) + CONVERGENCE_TOLERANCE_DEG / 2.0
    assert parse_site(a_record(site=a_site_response(convergence_deg=near))).convergence_deg == near


# --- the readers fail loudly -----------------------------------------------


def test_an_unknown_response_shape_names_what_it_did_see() -> None:
    """The ``/api/site`` key names are not known here -- this is why that is safe."""
    with pytest.raises(SiteError) as caught:
        parse_site(a_record(site={"middle": [CENTRE_LAT, CENTRE_LON]}))
    assert "middle" in str(caught.value)


def test_a_centre_may_be_a_pair_or_a_nested_object() -> None:
    site = parse_site(a_record(site=a_site_response(centre=[CENTRE_LAT, CENTRE_LON])))
    assert (site.centre_lat_deg, site.centre_lon_deg) == (CENTRE_LAT, CENTRE_LON)
    flat = a_site_response()
    del flat["centre"]
    flat["lat"], flat["lon"] = CENTRE_LAT, CENTRE_LON
    assert parse_site(a_record(site=flat)).centre_lon_deg == CENTRE_LON


def test_env_fields_may_arrive_as_the_text_that_key_value_lines_parse_to() -> None:
    """``/api/env`` is quoted in #102 as ``SITE_EXTENT=6500``, not as JSON."""
    site = parse_site(a_record(SITE_EXTENT="6500.0"))
    assert site.extent_m == EXTENT_M


@pytest.mark.parametrize(
    "bounds",
    [
        [-2.0, -1.0, 2.0],
        [2.0, 1.0, -2.0, -1.0],
        "not a list",
    ],
)
def test_bounds_must_be_four_numbers_in_order(bounds: object) -> None:
    with pytest.raises(SiteError):
        parse_site(a_record(site=a_site_response(bounds3413=bounds)))


def test_an_extent_that_is_not_positive_is_refused() -> None:
    with pytest.raises(SiteError, match="must be positive"):
        parse_site(a_record(SITE_EXTENT=0.0))


def test_a_record_with_no_provenance_is_refused() -> None:
    """An answer with no record of where it came from is the thing this stops."""
    record = a_record(site=None)
    record["records"]["env"]["provenance"] = {"request": "GET /api/env"}
    with pytest.raises(SiteError, match="where it came from"):
        parse_site(record)


def test_an_env_record_with_no_response_is_refused() -> None:
    """``/api/site`` may be unrecorded; ``/api/env`` may not, because the centre is in it."""
    record = a_record(site=None)
    record["records"]["env"]["response"] = None
    with pytest.raises(SiteError, match="the centre comes from it"):
        parse_site(record)


def test_the_fixture_path_points_into_the_repository() -> None:
    assert SITE_FIXTURE.name == "site.json"
    assert SITE_FIXTURE.parent.name == "arena"
    assert SITE_FIXTURE.is_file()
