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
are therefore the site's real *span*, which is a four-figure number, around a
centre of zero, which is not the site's — the real corners stay in the JSON,
which the guard does not walk.
"""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from whiteout.belief.geometry import SITE_CENTRE_LAT, SITE_CENTRE_LON, SITE_EXTENT_M
from whiteout.site import (
    BOUNDS_EXTENT_TOLERANCE,
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

#: Half the site's span in EPSG:3413 grid metres, as the arena reports it.
#: Grid metres, so 0.4952% under half of ``SITE_EXTENT``: that is the polar
#: stereographic scale factor at 72 deg N against a 70 deg N standard
#: parallel, and carrying it here is what keeps
#: :data:`whiteout.site.BOUNDS_EXTENT_TOLERANCE` honest rather than nominal.
HALF_SPAN_M = 3234.311145986

#: A bounds3413 in the right shape and the wrong place: the site's real span
#: around a centre of zero, which is not the site's centre. See the module
#: docstring on why the real corners cannot be written in a test file.
BOUNDS: list[float] = [-HALF_SPAN_M, -HALF_SPAN_M, HALF_SPAN_M, HALF_SPAN_M]


#: The fetching script, which is not a module of the package and is not
#: importable as one. Loaded by path the way ``tests/test_gate_config.py``
#: loads the gate, so that the redaction it does can be tested rather than
#: trusted.
ARENA_SITE = Path(__file__).resolve().parents[1] / "scripts" / "arena_site.py"


def _arena_site() -> ModuleType:
    spec = importlib.util.spec_from_file_location("arena_site", ARENA_SITE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_site_record() -> dict[str, Any]:
    """The committed record as raw JSON, for checks on the file itself."""
    loaded: dict[str, Any] = json.loads(SITE_FIXTURE.read_text(encoding="utf-8"))
    return loaded


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


# --- what the arena actually answered --------------------------------------
#
# Everything above was written before `/api/site` had ever been called. These
# were written after, against the SIM-5 arena at 10.99.4.1, and two of them
# are here because the guesses above were wrong: the bounds arrive as an
# object rather than a list, and `/api/env` arrives wrapped in a one-key
# envelope rather than as the bare `KEY=VALUE` text #102 quotes.


def test_the_committed_record_holds_a_fetched_site_endpoint() -> None:
    """The half that was ``null`` when this file was written is recorded now.

    ``host`` being set is what separates a fetch from a transcription: the
    ``/api/env`` half carried ``None`` there for as long as it was copied out
    of PR #102 rather than asked for.
    """
    site = load_site()
    assert site.site is not None
    assert site.site.host is not None and site.site.at is not None
    assert site.convergence_deg is not None
    assert site.bounds3413 is not None


def test_the_recorded_convergence_is_the_projections_own() -> None:
    """EPSG:3413 confirmed, which is the premise ``projection.py`` rests on.

    Not equal to the identity and not expected to be: the arena computes its
    convergence some other way, and the 0.0176 deg between them is 1 m across
    the site's half-extent. What the check is for is telling this grid apart
    from a different one, and it does.
    """
    site = load_site()
    assert site.convergence_deg is not None
    identity = expected_convergence_deg(site.centre_lon_deg)
    assert abs(abs(site.convergence_deg) - abs(identity)) < CONVERGENCE_TOLERANCE_DEG
    assert site.convergence_deg != identity


def test_the_committed_record_carries_no_credentials() -> None:
    """``/api/env`` returns the organisers' whole ``.env``. This repository is public.

    The fixture holds that file, so the guard belongs on the fixture and not
    only on the script that writes it: a hand-edit is exactly the way a token
    would arrive here.
    """
    module = _arena_site()
    env = load_site_record()["records"]["env"]
    for key, value in env["response"].items():
        if module.is_secret(key):
            assert value == module.REDACTED, f"{key} is not redacted"
    text = SITE_FIXTURE.read_text(encoding="utf-8")
    for line in text.splitlines():
        key, sep, rest = line.partition("=")
        if sep and module.is_secret(key.strip().strip('" ')):
            assert module.REDACTED in rest, f"a credential survives in the body: {key}"


def test_bounds_may_be_the_object_the_arena_sends() -> None:
    """``{xmin, ymin, xmax, ymax}``, which is what ``/api/site`` returns."""
    corners = dict(zip(("xmin", "ymin", "xmax", "ymax"), BOUNDS, strict=True))
    site = parse_site(a_record(site=a_site_response(bounds3413=corners)))
    assert site.bounds3413 == tuple(BOUNDS)


def test_bounds_may_be_the_object_spelled_the_other_way() -> None:
    corners = dict(zip(("min_x", "min_y", "max_x", "max_y"), BOUNDS, strict=True))
    assert parse_site(a_record(site=a_site_response(bounds3413=corners))).bounds3413 == tuple(
        BOUNDS
    )


def test_a_bounds_object_missing_a_corner_says_which() -> None:
    corners = dict(zip(("xmin", "ymin", "xmax"), BOUNDS, strict=False))
    with pytest.raises(SiteError) as caught:
        parse_site(a_record(site=a_site_response(bounds3413=corners)))
    assert "ymax" in str(caught.value)


def test_bounds_that_are_not_square_are_refused() -> None:
    """The arena renders a square site, and the box it reports is square."""
    oblong = [-HALF_SPAN_M, -HALF_SPAN_M, HALF_SPAN_M, HALF_SPAN_M / 2.0]
    with pytest.raises(SiteError, match="not square"):
        parse_site(a_record(site=a_site_response(bounds3413=oblong)))


def test_bounds_that_do_not_span_the_extent_are_refused() -> None:
    """A box agreeing on the centre and disagreeing on the size is two sites."""
    half = HALF_SPAN_M / 2.0
    with pytest.raises(SiteError, match="not this site's"):
        parse_site(a_record(site=a_site_response(bounds3413=[-half, -half, half, half])))


def test_the_grid_to_ground_scale_factor_is_inside_the_bound_and_not_at_it() -> None:
    """The one number that bound exists to tolerate, measured rather than assumed.

    ``BOUNDS`` is the real span, so this asserts on what the arena reported:
    grid metres run 0.4952% short of ground metres here. If that ever sat near
    the tolerance the tolerance would be wrong, so this fails while there is
    still room rather than after the bound has quietly become a fit.
    """
    off = abs(2.0 * HALF_SPAN_M - EXTENT_M) / EXTENT_M
    assert 0.004 < off < 0.006
    assert off < BOUNDS_EXTENT_TOLERANCE / 3.0


# --- the fetching script, which the gate never runs ------------------------


def test_the_env_envelope_is_opened_and_its_lines_parsed() -> None:
    """``/api/env`` answers ``{"env": "<the whole .env>"}``, not KEY=VALUE text.

    This is the shape that made the first attempt at the record unusable: the
    envelope parses as a JSON object, so it was taken as the fields
    themselves, and the one field recorded was called ``env``.
    """
    module = _arena_site()
    body = json.dumps({"env": "# a comment\nSITE_LAT=71.99196\nSITE_EXTENT=6500\n"})
    fields, kept = module.as_fields(json.loads(body), body)
    assert fields == {"SITE_LAT": 71.99196, "SITE_EXTENT": 6500}
    assert kept is not None and "SITE_LAT=71.99196" in kept


def test_a_credential_is_redacted_in_the_fields_and_in_the_body() -> None:
    module = _arena_site()
    body = json.dumps({"env": "SITE_LAT=71.99196\nMAPBOX_TOKEN=pk.a-real-one\n"})
    fields, kept = module.as_fields(json.loads(body), body)
    assert fields["MAPBOX_TOKEN"] == module.REDACTED
    assert fields["SITE_LAT"] == 71.99196
    assert kept is not None
    assert "pk.a-real-one" not in kept
    assert "MAPBOX_TOKEN=" in kept


def test_a_plain_json_object_is_still_taken_as_it_stands() -> None:
    module = _arena_site()
    fields, kept = module.as_fields({"SITE_LAT": CENTRE_LAT}, "{}")
    assert fields == {"SITE_LAT": CENTRE_LAT}
    assert kept is None


def test_the_script_refuses_to_guess_a_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in once. The other half, ``--write``, is not exercised here: it fetches."""
    monkeypatch.delenv("WHITEOUT_ARENA_ENDPOINT", raising=False)
    module = _arena_site()
    with pytest.raises(SystemExit):
        module.main([])
