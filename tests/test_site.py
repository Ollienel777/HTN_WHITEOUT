"""The site record: what it must contain, and what it must refuse.

Issue #112. Two jobs here.

The first is to keep ``whiteout/belief/geometry.py``'s site constants tied to
``whiteout/data/site.json``. They are the same three numbers in two places,
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

import whiteout.site
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
#: Grid metres, so 0.483% under half of ``SITE_EXTENT`` -- the order of the
#: polar stereographic scale factor at 72 deg N against a 70 deg N standard
#: parallel, though not equal to it; see ``whiteout/site.py``. Carrying the
#: real span here is what keeps
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


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    import time -- so this is what stops the two drifting apart, which is the
    failure PR #102 spent twenty hours inside.

    **Exact equality, deliberately.** Those three are a transcription of the
    record, and a transcription is either right or it is not; a tolerance here
    would let real drift accumulate under it. The looser
    :data:`~whiteout.site.CENTRE_AGREEMENT_M` answers a different question --
    "are these two endpoints describing the same site?" -- and
    :func:`test_a_coarsely_quoted_site_centre_does_not_move_the_parameters` is
    what keeps the loose bound from feeding this strict one.

    When this does fail, the record won: edit the literals, and re-measure
    what is derived from them. ``scripts/arena_site.py`` prints the same
    instruction at the moment of the fetch.
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
    """A ``null`` response means "never called", and that is an honest state.

    It is the state the committed file shipped in before the tunnel existed,
    and the state it would return to for a site nobody has asked yet —
    ``competition.lock`` pins three. The centre still comes from ``/api/env``;
    the two 3413 fields report as ``None`` rather than as a plausible number
    nobody measured.
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


def test_a_coarsely_quoted_site_centre_does_not_move_the_parameters() -> None:
    """The regression for the bug this module's design nearly shipped.

    ``/api/site`` answering ``71.9920`` where ``/api/env`` says ``71.991960``
    is an ordinary rounding: 4.4 m, far inside :data:`CENTRE_AGREEMENT_M`,
    which exists to tolerate exactly that. An earlier revision *preferred*
    ``/api/site``'s centre where the two overlapped, so this entirely
    legitimate answer would have moved ``centre_lat_deg`` off
    ``SITE_CENTRE_LAT`` — and
    :func:`test_the_geometry_constants_are_the_recorded_ones`, which compares
    exactly, would have gone red on the documented ``--write`` at the venue,
    with no stated answer for what to do about it.

    So the centre is ``/api/env``'s, always. ``/api/site``'s is corroboration,
    and its measured separation is reported rather than substituted.
    """
    coarse = round(CENTRE_LAT, 4)
    assert coarse != CENTRE_LAT
    site = parse_site(a_record(site=a_site_response(centre={"lat": coarse, "lon": CENTRE_LON})))
    assert site.centre_lat_deg == CENTRE_LAT == SITE_CENTRE_LAT
    assert site.centre_agreement_m == pytest.approx(4.4, abs=0.5)


def test_the_agreement_is_reported_and_not_merely_asserted() -> None:
    """A record that passes at 4 m and one that passes at 45 m are different records."""
    assert parse_site(a_record(site=a_site_response())).centre_agreement_m == pytest.approx(0.0)
    assert parse_site(a_record(site=None)).centre_agreement_m is None


def test_two_centres_that_disagree_are_refused() -> None:
    """One arena, one site, one centre. A kilometre apart is a bad record."""
    far = {"lat": CENTRE_LAT + 0.02, "lon": CENTRE_LON}
    with pytest.raises(SiteError, match="do not pick one"):
        parse_site(a_record(site=a_site_response(centre=far)))


@pytest.mark.parametrize("where", ["env", "site"])
def test_a_swapped_position_is_refused_as_a_site_error(where: str) -> None:
    """The mix-up ``GeoPoint`` exists to catch, raised as this module's own type.

    ``GeoPoint`` refuses ``lat=-94.82`` with :class:`~whiteout.geo.GeoError`,
    which is not a :class:`SiteError`: it would escape ``load_site``'s
    documented ``:raises``, and it would slip past ``arena_site.py``'s
    ``except SiteError`` and reach the operator as a traceback instead of the
    refusal message that was designed for them. Both halves of the record are
    checked, so a swapped ``/api/env`` pair is refused even when ``/api/site``
    has never been recorded.
    """
    if where == "env":
        record = a_record(site=None, SITE_LAT=CENTRE_LON, SITE_LON=CENTRE_LAT)
    else:
        swapped = {"lat": CENTRE_LON, "lon": CENTRE_LAT}
        record = a_record(site=a_site_response(centre=swapped))
    with pytest.raises(SiteError, match="is not a position"):
        parse_site(record)


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
    """Two shapes, not two spellings: both are ordinary ways to send a centre.

    The *key* has one accepted spelling plus ``center``, which is the same
    word. The invented variants an earlier revision accepted are gone — see
    the comment above the readers in ``whiteout/site.py``.
    """
    assert parse_site(a_record(site=a_site_response(centre=[CENTRE_LAT, CENTRE_LON]))).site
    flat = a_site_response()
    del flat["centre"]
    flat["lat"], flat["lon"] = CENTRE_LAT, CENTRE_LON
    assert parse_site(a_record(site=flat)).centre_agreement_m == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("accepted", "invented"),
    [("convergence_deg", "grid_convergence_deg"), ("bounds3413", "bounds_3413")],
)
def test_an_invented_spelling_is_not_quietly_accepted(accepted: str, invented: str) -> None:
    """A name nobody has seen an answer use fails loudly rather than being guessed at.

    An earlier revision accepted all three of these on the theory that the
    fetch should not fail at the venue for a key name. That was imagination,
    and it bought nothing: the failure names every key the response did have,
    the raw body is kept in the record either way, and widening the list from
    a real answer is a one-line change.
    """
    response = a_site_response()
    response[invented] = response.pop(accepted)
    with pytest.raises(SiteError) as caught:
        parse_site(a_record(site=response))
    assert invented in str(caught.value)


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


def test_the_record_ships_inside_the_package() -> None:
    """``load_site()`` must mean the same thing in a wheel as in a checkout.

    The record lives beside ``whiteout/site.py`` rather than in ``fixtures/``,
    which ``pyproject.toml`` excludes from the wheel, and
    ``[tool.setuptools.package-data]`` is what carries it there. This asserts
    both halves: the file is where the module looks, and the packaging still
    claims it.
    """
    assert SITE_FIXTURE.name == "site.json"
    assert SITE_FIXTURE.is_file()
    assert SITE_FIXTURE.parent == Path(whiteout.site.__file__).resolve().parent / "data"
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in pyproject
    assert 'whiteout = ["data/*.json"]' in pyproject


# --- the fetch script ------------------------------------------------------
#
# Only the two behaviours that are decisions rather than plumbing. The rest of
# the script needs an arena, and a test that mocks an arena end to end would be
# asserting the mock.


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("10.99.4.1", "http://10.99.4.1:8090"),
        ("10.99.4.1:8090", "http://10.99.4.1:8090"),
        ("http://arena.local", "http://arena.local:8090"),
        ("http://arena.local:9000", "http://arena.local:9000"),
        # The colon-counting version appended the port to the *path* and asked
        # for `/api:8090`. A host with a path in front of it is what somebody
        # pastes out of a browser.
        ("http://arena.local/api", "http://arena.local:8090"),
    ],
)
def test_the_endpoint_is_split_rather_than_counted(endpoint: str, expected: str) -> None:
    assert _arena_site().base_url(endpoint) == expected


def test_one_endpoint_failing_does_not_discard_the_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 on ``/api/site`` must not throw away a good ``/api/env`` answer.

    A run where one endpoint fails is an ordinary outcome — the fetch that
    recorded this file found ``/api/site`` answering a shape nothing here
    expected — and the other half's answer is worth keeping. The failed half is
    recorded the way the record already spells "never fetched", with the error
    in its provenance, and the run reports the failure to its caller.
    """

    module = _arena_site()

    def answer(url: str) -> tuple[Any, str]:
        if url.endswith("/api/site"):
            raise module.FetchError("HTTP Error 404: Not Found")
        return None, f"SITE_LAT={CENTRE_LAT}\nSITE_LON={CENTRE_LON}\nSITE_EXTENT={EXTENT_M}\n"

    monkeypatch.setattr(module, "fetch", answer)
    built, failures = module.record("127.0.0.1")

    assert failures and "404" in failures[0]
    assert built["records"]["site"]["response"] is None
    assert "404" in built["records"]["site"]["provenance"]["note"]
    # And what survives is a record whiteout.site accepts, with the env half
    # now a fetch rather than a transcription.
    site = parse_site(built)
    assert (site.centre_lat_deg, site.extent_m) == (CENTRE_LAT, EXTENT_M)
    assert site.convergence_deg is None
    assert built["records"]["env"]["provenance"]["host"] == "http://127.0.0.1:8090"


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
    grid metres run 0.483% short of ground metres here. If that ever sat near
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


@pytest.mark.parametrize(
    "value",
    [
        "pk.eyJ1IjoiZXhhbXBsZSIsImEiOiJjbGV4YW1wbGUifQ",
        "ghp_AAAAbbbbCCCCddddEEEEffff1111",
        "postgres://user:hunter2@db.internal:5432/arena",
        "AKIAIOSFODNN7EXAMPLE",
        "Aa1" + "b" * 40,
    ],
)
def test_a_credential_is_caught_by_its_shape_when_the_key_name_does_not_say(value: str) -> None:
    """The hole a key-name denylist cannot close, and the CI guard inherits.

    ``SECRET_KEY_MARKERS`` is a substring list over a ``.env`` this repository
    does not control, so ``GITHUB_PAT``, ``SENTRY_DSN`` and a ``DATABASE_URL``
    with an inline password all slip through it -- and the guard on the
    committed record reuses the same predicate, so whatever the denylist
    misses the guard misses too. The value is therefore read as well as the
    key, and either firing is enough.
    """
    module = _arena_site()
    assert not module.is_secret("HARMLESS_SETTING")
    assert module.is_secret_value(value)
    fields, _ = module.as_fields({"HARMLESS_SETTING": value}, "{}")
    assert fields["HARMLESS_SETTING"] == module.REDACTED


def test_the_shape_test_leaves_every_real_site_parameter_alone() -> None:
    """The half of that check which could misfire, run over what it must not touch.

    A tight rule is only useful if it is also quiet. This runs it over every
    field of the committed ``/api/env`` record -- a dotted registry host, a
    comma-separated month list, a space-separated colour, the asset lines --
    and fails if it fires on any of them.
    """
    module = _arena_site()
    env = load_site_record()["records"]["env"]["response"]
    misfired = [
        key
        for key, value in env.items()
        if module.is_secret_value(value) and not module.is_secret(key)
    ]
    assert not misfired, f"the value-shape test fired on site parameters: {misfired}"


def test_a_redaction_is_not_itself_read_as_a_credential() -> None:
    """Re-running the script over its own output must be a fixed point."""
    module = _arena_site()
    assert not module.is_secret_value(module.REDACTED)


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
