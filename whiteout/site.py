"""What the arena says about the site it renders, read from a recorded answer.

Issue #112. The site's parameters — where the rendered world is centred, how
big it is, and how its map grid is turned relative to true North — are facts
the arena will tell us if asked. Until this module they were hand-copied
constants in :mod:`whiteout.belief.geometry` with a citation in a comment, and
#102 is what that costs when the citation is a slide instead of the running
system.

So this module is deliberately small and does one thing: it reads a
**committed fixture** holding the arena's own answers, verbatim, with a note
saying where each one came from. Nothing here computes a site parameter, and
nothing here guesses one.

The fixture, and why the fetch is not in the product
-----------------------------------------------------

``fixtures/arena/site.json`` is the record. :func:`load_site` reads it and
never opens a socket; ``scripts/arena_site.py`` is the only thing that talks
to the arena, it refuses to run without an endpoint named on the command line,
and it overwrites the fixture only when asked with ``--write``. That split is
the whole design, and it has two reasons:

* **CI cannot reach the arena.** A module that fetched on import would make
  the gate depend on a WireGuard tunnel to a laptop in a venue.
* **A fetched number with no record is a number nobody can check.** The
  fixture carries the raw response and the provenance beside it, so the next
  person can see what was asked, of which host, and when — which is exactly
  what ``DEFAULT_STRAIT``'s centreline still cannot show.

Two endpoints, and what each is good for
-----------------------------------------

``GET :8090/api/env``
    The arena's environment, as ``SITE_LAT``, ``SITE_LON``, ``SITE_EXTENT``
    and the rest. This is where #102's correction came from, and its three
    site fields are transcribed into the fixture with that provenance.

``GET :8090/api/site``
    The site's own description: its centre, ``bounds3413`` — the extent in
    EPSG:3413, the polar stereographic grid the arena tiles its terrain in —
    and ``convergence_deg``, the angle between that grid's North and true
    North. Called and recorded, from the SIM-5 arena at ``10.99.4.1`` on
    2026-09-20. A record whose ``response`` is ``null`` still means *not
    recorded*, and :class:`SiteParameters` still reports the two grid fields
    as ``None`` in that case rather than inventing a plausible answer; that
    is the state the file shipped in before the tunnel existed, and the state
    it would return to on a site nobody has asked yet.

When both are recorded they overlap on the centre, and that overlap is
checked: :func:`parse_site` refuses a fixture whose two centres disagree by
more than :data:`CENTRE_AGREEMENT_M`. Two answers to the same question that
are allowed to differ are how #102 happened.

What ``convergence_deg`` is for, and what it is not for
--------------------------------------------------------

It is **not** a correction to apply to anything we compute. The reasoning is
written out in :mod:`whiteout.vision.projection`, under "Grid North, true
North, and why ``convergence_deg`` is not a bias here", and is summarised at
:func:`whiteout.geo.bearing_deg`; it is not repeated here, because a rule
repeated in three places is a rule two of which go stale.

What the value is good for is the one check this module does make. For the
polar aspect of a stereographic projection the convergence at longitude
:math:`\\lambda` is exactly :math:`\\lambda - \\lambda_0`, and EPSG:3413's
central meridian is 45° West. At the site that is about **49.8°**. If the
arena's answer disagrees with that identity by more than
:data:`CONVERGENCE_TOLERANCE_DEG`, then its grid is not the projection we
think it is, and the reasoning in ``projection.py`` — which rests on that grid
being a *rendering* frame we never enter — has to be re-derived rather than
trusted. So a disagreement raises :class:`SiteError` here instead of being
recorded quietly.

The tolerance is degrees rather than arcseconds on purpose. The identity is
exact, but the arena's sign convention is its own — a convergence may be
published as the angle from grid to true or from true to grid — so
:func:`parse_site` compares magnitudes, and the slack is there to absorb a
centre quoted to fewer decimals than ours, not to absorb a different
projection. Half a degree at the site's 3.25 km half-extent is 28 m; a wrong
projection is tens of degrees out, not tenths.

That slack turned out to be needed, and to be the right size. The arena
answers ``-49.80479318596525``; the identity at ``SITE_LON`` gives
``-49.822428``. The two agree to **0.0176°**, which is 1 m across the site's
half-extent and about a thirtieth of the tolerance. The residual is not
rounding — the centre is quoted to six decimals — so the arena is computing
its convergence some other way than the closed form, most likely by
differentiating the projection numerically. That is not worth chasing: what
the check is for is telling EPSG:3413 apart from some other grid, and tenths
of a degree is an answer of *yes, this grid*. The recorded ``bounds3413``
says the same thing independently — reprojecting ``SITE_LAT``/``SITE_LON``
into EPSG:3413 by the closed form lands on the centre of the recorded box to
**0.0 m**, and the box spans 6468.62 grid metres across 6500 ground metres,
which is the scale factor at 72° N against a 70° N standard parallel.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whiteout.geo import GeoPoint, geodetic_to_local

__all__ = [
    "BOUNDS_EXTENT_TOLERANCE",
    "BOUNDS_SQUARENESS_M",
    "CENTRE_AGREEMENT_M",
    "CONVERGENCE_TOLERANCE_DEG",
    "GRID_CENTRAL_MERIDIAN_DEG",
    "SITE_FIXTURE",
    "Provenance",
    "SiteError",
    "SiteParameters",
    "expected_convergence_deg",
    "load_site",
    "parse_site",
]

#: The committed record, relative to the repository root. It is not packaged
#: into the wheel — ``pyproject.toml`` excludes ``fixtures*`` — because
#: nothing in the run loop reads it: it is a source for geometry and for the
#: tests that pin geometry, and a caller outside the source tree passes its
#: own path to :func:`load_site`.
SITE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "arena" / "site.json"

#: EPSG:3413's central meridian, degrees East. NSIDC Sea Ice Polar
#: Stereographic North: latitude of true scale 70° N, central meridian 45° W.
GRID_CENTRAL_MERIDIAN_DEG = -45.0

#: How far a recorded convergence may sit from ``lon - GRID_CENTRAL_MERIDIAN``
#: before :func:`parse_site` refuses the fixture. See the module docstring:
#: this absorbs a rounded centre, not a different projection.
CONVERGENCE_TOLERANCE_DEG = 0.5

#: How far ``/api/env``'s site centre and ``/api/site``'s may sit apart,
#: metres, before :func:`parse_site` refuses the fixture. One arena, one
#: site, one centre; a kilometre of disagreement is a bug in the record and
#: not a rounding.
CENTRE_AGREEMENT_M = 50.0

#: How far ``bounds3413``'s span may sit from ``SITE_EXTENT`` as a fraction,
#: before :func:`parse_site` refuses the fixture. The two are not equal and
#: are not meant to be: ``SITE_EXTENT`` is ground metres, the bounds are
#: EPSG:3413 grid metres, and the polar stereographic scale factor between
#: them is 0.4952% at the recorded centre (6468.62 m of grid across 6500 m of
#: ground). This bound is wide enough to hold that and any neighbouring
#: latitude, and narrow enough to catch a bounds box of the wrong site, the
#: wrong extent, or the wrong units.
BOUNDS_EXTENT_TOLERANCE = 0.02

#: How far ``bounds3413``'s width and height may sit apart, metres, before
#: :func:`parse_site` refuses the fixture. The arena renders a square site and
#: the recorded box is square to the last decimal it quotes; a rectangle means
#: the box is not this site's.
BOUNDS_SQUARENESS_M = 1.0


class SiteError(ValueError):
    """A site record is missing, malformed, or disagrees with itself."""


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where one recorded answer came from.

    ``host`` and ``at`` are ``None`` for a field that was transcribed from an
    earlier measurement rather than fetched by ``scripts/arena_site.py``, and
    ``note`` says which. That asymmetry is the point of carrying provenance at
    all: a transcription and a fetch are not the same evidence.
    """

    request: str
    host: str | None
    at: str | None
    note: str


@dataclass(frozen=True, slots=True)
class SiteParameters:
    """The site the arena renders, as the arena describes it.

    :param centre_lat_deg: site centre, degrees North.
    :param centre_lon_deg: site centre, degrees East.
    :param extent_m: the site's square extent, metres. The whole of the
        world: there is no terrain, no water and no vessel outside it.
    :param convergence_deg: grid North against true North, degrees, or
        ``None`` when ``/api/site`` has not been recorded. **Not a correction
        to apply** — see the module docstring.
    :param bounds3413: ``(min_x, min_y, max_x, max_y)`` in EPSG:3413 metres,
        or ``None`` when ``/api/site`` has not been recorded. Carried because
        it is what the arena tiles terrain against; nothing of ours consumes
        it.
    :param env: where the ``/api/env`` half came from.
    :param site: where the ``/api/site`` half came from, or ``None`` when it
        has not been recorded.
    """

    centre_lat_deg: float
    centre_lon_deg: float
    extent_m: float
    convergence_deg: float | None
    bounds3413: tuple[float, float, float, float] | None
    env: Provenance
    site: Provenance | None

    @property
    def half_extent_m(self) -> float:
        """Half the site's extent — the furthest East or North the world goes."""
        return self.extent_m / 2.0


def expected_convergence_deg(lon_deg: float) -> float:
    """Convergence the polar stereographic identity gives at ``lon_deg``.

    :math:`\\lambda - \\lambda_0` wrapped into ``[-180, 180)``, with
    :math:`\\lambda_0` = :data:`GRID_CENTRAL_MERIDIAN_DEG`. Exact for the
    polar aspect: grid North runs along the central meridian, so at any other
    longitude the meridian is turned from it by the longitude difference.

    **The sign here is one of the two conventions**, from grid to true. The
    arena's is the arena's, so :func:`parse_site` compares this against a
    recorded value by magnitude.
    """
    if not math.isfinite(lon_deg):
        raise SiteError(f"longitude must be finite, got {lon_deg!r}")
    return ((lon_deg - GRID_CENTRAL_MERIDIAN_DEG) + 180.0) % 360.0 - 180.0


def load_site(path: Path | None = None) -> SiteParameters:
    """Read the committed site record. **Opens no socket, ever.**

    :param path: the record to read; :data:`SITE_FIXTURE` when ``None``.
    :raises SiteError: if the file is missing, is not JSON, or fails any of
        the checks in :func:`parse_site`.
    """
    where = SITE_FIXTURE if path is None else Path(path)
    try:
        text = where.read_text(encoding="utf-8")
    except OSError as exc:
        raise SiteError(
            f"no site record at {where}: run `python scripts/arena_site.py "
            f"--endpoint <host> --write` against a live arena to make one"
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SiteError(f"{where} is not JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SiteError(f"{where} holds {type(data).__name__}, not an object")
    return parse_site(data, where=str(where))


def parse_site(data: Mapping[str, Any], *, where: str = "<record>") -> SiteParameters:
    """Turn a loaded record into :class:`SiteParameters`, checking it.

    Split from :func:`load_site` so that the checks can be exercised against a
    record built in a test rather than only against the committed file — the
    interesting cases are the ones the committed file must never be in.
    """
    records = _mapping(data.get("records"), f"{where}: records")
    env_raw, env_where = _record(records, "env", where)
    site_raw, site_where = _record(records, "site", where)

    if env_raw is None:
        raise SiteError(f"{where}: the env record has no response, and the centre comes from it")
    env_lat = _number(_pick(env_raw, "SITE_LAT", "site_lat", "lat"), f"{env_where}: SITE_LAT")
    env_lon = _number(_pick(env_raw, "SITE_LON", "site_lon", "lon"), f"{env_where}: SITE_LON")
    extent_m = _number(
        _pick(env_raw, "SITE_EXTENT", "site_extent", "extent"), f"{env_where}: SITE_EXTENT"
    )
    if extent_m <= 0.0:
        raise SiteError(f"{env_where}: SITE_EXTENT must be positive, got {extent_m!r}")

    centre_lat, centre_lon = env_lat, env_lon
    convergence_deg: float | None = None
    bounds: tuple[float, float, float, float] | None = None

    if site_raw is not None:
        site_lat, site_lon = _centre(site_raw, site_where)
        _agree(env_lat, env_lon, site_lat, site_lon, where)
        # `/api/site` is the site endpoint's own answer about the site, so it
        # wins where the two overlap. The check above is what makes that a
        # preference rather than a second frame.
        centre_lat, centre_lon = site_lat, site_lon
        convergence_deg = _convergence(site_raw, centre_lon, site_where)
        bounds = _bounds(site_raw, extent_m, site_where)

    return SiteParameters(
        centre_lat_deg=centre_lat,
        centre_lon_deg=centre_lon,
        extent_m=extent_m,
        convergence_deg=convergence_deg,
        bounds3413=bounds,
        env=_provenance(records, "env", where),
        site=_provenance(records, "site", where) if site_raw is not None else None,
    )


# -- the tolerant readers ---------------------------------------------------
#
# Every reader below takes the response **as the arena sent it** and looks for
# the field under the spellings it might have used. That tolerance is
# deliberate and it is bounded. The spellings the arena actually uses are now
# known and recorded — `centre` as an object, `bounds3413` as an object of
# xmin/ymin/xmax/ymax, `convergence_deg` — and those come first below; the
# alternatives stay because one arena answering one way is not a contract, and
# `competition.lock` pins three sites. These fail loudly, naming the keys they
# did see, and the raw response is kept in the fixture so the next reader can widen the
# list from evidence instead of from imagination.


def _record(
    records: Mapping[str, Any], name: str, where: str
) -> tuple[Mapping[str, Any] | None, str]:
    """``(response, where)`` for one endpoint's record; response ``None`` if unrecorded."""
    entry = _mapping(records.get(name), f"{where}: records.{name}")
    response = entry.get("response")
    if response is None:
        return None, f"{where}: records.{name}"
    return _mapping(response, f"{where}: records.{name}.response"), f"{where}: records.{name}"


def _provenance(records: Mapping[str, Any], name: str, where: str) -> Provenance:
    entry = _mapping(records.get(name), f"{where}: records.{name}")
    raw = _mapping(entry.get("provenance"), f"{where}: records.{name}.provenance")
    request = raw.get("request")
    note = raw.get("note")
    if not isinstance(request, str) or not isinstance(note, str):
        raise SiteError(
            f"{where}: records.{name}.provenance needs a string 'request' and 'note' — "
            f"a recorded answer with no record of where it came from is the thing this "
            f"file exists to stop"
        )
    host = raw.get("host")
    at = raw.get("at")
    if host is not None and not isinstance(host, str):
        raise SiteError(f"{where}: records.{name}.provenance.host must be a string or null")
    if at is not None and not isinstance(at, str):
        raise SiteError(f"{where}: records.{name}.provenance.at must be a string or null")
    return Provenance(request=request, host=host, at=at, note=note)


def _centre(response: Mapping[str, Any], where: str) -> tuple[float, float]:
    """The site centre out of an ``/api/site`` response, however it spells it."""
    nested = _maybe(response, "centre", "center")
    if isinstance(nested, Mapping):
        return (
            _number(_pick(nested, "lat", "latitude"), f"{where}: centre.lat"),
            _number(_pick(nested, "lon", "lng", "longitude"), f"{where}: centre.lon"),
        )
    if isinstance(nested, list | tuple):
        if len(nested) != 2:
            raise SiteError(f"{where}: centre is a list of {len(nested)}, expected [lat, lon]")
        return (
            _number(nested[0], f"{where}: centre[0]"),
            _number(nested[1], f"{where}: centre[1]"),
        )
    lat = _maybe(response, "lat", "latitude", "site_lat")
    lon = _maybe(response, "lon", "lng", "longitude", "site_lon")
    if lat is None or lon is None:
        raise SiteError(
            f"{where}: no site centre. Looked for 'centre' or 'center' holding a "
            f"[lat, lon] or a {{lat, lon}}, and for a flat lat/lon pair; the response "
            f"has: {_keys(response)}"
        )
    return _number(lat, f"{where}: lat"), _number(lon, f"{where}: lon")


def _convergence(response: Mapping[str, Any], lon_deg: float, where: str) -> float:
    """The recorded convergence, checked against the projection's own identity."""
    value = _number(
        _pick(response, "convergence_deg", "convergence", "grid_convergence_deg"),
        f"{where}: convergence_deg",
    )
    expected = expected_convergence_deg(lon_deg)
    if abs(abs(value) - abs(expected)) > CONVERGENCE_TOLERANCE_DEG:
        raise SiteError(
            f"{where}: convergence_deg is {value!r}, but the polar stereographic identity "
            f"gives {expected:.4f} at longitude {lon_deg!r} against a {GRID_CENTRAL_MERIDIAN_DEG} "
            f"central meridian — more than {CONVERGENCE_TOLERANCE_DEG} apart in magnitude. The "
            f"arena's grid is then not the EPSG:3413 this build assumes, and "
            f"whiteout/vision/projection.py's 'Grid North, true North' section has to be "
            f"re-derived before this record is trusted"
        )
    return value


def _bounds(
    response: Mapping[str, Any], extent_m: float, where: str
) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)``, however the arena spelled the box.

    The arena sends an object — ``{"xmin": …, "ymin": …, "xmax": …, "ymax":
    …}`` — which is why that form comes first. A list of four is still read,
    because it is the other obvious spelling and reading it costs two lines.
    """
    raw = _pick(response, "bounds3413", "bounds_3413", "bounds")
    if isinstance(raw, Mapping):
        corners = (("xmin", "min_x"), ("ymin", "min_y"), ("xmax", "max_x"), ("ymax", "max_y"))
        values = []
        for name, alternative in corners:
            found = _maybe(raw, name, alternative)
            if found is None:
                raise SiteError(
                    f"{where}: bounds3413 has no {name} or {alternative}; it has: {_keys(raw)}"
                )
            values.append(_number(found, f"{where}: bounds3413.{name}"))
        min_x, min_y, max_x, max_y = values
    elif isinstance(raw, list | tuple):
        if len(raw) != 4:
            raise SiteError(f"{where}: bounds3413 has {len(raw)} entries, expected four")
        min_x, min_y, max_x, max_y = (
            _number(value, f"{where}: bounds3413[{index}]") for index, value in enumerate(raw)
        )
    else:
        raise SiteError(
            f"{where}: bounds3413 is {type(raw).__name__}, expected four numbers as a list "
            f"or as an object of xmin, ymin, xmax and ymax"
        )

    if min_x >= max_x or min_y >= max_y:
        raise SiteError(
            f"{where}: bounds3413 is not [min_x, min_y, max_x, max_y] — got "
            f"{[min_x, min_y, max_x, max_y]!r}"
        )
    _bounds_match_extent(min_x, min_y, max_x, max_y, extent_m, where)
    return min_x, min_y, max_x, max_y


def _bounds_match_extent(
    min_x: float, min_y: float, max_x: float, max_y: float, extent_m: float, where: str
) -> None:
    """Refuse a bounds box that is not this site's square of this size.

    Grid metres against ground metres, so the comparison is deliberately loose
    — see :data:`BOUNDS_EXTENT_TOLERANCE`. It is here because the two records
    otherwise only overlap on the centre, and a box agreeing on the centre
    while disagreeing on the size would say the endpoints describe different
    renders of the same place.
    """
    width = max_x - min_x
    height = max_y - min_y
    if abs(width - height) > BOUNDS_SQUARENESS_M:
        raise SiteError(
            f"{where}: bounds3413 is {width:.3f} m by {height:.3f} m, which is not square "
            f"to within {BOUNDS_SQUARENESS_M} m, but the arena renders a square site"
        )
    span = 0.5 * (width + height)
    off = abs(span - extent_m) / extent_m
    if off > BOUNDS_EXTENT_TOLERANCE:
        raise SiteError(
            f"{where}: bounds3413 spans {span:.1f} m against a SITE_EXTENT of {extent_m:.1f} m, "
            f"{off:.2%} apart and over the {BOUNDS_EXTENT_TOLERANCE:.0%} bound. Grid metres and "
            f"ground metres differ by the polar stereographic scale factor, which is under 1% "
            f"here, so a gap this size means the box is not this site's"
        )


def _agree(env_lat: float, env_lon: float, site_lat: float, site_lon: float, where: str) -> None:
    """Refuse a record whose two endpoints disagree about where the site is."""
    offset = geodetic_to_local(GeoPoint(env_lat, env_lon), GeoPoint(site_lat, site_lon))
    apart_m = math.hypot(offset.east_m, offset.north_m)
    if apart_m > CENTRE_AGREEMENT_M:
        raise SiteError(
            f"{where}: /api/env puts the site centre at ({env_lat}, {env_lon}) and /api/site "
            f"at ({site_lat}, {site_lon}), {apart_m:.0f} m apart against a "
            f"{CENTRE_AGREEMENT_M:.0f} m bound. One of the two records is of a different site "
            f"or a different run; do not pick one"
        )


def _pick(mapping: Mapping[str, Any], *names: str) -> Any:
    """First of ``names`` present in ``mapping``, or :class:`SiteError`."""
    found = _maybe(mapping, *names)
    if found is None:
        raise SiteError(
            f"none of {', '.join(names)} is in the response, which has: {_keys(mapping)}"
        )
    return found


def _maybe(mapping: Mapping[str, Any], *names: str) -> Any:
    """First of ``names`` present and not null, matched without regard to case."""
    folded = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        value = folded.get(name.lower())
        if value is not None:
            return value
    return None


def _keys(mapping: Mapping[str, Any]) -> str:
    return ", ".join(sorted(str(key) for key in mapping)) or "(nothing)"


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SiteError(f"{where} is {type(value).__name__}, expected an object")
    return value


def _number(value: Any, where: str) -> float:
    """A finite float, from a JSON number or from ``/api/env``'s ``KEY=VALUE`` text."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SiteError(f"{where} is {type(value).__name__}, expected a number")
    try:
        number = float(value)
    except ValueError as exc:
        raise SiteError(f"{where} is {value!r}, which is not a number") from exc
    if not math.isfinite(number):
        raise SiteError(f"{where} is {number!r}, which is not finite")
    return number
