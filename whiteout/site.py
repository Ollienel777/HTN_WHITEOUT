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

``whiteout/data/site.json`` is the record. :func:`load_site` reads it and
never opens a socket; ``scripts/arena_site.py`` is the only thing that talks
to the arena, it refuses to run without an endpoint named on the command line,
and it overwrites the record only when asked with ``--write``. That split is
the whole design, and it has two reasons:

* **CI cannot reach the arena.** A module that fetched on import would make
  the gate depend on a WireGuard tunnel to a laptop in a venue.
* **A fetched number with no record is a number nobody can check.** The
  record carries the raw response and the provenance beside it, so the next
  person can see what was asked, of which host, and when — which is exactly
  what ``DEFAULT_STRAIT``'s centreline still cannot show.

It sits inside the package rather than in ``fixtures/`` because it is this
module's data and not a test's: ``pyproject.toml`` excludes ``fixtures*`` from
the wheel, so a record kept there would make :func:`load_site` a function that
cannot work as documented anywhere but a source checkout. ``fixtures/`` holds
what the tests feed the product — episodes, frames, weights — and this is the
product's own reference data.

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
    North. **We have never called it.** The fixture therefore records it as
    *not recorded*, rather than inventing a plausible answer, and
    :class:`SiteParameters` reports ``convergence_deg`` and ``bounds3413`` as
    ``None`` until somebody runs the script against a live arena.

Which centre wins, and the two questions that are not the same question
-------------------------------------------------------------------------

**``/api/env``'s centre is the site centre, always**, even when ``/api/site``
has been recorded and quotes one too. ``/api/site``'s is *corroboration*: it is
checked against ``/api/env``'s and then reported as
:attr:`SiteParameters.centre_agreement_m`, the measured separation, rather
than substituted for it.

That was not the first design, and the reason it is this one is worth stating,
because two plausible-looking bounds here are answers to different questions.

*Is this the same site?* — :data:`CENTRE_AGREEMENT_M`, 50 m. Two endpoints of
one arena describing one site; a kilometre apart means one of the records is
of a different site or a different run, and picking either would be guessing.
Fifty metres is deliberately loose, because an endpoint is free to quote a
centre to four decimal places and four decimal places of latitude is 11 m.

*Is ``geometry.py``'s copy still the record's?* —
``tests/test_site.py::test_the_geometry_constants_are_the_recorded_ones``,
exact equality. Those three literals are a transcription of this record, and a
transcription is either right or it is not; a tolerance there would let real
drift accumulate under it, which is the failure this whole module exists to
stop.

Letting the looser bound feed the stricter check is how those two collapse
into one. If ``/api/site``'s coarser centre were preferred, an entirely
legitimate ``--write`` — the arena answering 71.9920 where ``/api/env`` says
71.991960, 4.4 m apart and well inside the corroboration bound — would move
:data:`whiteout.belief.geometry.SITE_CENTRE_LAT`'s counterpart and turn the
gate red for a rounding. So it is not preferred, and
:func:`test_a_coarsely_quoted_site_centre_does_not_move_the_parameters
<tests.test_site>` is the regression.

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
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whiteout.geo import GeoError, GeoPoint, geodetic_to_local

__all__ = [
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

#: The committed record, beside this module and inside the package, so that
#: :func:`load_site` works the same from a source checkout and from an
#: installed wheel. ``pyproject.toml``'s ``package-data`` is what puts it in
#: the wheel, and ``tests/test_site.py`` checks that it is still there.
SITE_FIXTURE = Path(__file__).resolve().parent / "data" / "site.json"

#: EPSG:3413's central meridian, degrees East. NSIDC Sea Ice Polar
#: Stereographic North: latitude of true scale 70° N, central meridian 45° W.
GRID_CENTRAL_MERIDIAN_DEG = -45.0

#: How far a recorded convergence may sit from ``lon - GRID_CENTRAL_MERIDIAN``
#: before :func:`parse_site` refuses the fixture. See the module docstring:
#: this absorbs a rounded centre, not a different projection.
CONVERGENCE_TOLERANCE_DEG = 0.5

#: How far ``/api/env``'s site centre and ``/api/site``'s may sit apart,
#: metres, before :func:`parse_site` refuses the record. One arena, one site,
#: one centre; a kilometre of disagreement is a bug in the record and not a
#: rounding. **It is a corroboration bound and not a precision one** — it is
#: loose on purpose, and nothing derived from the centre is allowed to move
#: within it. See the module docstring, "Which centre wins".
CENTRE_AGREEMENT_M = 50.0


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

    :param centre_lat_deg: site centre, degrees North, from ``/api/env``.
    :param centre_lon_deg: site centre, degrees East, from ``/api/env``.
    :param extent_m: the site's square extent, metres. The whole of the
        world: there is no terrain, no water and no vessel outside it.
    :param centre_agreement_m: how far ``/api/site``'s centre sits from the
        two above, metres, or ``None`` when ``/api/site`` has not been
        recorded. Reported rather than acted on: see the module docstring on
        why the centre is never taken from ``/api/site``.
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
    centre_agreement_m: float | None
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
    env_lat = _number(_pick(env_raw, "SITE_LAT", "lat"), f"{env_where}: SITE_LAT")
    env_lon = _number(_pick(env_raw, "SITE_LON", "lon"), f"{env_where}: SITE_LON")
    extent_m = _number(_pick(env_raw, "SITE_EXTENT", "extent"), f"{env_where}: SITE_EXTENT")
    if extent_m <= 0.0:
        raise SiteError(f"{env_where}: SITE_EXTENT must be positive, got {extent_m!r}")
    # Validated here and not only where the two centres are compared, so that a
    # swapped pair -- `lat=-94.82`, the mix-up the whole build guards against --
    # is refused even when `/api/site` has never been recorded.
    centre = _place(env_lat, env_lon, f"{env_where}: SITE_LAT/SITE_LON")

    agreement_m: float | None = None
    convergence_deg: float | None = None
    bounds: tuple[float, float, float, float] | None = None

    if site_raw is not None:
        # Corroboration only. `/api/env`'s centre is the site centre, and
        # `/api/site`'s is measured against it and then reported: see the
        # module docstring, "Which centre wins".
        agreement_m = _agreement_m(centre, _centre(site_raw, site_where), where)
        convergence_deg = _convergence(site_raw, env_lon, site_where)
        bounds = _bounds(site_raw, site_where)

    return SiteParameters(
        centre_lat_deg=env_lat,
        centre_lon_deg=env_lon,
        extent_m=extent_m,
        centre_agreement_m=agreement_m,
        convergence_deg=convergence_deg,
        bounds3413=bounds,
        env=_provenance(records, "env", where),
        site=_provenance(records, "site", where) if site_raw is not None else None,
    )


# -- the readers ------------------------------------------------------------
#
# Every reader below takes the response **as the arena sent it**, and each
# field has **one accepted spelling**, with two exceptions that are not
# guesses: `centre`/`center`, which is one word with two spellings, and the
# shape of the centre, which may be a nested object or a pair because both are
# ordinary ways to send one.
#
# An earlier revision accepted a handful of invented variants -- `bounds_3413`,
# `grid_convergence_deg`, `site_lat` -- on the theory that the fetch should not
# fail at the venue for a name nobody had seen. That was imagination rather
# than evidence, which is what this module's own docstring warns against, and
# it bought nothing: a miss fails loudly here, naming every key the response
# did have, the raw body is kept in the record either way, and widening the
# list afterwards is a one-line change made from the evidence in front of you.
# (Half of them were dead anyway: `_maybe` folds case, so `SITE_LAT` and
# `site_lat` were always the same lookup.)


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


def _centre(response: Mapping[str, Any], where: str) -> GeoPoint:
    """The site centre out of an ``/api/site`` response."""
    nested = _maybe(response, "centre", "center")
    if isinstance(nested, Mapping):
        return _place(
            _number(_pick(nested, "lat"), f"{where}: centre.lat"),
            _number(_pick(nested, "lon"), f"{where}: centre.lon"),
            f"{where}: centre",
        )
    if isinstance(nested, list | tuple):
        if len(nested) != 2:
            raise SiteError(f"{where}: centre is a list of {len(nested)}, expected [lat, lon]")
        return _place(
            _number(nested[0], f"{where}: centre[0]"),
            _number(nested[1], f"{where}: centre[1]"),
            f"{where}: centre",
        )
    lat = _maybe(response, "lat")
    lon = _maybe(response, "lon")
    if lat is None or lon is None:
        raise SiteError(
            f"{where}: no site centre. Looked for 'centre' or 'center' holding a "
            f"[lat, lon] or a {{lat, lon}}, and for a flat lat/lon pair; the response "
            f"has: {_keys(response)}"
        )
    return _place(_number(lat, f"{where}: lat"), _number(lon, f"{where}: lon"), f"{where}: lat/lon")


def _convergence(response: Mapping[str, Any], lon_deg: float, where: str) -> float:
    """The recorded convergence, checked against the projection's own identity."""
    value = _number(
        _pick(response, "convergence_deg"),
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


def _bounds(response: Mapping[str, Any], where: str) -> tuple[float, float, float, float]:
    raw = _pick(response, "bounds3413")
    if not isinstance(raw, list | tuple):
        raise SiteError(f"{where}: bounds3413 is {type(raw).__name__}, expected four numbers")
    if len(raw) != 4:
        raise SiteError(f"{where}: bounds3413 has {len(raw)} entries, expected four")
    min_x, min_y, max_x, max_y = (
        _number(value, f"{where}: bounds3413[{index}]") for index, value in enumerate(raw)
    )
    if min_x >= max_x or min_y >= max_y:
        raise SiteError(
            f"{where}: bounds3413 is not [min_x, min_y, max_x, max_y] — got {list(raw)!r}"
        )
    return min_x, min_y, max_x, max_y


def _place(lat_deg: float, lon_deg: float, where: str) -> GeoPoint:
    """A :class:`~whiteout.geo.GeoPoint`, with its refusal spelled as a :class:`SiteError`.

    :class:`~whiteout.geo.GeoPoint` raises :class:`~whiteout.geo.GeoError` on a
    position outside geodetic range — a swapped pair at this site reads
    ``lat=-94.82``, which is not a latitude. That refusal is wanted; its *type*
    is not, because it would escape this module's documented
    ``:raises SiteError:`` and slip straight past ``arena_site.py``'s
    ``except SiteError``, turning the designed refusal message into a
    traceback at the moment somebody is standing in front of the arena.
    """
    try:
        return GeoPoint(lat_deg, lon_deg)
    except GeoError as exc:
        raise SiteError(f"{where} is not a position: {exc}") from exc


def _agreement_m(env: GeoPoint, site: GeoPoint, where: str) -> float:
    """How far the two recorded centres sit apart, refusing a record they disagree in.

    The answer is returned rather than only asserted: a record that passes at
    4 m and one that passes at 45 m are different records, and
    :attr:`SiteParameters.centre_agreement_m` is where that shows.
    """
    offset = geodetic_to_local(env, site)
    apart_m = math.hypot(offset.east_m, offset.north_m)
    if apart_m > CENTRE_AGREEMENT_M:
        raise SiteError(
            f"{where}: /api/env puts the site centre at ({env.lat_deg}, {env.lon_deg}) and "
            f"/api/site at ({site.lat_deg}, {site.lon_deg}), {apart_m:.0f} m apart against a "
            f"{CENTRE_AGREEMENT_M:.0f} m bound. One of the two records is of a different site "
            f"or a different run; do not pick one"
        )
    return apart_m


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
