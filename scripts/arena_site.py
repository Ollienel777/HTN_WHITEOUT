#!/usr/bin/env python
"""Ask the arena what site it is rendering, and record the answer.

The only thing in this repository that talks to ``:8090``. Issue #112.

    python scripts/arena_site.py --endpoint 10.99.4.1          # show it
    python scripts/arena_site.py --endpoint 10.99.4.1 --write   # record it

The arena answers on the tunnel's first address. Which tunnel that is depends
on the config pack: the SIM-5 pack routes ``10.99.4.0/24``, so the host is
``10.99.4.1``. An earlier attempt at this recorded nothing because it probed
``10.99.0.1``, which is a different pack's /24.

**It is opt-in twice over.** It refuses to run without an endpoint named on
the command line or in ``WHITEOUT_ARENA_ENDPOINT``, and it leaves
``whiteout/data/site.json`` alone unless ``--write`` says otherwise. Nothing
imports this module: ``whiteout/site.py`` reads the record, the gate never
runs this, and CI could not reach the arena if it did.

What it records, and what it does not do to it
-----------------------------------------------

Both endpoints' bodies go into the record **as the arena sent them**, under
``response``, beside a note saying which host answered and when. This script
renames nothing and computes nothing — every check on the values lives in
:mod:`whiteout.site`, and it runs on the committed record rather than on the
wire, so that a bad answer fails in the gate for everybody rather than once,
here, for whoever happened to be holding the tunnel.

**A half-answer is recorded as a half-answer.** Either endpoint may fail on a
run; if one does, the other's answer is still written,
the failed endpoint keeps the ``null`` response that means "never fetched"
with the error in its provenance note, and the run exits 1 saying so. Losing
a fetch because the other endpoint was missing would defeat one of the two
reasons for running this at all.

**It says whether the answer moves the product.** After a successful
``--write`` it compares the recorded site against
``whiteout/belief/geometry.py``'s three constants and prints what changed. The
record is the source and those literals are its copy, so drift is meant to
fail ``tests/test_site.py`` — :func:`report_drift` is how the person holding
the arena hears about it before they commit rather than from CI afterwards.

``/api/env`` is fetched as well as ``/api/site``, even though the three site
fields came from it originally. The run that records ``/api/site`` should
record the thing it is cross-checked against at the same moment, from the same
host — which is what the committed record now holds, both halves fetched
together on 2026-09-20. Before that the ``/api/env`` half was a transcription
from PR #102, the weaker evidence of the two.

``/api/env`` answers with neither shape this script first guessed. It sends a
JSON object of a single key, ``{"env": "<the whole .env file>"}``, and the
``KEY=VALUE`` lines #102 quotes (``SITE_EXTENT=6500``) live inside that string
with the organisers' own comments around them. All three shapes are accepted
now — a plain JSON object, bare ``KEY=VALUE`` lines, and that envelope — and
the body is kept under ``body`` whenever the parse was not a plain JSON
object, so a further surprise in the format is visible in the diff rather
than swallowed here.

The one thing not kept as sent
------------------------------

That ``.env`` is the arena's own and it carries credentials, a live
``MAPBOX_TOKEN`` among them. **This repository is public.** So the single
alteration this script makes to what the arena sent is to replace the value
of any key that looks like a credential — :data:`SECRET_KEY_MARKERS` — with
:data:`REDACTED`, in the parsed fields and in the kept body alike. The key
itself stays, so a redaction reads as a redaction rather than as a gap, and
no site parameter matches those markers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from whiteout.site import SiteParameters

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "whiteout" / "data" / "site.json"


class FetchError(RuntimeError):
    """One endpoint did not answer. Fatal to that endpoint, not to the run."""


#: Where the arena answers when the endpoint names no port. ``ARENA.md`` §5
#: puts the tracks API on 8010; the environment and site endpoints are 8090.
DEFAULT_PORT = 8090

#: Seconds to wait on one request. The arena is on a LAN or a tunnel, so a
#: request that takes longer than this is a wrong host, not a slow one.
TIMEOUT_S = 10.0

#: The endpoints fetched, and the key each is recorded under.
ENDPOINTS = (("env", "/api/env"), ("site", "/api/site"))

#: A key holding a credential rather than a site parameter, matched as a
#: substring without regard to case. ``/api/env`` returns the organisers'
#: whole ``.env``, and this repository is public, so these values never reach
#: the fixture. Deliberately wider than the one key that matches today
#: (``MAPBOX_TOKEN``): the arena's ``.env`` is not ours and may grow another.
SECRET_KEY_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "APIKEY", "_KEY")

#: What a credential's value is replaced with. Says who did it, so that
#: whoever reads the fixture looks here rather than at the arena.
REDACTED = "<redacted by scripts/arena_site.py>"


def is_secret(key: str) -> bool:
    """Whether ``key``'s value is a credential and must not be recorded."""
    folded = str(key).upper()
    return any(marker in folded for marker in SECRET_KEY_MARKERS)


def base_url(endpoint: str) -> str:
    """``http://host:port`` from a bare host, a ``host:port``, or a full URL.

    Split with :func:`urllib.parse.urlsplit` rather than by counting colons.
    Counting gets ``http://arena.local/api`` wrong — one colon, so a port is
    appended to the *path* and the request goes to ``/api:8090`` — and a host
    with a path in front of it is exactly what somebody pastes out of a
    browser.
    """
    text = endpoint.strip().rstrip("/")
    if not text:
        raise SystemExit("--endpoint is empty")
    if "://" not in text:
        text = f"http://{text}"
    split = urlsplit(text)
    if not split.hostname:
        raise SystemExit(f"--endpoint {endpoint!r} names no host")
    host = f"[{split.hostname}]" if ":" in split.hostname else split.hostname
    port = split.port or DEFAULT_PORT
    return f"{split.scheme}://{host}:{port}"


def fetch(url: str) -> tuple[Any, str]:
    """``(parsed, body)`` for one endpoint. ``parsed`` is ``None`` if it is not JSON.

    :raises FetchError: on anything that stops the body arriving — a refused
        connection, a timeout, a 404. It is a plain exception rather than a
        :class:`SystemExit` because one endpoint failing must not throw away
        the other's answer: see :func:`record`.

    No proxy: the arena is a host on a tunnel, and a proxy in the way would
    mean the answer came from somewhere that is not the arena.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=TIMEOUT_S) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        raise FetchError(f"{exc}") from exc
    try:
        return json.loads(body), body
    except json.JSONDecodeError:
        return None, body


def as_fields(parsed: Any, body: str) -> tuple[dict[str, Any], str | None]:
    """``(fields, kept_body)`` — the response as an object, and the body if it differs.

    A JSON object is taken as it stands. The one-key envelope ``/api/env``
    actually sends, ``{"env": "<file>"}``, is opened and its contents read as
    ``KEY=VALUE`` lines, which is how #102 quotes them; so is a body that is
    not JSON at all. In both of those the text is kept alongside, redacted,
    so the next reader can see what was really sent.
    """
    text = body
    if isinstance(parsed, dict):
        inner = parsed.get("env")
        if len(parsed) == 1 and isinstance(inner, str):
            text = inner
        else:
            return redact_fields(parsed), None
    fields: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        fields[key.strip()] = _number_or_text(value.strip())
    if not fields:
        raise FetchError(
            f"the response is neither a JSON object nor KEY=VALUE lines:\n{body[:400]}"
        )
    return redact_fields(fields), redact_body(text)


def redact_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """``fields`` with every credential's value replaced by :data:`REDACTED`."""
    return {key: (REDACTED if is_secret(key) else value) for key, value in fields.items()}


def redact_body(body: str) -> str:
    """``body`` with every ``KEY=VALUE`` line whose key is a credential blanked.

    Line by line rather than by searching for the values themselves: a value
    that failed to parse as a line is a value we would otherwise write out,
    and the parse above and this have to agree about what a line is.
    """
    lines = []
    for line in body.splitlines():
        key, sep, _ = line.partition("=")
        lines.append(f"{key}{sep}{REDACTED}" if sep and is_secret(key.strip()) else line)
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def _number_or_text(value: str) -> Any:
    try:
        return float(value) if ("." in value or "e" in value.lower()) else int(value)
    except ValueError:
        return value


def record(endpoint: str) -> tuple[dict[str, Any], list[str]]:
    """``(record, failures)`` — fetch both endpoints off ``endpoint``.

    **One endpoint failing does not discard the other.** The first attempt at
    this record reached no arena at all, and the one that succeeded found
    ``/api/site`` answering a shape nothing here expected; a run where one of
    the two fails is an ordinary outcome, and throwing away the other's answer
    because of it would defeat half the reason to run this.
    A failed endpoint is recorded the way the record already spells "never
    fetched": a ``null`` response, with the error in its provenance note.

    The failures are returned rather than printed here, so that the caller
    decides what an incomplete run means. It means exit 1 and a loud line;
    what it does not mean is silence or a lost answer.
    """
    base = base_url(endpoint)
    at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records: dict[str, Any] = {}
    failures: list[str] = []
    for name, path in ENDPOINTS:
        url = f"{base}{path}"
        entry: dict[str, Any] = {
            "provenance": {
                "request": f"GET {url}",
                "host": base,
                "at": at,
                "note": f"Fetched by scripts/arena_site.py from a live arena at {at}.",
            }
        }
        try:
            parsed, body = fetch(url)
            fields, kept = as_fields(parsed, body)
        except FetchError as exc:
            failures.append(f"{url}: {exc}")
            entry["provenance"]["at"] = None
            entry["provenance"]["note"] = (
                f"Not recorded. scripts/arena_site.py asked {url} at {at} and got: {exc}"
            )
            entry["response"] = None
        else:
            entry["response"] = fields
            if kept is not None:
                entry["body"] = kept
        records[name] = entry
    built = {
        "schema_version": 1,
        "site": "fort_ross",
        "README": _readme(),
        "records": records,
    }
    return built, failures


def _readme() -> list[str]:
    """The committed file's own preamble, carried across a rewrite.

    Read off the existing fixture rather than spelled again here: the note is
    for whoever opens the JSON, and two copies of it would drift the first
    time one is edited.
    """
    try:
        existing = json.loads(FIXTURE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    readme = existing.get("README")
    return readme if isinstance(readme, list) else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the arena what site it renders, and record the answer."
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("WHITEOUT_ARENA_ENDPOINT", ""),
        help="arena host, host:port, or URL. Defaults to $WHITEOUT_ARENA_ENDPOINT.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"overwrite {FIXTURE.relative_to(REPO_ROOT).as_posix()} with the answer",
    )
    args = parser.parse_args(argv)
    if not args.endpoint:
        parser.error(
            "no arena endpoint. This script is the one thing here that leaves the machine, "
            "so it will not guess a host: pass --endpoint <host> or set "
            "WHITEOUT_ARENA_ENDPOINT."
        )

    built, failures = record(args.endpoint)
    text = json.dumps(built, indent=2, sort_keys=False) + "\n"
    for failure in failures:
        sys.stderr.write(f"NOT RECORDED: {failure}\n")
    if not args.write:
        sys.stdout.write(text)
        sys.stdout.write(
            f"\n# not written. Pass --write to record this in "
            f"{FIXTURE.relative_to(REPO_ROOT).as_posix()}.\n"
        )
        return 1 if failures else 0

    # Parse before overwriting: a record that whiteout.site refuses is a
    # record that would turn the gate red the moment it landed, and the
    # useful moment to hear about it is here, with the arena still reachable.
    #
    # Checked against the `whiteout` beside this script, not whichever one an
    # editable install happens to point at. The two are the same on a normal
    # checkout and differ on a worktree, and the checkout this writes the
    # fixture into is the one whose checks have to pass.
    sys.path.insert(0, str(REPO_ROOT))
    from whiteout.site import SiteError, parse_site

    try:
        site = parse_site(built, where=str(FIXTURE))
    except SiteError as exc:
        sys.stderr.write(f"the arena's answer does not check out, so nothing was written:\n{exc}\n")
        return 1
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(text, encoding="utf-8")
    sys.stdout.write(f"wrote {FIXTURE.relative_to(REPO_ROOT).as_posix()}\n")
    if site.centre_agreement_m is not None:
        sys.stdout.write(
            f"/api/site's centre agrees with /api/env's to {site.centre_agreement_m:.1f} m\n"
        )
    report_drift(site)
    return 1 if failures else 0


def report_drift(site: SiteParameters) -> None:
    """Say, here and now, whether this answer moves ``geometry.py``'s constants.

    The record is the source and those three literals are its copy, so a fetch
    that changes the record is *meant* to make
    ``tests/test_site.py::test_the_geometry_constants_are_the_recorded_ones``
    fail. What that test cannot do is tell somebody standing in front of the
    arena, before they commit, so this does — and it names the second half of
    the job, which is the part #102 had to discover: the grid shape and every
    entropy figure measured off it move with the site, and re-measuring them is
    not optional.
    """
    from whiteout.belief.geometry import SITE_CENTRE_LAT, SITE_CENTRE_LON, SITE_EXTENT_M

    constants = (
        ("SITE_CENTRE_LAT", SITE_CENTRE_LAT, site.centre_lat_deg),
        ("SITE_CENTRE_LON", SITE_CENTRE_LON, site.centre_lon_deg),
        ("SITE_EXTENT_M", SITE_EXTENT_M, site.extent_m),
    )
    drifted = [
        f"  {name:<16} {held!r} -> {recorded!r}"
        for name, held, recorded in constants
        if held != recorded
    ]
    if not drifted:
        sys.stdout.write("whiteout/belief/geometry.py's site constants still match the record\n")
        return
    sys.stdout.write(
        "\nTHE SITE MOVED. whiteout/belief/geometry.py's constants no longer match the\n"
        "record, so the gate will be red until they do:\n"
        + "\n".join(drifted)
        + "\n\nThe record is the source and those literals are the copy, so edit the copy --\n"
        "and then re-measure what is derived from it. PR #102 moved the same three and\n"
        "the grid went from 251x20 to 63x16, which moved every entropy figure stated in\n"
        "whiteout/belief/grid.py and asserted in tests/test_belief_grid.py.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
