#!/usr/bin/env python
"""Ask the arena what site it is rendering, and record the answer.

The only thing in this repository that talks to ``:8090``. Issue #112.

    python scripts/arena_site.py --endpoint 10.99.0.1          # show it
    python scripts/arena_site.py --endpoint 10.99.0.1 --write   # record it

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

**A half-answer is recorded as a half-answer.** ``/api/site`` has never been
called and may 404; if it does, the good ``/api/env`` answer is still written,
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
fields we already have came from it. Two reasons: the run that records
``/api/site`` should record the thing it is cross-checked against at the same
moment, from the same host; and ``/api/env``'s answer is currently a
transcription from PR #102 rather than a fetch, which is the weaker evidence
of the two and worth upgrading the first time anyone can.

``/api/env`` may answer with JSON or with ``KEY=VALUE`` lines — #102 quotes it
as ``SITE_EXTENT=6500``, which is the latter. Both are accepted, and the
untouched body is kept under ``body`` whenever the parse was not a plain JSON
object, so a surprise in the format is visible in the diff rather than
swallowed here.
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

    A JSON object is taken as it stands. Anything else is read as
    ``KEY=VALUE`` lines, which is how ``/api/env`` is quoted in #102, and the
    raw body is kept alongside so the next reader can see what was really
    sent.
    """
    if isinstance(parsed, dict):
        return parsed, None
    fields: dict[str, Any] = {}
    for line in body.splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        fields[key.strip()] = _number_or_text(value.strip())
    if not fields:
        raise FetchError(
            f"the response is neither a JSON object nor KEY=VALUE lines:\n{body[:400]}"
        )
    return fields, body


def _number_or_text(value: str) -> Any:
    try:
        return float(value) if ("." in value or "e" in value.lower()) else int(value)
    except ValueError:
        return value


def record(endpoint: str) -> tuple[dict[str, Any], list[str]]:
    """``(record, failures)`` — fetch both endpoints off ``endpoint``.

    **One endpoint failing does not discard the other.** ``/api/site`` has
    never been called, so a 404 on it is a live possibility, and throwing away
    a good ``/api/env`` answer because of one would defeat half the reason to
    run this at all — upgrading that half from #102's transcription to a fetch.
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
