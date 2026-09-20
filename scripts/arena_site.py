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
``fixtures/arena/site.json`` alone unless ``--write`` says otherwise. Nothing
imports this module: ``whiteout/site.py`` reads the fixture, the gate never
runs this, and CI could not reach the arena if it did.

What it records, and what it does not do to it
-----------------------------------------------

Both endpoints' bodies go into the fixture **as the arena sent them**, under
``response``, beside a note saying which host answered and when. This script
renames nothing and computes nothing — every check on the values lives in
:mod:`whiteout.site`, and it runs on the committed record rather than on the
wire, so that a bad answer fails in the gate for everybody rather than once,
here, for whoever happened to be holding the tunnel.

``/api/env`` is fetched as well as ``/api/site``, even though the three site
fields we already have came from it. Two reasons: the run that records
``/api/site`` should record the thing it is cross-checked against at the same
moment, from the same host; and ``/api/env``'s answer is currently a
transcription from PR #102 rather than a fetch, which is the weaker evidence
of the two and worth upgrading the first time anyone can.

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "fixtures" / "arena" / "site.json"

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
    """``http://host:port`` from a bare host, a ``host:port``, or a full URL."""
    text = endpoint.strip().rstrip("/")
    if not text:
        raise SystemExit("--endpoint is empty")
    if "://" not in text:
        text = f"http://{text}"
    if text.count(":") < 2:
        text = f"{text}:{DEFAULT_PORT}"
    return text


def fetch(url: str) -> tuple[Any, str]:
    """``(parsed, body)`` for one endpoint. ``parsed`` is ``None`` if it is not JSON.

    No proxy handler and no redirect following beyond urllib's default: the
    arena is a host on a tunnel, and a proxy in the way would mean the answer
    came from somewhere that is not the arena.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=TIMEOUT_S) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"{url}: {exc}") from exc
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
        raise SystemExit(
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


def record(endpoint: str) -> dict[str, Any]:
    """Fetch both endpoints off ``endpoint`` and build the fixture's contents."""
    base = base_url(endpoint)
    at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records: dict[str, Any] = {}
    for name, path in ENDPOINTS:
        url = f"{base}{path}"
        parsed, body = fetch(url)
        fields, kept = as_fields(parsed, body)
        entry: dict[str, Any] = {
            "provenance": {
                "request": f"GET {url}",
                "host": base,
                "at": at,
                "note": f"Fetched by scripts/arena_site.py from a live arena at {at}.",
            },
            "response": fields,
        }
        if kept is not None:
            entry["body"] = kept
        records[name] = entry
    return {
        "schema_version": 1,
        "site": "fort_ross",
        "README": _readme(),
        "records": records,
    }


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

    built = record(args.endpoint)
    text = json.dumps(built, indent=2, sort_keys=False) + "\n"
    if not args.write:
        sys.stdout.write(text)
        sys.stdout.write(
            f"\n# not written. Pass --write to record this in "
            f"{FIXTURE.relative_to(REPO_ROOT).as_posix()}.\n"
        )
        return 0

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
        parse_site(built, where=str(FIXTURE))
    except SiteError as exc:
        sys.stderr.write(f"the arena's answer does not check out, so nothing was written:\n{exc}\n")
        return 1
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(text, encoding="utf-8")
    sys.stdout.write(f"wrote {FIXTURE.relative_to(REPO_ROOT).as_posix()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
