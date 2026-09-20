"""Read the arena's ground truth, so tracking accuracy stops being unfalsifiable.

**This is a measurement instrument, not an input.** Direct simulator ground
truth must not be used where the challenge rules prohibit it, and nothing here
is wired into the system that does the finding. No module under ``whiteout/``
imports this file; no detection, belief, policy or tracking path reads it. That
is the repository's existing rule rather than a new one — ``SPEC.md`` §4 says
truth is written by the world and read only by the scorer and the viewer, and
the coordinator never sees it. This script exists to answer "how far off were
we?" after the fact, and for nothing else.

What it reads
--------------

gzweb relays Gazebo's ``~/pose/info`` topic over a WebSocket on the same port
that serves the viewer (:8080). It pushes on connect — no subscription is
needed — and every model's true pose is in it, the target vessel included.

The only bytes this sends are the RFC6455 handshake. It never publishes, never
changes Gazebo state, and never posts a track.

The transform, measured rather than assumed
--------------------------------------------

Gazebo's world x/y are **EPSG:3413 grid** axes, and grid north is not true
north. At this site the grid convergence is ``-49.80°``, which
``GET :8090/api/site`` reports directly.

Measured against MAVLink lat/lon for the same aircraft:

.. code-block:: text

    conversion                              quadcopter   fixed-wing
    no rotation                               1547.6 m      771.7 m
    rotate by -convergence                       5.7 m       66.0 m
    rotate by -convergence, times k             16.4 m       68.5 m

Two things fall out, and both are the opposite of the obvious guess.

**The convergence rotation is the whole game.** Omitting it is a ~1500 m
error — two orders of magnitude larger than the accuracy being measured.

**Do not apply the EPSG:3413 point-scale factor.** It makes the fit *worse*,
because the sponsor already corrected for it when building the terrain: their
own build log reads ``scale factor k=0.994202; footprint is 6500.0 m on the
ground [corrected]``. Applying it again double-counts, and a double-counted
scale is exactly the kind of error that yields a confident wrong accuracy
figure — the worst artifact this tool could produce.

The fixed-wing's 66 m residual above is a **time-sync artifact, not a
transform error**: it flies at ~12 m/s and the two samples are seconds apart.
The quadcopter, nearly stationary, shows the transform's real quality. Any
error series this tool produces for a *moving* object carries the same
sampling lag, and the docstring says so rather than letting a reader take
66 m for the floor.

Usage
------

.. code-block:: text

    python scripts/truth_probe.py --host 10.99.4.1 --seconds 60 \\
        --out artifacts/truth.jsonl

    python scripts/truth_probe.py --host 10.99.4.1 --seconds 60 \\
        --compare http://10.99.4.1:8010 --track "Sierra One"

**Network access is opt-in.** With no ``--host`` this opens no socket and
exits with usage, so ``python scripts/gate.py`` stays green on a machine with
no arena.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import socket
import struct
import sys
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

#: The model whose truth we want. Everything else in the stream is ours.
DEFAULT_TARGET = "target_vessel"

#: gzweb serves the viewer and relays the pose topic on the same port.
DEFAULT_GZWEB_PORT = 8080

#: Where the site's centre and grid convergence come from. Read, never assumed.
SITE_PATH = "/api/site"
DEFAULT_CONTROL_PORT = 8090


class ProbeError(RuntimeError):
    """The probe could not read what it was asked to read."""


@dataclass(frozen=True)
class Site:
    """The arena's centre and how its grid is turned from true north."""

    lat_deg: float
    lon_deg: float
    convergence_deg: float


@dataclass(frozen=True)
class TruthFix:
    """One model's true position at one instant."""

    wall: float
    name: str
    east_m: float
    north_m: float
    up_m: float
    lat_deg: float
    lon_deg: float


def read_site(host: str, port: int = DEFAULT_CONTROL_PORT) -> Site:
    """Fetch the site's centre and grid convergence from the arena itself."""
    url = f"http://{host}:{port}{SITE_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=10) as answer:
            payload: dict[str, Any] = json.loads(answer.read())
    except (OSError, ValueError) as error:
        raise ProbeError(f"could not read {url}: {error}") from error
    centre = payload.get("centre") or {}
    try:
        return Site(
            lat_deg=float(centre["lat"]),
            lon_deg=float(centre["lon"]),
            convergence_deg=float(payload["convergence_deg"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProbeError(f"{url} did not carry centre and convergence") from error


def grid_to_local(site: Site, x_m: float, y_m: float) -> tuple[float, float]:
    """Turn EPSG:3413 grid metres into East/North metres about the site centre.

    A rotation by minus the grid convergence, and **no scale factor** — see
    the module docstring for the measurement behind both halves of that.
    """
    angle = math.radians(-site.convergence_deg)
    east = x_m * math.cos(angle) - y_m * math.sin(angle)
    north = x_m * math.sin(angle) + y_m * math.cos(angle)
    return east, north


def _handshake(host: str, port: int, timeout: float) -> tuple[socket.socket, bytes]:
    stream = socket.create_connection((host, port), timeout=timeout)
    stream.settimeout(1.0)
    key = base64.b64encode(os.urandom(16)).decode()
    stream.sendall(
        (
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode()
    )
    buffered = b""
    while b"\r\n\r\n" not in buffered:
        chunk = stream.recv(4096)
        if not chunk:
            raise ProbeError(f"{host}:{port} closed during the handshake")
        buffered += chunk
    head, rest = buffered.split(b"\r\n\r\n", 1)
    if b"101" not in head.split(b"\r\n")[0]:
        raise ProbeError(f"{host}:{port} did not upgrade: {head.splitlines()[0]!r}")
    stream.settimeout(1.0)
    # The leftover bytes are returned rather than attached to the socket:
    # socket defines __slots__, so an arbitrary attribute raises.
    return stream, rest


def _frames(stream: socket.socket, rest: bytes, seconds: float) -> Iterator[dict[str, Any]]:
    """Yield decoded text frames. Server frames are unmasked, so this is short."""
    buffered = rest
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            chunk = stream.recv(65536)
        except TimeoutError:
            continue
        except OSError:
            return
        if not chunk:
            return
        buffered += chunk
        while len(buffered) >= 2:
            length = buffered[1] & 127
            offset = 2
            if length == 126:
                if len(buffered) < 4:
                    break
                length = struct.unpack(">H", buffered[2:4])[0]
                offset = 4
            elif length == 127:
                if len(buffered) < 10:
                    break
                length = struct.unpack(">Q", buffered[2:10])[0]
                offset = 10
            if len(buffered) < offset + length:
                break
            payload = buffered[offset : offset + length]
            buffered = buffered[offset + length :]
            try:
                yield json.loads(payload.decode("utf-8", "replace"))
            except ValueError:
                continue


def truth_stream(
    host: str,
    site: Site,
    *,
    port: int = DEFAULT_GZWEB_PORT,
    seconds: float = 30.0,
    target: str = DEFAULT_TARGET,
) -> Iterator[TruthFix]:
    """Yield the target's true position for ``seconds``.

    Nested links (``model::link::part``) are skipped: only the model's own
    pose is a position, and the links are its geometry.
    """
    stream, rest = _handshake(host, port, timeout=10.0)
    try:
        for message in _frames(stream, rest, seconds):
            body = message.get("msg") or {}
            name = body.get("name")
            if name != target:
                continue
            position = body.get("position") or {}
            try:
                x_m = float(position["x"])
                y_m = float(position["y"])
                up_m = float(position["z"])
            except (KeyError, TypeError, ValueError):
                continue
            east, north = grid_to_local(site, x_m, y_m)
            lat, lon = _place(site, east, north)
            yield TruthFix(
                wall=time.time(),
                name=name,
                east_m=east,
                north_m=north,
                up_m=up_m,
                lat_deg=lat,
                lon_deg=lon,
            )
    finally:
        stream.close()


def _place(site: Site, east_m: float, north_m: float) -> tuple[float, float]:
    """East/North about the site centre, as a position.

    The conversion itself is ``whiteout.geo``'s — this script does not own a
    second one. Imported inside the function so that ``--help`` works from a
    checkout with no dependencies installed.
    """
    from whiteout.geo import GeoPoint, LocalPoint, local_to_geodetic

    point = local_to_geodetic(
        GeoPoint(site.lat_deg, site.lon_deg),
        LocalPoint(east_m=east_m, north_m=north_m),
    )
    return point.lat_deg, point.lon_deg


def posted_track(endpoint: str, name: str) -> tuple[float, float] | None:
    """The position we most recently reported for ``name``, or ``None``.

    Read-only: a ``GET``, never a post.
    """
    url = f"{endpoint.rstrip('/')}/api/tracks"
    try:
        with urllib.request.urlopen(url, timeout=10) as answer:
            payload = json.loads(answer.read())
    except (OSError, ValueError) as error:
        raise ProbeError(f"could not read {url}: {error}") from error
    for row in payload.get("tracks", []):
        if row.get("name") == name and row.get("lat") is not None:
            return float(row["lat"]), float(row["lon"])
    return None


def _separation_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    from whiteout.geo import GeoPoint, geodetic_to_local

    offset = geodetic_to_local(GeoPoint(a[0], a[1]), GeoPoint(b[0], b[1]))
    return math.hypot(offset.east_m, offset.north_m)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the arena's ground truth for measurement. Never an input to "
            "detection, belief or tracking."
        )
    )
    parser.add_argument("--host", help="arena address, e.g. 10.99.4.1. Required to open a socket.")
    parser.add_argument("--port", type=int, default=DEFAULT_GZWEB_PORT)
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--out", help="write JSONL here")
    parser.add_argument("--compare", help="tracks API base URL, to diff against what we posted")
    parser.add_argument("--track", default="Sierra One", help="track name to compare against")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.host:
        # No host, no socket. The gate runs here with no arena.
        build_parser().print_usage(sys.stderr)
        print(
            "truth_probe: --host is required; without it this opens no socket",
            file=sys.stderr,
        )
        return 2
    try:
        site = read_site(args.host, args.control_port)
    except ProbeError as error:
        print(f"truth_probe: {error}", file=sys.stderr)
        return 1
    print(
        f"site {site.lat_deg:.6f},{site.lon_deg:.6f} "
        f"convergence {site.convergence_deg:.4f}deg (rotation applied, no scale factor)",
        file=sys.stderr,
    )

    reported: tuple[float, float] | None = None
    if args.compare:
        try:
            reported = posted_track(args.compare, args.track)
        except ProbeError as error:
            print(f"truth_probe: {error}", file=sys.stderr)
            return 1
        if reported is None:
            print(f"truth_probe: no track named {args.track!r} to compare", file=sys.stderr)

    handle = open(args.out, "w", encoding="utf-8") if args.out else None
    errors: list[float] = []
    count = 0
    try:
        for fix in truth_stream(
            args.host,
            site,
            port=args.port,
            seconds=args.seconds,
            target=args.target,
        ):
            count += 1
            row: dict[str, Any] = {
                "wall": fix.wall,
                "name": fix.name,
                "east_m": round(fix.east_m, 2),
                "north_m": round(fix.north_m, 2),
                "lat": fix.lat_deg,
                "lon": fix.lon_deg,
            }
            if reported is not None:
                separation = _separation_m(reported, (fix.lat_deg, fix.lon_deg))
                errors.append(separation)
                row["posted_error_m"] = round(separation, 1)
            if handle is not None:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    except ProbeError as error:
        print(f"truth_probe: {error}", file=sys.stderr)
        return 1
    finally:
        if handle is not None:
            handle.close()

    print(f"truth_probe: {count} fixes of {args.target!r}", file=sys.stderr)
    if errors:
        errors.sort()
        middle = errors[len(errors) // 2]
        print(
            f"truth_probe: error against {args.track!r} — "
            f"min {errors[0]:.0f} m, median {middle:.0f} m, max {errors[-1]:.0f} m. "
            f"A moving target carries the sampling lag between the two feeds; "
            f"see the module docstring.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
