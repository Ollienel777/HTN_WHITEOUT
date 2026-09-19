"""Client for Dominion Dynamics' tracks API, and the poster that feeds it.

``hackathon/ARENA.md`` §5 specifies the whole surface:

.. code-block:: text

    POST /api/tracks  {"name","lat","lon"}                    -> created: true
    POST /api/tracks  {"name","lat","lon","heading","speed"}  -> created: false
    GET  /api/tracks                                          -> {"tracks": [...]}

``name`` is the identity key. The first post under a name mints a uuid and
answers ``created: true``; every later post under the same name updates that
record, answers ``created: false``, and increments its ``fixes`` count. So
"tracking" is not one call — it is the same call, repeated, for as long as the
vessel is held.

Two design points that the acceptance criteria turn on
------------------------------------------------------

**A failed post is retried, because a lost update is a lost scoring second.**
:class:`TracksClient` retries on connection failure, timeout and 5xx, with a
bounded exponential backoff. It does *not* retry a 4xx: the server rejecting
our body is a bug on our side, and retrying it just sends the same wrong thing
three times.

**Posting must not stall the control loop, and a stale fix must never be sent
as if it were fresh.** Those two pull in opposite directions, and
:class:`TrackPoster` resolves both the same way: a **latest-wins** queue, one
slot per track name. :meth:`TrackPoster.submit` never blocks and never raises;
it replaces whatever is still pending for that name. If the endpoint is slow
the worker falls behind and the intermediate fixes are *superseded* rather
than queued — which is what we want, because a stale position posted behind
five older ones scores worse than the newest position posted once.

Superseded fixes are counted (:attr:`TrackPoster.superseded`) rather than
dropped silently: a steadily rising count means the endpoint cannot keep up
with the tick rate, and that is worth being able to see.

**The address is configuration, never a constant.** ``WHITEOUT_TRACKS_ENDPOINT``,
read by :func:`endpoint_from_env`. Absent, it raises rather than falling back
to an address that happened to work once — the arena is handed out per team,
and a baked-in IP is a silent no-score.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

__all__ = [
    "ENDPOINT_ENV",
    "TrackAck",
    "TrackFix",
    "TrackPoster",
    "TrackRecord",
    "TracksClient",
    "TracksError",
    "endpoint_from_env",
]

#: Environment variable naming the tracks API's base URL, for example
#: ``http://10.99.4.1:8010``. ``SPEC.md`` §7 keeps every endpoint in the
#: environment; this is that table's tracks row.
ENDPOINT_ENV = "WHITEOUT_TRACKS_ENDPOINT"

#: Retried: the endpoint was unreachable, slow, or broke on its own side.
_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class TracksError(RuntimeError):
    """A tracks API call that could not be completed."""


@dataclass(frozen=True)
class TrackFix:
    """One report of where a track is, and optionally where it is going.

    ``heading_deg`` and ``speed_mps`` are optional in the API and are left out
    of the payload entirely when ``None``. An omitted field and a field sent
    as ``null`` are not the same thing to a server we did not write, and there
    is no reason to make it decide which we meant.
    """

    name: str
    lat_deg: float
    lon_deg: float
    heading_deg: float | None = None
    speed_mps: float | None = None

    def payload(self) -> dict[str, Any]:
        """The JSON body for ``POST /api/tracks``."""
        body: dict[str, Any] = {
            "name": self.name,
            "lat": self.lat_deg,
            "lon": self.lon_deg,
        }
        if self.heading_deg is not None:
            body["heading"] = self.heading_deg
        if self.speed_mps is not None:
            body["speed"] = self.speed_mps
        return body


@dataclass(frozen=True)
class TrackAck:
    """What the API answered to a post.

    ``created`` is the useful field: ``True`` the first time a name is seen,
    ``False`` on every update to it. A run that only ever sees ``True`` is
    minting a new track per fix instead of maintaining one, and scores as many
    one-fix tracks rather than as one long one.
    """

    ok: bool
    created: bool
    name: str
    uuid: str | None = None


@dataclass(frozen=True)
class TrackRecord:
    """One row of ``GET /api/tracks``."""

    name: str
    uuid: str | None
    lat_deg: float | None
    lon_deg: float | None
    fixes: int
    heading_deg: float | None = None
    speed_mps: float | None = None


def endpoint_from_env(env: dict[str, str] | None = None) -> str:
    """The tracks endpoint from ``WHITEOUT_TRACKS_ENDPOINT``.

    Raises :class:`TracksError` when unset. There is deliberately no default:
    the arena's address is handed out per team, and an endpoint quietly
    pointing at nothing posts nothing and scores nothing without ever failing.
    """
    source = os.environ if env is None else env
    value = source.get(ENDPOINT_ENV, "").strip()
    if not value:
        raise TracksError(
            f"{ENDPOINT_ENV} is unset; set it to the tracks API base URL, "
            f"for example http://10.99.4.1:8010"
        )
    return value.rstrip("/")


class _Retryable(Exception):
    """Internal: a failure worth another attempt."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


class TracksClient:
    """Blocking, retrying client for one tracks endpoint.

    Thread-safe: it holds no per-call state, so :class:`TrackPoster` can drive
    one instance from its worker while a test drives the same instance
    directly.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 2.0,
        attempts: int = 3,
        backoff_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if attempts < 1:
            raise TracksError(f"attempts must be at least 1, got {attempts!r}")
        self._endpoint = endpoint.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._attempts = int(attempts)
        self._backoff_s = float(backoff_s)
        self._sleep = sleep

    @property
    def endpoint(self) -> str:
        """The base URL this client posts to."""
        return self._endpoint

    def post(self, fix: TrackFix) -> TrackAck:
        """Create or update one track, retrying what is worth retrying.

        Raises :class:`TracksError` when every attempt failed, so the caller
        is told rather than the update being dropped on the floor.
        """
        body = json.dumps(fix.payload()).encode("utf-8")
        last: BaseException | None = None
        for attempt in range(self._attempts):
            try:
                raw = self._request("POST", "/api/tracks", body)
            except _Retryable as error:
                last = error.cause
                if attempt + 1 < self._attempts:
                    self._sleep(self._backoff_s * (2**attempt))
                continue
            answer = json.loads(raw)
            return TrackAck(
                ok=bool(answer.get("ok", False)),
                created=bool(answer.get("created", False)),
                name=str(answer.get("name", fix.name)),
                uuid=answer.get("uuid"),
            )
        raise TracksError(
            f"posting {fix.name!r} to {self._endpoint} failed after "
            f"{self._attempts} attempts: {last}"
        )

    def list(self) -> list[TrackRecord]:
        """Every track the endpoint currently holds."""
        last: BaseException | None = None
        for attempt in range(self._attempts):
            try:
                raw = self._request("GET", "/api/tracks", None)
            except _Retryable as error:
                last = error.cause
                if attempt + 1 < self._attempts:
                    self._sleep(self._backoff_s * (2**attempt))
                continue
            answer = json.loads(raw)
            rows = answer.get("tracks", [])
            return [_record(row) for row in rows]
        raise TracksError(f"listing tracks at {self._endpoint} failed: {last}")

    def _request(self, method: str, path: str, body: bytes | None) -> bytes:
        request = urllib.request.Request(f"{self._endpoint}{path}", data=body, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as answer:
                return bytes(answer.read())
        except urllib.error.HTTPError as error:
            if error.code in _RETRY_STATUS:
                raise _Retryable(error) from error
            # A 4xx is our bug. Retrying sends the same wrong body again.
            raise TracksError(
                f"{method} {path} rejected with {error.code}: {error.reason}"
            ) from error
        except (urllib.error.URLError, OSError) as error:
            raise _Retryable(error) from error


def _record(row: dict[str, Any]) -> TrackRecord:
    return TrackRecord(
        name=str(row.get("name", "")),
        uuid=row.get("uuid"),
        lat_deg=row.get("lat"),
        lon_deg=row.get("lon"),
        fixes=int(row.get("fixes", 0)),
        heading_deg=row.get("heading"),
        speed_mps=row.get("speed"),
    )


class TrackPoster:
    """Non-blocking, latest-wins posting of fixes to the tracks API.

    :meth:`submit` returns immediately and never raises. There is one slot per
    track name: submitting again for a name that has not been sent yet
    *replaces* the pending fix rather than queueing behind it, so a slow
    endpoint costs us intermediate fixes and never costs us freshness.

    Use it as a context manager, or call :meth:`close` — both drain what is
    pending before returning, so a run that ends does not lose its last fix.
    """

    def __init__(self, client: TracksClient, *, poll_s: float = 0.02) -> None:
        self._client = client
        self._poll_s = float(poll_s)
        self._pending: dict[str, TrackFix] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._posted = 0
        self._failed = 0
        self._superseded = 0
        self._worker = threading.Thread(target=self._run, name="track-poster", daemon=True)
        self._worker.start()

    def submit(self, fix: TrackFix) -> None:
        """Queue a fix. Never blocks, never raises, latest wins per name."""
        with self._lock:
            if fix.name in self._pending:
                self._superseded += 1
            self._pending[fix.name] = fix
        self._wake.set()

    @property
    def posted(self) -> int:
        """Fixes the endpoint accepted."""
        with self._lock:
            return self._posted

    @property
    def failed(self) -> int:
        """Fixes that exhausted their retries and were given up on."""
        with self._lock:
            return self._failed

    @property
    def superseded(self) -> int:
        """Fixes replaced before they were ever sent.

        Rising steadily means the endpoint cannot keep up with the tick rate.
        """
        with self._lock:
            return self._superseded

    def _take(self) -> TrackFix | None:
        with self._lock:
            if not self._pending:
                return None
            name = next(iter(self._pending))
            return self._pending.pop(name)

    def _run(self) -> None:
        while True:
            self._wake.wait(self._poll_s)
            self._wake.clear()
            while True:
                fix = self._take()
                if fix is None:
                    break
                try:
                    self._client.post(fix)
                except TracksError:
                    with self._lock:
                        self._failed += 1
                else:
                    with self._lock:
                        self._posted += 1
            if self._stop.is_set():
                with self._lock:
                    if not self._pending:
                        return

    def close(self, timeout_s: float = 5.0) -> None:
        """Drain what is pending, then stop the worker."""
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout_s)

    def __enter__(self) -> TrackPoster:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
