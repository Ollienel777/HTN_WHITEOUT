"""A stand-in for Dominion Dynamics' tracks API.

Everything above the tracks client — the track-maintenance loop, the policy,
the CLI — has to be buildable and testable without the arena. The arena is one
laptop on a WireGuard network, it is not on CI, and it is not available at four
in the morning. So this reproduces the parts of the real endpoint that the rest
of the system depends on, and nothing else.

What it reproduces, because behaviour above it turns on these
-------------------------------------------------------------

- ``name`` is the identity key: first post mints a uuid and answers
  ``created: true``, every later post under that name updates it and answers
  ``created: false``.
- ``fixes`` counts updates, starting at 1 for the create. This is how the real
  API shows tracking duration, so it is the number a test asserts against when
  it wants to know that a track was *maintained* rather than re-created.
- ``name`` is required, and a body without one is a 400 with
  ``{"ok": false, "error": "name is required"}`` — the observed wording.
- ``heading`` and ``speed`` are stored when present and left as ``null`` when
  not, so a client that omits them can tell the difference on read-back.

What it deliberately does not reproduce
----------------------------------------

Scoring. The real endpoint is the sponsor's, and whatever it computes from
these rows is theirs. Faking a score here would invite us to optimise against
our own invention, which is the one mistake this project cannot afford to make
twice.

It binds an ephemeral port by default, so parallel worktrees and parallel test
runs never collide (``SPEC.md`` §6).
"""

from __future__ import annotations

import json
import threading
import uuid as _uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

__all__ = ["StubTracksServer", "open_stub_tracks_server"]

_HOST = "127.0.0.1"
_HINT = 'POST /api/tracks with {"name":"...","lat":0,"lon":0}'


class _Store:
    """The tracks the stub is holding, and the lock that guards them."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []

    def upsert(self, body: dict[str, Any]) -> dict[str, Any]:
        name = body["name"]
        with self.lock:
            row = self.rows.get(name)
            created = row is None
            if row is None:
                row = {
                    "name": name,
                    "uuid": f"entity-{_uuid.uuid4().hex[:10]}",
                    "fixes": 0,
                    "heading": None,
                    "speed": None,
                }
                self.rows[name] = row
                self.order.append(name)
            row["lat"] = body.get("lat")
            row["lon"] = body.get("lon")
            if "heading" in body:
                row["heading"] = body["heading"]
            if "speed" in body:
                row["speed"] = body["speed"]
            row["fixes"] = int(row["fixes"]) + 1
            answer = dict(row)
        answer["ok"] = True
        answer["created"] = created
        return answer

    def listing(self) -> dict[str, Any]:
        with self.lock:
            rows = [dict(self.rows[name]) for name in self.order]
        return {"ok": True, "count": len(rows), "tracks": rows}


class _Handler(BaseHTTPRequestHandler):
    """One request. ``store`` is attached to the server, not to the handler."""

    protocol_version = "HTTP/1.1"

    @property
    def _store(self) -> _Store:
        store = getattr(self.server, "store", None)
        assert isinstance(store, _Store)
        return store

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence. A stub that prints a line per fix drowns pytest output."""

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.rstrip("/") != "/api/tracks":
            self._send(404, {"ok": False, "error": "not found", "hint": _HINT})
            return
        self._send(200, self._store.listing())

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.rstrip("/") != "/api/tracks":
            self._send(404, {"ok": False, "error": "not found", "hint": _HINT})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "body must be JSON"})
            return
        if not isinstance(body, dict) or not str(body.get("name", "")).strip():
            self._send(400, {"ok": False, "error": "name is required"})
            return
        self._send(200, self._store.upsert(body))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    store: _Store


class StubTracksServer:
    """A running stub, with the URL a :class:`TracksClient` should be given.

    Use it as a context manager. :attr:`endpoint` is the base URL, so it drops
    straight into ``TracksClient(server.endpoint)`` and into
    ``WHITEOUT_TRACKS_ENDPOINT``.
    """

    def __init__(self, port: int = 0) -> None:
        self._server = _Server((_HOST, port), _Handler)
        self._server.store = _Store()
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.02),
            name="stub-tracks",
            daemon=True,
        )
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound port. Ephemeral unless one was asked for."""
        return int(self._server.server_address[1])

    @property
    def endpoint(self) -> str:
        """Base URL, for example ``http://127.0.0.1:53124``."""
        return f"http://{_HOST}:{self.port}"

    def tracks(self) -> list[dict[str, Any]]:
        """The stored rows, read directly rather than over HTTP."""
        listing = self._server.store.listing()
        rows: list[dict[str, Any]] = listing["tracks"]
        return rows

    def close(self) -> None:
        """Stop serving and release the port."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(5.0)

    def __enter__(self) -> StubTracksServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_stub_tracks_server(port: int = 0) -> StubTracksServer:
    """Start a stub tracks API on ``port``, ephemeral when zero."""
    return StubTracksServer(port)
