"""The viewer's dev server: ``whiteout serve``.

``hackathon/SPEC.md`` §6 states the two rules this module exists to keep:

    take any port from ``PORT``, with no reuse of a running server. The viewer
    dev server (``whiteout.cli serve``) binds ``PORT`` or an ephemeral port,
    never a fixed 8000. Parallel worktrees run side by side.

So there is no default port. ``PORT`` unset means port 0, which the kernel
answers with a free one, and the chosen port is printed. ``PORT`` set means
that port and no other: if something is already listening there the server
refuses rather than attaching to whatever is, because "the viewer came up" and
"the viewer came up **from this worktree**" have to be the same statement when
five worktrees are running at once. ``allow_reuse_address`` is off for the
same reason.

The server is a plain static file server over the repository root, so that the
viewer at ``/viz/`` can read the episode fixtures at ``/fixtures/`` beside it.
Nothing here imports from ``viz/`` — it only hands its files out (``SPEC.md``
§5 "Conventions").
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__all__ = [
    "VIEWER_INDEX",
    "VIEWER_ROOT",
    "ServeError",
    "open_viewer_server",
    "resolve_port",
    "viewer_url",
]

#: The tree handed out. ``whiteout/serve.py`` -> ``whiteout/`` -> the root.
VIEWER_ROOT = Path(__file__).resolve().parent.parent

#: The page ``serve`` points the browser at, relative to :data:`VIEWER_ROOT`.
VIEWER_INDEX = "viz/index.html"

#: Loopback only. The viewer is a debugging tool for the machine it runs on,
#: and a venue network is not somewhere to publish one by accident.
HOST = "127.0.0.1"


class ServeError(RuntimeError):
    """The server could not be started, with a reason for the operator."""


def resolve_port(environ: Mapping[str, str] | None = None) -> int:
    """``PORT`` as an integer, or 0 for "let the kernel pick".

    Raises :class:`ServeError` for a ``PORT`` that is not a port, rather than
    falling back to a default: a typo in the variable that silently produced
    some other port would hand the operator a URL for a server that is not the
    one they configured.
    """
    source = os.environ if environ is None else environ
    raw = source.get("PORT", "").strip()
    if not raw:
        return 0
    try:
        port = int(raw)
    except ValueError:
        raise ServeError(f"PORT={raw!r} is not an integer") from None
    if not 0 <= port <= 65535:
        raise ServeError(f"PORT={port} is outside 0-65535")
    return port


class _ViewerServer(ThreadingHTTPServer):
    """A static server that refuses to share a port with anything.

    ``allow_reuse_address`` is ``False`` deliberately (``SPEC.md`` §6, "no
    reuse of a running server"): with it on, a port left in ``TIME_WAIT`` by
    another worktree's server binds silently, and the operator cannot tell
    which tree answered them.
    """

    allow_reuse_address = False
    daemon_threads = True


class _ViewerHandler(SimpleHTTPRequestHandler):
    """Static files, plus ``/`` -> the viewer."""

    #: `.jsonl` is not in the system mime table, and an episode log served as
    #: `application/octet-stream` is one a browser offers to download rather
    #: than one `fetch` reads.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".jsonl": "application/x-ndjson",
    }

    def do_GET(self) -> None:  # noqa: N802 — the base class names it
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/" + VIEWER_INDEX)
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        # The viewer is edited and reloaded continuously while the rest of the
        # build is debugged through it; a cached tokens.css is a wasted minute
        # every time.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def open_viewer_server(port: int, root: Path | None = None) -> ThreadingHTTPServer:
    """Bind a viewer server on ``port``, or raise :class:`ServeError`.

    The caller owns the returned server: call ``serve_forever`` on it, and
    ``server_close`` when done. ``port`` 0 binds an ephemeral port, which
    ``server.server_address[1]`` then reports.
    """
    tree = VIEWER_ROOT if root is None else root
    index = tree / VIEWER_INDEX
    if not index.is_file():
        raise ServeError(
            f"no viewer at {index}; `serve` hands out the source tree, so run it "
            f"from a checkout rather than from an installed wheel"
        )
    handler = partial(_ViewerHandler, directory=str(tree))
    try:
        return _ViewerServer((HOST, port), handler)
    except OSError as error:
        raise ServeError(
            f"could not bind {HOST}:{port}: {error}. Something is already listening "
            f"there, or the port is still closing from a server that just stopped — "
            f"set PORT to a free one, or unset it for an ephemeral port."
        ) from error


def viewer_url(server: ThreadingHTTPServer) -> str:
    """The URL of the viewer page on a bound ``server``.

    ``server_port`` and not ``server_address[1]``: the address is a plain
    tuple whose members are typed loosely enough that mypy cannot tell a host
    string from raw bytes, and the port is the only part that is not already
    known from :data:`HOST`.
    """
    return f"http://{HOST}:{server.server_port}/{VIEWER_INDEX}"
