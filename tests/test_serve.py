"""``whiteout serve``: the port rules in ``hackathon/SPEC.md`` §6.

    take any port from ``PORT``, with no reuse of a running server. The viewer
    dev server (``whiteout.cli serve``) binds ``PORT`` or an ephemeral port,
    never a fixed 8000. Parallel worktrees run side by side.

Every test here binds loopback only, on a port the kernel chose, and closes
what it opened. None of them starts a transport or leaves the machine.
"""

from __future__ import annotations

import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from whiteout.cli import main
from whiteout.serve import (
    VIEWER_INDEX,
    VIEWER_ROOT,
    ServeError,
    open_viewer_server,
    resolve_port,
    viewer_url,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _a_free_port() -> int:
    """A port nothing was listening on a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def running() -> Iterator[ThreadingHTTPServer]:
    server = open_viewer_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_port_is_respected() -> None:
    wanted = _a_free_port()
    assert resolve_port({"PORT": str(wanted)}) == wanted
    server = open_viewer_server(wanted)
    try:
        assert server.server_address[1] == wanted
    finally:
        server.server_close()


def test_an_unset_port_is_ephemeral_and_never_a_fixed_one() -> None:
    assert resolve_port({}) == 0
    assert resolve_port({"PORT": "   "}) == 0
    first = open_viewer_server(0)
    second = open_viewer_server(0)
    try:
        # Two worktrees' servers side by side, which is the point of the rule.
        assert first.server_address[1] not in (0, 8000)
        assert second.server_address[1] not in (0, 8000)
        assert first.server_address[1] != second.server_address[1]
    finally:
        first.server_close()
        second.server_close()


def test_a_running_server_is_never_reused(running: ThreadingHTTPServer) -> None:
    """The second bind fails loudly rather than attaching to the first.

    This is what ``allow_reuse_address = False`` buys: with it on, a port left
    behind by another worktree's server is taken silently, and the URL printed
    no longer identifies which tree answers on it.
    """
    port = int(running.server_address[1])
    with pytest.raises(ServeError) as raised:
        open_viewer_server(port)
    assert str(port) in str(raised.value)
    # On Linux the bind fails whatever `allow_reuse_address` says, because
    # SO_REUSEADDR only reaches a socket in TIME_WAIT. It is asserted
    # separately because on Windows it does let a live listening socket be
    # taken over, and that is the platform half the team builds on.
    assert type(running).allow_reuse_address is False


@pytest.mark.parametrize("value", ["eight", " -1 ", "70000", "1.5"])
def test_a_bad_port_is_diagnosed_rather_than_defaulted(value: str) -> None:
    with pytest.raises(ServeError):
        resolve_port({"PORT": value})


def test_the_cli_diagnoses_a_bad_port_without_binding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PORT", "not-a-port")
    assert main(["serve"]) == 1
    assert "PORT" in capsys.readouterr().err


def _get(server: ThreadingHTTPServer, path: str) -> tuple[int, bytes]:
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    with urllib.request.urlopen(url, timeout=5) as response:
        return int(response.status), response.read()


def test_it_serves_the_viewer_and_the_bundled_episode(running: ThreadingHTTPServer) -> None:
    status, body = _get(running, f"/{VIEWER_INDEX}")
    assert status == 200
    assert b"WHITEOUT run viewer" in body

    # The viewer reads the episode from a sibling directory, so the server has
    # to hand out the tree, not just viz/.
    status, body = _get(running, "/fixtures/episodes/demo.jsonl")
    assert status == 200
    assert b'"schema_version"' in body


def test_the_root_leads_to_the_viewer(running: ThreadingHTTPServer) -> None:
    status, _ = _get(running, "/")
    assert status == 200
    assert VIEWER_INDEX in viewer_url(running)


def test_a_missing_path_is_a_404_not_a_traceback(running: ThreadingHTTPServer) -> None:
    with pytest.raises(urllib.error.HTTPError) as raised:
        _get(running, "/no/such/file.jsonl")
    assert raised.value.code == 404


def test_the_served_tree_is_this_worktree() -> None:
    """A gate in another worktree must not be answered by this one's viewer."""
    assert VIEWER_ROOT == REPO_ROOT
    assert (VIEWER_ROOT / VIEWER_INDEX).is_file()


def test_serving_a_tree_without_a_viewer_is_diagnosed(tmp_path: Path) -> None:
    with pytest.raises(ServeError) as raised:
        open_viewer_server(0, root=tmp_path)
    assert VIEWER_INDEX in str(raised.value)
