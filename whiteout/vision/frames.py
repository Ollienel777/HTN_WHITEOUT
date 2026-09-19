"""Frames in, from a live camera or from a recording of one.

``hackathon/ARENA.md`` §4; the live ports are on issue #63. Every asset serves
**MJPEG over HTTP**:

| asset | camera port |
|---|---|
| quadcopter | 8600 (gimbal) |
| fixed-wing | 8610 (FPV) |
| tower-1 | 8630 (EO) |
| tower-2 | 8640 (EO) |

An MJPEG stream is a ``multipart/x-mixed-replace`` body that never ends: one
part per frame, each a complete JPEG. **A recording of that stream is the
same bytes on disk**, which is the whole design here — one parser,
:func:`read_mjpeg`, reads either, and :func:`recorded_frames` and
:func:`live_frames` differ only in what they open and how they stamp time.

That is what lets the detector be developed offline. Record once, against the
live arena::

    curl -N http://10.99.4.1:8600/ > fixtures/frames/quadcopter.mjpeg

and from then on the detector, and this module's tests, run against the file
with no arena and no socket. The gate never opens one (``SPEC.md`` §6).

**Frames carry encoded JPEG bytes, not pixels.** Decoding needs an image
library, the choice of one belongs to the detector, and the projection in
:mod:`whiteout.vision.projection` never looks at the imagery at all — it
needs the pose and the published intrinsics, not the frame. Keeping
:class:`Frame` at the encoded bytes means this module has no dependency
beyond the standard library and that a recording round-trips byte for byte.

**Timestamps are injected, never taken from a clock inside the parser.**
``SPEC.md`` §5 makes determinism a hard rule, and a frame reader that called
:func:`time.time` would make every offline replay a different episode.
:func:`recorded_frames` derives ``t`` from the frame index and a frame rate,
so a file replays identically every time; :func:`live_frames` is the only
place a wall clock is read, and it is the only one that cannot run in the
gate.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import IO

from whiteout.vision.camera import VisionError

__all__ = [
    "DEFAULT_FPS",
    "Frame",
    "FrameError",
    "live_frames",
    "read_mjpeg",
    "recorded_frames",
]

#: Frames per second assumed when replaying a recording that carries no
#: timing of its own. ``ARENA.md`` §6 runs the sim at ``SPEEDUP=1``, so a
#: recording is real time and this only has to be near enough to keep a
#: replay's clock honest.
DEFAULT_FPS = 10.0

#: Bytes pulled from the stream per underlying read.
_CHUNK = 65536

#: JPEG start-of-image marker. Every part of an MJPEG stream begins with it.
_JPEG_SOI = b"\xff\xd8"


class FrameError(VisionError):
    """A frame stream could not be parsed.

    Raised when no multipart boundary can be found, when a part's
    ``Content-Length`` is not a number, and when a part's payload is not a
    JPEG — the last of which is the one that catches a boundary parsed
    wrongly, because a mis-split part almost never starts with the
    start-of-image marker.

    A stream that simply *ends* is not an error. A live camera's connection
    drops and a recording is whatever was on disk when the operator stopped
    ``curl``, so a truncated trailing part ends the iteration and the frames
    already read stand.
    """


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame, as the encoded bytes that came off the wire.

    :param asset_id: which camera this came from, e.g. ``"tower-1"``.
    :param seq: 0-based index within this stream.
    :param t: seconds; see the module docstring on where it comes from.
    :param jpeg: the complete JPEG, start-of-image marker onward.
    """

    asset_id: str
    seq: int
    t: float
    jpeg: bytes


def _fill(stream: IO[bytes], buf: bytearray) -> bool:
    chunk = stream.read(_CHUNK)
    if not chunk:
        return False
    buf += chunk
    return True


def _read_line(stream: IO[bytes], buf: bytearray) -> bytes | None:
    """Return the next line without its terminator, or ``None`` at end of stream."""
    while True:
        index = buf.find(b"\n")
        if index >= 0:
            line = bytes(buf[:index])
            del buf[: index + 1]
            return line.rstrip(b"\r")
        if not _fill(stream, buf):
            return None


def _read_exactly(stream: IO[bytes], buf: bytearray, count: int) -> bytes | None:
    while len(buf) < count:
        if not _fill(stream, buf):
            return None
    out = bytes(buf[:count])
    del buf[:count]
    return out


def _read_until(stream: IO[bytes], buf: bytearray, sep: bytes) -> bytes | None:
    """Return everything before the next ``sep``, consuming it; ``None`` at end."""
    start = 0
    while True:
        index = buf.find(sep, start)
        if index >= 0:
            out = bytes(buf[:index])
            del buf[: index + len(sep)]
            return out
        start = max(0, len(buf) - len(sep) + 1)
        if not _fill(stream, buf):
            return None


def _find_first_delimiter(stream: IO[bytes], buf: bytearray, delimiter: bytes | None) -> bytes:
    """Consume the preamble and return the delimiter line, discovering it if needed.

    A recording has no HTTP headers, so the boundary has to come out of the
    body: it is the first line that opens with ``--``, which is exactly how
    RFC 2046 defines the delimiter.
    """
    while True:
        line = _read_line(stream, buf)
        if line is None:
            raise FrameError("no multipart boundary found in the stream")
        if delimiter is None:
            if line.startswith(b"--") and len(line) > 2:
                return line
        elif line == delimiter or line == delimiter + b"--":
            return delimiter


def read_mjpeg(
    stream: IO[bytes],
    asset_id: str,
    timestamps: Callable[[int], float],
    *,
    boundary: bytes | None = None,
) -> Iterator[Frame]:
    """Yield :class:`Frame` objects from a ``multipart/x-mixed-replace`` stream.

    :param stream: any binary reader — an open file, or the object
        :func:`urllib.request.urlopen` returns.
    :param asset_id: stamped onto every frame.
    :param timestamps: given a frame's index, return its ``t`` in seconds.
    :param boundary: the boundary **without** its leading ``--``, as an HTTP
        ``Content-Type`` header gives it. Omitted, the boundary is read out of
        the body, which is what a recording needs.
    :raises FrameError: see that class.

    Both framings in the wild are handled: a part with a ``Content-Length``
    header is read by length, and one without is read up to the next
    delimiter. Reading by length is preferred where it is offered because a
    JPEG's entropy-coded bytes can in principle contain the delimiter, and
    length is exact.
    """
    buf = bytearray()
    delimiter = _find_first_delimiter(stream, buf, None if boundary is None else b"--" + boundary)
    seq = 0
    while True:
        content_length: int | None = None
        while True:
            line = _read_line(stream, buf)
            if line is None:
                return
            if not line:
                break
            name, separator, value = line.partition(b":")
            if separator and name.strip().lower() == b"content-length":
                try:
                    content_length = int(value.strip())
                except ValueError as exc:
                    raise FrameError(f"part {seq}: bad Content-Length {value!r}") from exc

        closing: bool
        if content_length is None:
            payload = _read_until(stream, buf, b"\r\n" + delimiter)
            if payload is None:
                return
            tail = _read_line(stream, buf)
            closing = tail is None or tail == b"--"
        else:
            read = _read_exactly(stream, buf, content_length)
            if read is None:
                return
            payload = read
            closing = _skip_to_delimiter(stream, buf, delimiter)

        if not payload.startswith(_JPEG_SOI):
            raise FrameError(
                f"part {seq} of {asset_id} is not a JPEG: it begins {payload[:4]!r}, not "
                f"{_JPEG_SOI!r}. The multipart boundary is probably wrong."
            )
        yield Frame(asset_id=asset_id, seq=seq, t=timestamps(seq), jpeg=payload)
        seq += 1
        if closing:
            return


def _skip_to_delimiter(stream: IO[bytes], buf: bytearray, delimiter: bytes) -> bool:
    """Consume up to and including the next delimiter; return whether it closed the stream."""
    while True:
        line = _read_line(stream, buf)
        if line is None:
            return True
        if line == delimiter:
            return False
        if line == delimiter + b"--":
            return True


def recorded_frames(
    path: str | os.PathLike[str],
    asset_id: str,
    *,
    fps: float = DEFAULT_FPS,
    boundary: bytes | None = None,
) -> Iterator[Frame]:
    """Replay a recorded MJPEG stream from disk.

    :param path: the recording, as written by ``curl -N <camera url> > file``.
    :param asset_id: stamped onto every frame.
    :param fps: frames per second to derive ``t`` from, ``t = seq / fps``.
    :param boundary: normally omitted; the boundary is read out of the file.

    The file is held open for the life of the iterator and closed when it is
    exhausted or the generator is closed, so the caller may stop early.
    """
    if fps <= 0.0:
        raise FrameError(f"fps must be positive, got {fps!r}")
    with open(path, "rb") as handle:
        yield from read_mjpeg(handle, asset_id, lambda seq: seq / fps, boundary=boundary)


def live_frames(
    url: str,
    asset_id: str,
    *,
    timeout: float = 5.0,
    clock: Callable[[], float] = time.time,
) -> Iterator[Frame]:
    """Read frames from a live camera's MJPEG endpoint.

    :param url: the camera's stream, e.g. ``http://10.99.4.1:8600/``. The
        arena's address is configuration and is never hardcoded (issue #63).
    :param asset_id: stamped onto every frame.
    :param timeout: seconds, passed to :func:`urllib.request.urlopen`.
    :param clock: the wall clock, injected so a test can pin it.

    **This is the one function here that opens a socket**, so it is the one
    the gate never reaches. Everything it does after the connection is
    :func:`read_mjpeg`, which the tests exercise against bytes; what is not
    covered offline is the connection itself and the boundary coming out of
    the ``Content-Type`` header, and both of those need the live arena.
    """
    from urllib.request import urlopen

    boundary: bytes | None = None
    with urlopen(url, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        for parameter in content_type.split(";")[1:]:
            name, separator, value = parameter.partition("=")
            if separator and name.strip().lower() == "boundary":
                boundary = value.strip().strip('"').encode("ascii")
        yield from read_mjpeg(response, asset_id, lambda _seq: clock(), boundary=boundary)
