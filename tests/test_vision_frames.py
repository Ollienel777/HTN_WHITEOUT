"""Reading MJPEG frames from a recording and from a live-shaped stream.

Issue #65's last acceptance criterion is that frames come from a recorded
file *as well as* a live asset, so the detector can be developed offline. The
design that satisfies it is one parser over a binary stream, and these tests
are the proof: the same bytes are fed to :func:`~whiteout.vision.frames.read_mjpeg`
as an in-memory stream and to :func:`~whiteout.vision.frames.recorded_frames`
as a file on disk, and the frames that come back are asserted equal.

The bytes themselves are synthetic. A real MJPEG part is a JPEG of a foggy
strait; what this module has to get right is the multipart framing around it,
so the payloads here are the shortest thing that is recognisably a JPEG — a
start-of-image marker, some body, an end-of-image marker. ``ARENA.md`` §4 and
``SPEC.md`` §6 both rule out reaching the arena from the gate, so nothing here
opens a socket.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from whiteout.vision.frames import DEFAULT_FPS, Frame, FrameError, read_mjpeg, recorded_frames

BOUNDARY = b"--arcticframe"

#: Three synthetic JPEGs, distinguishable by their body.
PAYLOADS = (
    b"\xff\xd8\x00first\xff\xd9",
    b"\xff\xd8\x00second\xff\xd9",
    b"\xff\xd8\x00third\xff\xd9",
)


def multipart(*, with_content_length: bool, closing: bool = True, preamble: bytes = b"") -> bytes:
    """Build an MJPEG body in the shape a camera serves it.

    ``with_content_length`` picks between the two framings seen in the wild:
    a part that declares its length, and one that does not and is terminated
    by the next delimiter.
    """
    out = bytearray(preamble)
    for payload in PAYLOADS:
        out += BOUNDARY + b"\r\n"
        out += b"Content-Type: image/jpeg\r\n"
        if with_content_length:
            out += b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n"
        out += b"\r\n"
        out += payload + b"\r\n"
    out += BOUNDARY + (b"--\r\n" if closing else b"\r\n")
    return bytes(out)


def frames_of(body: bytes, boundary: bytes | None = None) -> list[Frame]:
    stream = read_mjpeg(io.BytesIO(body), "tower-1", lambda seq: seq / 10.0, boundary=boundary)
    return list(stream)


@pytest.mark.parametrize("with_content_length", [True, False])
def test_every_part_comes_back_as_a_frame(with_content_length: bool) -> None:
    """Both framings yield the three payloads, byte for byte, in order.

    Byte-for-byte matters: the frame is handed to a detector that will decode
    it, and a parser that leaves a trailing ``\\r\\n`` on the payload or eats
    the final ``\\xff\\xd9`` produces JPEGs that decode with a warning on one
    library and not at all on the next.
    """
    frames = frames_of(multipart(with_content_length=with_content_length))
    assert [frame.jpeg for frame in frames] == list(PAYLOADS)
    assert [frame.seq for frame in frames] == [0, 1, 2]
    assert [frame.asset_id for frame in frames] == ["tower-1"] * 3
    assert [frame.t for frame in frames] == [0.0, 0.1, 0.2]


def test_the_boundary_is_discovered_from_the_body_and_can_also_be_given() -> None:
    """A recording has no HTTP headers, so the boundary comes out of the body.

    Passing it explicitly — which is what :func:`live_frames` does with the
    ``Content-Type`` header's ``boundary`` parameter, *without* its leading
    ``--`` — must give the same frames.
    """
    body = multipart(with_content_length=True)
    discovered = frames_of(body)
    declared = frames_of(body, boundary=b"arcticframe")
    assert discovered == declared
    assert len(discovered) == 3


def test_a_preamble_before_the_first_boundary_is_skipped() -> None:
    """RFC 2046 allows text before the first delimiter, and servers emit it."""
    body = multipart(with_content_length=True, preamble=b"ignore me\r\n\r\n")
    assert [frame.jpeg for frame in frames_of(body)] == list(PAYLOADS)


def test_a_stream_that_never_closes_still_yields_what_it_carried() -> None:
    """A live camera's stream is open-ended; ending it is not an error.

    The generator simply stops when the bytes run out, which is what happens
    when the operator kills ``curl`` or the link drops mid-flight.
    """
    body = multipart(with_content_length=True, closing=False)
    assert [frame.jpeg for frame in frames_of(body)] == list(PAYLOADS)


def test_a_truncated_final_part_is_dropped_and_the_rest_stand() -> None:
    """Half a frame is not a frame, and losing it must not lose the others."""
    body = multipart(with_content_length=True)
    truncated = body[: body.rindex(PAYLOADS[2]) + 4]
    frames = frames_of(truncated)
    assert [frame.jpeg for frame in frames] == [PAYLOADS[0], PAYLOADS[1]]


def test_a_recorded_file_and_a_live_shaped_stream_give_identical_frames(
    tmp_path: Path,
) -> None:
    """The acceptance criterion, asserted directly.

    The same bytes, once through :func:`read_mjpeg` as a stream and once
    through :func:`recorded_frames` as a file, produce equal frames. That is
    what lets the detector be written and regression-tested with no arena:
    record the stream once, replay the file forever.
    """
    body = multipart(with_content_length=True)
    path = tmp_path / "tower-1.mjpeg"
    path.write_bytes(body)

    live_shaped = frames_of(body)
    replayed = list(recorded_frames(path, "tower-1", fps=10.0))
    assert replayed == live_shaped


def test_a_replayed_file_is_deterministic_and_timed_from_the_frame_rate(
    tmp_path: Path,
) -> None:
    """``t = seq / fps``: two replays of one file are identical.

    ``SPEC.md`` §5 makes determinism a hard rule. A frame reader that stamped
    frames from a wall clock would make every offline replay a different
    episode and every log comparison useless.
    """
    path = tmp_path / "quadcopter.mjpeg"
    path.write_bytes(multipart(with_content_length=True))

    first = list(recorded_frames(path, "quadcopter", fps=4.0))
    second = list(recorded_frames(path, "quadcopter", fps=4.0))
    assert first == second
    assert [frame.t for frame in first] == [0.0, 0.25, 0.5]

    default = list(recorded_frames(path, "quadcopter"))
    assert [frame.t for frame in default] == [0.0, 1.0 / DEFAULT_FPS, 2.0 / DEFAULT_FPS]


def test_a_caller_may_stop_early_and_the_file_is_closed(tmp_path: Path) -> None:
    """The detector reads a few frames and walks away; the handle must not leak."""
    path = tmp_path / "fixed-wing.mjpeg"
    path.write_bytes(multipart(with_content_length=True))

    stream = recorded_frames(path, "fixed-wing")
    assert next(stream).jpeg == PAYLOADS[0]
    stream.close()


def test_a_body_with_no_boundary_at_all_is_refused() -> None:
    """A file that is not a multipart stream is a mistake worth reporting."""
    with pytest.raises(FrameError, match="no multipart boundary"):
        frames_of(b"\xff\xd8 just a bare jpeg \xff\xd9")


def test_a_boundary_that_never_occurs_is_refused() -> None:
    """Asking for a delimiter the body does not contain is not a silent empty stream."""
    with pytest.raises(FrameError, match="no multipart boundary"):
        frames_of(multipart(with_content_length=True), boundary=b"nothing-like-it")


def test_a_part_that_is_not_a_jpeg_is_refused() -> None:
    """The payload check is what turns a mis-split into a diagnostic.

    A part that does not open with the start-of-image marker means the
    framing was read wrongly — the wrong boundary, or a length off by a few
    bytes — and saying so beats handing a detector a bag of bytes that
    decodes to nothing.
    """
    body = BOUNDARY + b"\r\nContent-Length: 5\r\n\r\nnotjp\r\n" + BOUNDARY + b"--\r\n"
    with pytest.raises(FrameError, match="not a JPEG"):
        frames_of(body)


def test_a_bad_content_length_is_refused() -> None:
    """A header that is not a number cannot be a length."""
    body = (
        BOUNDARY + b"\r\nContent-Length: ?\r\n\r\n" + PAYLOADS[0] + b"\r\n" + BOUNDARY + b"--\r\n"
    )
    with pytest.raises(FrameError, match="bad Content-Length"):
        frames_of(body)


def test_a_non_positive_frame_rate_is_refused(tmp_path: Path) -> None:
    """``t = seq / fps`` needs an fps."""
    path = tmp_path / "empty.mjpeg"
    path.write_bytes(multipart(with_content_length=True))
    with pytest.raises(FrameError, match="fps must be positive"):
        list(recorded_frames(path, "tower-2", fps=0.0))
