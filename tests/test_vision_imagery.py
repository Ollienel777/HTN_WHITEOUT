"""The frame-source adapter: one interface, the real one and the fake.

``CLAUDE.md``'s adapter rule says every external service sits behind an
interface with a fixture-backed fake, chosen by an environment variable, so
that the app, CI and the demo all run without keys or a network. The arena's
cameras are that service, and this is the seam.

Two things here are worth a reviewer's attention.

The **fixture directory is the deliverable**, not a test artefact. Its round
trip — render, write, read back, get the same pixels, poses and labels — is
what makes "point the harness at real frames when they arrive" a true
statement rather than an intention.

The **MJPEG path is tested with the decoder stubbed out**, because there is no
JPEG decoder in this environment at all (see
:mod:`whiteout.vision.imagery`). What is covered is the plumbing either side
of the decode: the multipart stream becomes frames, they carry the right asset
and camera, a decode of the wrong size is refused, and a label never appears
on a frame that came from a camera. What is not covered is turning JPEG bytes
into pixels, and :func:`test_decode_jpeg_luma_says_what_is_missing` pins the
failure the absence produces.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import IO, cast

import numpy as np
import pytest

from whiteout.vision import imagery
from whiteout.vision.camera import TOWER_CAMERA, VisionError
from whiteout.vision.imagery import (
    FIXTURE_ENV,
    FRAMES_ENV,
    MJPEG_ENV,
    FixtureFrames,
    Label,
    LumaFrame,
    MjpegFrames,
    decode_jpeg_luma,
    frame_source_from_env,
    read_pgm,
    write_fixtures,
    write_pgm,
)
from whiteout.vision.projection import CameraPose
from whiteout.vision.scene import CLEAR, render_scene

POSE = CameraPose(71.985, -94.40, 60.0, 75.0, -22.0)


# --------------------------------------------------------------------------
# PGM: pixels on disk, with no dependency
# --------------------------------------------------------------------------


def test_a_frame_survives_a_pgm_round_trip(tmp_path: Path) -> None:
    scene = render_scene(TOWER_CAMERA, np.random.default_rng(1), params=CLEAR)
    path = tmp_path / "frame.pgm"
    write_pgm(path, scene.luma)
    assert np.array_equal(read_pgm(path), scene.luma)


def test_a_pgm_header_may_carry_comments(tmp_path: Path) -> None:
    path = tmp_path / "commented.pgm"
    path.write_bytes(b"P5\n# written by hand\n2 2\n255\n\x01\x02\x03\x04")
    assert np.array_equal(read_pgm(path), np.array([[1, 2], [3, 4]], dtype=np.uint8))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"P2\n2 2\n255\n1 2 3 4\n", "not a binary PGM"),
        (b"P5\n2 2\n65535\n\x00\x01\x00\x02\x00\x03\x00\x04", "only 8-bit"),
        (b"P5\n4 4\n255\n\x01\x02", "truncated"),
        (b"P5\n2 2", "truncated PGM header"),
    ],
)
def test_a_pgm_this_reader_does_not_speak_is_refused(
    tmp_path: Path, body: bytes, message: str
) -> None:
    path = tmp_path / "bad.pgm"
    path.write_bytes(body)
    with pytest.raises(VisionError, match=message):
        read_pgm(path)


def test_writing_something_that_is_not_an_8_bit_image_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VisionError, match="2-D"):
        write_pgm(tmp_path / "x.pgm", np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(VisionError, match="uint8"):
        write_pgm(tmp_path / "y.pgm", np.zeros((2, 2), dtype=np.uint8).astype(np.float64))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the fixture directory: what real frames will arrive as
# --------------------------------------------------------------------------


def a_frame(seq: int, *, present: bool) -> LumaFrame:
    scene = render_scene(
        TOWER_CAMERA, np.random.default_rng(seq), params=CLEAR, with_vessel=present
    )
    label = (
        Label(present=True, px=scene.vessel_px[0], py=scene.vessel_px[1], tolerance_px=11.0)
        if present and scene.vessel_px
        else Label(present=False)
    )
    return LumaFrame(
        asset_id="tower-1",
        camera=TOWER_CAMERA,
        seq=seq,
        t=seq / 10.0,
        luma=scene.luma,
        pose=POSE,
        ground_alt_m=3.5,
        label=label,
    )


def test_a_fixture_directory_round_trips_pixels_poses_and_labels(tmp_path: Path) -> None:
    written = [a_frame(0, present=True), a_frame(1, present=False)]
    write_fixtures(
        tmp_path,
        [(frame, f"{frame.seq:04d}.pgm") for frame in written],
        note="a test corpus, not arena imagery",
    )
    read = list(FixtureFrames(tmp_path).frames())
    assert len(read) == 2
    for before, after in zip(written, read, strict=True):
        assert np.array_equal(before.luma, after.luma)
        assert after.asset_id == before.asset_id
        assert after.camera is TOWER_CAMERA
        assert after.seq == before.seq
        assert after.t == pytest.approx(before.t)
        assert after.pose == before.pose
        assert after.ground_alt_m == pytest.approx(3.5)
        assert after.label == before.label


def test_a_manifest_says_where_its_frames_came_from(tmp_path: Path) -> None:
    """Whoever reads a fixture directory has to be able to tell.

    A directory of PGMs looks the same whether it came from Dominion
    Dynamics' render or from :mod:`whiteout.vision.scene`, and a
    false-positive rate means something very different in each case.
    """
    manifest = write_fixtures(
        tmp_path, [(a_frame(0, present=False), "0000.pgm")], note="synthetic, not the arena"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["version"] == imagery.MANIFEST_VERSION
    assert "synthetic" in payload["note"]


def test_a_manifest_of_another_version_is_refused(tmp_path: Path) -> None:
    write_fixtures(tmp_path, [(a_frame(0, present=False), "0000.pgm")], note="x")
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = imagery.MANIFEST_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VisionError, match="this reader speaks"):
        list(FixtureFrames(tmp_path).frames())


def test_a_missing_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VisionError, match="no fixture manifest"):
        list(FixtureFrames(tmp_path).frames())


def test_a_frame_that_is_not_its_cameras_size_is_refused(tmp_path: Path) -> None:
    write_fixtures(tmp_path, [(a_frame(0, present=False), "0000.pgm")], note="x")
    write_pgm(tmp_path / "0000.pgm", np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(VisionError, match="publishes"):
        list(FixtureFrames(tmp_path).frames())


def test_an_unknown_camera_is_refused(tmp_path: Path) -> None:
    write_fixtures(tmp_path, [(a_frame(0, present=False), "0000.pgm")], note="x")
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frames"][0]["camera"] = "submarine"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VisionError, match="unknown camera"):
        list(FixtureFrames(tmp_path).frames())


def test_an_absent_vessel_cannot_carry_a_pixel() -> None:
    with pytest.raises(VisionError, match="cannot have a pixel"):
        Label(present=False, px=1.0, py=2.0)


def test_a_present_vessel_needs_both_coordinates_or_neither() -> None:
    with pytest.raises(VisionError, match="both px and py"):
        Label(present=True, px=1.0)
    unlocalised = Label(present=True)
    assert not unlocalised.localised


# --------------------------------------------------------------------------
# the committed fixtures
# --------------------------------------------------------------------------


def test_the_committed_fixtures_load_and_are_labelled() -> None:
    """``fixtures/frames/synthetic`` is what the harness reads with no arguments."""
    root = Path(__file__).resolve().parent.parent
    frames = list(FixtureFrames(root / imagery.DEFAULT_FIXTURE_DIR).frames())
    assert frames, "the committed fixture directory is empty"
    assert {frame.camera.name for frame in frames} == {"quadcopter", "fixed-wing", "tower"}
    assert any(frame.label is not None and frame.label.present for frame in frames)
    assert any(frame.label is not None and not frame.label.present for frame in frames)
    for frame in frames:
        assert frame.pose is not None
        assert frame.luma.shape == (frame.camera.height, frame.camera.width)


def test_the_committed_manifest_says_the_frames_are_synthetic() -> None:
    root = Path(__file__).resolve().parent.parent
    payload = json.loads(
        (root / imagery.DEFAULT_FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert "NOT arena imagery" in payload["note"]


# --------------------------------------------------------------------------
# the real source
# --------------------------------------------------------------------------

BOUNDARY = b"--arcticframe"


def multipart(count: int) -> bytes:
    out = bytearray()
    for index in range(count):
        payload = b"\xff\xd8" + bytes([index]) * 8 + b"\xff\xd9"
        out += BOUNDARY + b"\r\n"
        out += b"Content-Type: image/jpeg\r\n"
        out += b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
        out += payload + b"\r\n"
    out += BOUNDARY + b"--\r\n"
    return bytes(out)


def test_a_recording_becomes_frames_of_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real adapter's plumbing, with the one step this machine cannot do stubbed."""
    recording = tmp_path / "tower-1.mjpeg"
    recording.write_bytes(multipart(3))
    decoded = np.full((TOWER_CAMERA.height, TOWER_CAMERA.width), 7, dtype=np.uint8)
    monkeypatch.setattr(imagery, "decode_jpeg_luma", lambda _jpeg: decoded)

    source = MjpegFrames(str(recording), "tower-1", TOWER_CAMERA, pose=POSE, fps=10.0)
    frames = list(source.frames())
    assert [frame.seq for frame in frames] == [0, 1, 2]
    assert [frame.t for frame in frames] == pytest.approx([0.0, 0.1, 0.2])
    for frame in frames:
        assert frame.asset_id == "tower-1"
        assert frame.camera is TOWER_CAMERA
        assert frame.pose is POSE
        assert np.array_equal(frame.luma, decoded)
        assert frame.label is None, "a live camera cannot know what is in its own frame"


def test_a_decode_of_the_wrong_size_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = tmp_path / "tower-1.mjpeg"
    recording.write_bytes(multipart(1))
    monkeypatch.setattr(
        imagery, "decode_jpeg_luma", lambda _jpeg: np.zeros((240, 320), dtype=np.uint8)
    )
    source = MjpegFrames(str(recording), "tower-1", TOWER_CAMERA)
    with pytest.raises(VisionError, match="publishes"):
        list(source.frames())


def test_decode_jpeg_luma_says_what_is_missing() -> None:
    """No decoder is installed here, and the message has to be actionable.

    Skipped the moment somebody adds one — at which point this path is real
    and the skip is the signal to test it properly.
    """
    for module in ("PIL", "cv2"):
        if _importable(module):
            pytest.skip(f"{module} is installed; the real decode path is live")
    with pytest.raises(VisionError, match="no JPEG decoder"):
        decode_jpeg_luma(b"\xff\xd8\xff\xd9")


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------
# the env-var switch
# --------------------------------------------------------------------------


def test_the_fake_is_the_default(tmp_path: Path) -> None:
    source = frame_source_from_env({}, root=tmp_path)
    assert isinstance(source, FixtureFrames)
    assert source.directory == tmp_path / imagery.DEFAULT_FIXTURE_DIR


def test_the_fixture_directory_can_be_pointed_at_real_frames(tmp_path: Path) -> None:
    source = frame_source_from_env({FIXTURE_ENV: str(tmp_path / "recorded")}, root=tmp_path)
    assert isinstance(source, FixtureFrames)
    assert source.directory == tmp_path / "recorded"


def test_the_real_source_is_selected_by_the_environment(tmp_path: Path) -> None:
    source = frame_source_from_env(
        {
            FRAMES_ENV: "mjpeg",
            MJPEG_ENV: "http://10.99.4.1:8630/",
            "WHITEOUT_VISION_ASSET": "tower-1",
            "WHITEOUT_VISION_CAMERA": "tower",
        },
        root=tmp_path,
    )
    assert isinstance(source, MjpegFrames)
    assert source.asset_id == "tower-1"
    assert source.camera is TOWER_CAMERA


def test_the_real_source_without_a_source_fails_loudly(tmp_path: Path) -> None:
    """It must not fall back to the fake.

    A silent fallback would let somebody report a synthetic false-positive
    rate as an arena one, which is the single mistake this whole ticket is
    trying not to make.
    """
    with pytest.raises(VisionError, match=MJPEG_ENV):
        frame_source_from_env({FRAMES_ENV: "mjpeg"}, root=tmp_path)


def test_an_unknown_selector_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(VisionError, match="not one of"):
        frame_source_from_env({FRAMES_ENV: "webcam"}, root=tmp_path)


def test_an_unknown_camera_in_the_environment_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(VisionError, match="not one of"):
        frame_source_from_env(
            {FRAMES_ENV: "mjpeg", MJPEG_ENV: "x.mjpeg", "WHITEOUT_VISION_CAMERA": "periscope"},
            root=tmp_path,
        )


def test_the_stream_reader_is_the_one_from_frames_py(tmp_path: Path) -> None:
    """No second MJPEG parser. ``read_mjpeg`` already does this (PR #72)."""
    from whiteout.vision.frames import read_mjpeg

    stream = cast(IO[bytes], io.BytesIO(multipart(2)))
    assert len(list(read_mjpeg(stream, "tower-1", lambda seq: seq / 10.0))) == 2
