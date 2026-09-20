"""Where frames come from, and what shape they arrive in.

The detector in :mod:`whiteout.vision.detect` takes **pixels**, not JPEG
bytes. This module is the seam either side of that: one interface, a real
implementation that reads the arena's MJPEG, and a fixture-backed fake that
reads a directory off disk. ``CLAUDE.md``'s adapter rule, and the reason the
gate can measure a detector with no network, no arena and no image library.

    ==========================  =====================================
    ``WHITEOUT_VISION_FRAMES``  what :func:`frame_source_from_env` returns
    ==========================  =====================================
    unset, or ``fixture``       :class:`FixtureFrames` — the fake
    ``mjpeg``                   :class:`MjpegFrames` — the real one
    ==========================  =====================================

**The fixture directory is the point of this module.** It is a ``manifest.json``
beside a set of PGM files, each entry carrying the frame's camera, its pose,
and — where someone has labelled it — where the vessel actually is. Today it
holds synthetic frames, because no arena imagery exists here. When a recording
does exist, the work is: decode it to PGM, write a manifest, label the frames,
and run ``scripts/score_detector.py`` at the directory. Not a rewrite of
anything.

**PGM, specifically.** The frames have to be *pixels* on disk, because a JPEG
on disk is only pixels if something can decode it, and nothing in this
project's dependencies can (below). Binary PGM is eleven lines of standard
library, it is the same bytes on every platform, and ``head -c 64`` tells you
what you are looking at. ``.npy`` would have been smaller to write and worse
to read.

**There is no JPEG decoder in this environment, and adding one is not this
ticket's decision to take.** ``pyproject.toml`` has NumPy, SciPy and NetworkX;
Pillow and OpenCV are both absent. :func:`decode_jpeg_luma` therefore uses
whichever of the two is installed and raises a :class:`~whiteout.vision.camera.VisionError`
naming the choice when neither is — so the real path is complete the moment
somebody decides, and until then it fails with an instruction rather than an
``ImportError``. Nothing in the gate reaches it.

**Calling** :func:`~whiteout.vision.frames.live_frames` **re-tiers two known
defects.** PR #72's review recorded three Mediums against it — an unguarded
``urlopen`` whose failure is not a ``VisionError``, unbounded buffering on a
stream that never ends, and a corrupt frame accepted on ``Content-Length:
-1`` — and tiered them Medium *because nothing called it*. :class:`MjpegFrames`
with an ``http://`` source calls it. They are not fixed here, and they are not
this ticket's; the pull request says so.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from whiteout.vision.camera import CAMERAS, CameraModel, VisionError
from whiteout.vision.frames import DEFAULT_FPS, Frame, live_frames, recorded_frames
from whiteout.vision.projection import CameraPose

__all__ = [
    "DEFAULT_FIXTURE_DIR",
    "FIXTURE_ENV",
    "FRAMES_ENV",
    "MJPEG_ENV",
    "FixtureFrames",
    "FrameSource",
    "Label",
    "LumaFrame",
    "MjpegFrames",
    "decode_jpeg_luma",
    "frame_source_from_env",
    "read_pgm",
    "write_fixtures",
    "write_pgm",
]

#: Chooses the implementation. ``fixture`` (the default) or ``mjpeg``.
FRAMES_ENV = "WHITEOUT_VISION_FRAMES"

#: The fixture directory :class:`FixtureFrames` reads when none is passed.
FIXTURE_ENV = "WHITEOUT_VISION_FIXTURES"

#: A path or an ``http://`` URL for :class:`MjpegFrames`.
MJPEG_ENV = "WHITEOUT_VISION_MJPEG"

#: Where the committed synthetic fixtures live, relative to the repository root.
DEFAULT_FIXTURE_DIR = "fixtures/frames/synthetic"

#: The manifest schema this module reads and writes. Bumped when a field's
#: meaning changes, so a stale fixture directory fails loudly rather than
#: being scored under the wrong assumptions.
MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class Label:
    """What a human, or the generator, says is in a frame.

    :param present: whether the vessel is in this frame at all. A frame with
        ``present=False`` is what a false-positive rate is measured on, and it
        is the cheapest kind of label to produce from a real recording — a
        stretch of empty channel needs no pixel-picking.
    :param px: the vessel's column, and :param py: its row, in the pixel
        convention of :mod:`whiteout.vision.camera`. ``None`` when absent, and
        also allowed when the vessel is present but nobody has picked the
        pixel: such a frame counts toward neither localisation nor the
        false-positive rate.
    :param tolerance_px: how far a detection may be from ``(px, py)`` and
        still count as the vessel.
    """

    present: bool
    px: float | None = None
    py: float | None = None
    tolerance_px: float = 12.0

    def __post_init__(self) -> None:
        if self.present and (self.px is None) != (self.py is None):
            raise VisionError(f"a label needs both px and py or neither, got {self!r}")
        if not self.present and (self.px is not None or self.py is not None):
            raise VisionError(f"an absent vessel cannot have a pixel, got {self!r}")
        if self.tolerance_px <= 0.0:
            raise VisionError(f"tolerance_px must be positive, got {self.tolerance_px!r}")

    @property
    def localised(self) -> bool:
        """Whether this label pins a pixel, and so can score localisation."""
        return self.present and self.px is not None and self.py is not None


@dataclass(frozen=True, slots=True)
class LumaFrame:
    """One frame, decoded, with everything the detector needs around it.

    :param asset_id: which camera, e.g. ``"tower-1"``.
    :param camera: its published intrinsics.
    :param seq: index within the stream.
    :param t: seconds.
    :param luma: ``uint8``, shape ``(camera.height, camera.width)``.
    :param pose: where the camera was and where it pointed. ``None`` from a
        live stream that carries no telemetry with it — the caller joins the
        MAVLink pose on by timestamp and cannot do so here.
    :param ground_alt_m: the water plane's altitude in ``pose``'s datum, as the
        fixture records it. Issue #76; never assumed.
    :param label: the truth, from a fixture manifest. Always ``None`` from a
        camera, which is the whole reason a measured false-positive rate needs
        a labelled recording and cannot be had from a live feed.
    """

    asset_id: str
    camera: CameraModel
    seq: int
    t: float
    luma: NDArray[np.uint8]
    pose: CameraPose | None = None
    ground_alt_m: float = 0.0
    label: Label | None = None


class FrameSource(Protocol):
    """Anything that yields frames the detector can read."""

    def frames(self) -> Iterator[LumaFrame]:
        """Yield frames in order. May be infinite; a caller bounds it."""


def read_pgm(path: str | os.PathLike[str]) -> NDArray[np.uint8]:
    """Read a binary (``P5``) PGM and return its samples as ``uint8``.

    Only the shape this project writes is accepted: ``P5``, a maximum value of
    255, comments allowed between tokens. A 16-bit PGM is refused rather than
    silently truncated, because a truncated frame would score as a detector
    result rather than as a file problem.

    The raster is required to fill the rest of the file **exactly**, and that
    is the strictest thing here on purpose. The format puts a single
    whitespace byte between the header and the samples, so the samples start
    at a position this reader has to count to; a writer that ends its header
    ``255\n\n`` puts every sample one byte late. Reading ``width * height``
    bytes from the wrong offset succeeds — the count is right, the file is
    long enough — and returns the frame shifted by a pixel with its last row
    wrapped, which is a plausible image and scores as one. An exact fit is the
    only check that separates that from a correct file.
    """
    data = Path(path).read_bytes()
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while index < len(data) and data[index : index + 1].isspace():
            index += 1
        if index < len(data) and data[index : index + 1] == b"#":
            while index < len(data) and data[index : index + 1] not in (b"\n", b"\r"):
                index += 1
            continue
        start = index
        while index < len(data) and not data[index : index + 1].isspace():
            index += 1
        if start == index:
            raise VisionError(f"{path}: truncated PGM header")
        tokens.append(data[start:index])
    if index >= len(data) or not data[index : index + 1].isspace():
        raise VisionError(f"{path}: PGM header is not followed by whitespace and a raster")
    index += 1  # exactly one whitespace byte separates the header from the raster
    magic, width_token, height_token, maximum_token = tokens
    if magic != b"P5":
        raise VisionError(f"{path}: not a binary PGM, magic is {magic!r} and not b'P5'")
    try:
        width, height, maximum = (int(t) for t in (width_token, height_token, maximum_token))
    except ValueError as error:
        raise VisionError(f"{path}: unreadable PGM header") from error
    if maximum != 255:
        raise VisionError(f"{path}: only 8-bit PGM is supported, maximum value is {maximum}")
    expected = width * height
    raster = data[index:]
    if len(raster) != expected:
        raise VisionError(
            f"{path}: the header promises {expected} samples ({width}x{height}) and "
            f"{len(raster)} bytes follow it. Too few is a truncated file; too many is "
            f"padding after the header, which would shift every sample and read as a "
            f"picture rather than as a fault"
        )
    return np.frombuffer(raster, dtype=np.uint8).reshape(height, width).copy()


def write_pgm(path: str | os.PathLike[str], luma: NDArray[np.uint8]) -> None:
    """Write a 2-D ``uint8`` array as a binary PGM."""
    if luma.ndim != 2:
        raise VisionError(f"a PGM is 2-D, got shape {luma.shape!r}")
    if luma.dtype != np.uint8:
        raise VisionError(f"a PGM holds uint8 samples, got {luma.dtype!r}")
    height, width = luma.shape
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.ascontiguousarray(luma).tobytes())


def decode_jpeg_luma(jpeg: bytes) -> NDArray[np.uint8]:
    """Decode a JPEG to a 2-D ``uint8`` luma array.

    :raises VisionError: when no decoder is installed — see the module
        docstring. The message names what to install, because the alternative
        is an ``ImportError`` from three frames down that reads as a bug.

    Pillow first, then OpenCV, because Pillow is the smaller of the two and
    ``.convert("L")`` is exactly ITU-R 601 luma. **Pillow is now a declared
    dependency** (#105): detection is a vision problem and the arena's
    cameras are MJPEG, so a decoder is on the critical path rather than
    optional. OpenCV stays as a fallback for an environment that has it
    instead, and the final error stays for one that has neither.
    """
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        import io

        with Image.open(io.BytesIO(jpeg)) as opened:
            return np.asarray(opened.convert("L"), dtype=np.uint8)
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise VisionError(
            "no JPEG decoder is installed, so arena frames cannot be turned into pixels. "
            "Install Pillow (or OpenCV) and add it to pyproject.toml — that is a dependency "
            "decision, not something this module takes on its own. Until then, point "
            f"{FRAMES_ENV}=fixture at a directory of PGM frames."
        ) from error
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise VisionError("OpenCV could not decode a frame of this MJPEG stream")
    return np.asarray(decoded, dtype=np.uint8)


def _pose_from(payload: Mapping[str, object] | None) -> CameraPose | None:
    if payload is None:
        return None
    try:
        return CameraPose(
            lat_deg=float(payload["lat_deg"]),  # type: ignore[arg-type]
            lon_deg=float(payload["lon_deg"]),  # type: ignore[arg-type]
            alt_m=float(payload["alt_m"]),  # type: ignore[arg-type]
            yaw_deg=float(payload["yaw_deg"]),  # type: ignore[arg-type]
            pitch_deg=float(payload["pitch_deg"]),  # type: ignore[arg-type]
            roll_deg=float(payload.get("roll_deg", 0.0)),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VisionError(f"unreadable pose in a fixture manifest: {payload!r}") from error


def _label_from(payload: Mapping[str, object] | None) -> Label | None:
    if payload is None:
        return None
    present = bool(payload["present"])
    px = payload.get("px")
    py = payload.get("py")
    return Label(
        present=present,
        px=None if px is None else float(px),  # type: ignore[arg-type]
        py=None if py is None else float(py),  # type: ignore[arg-type]
        tolerance_px=float(payload.get("tolerance_px", 12.0)),  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class FixtureFrames:
    """The fake: frames and their truth, read from a directory on disk.

    :param directory: holds ``manifest.json`` and the PGM files it names.

    This is what the gate runs and what ``scripts/score_detector.py`` is
    pointed at. It is also what a recording of the arena becomes, which is why
    it carries poses and labels rather than pixels alone.
    """

    directory: Path

    def frames(self) -> Iterator[LumaFrame]:
        manifest_path = self.directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise VisionError(f"no fixture manifest at {manifest_path}") from error
        except json.JSONDecodeError as error:
            raise VisionError(f"{manifest_path} is not JSON: {error}") from error
        version = manifest.get("version")
        if version != MANIFEST_VERSION:
            raise VisionError(
                f"{manifest_path} is version {version!r}; this reader speaks {MANIFEST_VERSION}"
            )
        for index, entry in enumerate(manifest.get("frames", [])):
            camera_name = entry["camera"]
            if camera_name not in CAMERAS:
                raise VisionError(
                    f"{manifest_path} frame {index}: unknown camera {camera_name!r}, "
                    f"expected one of {sorted(CAMERAS)}"
                )
            camera = CAMERAS[camera_name]
            luma = read_pgm(self.directory / entry["file"])
            if luma.shape != (camera.height, camera.width):
                raise VisionError(
                    f"{entry['file']} is {luma.shape[1]}x{luma.shape[0]}, but camera "
                    f"{camera_name!r} publishes {camera.width}x{camera.height}"
                )
            yield LumaFrame(
                asset_id=entry["asset_id"],
                camera=camera,
                seq=int(entry.get("seq", index)),
                t=float(entry.get("t", index / DEFAULT_FPS)),
                luma=luma,
                pose=_pose_from(entry.get("pose")),
                ground_alt_m=float(entry.get("ground_alt_m", 0.0)),
                label=_label_from(entry.get("label")),
            )


@dataclass(frozen=True, slots=True)
class MjpegFrames:
    """The real one: MJPEG from the arena, or from a recording of it.

    :param source: a filesystem path, or an ``http://`` URL — a camera port
        from ``ARENA.md`` §4. The address is configuration and is never
        hardcoded (issue #63).
    :param asset_id: stamped onto every frame.
    :param camera: the asset's published intrinsics.
    :param pose: the pose to stamp on every frame, when one is known. A live
        camera's pose changes frame to frame and comes from MAVLink, so the
        honest default is ``None`` and a caller that has telemetry joins it on
        itself.
    :param fps: replay rate for a recording; ignored for a live stream, which
        is stamped from the clock.
    :param max_consecutive_bad: how many undecodable frames in a row this
        stream tolerates before it gives up; see :meth:`frames`.

    A URL routes through :func:`~whiteout.vision.frames.live_frames` and so
    opens a socket. The gate never constructs this.
    """

    source: str
    asset_id: str
    camera: CameraModel
    pose: CameraPose | None = None
    ground_alt_m: float = 0.0
    fps: float = DEFAULT_FPS
    max_consecutive_bad: int = 30

    def _raw(self) -> Iterator[Frame]:
        if self.source.startswith(("http://", "https://")):
            return live_frames(self.source, self.asset_id)
        return recorded_frames(self.source, self.asset_id, fps=self.fps)

    def frames(self) -> Iterator[LumaFrame]:
        """Yield decoded frames, stepping over the ones that will not decode.

        **A bad frame must not end the stream.** These are yielded from a
        generator, so raising here does not merely report the frame — it
        closes the iterator for good, and the caller sees an iterator that
        *ended*, which is what a stream reaching its last frame also looks
        like. One truncated JPEG in a live MJPEG feed would retire that camera
        for the rest of the episode and four sensors would quietly become
        three, on a run that is judged once and has no retry.

        So a frame that will not decode, or that decodes to the wrong size, is
        skipped and the stream goes on. What is *not* survivable is every
        frame failing: no decoder installed, the wrong camera configured
        against this port, or a source that is not MJPEG at all. Those fail on
        frame after frame, so ``max_consecutive_bad`` in a row raises — with
        the last failure chained, because that message is the one that says
        which of the three it was. A run of good frames resets the count; the
        threshold is for a stream that is broken, not one that is lossy.
        """
        bad = 0
        # The arena does not serve the resolutions ARENA.md section 4
        # publishes - it upscales, keeping the aspect ratio - so the raster is
        # adopted from the stream and the published fields of view are kept.
        # See CameraModel.at_frame_size, which refuses a change of aspect.
        camera = self.camera
        for frame in self._raw():
            try:
                luma = decode_jpeg_luma(frame.jpeg)
                if luma.shape != (camera.height, camera.width):
                    # Rescale off the *published* camera every time, never off
                    # the last adopted one: a stream that changes size twice
                    # must not ratchet through an aspect the published pair
                    # never described.
                    camera = self.camera.at_frame_size(luma.shape[1], luma.shape[0])
            except VisionError as error:
                bad += 1
                if bad >= self.max_consecutive_bad:
                    raise VisionError(
                        f"{self.asset_id}: {bad} consecutive frames of {self.source!r} could "
                        f"not be read as {self.camera.name!r} imagery, so this is the stream "
                        f"and not a dropped frame: {error}"
                    ) from error
                continue
            bad = 0
            yield LumaFrame(
                asset_id=self.asset_id,
                camera=camera,
                seq=frame.seq,
                t=frame.t,
                luma=luma,
                pose=self.pose,
                ground_alt_m=self.ground_alt_m,
                label=None,
            )


def frame_source_from_env(
    environment: Mapping[str, str] | None = None, *, root: Path | None = None
) -> FrameSource:
    """Build the frame source this environment selects.

    :param environment: defaults to :data:`os.environ`; injected so a test can
        pin it without touching the process.
    :param root: what a relative fixture directory is taken against —
        :data:`DEFAULT_FIXTURE_DIR` and a relative ``WHITEOUT_VISION_FIXTURES``
        alike. An absolute value in the environment is used as it stands.
        Defaults to the working directory.
    :raises VisionError: on an unknown selector, or on ``mjpeg`` with no
        source set — loudly, rather than silently falling back to the fake,
        which would report a measurement against synthetic frames as a
        measurement against the arena.
    """
    env = os.environ if environment is None else environment
    choice = env.get(FRAMES_ENV, "fixture").strip().lower()
    base = Path.cwd() if root is None else root
    if choice in ("", "fixture"):
        # A relative override resolves against ``root`` like the default does.
        # Reading it as relative to the *process* directory instead made the
        # injected root a half-measure: a test could pin where the default
        # points and not where the environment's own value points, and the two
        # are the same knob.
        override = env.get(FIXTURE_ENV)
        directory = base / DEFAULT_FIXTURE_DIR if not override else base / override
        return FixtureFrames(directory)
    if choice == "mjpeg":
        source = env.get(MJPEG_ENV)
        if not source:
            raise VisionError(f"{FRAMES_ENV}=mjpeg needs {MJPEG_ENV} set to a path or a URL")
        asset_id = env.get("WHITEOUT_VISION_ASSET", "quadcopter")
        camera_name = env.get("WHITEOUT_VISION_CAMERA", "quadcopter")
        if camera_name not in CAMERAS:
            raise VisionError(
                f"WHITEOUT_VISION_CAMERA={camera_name!r} is not one of {sorted(CAMERAS)}"
            )
        return MjpegFrames(source=source, asset_id=asset_id, camera=CAMERAS[camera_name])
    raise VisionError(f"{FRAMES_ENV}={choice!r} is not one of 'fixture' or 'mjpeg'")


def write_fixtures(
    directory: str | os.PathLike[str], entries: Sequence[tuple[LumaFrame, str]], *, note: str
) -> Path:
    """Write a fixture directory: the PGM files, and the manifest naming them.

    :param entries: pairs of frame and the file name to write it under.
    :param note: recorded in the manifest, to say where these frames came from.
        A reader of a fixture directory needs to know whether they are looking
        at the arena or at :mod:`whiteout.vision.scene`, and a file name will
        not tell them.
    :returns: the manifest's path.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for frame, name in entries:
        write_pgm(destination / name, frame.luma)
        records.append(
            {
                "file": name,
                "asset_id": frame.asset_id,
                "camera": frame.camera.name,
                "seq": frame.seq,
                "t": frame.t,
                "pose": None if frame.pose is None else asdict(frame.pose),
                "ground_alt_m": frame.ground_alt_m,
                "label": None if frame.label is None else asdict(frame.label),
            }
        )
    manifest = destination / "manifest.json"
    manifest.write_text(
        json.dumps({"version": MANIFEST_VERSION, "note": note, "frames": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
