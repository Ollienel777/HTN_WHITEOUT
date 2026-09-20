#!/usr/bin/env python
"""Score the vessel detector, and say what it was scored against.

Issue #66 asks for a false-positive rate "measured on real frames, with the
number in the PR". **There are no real frames.** The arena is behind WireGuard
on the operator's machine and nothing in this repository has ever seen a
Dominion Dynamics render. So this harness measures the detector against
:mod:`whiteout.vision.scene`'s synthetic channel, prints the numbers, and
prints — every time, in the header — what they are numbers about.

The shape is deliberate: **when a recording arrives, measuring against it is
running this script at a directory.** Decode the MJPEG to PGM, write a
``manifest.json`` beside it (:func:`whiteout.vision.imagery.write_fixtures`
writes that format, and
``python scripts/score_detector.py --write-fixtures DIR`` shows one), label
the frames, then::

    python scripts/score_detector.py --frames path/to/real/frames

Nothing in :mod:`whiteout.vision.detect` changes for that.

Usage::

    python scripts/score_detector.py                    # the committed fixtures
    python scripts/score_detector.py --frames DIR       # any fixture directory
    python scripts/score_detector.py --synthetic        # a large generated corpus
    python scripts/score_detector.py --synthetic --sweep
    python scripts/score_detector.py --write-fixtures DIR --per-camera 1
    python scripts/score_detector.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Run as `python scripts/score_detector.py` from anywhere, and against
    # *this* tree: an ambient editable install can point at another worktree
    # (scripts/gate.py has the long version of this story).
    sys.path.insert(0, str(REPO_ROOT))

from scripts.jpeg_quantisation import quantise  # noqa: E402
from whiteout.vision.camera import CAMERAS  # noqa: E402
from whiteout.vision.detect import DEFAULT_PARAMS, DetectorParams, detect_vessel  # noqa: E402
from whiteout.vision.imagery import (  # noqa: E402
    DEFAULT_FIXTURE_DIR,
    FixtureFrames,
    Label,
    LumaFrame,
    write_fixtures,
)
from whiteout.vision.projection import CameraPose  # noqa: E402
from whiteout.vision.scene import CLEAR, SceneParams, render_scene  # noqa: E402

#: The four cameras of ``ARENA.md`` §3, with poses that put the **whole frame
#: on water inside the projection's range bound**. They are placeholders for
#: geometry, not a plan: the towers' lat/lon are an ``.env`` setting and a
#: strategic lever (§3), and the aircraft's attitudes come from MAVLink. The
#: ranges these give across a frame are 89 m to 822 m for a tower, which is
#: the right order for a channel 2 km across.
#:
#: The sky and over-range mask is therefore *not* exercised by this corpus —
#: a frame half full of sky would depress the detection rate for a reason that
#: has nothing to do with the detector. ``tests/test_vision_detect.py`` covers
#: it directly instead.
FLEET: tuple[tuple[str, str, CameraPose], ...] = (
    ("quadcopter", "quadcopter", CameraPose(71.99, -94.50, 120.0, 90.0, -55.0)),
    ("fixed-wing", "fixed-wing", CameraPose(71.99, -94.52, 200.0, 90.0, -35.0)),
    ("tower-1", "tower", CameraPose(71.985, -94.40, 60.0, 75.0, -22.0)),
    ("tower-2", "tower", CameraPose(71.995, -94.60, 60.0, 250.0, -22.0)),
)

#: Water is the datum's zero here. Issue #76 has not settled what the arena
#: reports altitude against; this is the synthetic corpus's own convention and
#: is carried on every frame rather than assumed downstream.
GROUND_ALT_M = 0.0


@dataclass(frozen=True, slots=True)
class Tally:
    """What one corpus produced."""

    frames: int
    labelled: int
    with_vessel: int
    without_vessel: int
    detections: int
    hits: int
    misses: int
    wrong: int
    false_on_empty: int
    errors_px: tuple[float, ...]
    seconds: float

    @property
    def detection_rate(self) -> float:
        """Share of localisable vessel frames the detector got right."""
        return self.hits / self.with_vessel if self.with_vessel else float("nan")

    @property
    def false_positive_rate(self) -> float:
        """Detections on frames labelled *no vessel*, per such frame.

        The headline number issue #66 asks for.
        """
        return self.false_on_empty / self.without_vessel if self.without_vessel else float("nan")

    @property
    def wrong_per_frame(self) -> float:
        """Every detection that is **not** the vessel, over every labelled frame.

        Wider than :attr:`false_positive_rate` and the one that matters for
        the belief update: a detection 300 px from the hull poisons it exactly
        as a detection on an empty frame does, and posts the same wrong
        lat/lon to the tracks API.
        """
        return self.wrong / self.labelled if self.labelled else float("nan")

    @property
    def ms_per_frame(self) -> float:
        return 1000.0 * self.seconds / self.frames if self.frames else float("nan")

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0.0 else float("nan")

    def to_dict(self) -> dict[str, object]:
        errors = np.array(self.errors_px) if self.errors_px else np.array([float("nan")])
        return {
            "frames": self.frames,
            "labelled": self.labelled,
            "with_vessel": self.with_vessel,
            "without_vessel": self.without_vessel,
            "detections": self.detections,
            "hits": self.hits,
            "misses": self.misses,
            "wrong": self.wrong,
            "false_on_empty": self.false_on_empty,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "wrong_per_frame": self.wrong_per_frame,
            "localisation_px_median": float(np.median(errors)),
            "localisation_px_p95": float(np.percentile(errors, 95)),
            "ms_per_frame": self.ms_per_frame,
            "fps": self.fps,
            "per_camera_hz_with_four": self.fps / len(FLEET),
        }


def synthetic_frames(
    per_camera: int, seed: int, params: SceneParams
) -> Iterator[tuple[LumaFrame, str]]:
    """Render a corpus: half the frames hold the vessel and half do not.

    Every draw comes from a generator seeded from ``seed``, the asset and the
    frame index, so a corpus is reproducible frame by frame and a surprising
    number can be re-rendered on its own.

    Presence alternates on the **asset and** the index rather than the index
    alone, so that even one frame per camera is a corpus with both cases in
    it — which is what the small committed fixture directory is.
    """
    for asset_index, (asset_id, camera_name, pose) in enumerate(FLEET):
        camera = CAMERAS[camera_name]
        for index in range(per_camera):
            rng = np.random.default_rng((seed, asset_index, index))
            present = (asset_index + index) % 2 == 0
            scene = render_scene(camera, rng, params=params, with_vessel=present)
            label = (
                Label(
                    present=True,
                    px=scene.vessel_px[0] if scene.vessel_px else None,
                    py=scene.vessel_px[1] if scene.vessel_px else None,
                    tolerance_px=max(8.0, scene.vessel_length_px),
                )
                if present
                else Label(present=False)
            )
            yield (
                LumaFrame(
                    asset_id=asset_id,
                    camera=camera,
                    seq=index,
                    t=index / 10.0,
                    luma=scene.luma,
                    pose=pose,
                    ground_alt_m=GROUND_ALT_M,
                    label=label,
                ),
                f"{asset_id}-{index:04d}.pgm",
            )


def score(frames: Sequence[LumaFrame], params: DetectorParams) -> Tally:
    """Run the detector over every frame and count what came out.

    A frame with no pose is skipped, not guessed at: the detector needs one to
    mask the sky and to refuse a pixel that will not project, and inventing a
    pose would make the score a score of a different detector.
    """
    labelled = with_vessel = without_vessel = 0
    detections = hits = misses = wrong = false_on_empty = 0
    errors: list[float] = []
    elapsed = 0.0
    counted = 0
    for frame in frames:
        if frame.pose is None:
            continue
        counted += 1
        started = time.perf_counter()
        found = detect_vessel(
            frame.luma,
            frame.camera,
            frame.pose,
            asset_id=frame.asset_id,
            seq=frame.seq,
            t=frame.t,
            ground_alt_m=frame.ground_alt_m,
            params=params,
        )
        elapsed += time.perf_counter() - started
        if found is not None:
            detections += 1
        label = frame.label
        if label is None:
            continue
        labelled += 1
        if not label.present:
            without_vessel += 1
            if found is not None:
                false_on_empty += 1
                wrong += 1
            continue
        if not label.localised:
            # Present but unlabelled: it can neither be a hit nor a false
            # alarm, so it counts toward nothing but the frame total.
            labelled -= 1
            continue
        with_vessel += 1
        if found is None:
            misses += 1
            continue
        assert label.px is not None and label.py is not None
        error = math.hypot(found.px - label.px, found.py - label.py)
        if error <= label.tolerance_px:
            hits += 1
            errors.append(error)
        else:
            wrong += 1
    return Tally(
        frames=counted,
        labelled=labelled,
        with_vessel=with_vessel,
        without_vessel=without_vessel,
        detections=detections,
        hits=hits,
        misses=misses,
        wrong=wrong,
        false_on_empty=false_on_empty,
        errors_px=tuple(errors),
        seconds=elapsed,
    )


def _print_tally(title: str, tally: Tally) -> None:
    facts = tally.to_dict()
    print(f"\n{title}")
    print(f"  frames                  {tally.frames}  ({tally.labelled} labelled)")
    print(f"  with vessel / without   {tally.with_vessel} / {tally.without_vessel}")
    print(f"  detections emitted      {tally.detections}")
    print(f"  detection rate          {tally.detection_rate:.3f}  ({tally.hits} hits)")
    print(
        f"  false positives         {tally.false_positive_rate:.3f} per empty frame  "
        f"({tally.false_on_empty} of {tally.without_vessel})"
    )
    print(
        f"  any wrong detection     {tally.wrong_per_frame:.3f} per labelled frame  "
        f"({tally.wrong} of {tally.labelled})"
    )
    print(
        f"  localisation error      {facts['localisation_px_median']:.2f} px median, "
        f"{facts['localisation_px_p95']:.2f} px p95"
    )
    print(
        f"  throughput              {tally.ms_per_frame:.1f} ms/frame, {tally.fps:.1f} fps, "
        f"{tally.fps / len(FLEET):.1f} Hz per camera across {len(FLEET)}"
    )


def _sweep(per_camera: int, seed: int, params: DetectorParams) -> list[dict[str, object]]:
    """Detection and false-positive rates against the two scene assumptions.

    Contrast and fog are the quantities ``ARENA.md`` does not pin and the
    detector is most sensitive to, so a single headline number would be a
    number about a guess. The ladder says which guess.
    """
    rows: list[dict[str, object]] = []
    print("\ncontrast ladder (hull counts below local water, fog 0.10)")
    print("  contrast  det rate  fp/empty  wrong/frame")
    for contrast in (12.0, 17.0, 25.0, 35.0):
        scene = replace(CLEAR, vessel_contrast=contrast)
        tally = score([f for f, _ in synthetic_frames(per_camera, seed, scene)], params)
        rows.append({"contrast": contrast, **tally.to_dict()})
        print(
            f"  {contrast:8.0f}  {tally.detection_rate:8.3f}  "
            f"{tally.false_positive_rate:8.3f}  {tally.wrong_per_frame:11.3f}"
        )
    print("\nfog ladder (hull contrast 17 counts)")
    print("  fog       det rate  fp/empty  wrong/frame")
    for fog in (0.0, 0.2, 0.4, 0.6, 0.8):
        scene = replace(CLEAR, fog=fog)
        tally = score([f for f, _ in synthetic_frames(per_camera, seed, scene)], params)
        rows.append({"fog": fog, **tally.to_dict()})
        print(
            f"  {fog:8.2f}  {tally.detection_rate:8.3f}  "
            f"{tally.false_positive_rate:8.3f}  {tally.wrong_per_frame:11.3f}"
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=None,
        help=f"a fixture directory to score; defaults to {DEFAULT_FIXTURE_DIR}",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="generate a corpus instead of reading one, for a larger sample",
    )
    parser.add_argument("--per-camera", type=int, default=50, help="synthetic frames per camera")
    parser.add_argument("--seed", type=int, default=7, help="corpus seed")
    parser.add_argument(
        "--sweep", action="store_true", help="also walk the contrast and fog ladders"
    )
    parser.add_argument(
        "--write-fixtures",
        type=Path,
        default=None,
        help="render a corpus into this directory and exit",
    )
    parser.add_argument(
        "--jpeg",
        type=int,
        default=None,
        metavar="QUALITY",
        help=(
            "put every frame through baseline JPEG luma quantisation at this quality "
            "(1-100) before scoring it. The arena publishes JPEG and the generator does "
            "not, so a number measured without this is a number about uncompressed "
            "frames; see scripts/jpeg_quantisation.py and issue #90"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the numbers as JSON too")
    args = parser.parse_args(argv)

    params = DEFAULT_PARAMS

    if args.write_fixtures is not None:
        entries = list(synthetic_frames(args.per_camera, args.seed, CLEAR))
        manifest = write_fixtures(
            args.write_fixtures,
            entries,
            note=(
                f"Synthetic Bellot Strait frames from whiteout.vision.scene, seed {args.seed}, "
                f"{args.per_camera} per camera. NOT arena imagery: no frame from Dominion "
                f"Dynamics' render exists in this repository. Regenerate with "
                f"scripts/score_detector.py --write-fixtures."
            ),
        )
        print(f"wrote {len(entries)} frames and {manifest}")
        return 0

    print("=" * 72)
    if args.synthetic:
        source = (
            f"SYNTHETIC frames from whiteout.vision.scene — seed {args.seed}, "
            f"{args.per_camera} per camera, {len(FLEET)} cameras"
        )
        frames = [frame for frame, _ in synthetic_frames(args.per_camera, args.seed, CLEAR)]
    else:
        directory = args.frames if args.frames is not None else REPO_ROOT / DEFAULT_FIXTURE_DIR
        source = f"fixture directory {directory}"
        frames = list(FixtureFrames(Path(directory)).frames())
    if args.jpeg is not None:
        frames = [replace(frame, luma=quantise(frame.luma, args.jpeg)) for frame in frames]
        source += f", through JPEG luma quantisation at quality {args.jpeg}"
    print(f"scored against: {source}")
    print(
        "THIS IS NOT A MEASUREMENT AGAINST ARENA IMAGERY unless the directory above holds\n"
        "frames recorded from the arena. No such frames exist in this repository; recording\n"
        "them is issue #63 and a human action."
    )
    if args.jpeg is not None:
        print(
            "--jpeg is a STAND-IN for the arena's encoder, not a model of it: the transform\n"
            "that flattens pixel-to-pixel variation, with no chroma and no entropy coding.\n"
            "It shows that a number measured uncompressed does not survive compression. It\n"
            "does not say what the arena's own numbers are."
        )
    print("=" * 72)

    tally = score(frames, params)
    _print_tally("overall", tally)
    payload: dict[str, object] = {"source": source, "overall": tally.to_dict()}
    if args.sweep:
        payload["sweeps"] = _sweep(args.per_camera, args.seed, params)
    if args.json:
        print("\n" + json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
