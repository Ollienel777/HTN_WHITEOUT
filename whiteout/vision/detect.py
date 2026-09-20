"""Find the shadow vessel in one frame, or say nothing.

Issue #66. The scene, from ``hackathon/ARENA.md`` §2 and §4: a **dark** hull on
**dark** water among **bright** ice floes, in a channel 2 km across, with no
other moving object anywhere in the world. That description is most of the
algorithm, and none of it needs a model.

Three things in this channel are dark and roughly hull-sized, and each is
refused for a different reason:

``ice, and the glint on the water``
    Brighter than the water, so they lose on polarity alone and are excluded
    from the statistic entirely rather than merely outscored.
``a crack through a floe``
    Open water seen through ice: the hull's polarity, the hull's size, and
    more than the hull's depth. What gives it away is its **surround** — a
    vessel in a 2 km channel sits on open water, a crack is bounded by ice —
    so the statistic is centre against surround with a cap on how much of the
    surround may be ice.
``a floe's shadow on the water``
    The hard one, because its surround is water and gives nothing away. It is
    told apart by **depth**: a shadow on water this dark has little room left
    to darken, and the hull is the darkest thing in the channel.

And the geometry refuses a great deal before any of that. The camera pose is
known and the published field of view is exact, so sky — and water so far down
the boresight that one pixel of angle is kilometres of range — cannot hold the
target and is masked before anything is picked, from
:func:`~whiteout.vision.projection.camera_basis`.

**The two numbers a candidate is judged on.** Write ``D = -r / sigma``, the
residual against the row's water level over the row's *sensor noise*, positive
where a pixel is darker than the water. Over a centre box of half-width ``k``
and a surround annulus out to ``3k + 2``, counting non-ice pixels only:

.. math::

    \\text{depth} = \\bar{D}_\\text{centre} - \\bar{D}_\\text{surround}
    \\qquad
    R = \\frac{\\text{depth}}
             {\\sqrt{1/n_\\text{centre} + 1/n_\\text{surround}}}

``depth`` is *how much darker*, in units of the frame's own noise, and does not
grow with the patch's size. ``R`` is *how confidently darker*, a two-sample z,
and grows with the square root of it. **Both are needed, and using only the
second was a bug.** ``R`` alone ranks a broad shallow shadow above a small deep
hull and picks the shadow every time; ``depth`` alone throws away the evidence
that a hundred consistent pixels carry. So depth sets a floor and ``R`` ranks
what clears it — depth read at the scale that maximises depth, ``R`` at the
scale that maximises ``R``.

Under water alone ``R`` is roughly standard normal, so its threshold is in
units a reader can reason about — but only roughly, because the optics
correlate neighbouring pixels. The working thresholds are **calibrated on
frames known to hold no vessel** rather than read off a table. That
calibration is ``scripts/score_detector.py``, and the numbers it produced are
in the pull request.

**Degrading honestly is the point, not a nicety.** ``ARENA.md`` §5 scores
*accuracy*, so a confident wrong lat/lon is worse than silence. Five separate
things here can only ever turn a detection into ``None``, never the other way
round:

1. ``depth`` must clear :attr:`DetectorParams.min_depth_sigma`. This is what
   fog eats: haze compresses the hull's contrast in counts while leaving the
   sensor's noise where it was, so depth falls and the floor stops being met.
2. ``peak_sigma`` must clear :attr:`DetectorParams.min_peak_sigma`.
3. The **runner-up** matters. A frame with two equally good dark patches is a
   frame this detector does not understand — there is exactly one vessel in
   the world — so the margin between best and second best multiplies into the
   confidence, and an ambiguous frame falls under the confidence floor.
4. Confidence must clear :attr:`DetectorParams.min_confidence`.
5. The winning pixel is put through
   :func:`~whiteout.vision.projection.project_pixel_to_ground` before it is
   returned. If it will not project — above the horizon, past the range where
   a flat water plane means anything — there is no detection. Nothing
   downstream has to re-check, and :meth:`VesselDetection.to_ground` cannot
   raise on a detection this module emitted.

**One degradation is recorded rather than refused**, and it is the one that
is not about the imagery: whether the frame and the pose it is projected with
belonged to the same instant. It rides along as
:attr:`VesselDetection.sync`, whose docstring argues why a skewed pose
qualifies a detection instead of cancelling it, and where the refusal does
live.

**Where it stops working**, measured rather than guessed: very heavy floe
cover, where the channel reads as a cracked sheet rather than as water with
ice in it. The false-alarm rate climbs sharply there and no threshold in this
module fixes it, because a crack in a sheet and a hull differ in one frame
only by their surround, and in a sheet there is no water surround to have.
The cue that does separate them is the one issue #66 also names and this
module deliberately does not use: **the vessel is the only thing in the world
that moves.** That is a follow-up, not a knob.

The datum question is issue **#76**: ``ground_alt_m`` is a parameter here and
is carried on the detection, so the projection that admitted a pixel and the
projection a caller later performs are the same one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whiteout.types import PoseSync
from whiteout.vision.camera import CameraModel, VisionError
from whiteout.vision.projection import (
    CameraPose,
    GeoPoint,
    ProjectionError,
    camera_basis,
    max_flat_plane_range_m,
    project_pixel_to_ground,
)

__all__ = [
    "DEFAULT_PARAMS",
    "DetectorParams",
    "VesselDetection",
    "detect_vessel",
]

_Float = NDArray[np.float64]

#: The per-row quantile taken as the water's level. See :func:`_row_background`
#: for why it is not the median, and ``tests/test_vision_detect.py`` for the
#: floe density that settles it.
WATER_QUANTILE = 10.0

#: The upper end of the water's spread. Also low, for the same reason
#: :data:`WATER_QUANTILE` is: the pair has to stay inside the water however
#: much of the row is floe. On clean water the gap between the two is
#: ``Phi^-1(0.30) - Phi^-1(0.10) = 0.758`` standard deviations.
WATER_SPREAD_QUANTILE = 30.0


@dataclass(frozen=True, slots=True)
class DetectorParams:
    """Every threshold the detector has, in one place.

    :param scales: centre-box half-widths, pixels. A hull spans roughly 5 to
        22 pixels at the fleet's fields of view over a 2 km channel, so three
        boxes — 3x3, 5x5, 9x9 — bracket it, and the strongest response across
        them wins.
    :param surround_gain: the surround annulus runs out to
        ``surround_gain * k + 2`` for a centre half-width ``k``.
    :param ice_spread_gain: a pixel more than this many *water spreads*
        above the row's water level is ice or glint, and is excluded from both
        the centre and the surround. Excluded, not clamped: a floe edge inside
        the surround would otherwise drag the local water level bright and
        manufacture contrast for anything dark beside it. The default of 7
        puts the cut about four standard deviations above clean water's mean —
        see :func:`_row_background` for the arithmetic, and for why this is
        *not* measured in the sensor-noise sigma the detection statistic
        uses.
    :param min_centre_px: refuse a candidate whose centre box holds fewer
        non-ice pixels than this. A centre that is mostly floe has no water
        level to be dark against.
    :param min_surround_px: the same for the surround.
    :param max_centre_ice: largest fraction of the centre box that may be ice.
    :param max_surround_ice: largest fraction of the surround annulus that may
        be ice. **This is the crack rejector**: see :func:`_response`. Raising
        it lets the detector see a vessel threading a floe field and lets the
        cracks through the floes in with it; the two move together and there
        is no arena imagery here to settle where the knee is.
    :param min_depth_sigma: how far below the surrounding water's level the
        centre must sit, in robust sigmas, regardless of how *significant*
        that is. The shadow rejector, and the assumption the detector rests on
        most heavily; :func:`_response` explains why significance alone is not
        enough.
    :param min_depth_counts: the same floor as ``min_depth_sigma``, but in
        **8-bit counts** and so unable to be met by a small denominator. Every
        other gate here is a ratio to the per-row noise, which makes all of
        them vulnerable in the same way to that noise being measured too low —
        and on JPEG imagery it is (:func:`_row_background`). This one asks
        instead how much darker the hull actually is, which is the quantity
        fog compresses, so it holds the bargain the module makes: a degraded
        frame yields ``None``. The default sits below the hull's contrast
        through thin fog and above what a floe's shadow on open water reaches.

        **The default is 12 and not 4, and #90 is why.** The arena publishes
        JPEG, and at 4 the detector reported a vessel on more than half the
        empty frames of a quantised corpus. Swept over three seeds, 24 frames
        per camera, ``scripts/score_detector.py --synthetic``:

        .. code-block:: text

            seed   counts        uncompressed            quality 75
                             recall      fp          recall      fp
            11        4       0.583     0.021         0.312     0.521
            11       12       0.500     0.000         0.438     0.104
            23        4       0.646     0.000         0.312     0.521
            23       12       0.542     0.000         0.542     0.250
            47        4       0.604     0.042         0.354     0.542
            47       12       0.479     0.000         0.375     0.062

        On the compressed corpus 12 **dominates** 4 on every seed — more
        detections *and* between two and eight times fewer false alarms — and
        costs about a tenth of the recall on frames that will never reach us,
        since the arena does not serve uncompressed. A false positive posts a
        lat/lon, and *accuracy* is scored, so that is the direction to err in.

        **Two things this number is not.** It is not tuned against Dominion
        Dynamics' encoder, only against ``scripts/jpeg_quantisation.py``,
        which has no chroma and no entropy coding. And it is not a monotone
        knob: raising this floor deletes rivals before ``runner_up`` is read
        (:func:`detect_vessel`), so a tighter gate can *raise* the detection
        rate — 4 to 10 does exactly that on seed 11. Part of the gain above is
        that interaction rather than a stricter detector, which is why #90
        stays open and why real frames (#63) settle it and this does not.
    :param min_sigma: floor under the per-row robust scale, 8-bit counts. A
        row that is genuinely flat would otherwise divide by nearly zero and
        turn sensor quantisation into a detection. It is a floor and not an
        estimate, so a row that hits it is a row whose statistics are *not*
        measured — which is why ``min_depth_counts`` exists to backstop it.
    :param min_peak_sigma: the response a candidate must reach to be
        considered at all.
    :param confident_sigma: the response at which contrast stops adding
        confidence.
    :param margin_fraction: how far clear of the runner-up the peak must
        stand, **as a fraction of the peak**, for the margin to stop costing
        confidence. A fraction and not an absolute gap: the statistic's scale
        moves with the target's size and the frame's noise, so a fixed gap of
        four sigmas is decisive next to a peak of twelve and meaningless next
        to a peak of sixty — and sixty against fifty-eight is two candidate
        hulls, which is the case this exists to catch.
    :param exclusion_px: the runner-up is taken outside a box this wide about
        the peak, so the peak's own skirt is not read as a second target.
    :param min_confidence: the floor a detection must clear to be emitted.
    :param max_range_m: furthest ground range a detection may project to.
        ``None`` uses :func:`~whiteout.vision.projection.max_flat_plane_range_m`
        for the camera's height. Bellot Strait is 2 km across, so a caller that
        knows where the asset is pointing should pass something far tighter.
    """

    scales: tuple[int, ...] = (1, 2, 4)
    surround_gain: int = 3
    ice_spread_gain: float = 7.0
    min_centre_px: int = 4
    min_surround_px: int = 24
    max_centre_ice: float = 0.15
    max_surround_ice: float = 0.10
    min_depth_sigma: float = 3.6
    min_depth_counts: float = 12.0
    min_sigma: float = 0.75
    min_peak_sigma: float = 11.0
    confident_sigma: float = 26.0
    margin_fraction: float = 0.25
    exclusion_px: int = 12
    min_confidence: float = 0.25
    max_range_m: float | None = None

    def __post_init__(self) -> None:
        if not self.scales or any(k < 1 for k in self.scales):
            raise VisionError(f"scales must be a non-empty tuple of positive ints, got {self!r}")
        if self.surround_gain < 2:
            raise VisionError(f"surround_gain must be at least 2, got {self.surround_gain!r}")
        if self.min_sigma <= 0.0:
            raise VisionError(f"min_sigma must be positive, got {self.min_sigma!r}")
        if self.min_depth_counts < 0.0:
            raise VisionError(
                f"min_depth_counts must be non-negative, got {self.min_depth_counts!r}"
            )
        for name, fraction in (
            ("max_centre_ice", self.max_centre_ice),
            ("max_surround_ice", self.max_surround_ice),
        ):
            if not 0.0 <= fraction <= 1.0:
                raise VisionError(f"{name} must lie in [0, 1], got {fraction!r}")
        if self.confident_sigma <= self.min_peak_sigma:
            raise VisionError(
                f"confident_sigma must exceed min_peak_sigma, got "
                f"{self.confident_sigma!r} against {self.min_peak_sigma!r}"
            )
        if self.min_peak_sigma <= 0.0:
            # The margin is a fraction *of the peak*, so a non-positive floor
            # would let a zero peak through and divide by it.
            raise VisionError(f"min_peak_sigma must be positive, got {self.min_peak_sigma!r}")
        if not 0.0 < self.margin_fraction <= 1.0:
            raise VisionError(f"margin_fraction must lie in (0, 1], got {self.margin_fraction!r}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise VisionError(f"min_confidence must lie in [0, 1], got {self.min_confidence!r}")
        if self.max_range_m is not None and not (
            math.isfinite(self.max_range_m) and self.max_range_m > 0.0
        ):
            raise VisionError(f"max_range_m must be positive and finite, got {self.max_range_m!r}")


#: The tuning `scripts/score_detector.py` was run against. See the PR for the
#: false-positive and detection rates it produces on synthetic frames.
DEFAULT_PARAMS = DetectorParams()


@dataclass(frozen=True, slots=True)
class VesselDetection:
    """One hit: where in the frame, how sure, and what the camera was doing.

    :param asset_id: which camera saw it, e.g. ``"tower-1"``.
    :param seq: the frame's index in its stream.
    :param t: the frame's timestamp, seconds.
    :param px: pixel column, and :param py: pixel row, in the convention of
        :mod:`whiteout.vision.camera` — origin at the frame's top-left corner.
        Sub-pixel: this is an intensity-weighted centroid, not a box centre.
    :param confidence: 0 to 1. See the module docstring for what lowers it.
    :param peak_sigma: the detection statistic at the winning pixel.
    :param depth_sigma: how far below the surrounding water the hull sits, in
        units of the frame's own sensor noise. The physical half of the
        evidence — ``peak_sigma`` says how sure, this says how dark — and the
        quantity fog eats.
    :param runner_up_sigma: the best response elsewhere in the frame. The gap
        between the two is half of ``confidence``, and is worth logging on its
        own: a frame where it is small is a frame to distrust.
    :param area_px: how many pixels carried the hull's contrast. A crude size,
        useful as a sanity check against range.
    :param scale_px: the centre half-width that won.
    :param camera: the published intrinsics this frame came through.
    :param pose: where the camera was and where it pointed for this frame —
        the thing that makes the pixel a place. Carried, not looked up, because
        a pose fetched later is a different pose.
    :param ground_alt_m: the water plane's altitude in ``pose.alt_m``'s datum,
        as it was when this detection was admitted. Issue #76.
    :param max_range_m: the range bound that admitted it.
    :param sync: whether ``pose`` and this frame belonged to one instant
        (:class:`~whiteout.types.PoseSync`), or ``None`` when the caller did
        not establish it. Carried, not recomputed: a :class:`CameraPose` has no
        clock in it, so this is the one place the answer can be kept.

    **An unsynchronised pose flags the detection; it does not refuse it.**
    Both halves of that are deliberate.

    Refusing here would be refusing the wrong thing. Skew does not make the
    hull absent from the frame — the pixel evidence is exactly as good — it
    makes the *place* wrong, by 5.5 m per quarter-second at the fixed-wing's
    22 m/s. A ``None`` from this module means "no vessel, or too degraded a
    frame to say", and every gate that produces one is a statement about the
    imagery (the module docstring lists all five). Adding a sixth that is a
    statement about the *telemetry* would let one adapter that never reports a
    measurement time silence a whole camera — the arena's adapter is exactly
    that adapter (it has not established the ``time_boot_ms`` offset, and
    ``None`` is the correct thing for it to report) — and would throw away the
    presence and the coverage along with the position.

    Nor is flagging the soft option. ``sync`` is on the detection, in the
    :class:`~whiteout.tracks.maintain.Sighting` it becomes and in the episode
    log's :class:`~whiteout.types.Contact`, so no consumer can trust the fix
    without having been told, and the viewer and the scorer can see which
    fixes were assembled out of two different instants.

    The refusal belongs one seam further on, where a pixel becomes a lat/lon
    somebody is scored on: :meth:`whiteout.vision.sightings.VisionSightings.
    _sight` drops a ``telemetry_stale`` pair, because there the skew is
    *measured* and exceeds a bound the operator set. What is merely unknown is
    carried forward and labelled, because refusing the unknown would make the
    fleet blind in the arena we are actually flying in.
    """

    asset_id: str
    seq: int
    t: float
    px: float
    py: float
    confidence: float
    peak_sigma: float
    depth_sigma: float
    runner_up_sigma: float
    area_px: float
    scale_px: int
    camera: CameraModel
    pose: CameraPose
    ground_alt_m: float
    max_range_m: float
    sync: PoseSync | None = None

    def to_ground(self) -> GeoPoint:
        """Project this detection onto the water plane.

        Cannot raise for a detection :func:`detect_vessel` returned: the same
        call, with the same arguments, is what admitted the pixel.
        """
        return project_pixel_to_ground(
            self.camera,
            self.pose,
            self.px,
            self.py,
            ground_alt_m=self.ground_alt_m,
            max_range_m=self.max_range_m,
        )


def _as_image(luma: NDArray[np.generic], camera: CameraModel) -> _Float:
    if luma.ndim != 2:
        raise VisionError(f"{camera.name}: frame must be 2-D luma, got shape {luma.shape!r}")
    if luma.shape != (camera.height, camera.width):
        raise VisionError(
            f"{camera.name}: frame is {luma.shape[1]}x{luma.shape[0]}, but this camera's "
            f"published frame is {camera.width}x{camera.height}"
        )
    image = np.asarray(luma, dtype=np.float64)
    if not bool(np.all(np.isfinite(image))):
        raise VisionError(f"{camera.name}: frame holds non-finite samples")
    return image


def _row_background(image: _Float, min_sigma: float) -> tuple[_Float, _Float, _Float]:
    """Per-row water level, the water's **spread**, and the **sensor noise**.

    Per *row* because fog's contribution grows with range and range grows up
    the frame, so both have a strong vertical gradient and almost none
    horizontally.

    **A low quantile, not the median.** The level's only job is to set the ice
    threshold, and the median holds only while water is the majority of a row.
    Bellot Strait is a channel *with ice floes in it*; at around half cover the
    median jumps to the ice, every floe stops being masked, and open water
    starts reading as a hole a hundred counts deep — a detector that was
    silent through light ice becomes confidently wrong through heavy ice,
    which is the exact failure issue #66 says costs us most.
    :data:`WATER_QUANTILE` holds to three-quarters cover instead. It costs
    nothing elsewhere: a per-row constant cancels in the centre-against-
    surround difference, so shifting the level moves the ice mask and nothing
    else.

    **Two scales come back, and confusing them is a bug this code has already
    had.** The ice threshold needs the spread of the *water's appearance* —
    swell, the sun's angle across the channel, everything smooth — because it
    is asking "is this pixel one of the bright things". The detection
    statistic needs the spread of the *sensor's noise*, because it is asking
    "how many samples' worth of evidence is this". They differ by a factor of
    two or more on any frame with a sea state, and using the noise for the ice
    cut classifies a third of open water as floe, which then fails the
    surround's ice-fraction test and makes the detector blind on exactly the
    calm, clear frames it should be surest on.

    The water's spread is taken between two low quantiles,
    :data:`WATER_QUANTILE` and :data:`WATER_SPREAD_QUANTILE`, so that it is
    measured on water however much ice is in the row; on clean water their
    gap is ``0.758`` standard deviations, which is where
    :attr:`DetectorParams.ice_spread_gain`'s default comes from.

    The scale is the one thing here that must not be a spread of *content*.
    An ordinary robust spread of the row — a quartile, a MAD — measures the
    swell, the shadows and the floe edges, all of which are structure the
    centre-against-surround statistic already removes, and all of which grow
    when the sea gets interesting. Dividing by that makes the detector blinder
    exactly when there is more to look at. So the scale is estimated from
    **adjacent-pixel differences** instead,

    .. math::

        \\sigma = \\frac{1.4826}{\\sqrt{2}}\\;
                 \\operatorname{median}_j \\bigl| I_{i,j+1} - I_{i,j} \\bigr|

    which is blind to anything smooth and sees only what varies pixel to
    pixel. Floe edges are steps, and are a minority of any row, so the median
    walks past them. What comes out is the noise the camera adds, which is
    what "how many sigmas deep" should be measured in — and it is why fog
    turns detections off: haze compresses the hull's depth in counts while
    leaving the noise where it was.

    **This assumes the noise is white, and on arena imagery it is not.** The
    arena publishes JPEG, which is a low-pass filter: quantising the
    high-frequency coefficients of every 8x8 block flattens adjacent-pixel
    differences while leaving a hull-sized blob — and a floe's shadow —
    untouched. Put :mod:`whiteout.vision.scene`'s frames through baseline luma
    quantisation (``scripts/jpeg_quantisation.py``) and this estimate falls to
    a third of the noise actually present at quality 75, and onto the
    :attr:`DetectorParams.min_sigma` floor at quality 60. Every statistic this
    module quotes in sigmas is a ratio to this number, so it is the one place
    where being wrong opens every gate at once, which is why
    :attr:`DetectorParams.min_depth_counts` exists alongside the sigma floor
    and is not measured in sigmas at all. Issue #90 has the measurement and
    the candidate fixes; none of them can be chosen without real frames.
    """
    low, high = np.percentile(image, (WATER_QUANTILE, WATER_SPREAD_QUANTILE), axis=1)
    residual = image - low[:, None]
    spread = np.maximum(high - low, min_sigma)
    steps = np.abs(np.diff(image, axis=1))
    sigma = np.maximum(1.4826 / math.sqrt(2.0) * np.median(steps, axis=1), min_sigma)
    return residual, sigma[:, None], spread[:, None]


def _integral(values: _Float, pad: int, shape: tuple[int, int]) -> _Float:
    """A summed-area table, pre-padded so every box sum is four slices.

    The table proper is ``(H+1, W+1)`` with a zero row and column in front. It
    is then edge-replicated by ``pad`` on all sides, which is what makes
    :func:`_box_sum` pure slicing: for any radius up to ``pad``, clipping a
    box at the frame's edge and reading the replicated edge of the table give
    the same four numbers. Clipping with index arrays instead costs four
    gathers of a whole frame per box per scale, and that was most of the
    detector's budget.
    """
    running = np.cumsum(values, axis=0, dtype=np.float64)
    np.cumsum(running, axis=1, out=running)
    table = np.zeros((shape[0] + 1, shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = running
    return np.pad(table, ((pad, pad + 1), (pad, pad + 1)), mode="edge")


def _box_sum(padded: _Float, radius: int, pad: int, shape: tuple[int, int]) -> _Float:
    """Sum over the ``(2r+1)`` square about every pixel, clipped at the edges."""
    height, width = shape
    low = pad - radius
    high = pad + radius + 1
    return (
        padded[high : high + height, high : high + width]
        - padded[low : low + height, high : high + width]
        - padded[high : high + height, low : low + width]
        + padded[low : low + height, low : low + width]
    )


def _water_mask(
    camera: CameraModel, pose: CameraPose, height_above_plane: float, max_range_m: float
) -> NDArray[np.bool_]:
    """Pixels whose ray descends steeply enough to be worth projecting.

    The up-component of the ray through a pixel is a linear combination of the
    camera's ENU basis, so one call to
    :func:`~whiteout.vision.projection.camera_basis` vectorises over the whole
    frame without restating the projection. Requiring
    ``-v_up >= h / max_range`` is the range cut, approximately: the exact
    ground range also divides by the horizontal part of the ray, which is
    between 1 and about 1.4 across a frame. So this admits a little more than
    it should, and :func:`project_pixel_to_ground` — the authority — rejects
    the remainder for the one pixel that wins.
    """
    right, down, forward = camera_basis(pose)
    columns = (np.arange(camera.width) + 0.5 - camera.cx) / camera.fx
    rows = (np.arange(camera.height) + 0.5 - camera.cy) / camera.fy
    up = rows[:, None] * down[2] + columns[None, :] * right[2] + forward[2]
    return up <= -(height_above_plane / max_range_m)


def _response(
    weighted_integral: _Float,
    water_integral: _Float,
    radius: int,
    params: DetectorParams,
    pad: int,
    shape: tuple[int, int],
) -> tuple[_Float, _Float]:
    """Centre-against-surround z and depth at one scale; ``-inf`` where neither forms.

    The two summed-area tables are built once by the caller and reused across
    scales, because they do not depend on the box size.

    Four ways a pixel is refused here rather than scored:

    - too few non-ice pixels in the centre to have a level at all;
    - too few in the surround to compare against;
    - **too much ice in the surround**. This is the crack rejector. Open water
      seen through a crack in a floe has the hull's polarity, the hull's size
      and more than the hull's depth, and the only thing in one frame that
      tells them apart is what is around them: a vessel in a 2 km channel sits
      on open water, a crack is bounded by ice. Excluding ice from the *level*
      is not enough on its own — it makes a crack look cleaner, not worse — so
      the fraction is checked too.
    - **not deep enough**, ``min_depth_sigma``. This is the shadow rejector,
      and it is the assumption the whole detector rests on. The z above grows
      with the square root of the area, so a broad shallow patch — a floe's
      shadow lying on open water, whose surround gives nothing away —
      outscores a small deep one and would otherwise win every frame. Asking
      *how much darker* rather than *how confidently darker* separates them,
      because a hull is the darkest thing in the water and a shadow on water
      this dark has little room left to darken. It is also what makes fog turn
      detections off rather than move them: haze compresses contrast, depth in
      sigmas falls, and the floor stops being met.
    """
    outer = params.surround_gain * radius + 2
    centre_sum = _box_sum(weighted_integral, radius, pad, shape)
    centre_n = _box_sum(water_integral, radius, pad, shape)
    surround_sum = _box_sum(weighted_integral, outer, pad, shape)
    surround_sum -= centre_sum
    surround_n = _box_sum(water_integral, outer, pad, shape)
    surround_n -= centre_n

    need_centre, need_surround = _admissibility(radius, outer, params)
    enough = (centre_n >= need_centre) & (surround_n >= need_surround)
    safe_centre = np.maximum(centre_n, 1.0)
    safe_surround = np.maximum(surround_n, 1.0)
    centre_sum /= safe_centre
    surround_sum /= safe_surround
    centre_sum -= surround_sum
    depth = np.where(enough, centre_sum, -np.inf)
    np.reciprocal(safe_centre, out=safe_centre)
    np.reciprocal(safe_surround, out=safe_surround)
    safe_centre += safe_surround
    np.sqrt(safe_centre, out=safe_centre)
    centre_sum /= safe_centre
    return np.where(enough, centre_sum, -np.inf), depth


def _box_at(padded: _Float, radius: int, pad: int, row: int, column: int) -> float:
    """One box sum, at one pixel. The scalar form of :func:`_box_sum`."""
    low_row, high_row = row + pad - radius, row + pad + radius + 1
    low_column, high_column = column + pad - radius, column + pad + radius + 1
    return float(
        padded[high_row, high_column]
        - padded[low_row, high_column]
        - padded[high_row, low_column]
        + padded[low_row, low_column]
    )


def _admissibility(radius: int, outer: int, params: DetectorParams) -> tuple[float, float]:
    """How many water pixels a centre box and its surround must hold.

    Both the scorer (:func:`_response`) and the centroid's scale
    (:func:`_deepest_scale`) ask this, so that the box a detection's
    ``px``/``py`` come from is always a box the scorer would have accepted.
    They disagreed once: the scale asked only for a single water pixel, so a
    radius the scorer refused as mostly ice could still win the centroid and
    drag the reported pixel across a floe edge.
    """
    centre_area = float((2 * radius + 1) ** 2)
    surround_area = float((2 * outer + 1) ** 2) - centre_area
    return (
        max(float(params.min_centre_px), (1.0 - params.max_centre_ice) * centre_area),
        max(float(params.min_surround_px), (1.0 - params.max_surround_ice) * surround_area),
    )


def _deepest_scale(
    weighted_integral: _Float,
    water_integral: _Float,
    params: DetectorParams,
    pad: int,
    shape: tuple[int, int],
    row: int,
    column: int,
) -> int | None:
    """Which centre box the winning pixel is deepest in, or ``None``.

    ``None`` when no scale passes :func:`_admissibility` at this pixel. The
    caller drops the detection rather than reporting a centroid taken from a
    box the scorer would have refused.

    Asked once, at one pixel, rather than tracked across the frame: keeping a
    per-pixel winning scale costs a whole-frame write per scale and is read in
    exactly one place, the centroid's window. Four table lookups per scale
    here replace that.
    """
    best_radius: int | None = None
    best_depth = -np.inf
    for radius in params.scales:
        outer = params.surround_gain * radius + 2
        centre_sum = _box_at(weighted_integral, radius, pad, row, column)
        centre_n = _box_at(water_integral, radius, pad, row, column)
        surround_sum = _box_at(weighted_integral, outer, pad, row, column) - centre_sum
        surround_n = _box_at(water_integral, outer, pad, row, column) - centre_n
        need_centre, need_surround = _admissibility(radius, outer, params)
        if centre_n < need_centre or surround_n < need_surround:
            continue
        depth = centre_sum / centre_n - surround_sum / surround_n
        if depth > best_depth:
            best_depth = depth
            best_radius = radius
    return best_radius


def _centroid(
    darkness: _Float, ice: NDArray[np.bool_], row: int, column: int, radius: int
) -> tuple[float, float, float]:
    """Intensity-weighted centroid of the dark pixels in the winning box.

    Returns ``(px, py, area)`` in the pixel convention of
    :mod:`whiteout.vision.camera`: the ``+ 0.5`` that turns an index into the
    centre of that pixel is applied here, once, and nowhere else.
    """
    height, width = darkness.shape
    y0, y1 = max(0, row - radius), min(height, row + radius + 1)
    x0, x1 = max(0, column - radius), min(width, column + radius + 1)
    patch = darkness[y0:y1, x0:x1]
    usable = patch[~ice[y0:y1, x0:x1]]
    floor = 0.5 * float(np.max(usable)) if usable.size else 0.0
    weights = np.where(ice[y0:y1, x0:x1] | (patch < floor), 0.0, patch)
    total = float(np.sum(weights))
    if total <= 0.0:
        return column + 0.5, row + 0.5, 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    px = float(np.sum(weights * (xs + 0.5))) / total
    py = float(np.sum(weights * (ys + 0.5))) / total
    return px, py, float(np.count_nonzero(weights))


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def detect_vessel(
    luma: NDArray[np.generic],
    camera: CameraModel,
    pose: CameraPose,
    *,
    asset_id: str,
    seq: int = 0,
    t: float = 0.0,
    ground_alt_m: float = 0.0,
    params: DetectorParams = DEFAULT_PARAMS,
    sync: PoseSync | None = None,
) -> VesselDetection | None:
    """Return the shadow vessel's pixel in this frame, or ``None``.

    :param luma: the frame as a 2-D array of luminance samples, shape
        ``(camera.height, camera.width)``. Any numeric dtype; ``uint8`` is
        what a camera gives. **Not** JPEG bytes — decoding is
        :mod:`whiteout.vision.imagery`'s job, and keeping this function at
        pixels is what lets it be measured without a decoder.
    :param camera: the asset's published intrinsics.
    :param pose: where the camera was and where it pointed for this frame.
    :param asset_id: stamped onto the detection.
    :param seq: the frame's index in its stream.
    :param t: the frame's timestamp, seconds.
    :param ground_alt_m: altitude of the water plane in ``pose.alt_m``'s
        datum. Issue #76 has not settled which datum the arena reports, so
        this is a parameter with the same meaning and the same default as
        :func:`~whiteout.vision.projection.project_pixel_to_ground`'s.
    :param params: the thresholds; see :class:`DetectorParams`.
    :param sync: how ``pose`` and this frame stood in time, from
        :meth:`whiteout.types.PoseSync.for_frame`. Stamped onto the detection
        and used for nothing else here: it can only ever *qualify* a detection,
        never gate one, and :class:`VesselDetection` argues why. ``None`` — the
        default — says the caller did not establish it, which is what a
        synthetic frame and a fixture recording have to say.
    :returns: a :class:`VesselDetection`, or ``None`` when the frame holds no
        vessel, when it is too degraded to say, or when the best candidate is
        somewhere the geometry says a vessel cannot be.
    :raises VisionError: if the frame is the wrong shape for this camera or
        holds non-finite samples, or if the camera is not above the water
        plane. These are *malformed inputs*, not empty frames; an empty frame
        is ``None``.

    Stateless and deterministic: the same frame, camera, pose and parameters
    give the same answer, with no clock and no generator anywhere in it.
    """
    image = _as_image(luma, camera)
    height_above_plane = pose.alt_m - ground_alt_m
    if height_above_plane <= 0.0:
        raise VisionError(
            f"{asset_id}: camera is not above the water plane — altitude {pose.alt_m!r} m "
            f"against a plane at {ground_alt_m!r} m"
        )
    max_range_m = (
        max_flat_plane_range_m(height_above_plane)
        if params.max_range_m is None
        else params.max_range_m
    )

    residual, sigma, spread = _row_background(image, params.min_sigma)
    darkness = -residual / sigma
    ice = residual > params.ice_spread_gain * spread

    searchable = _water_mask(camera, pose, height_above_plane, max_range_m) & ~ice
    margin = params.surround_gain * max(params.scales) + 2
    edge = np.zeros_like(searchable)
    edge[margin : camera.height - margin, margin : camera.width - margin] = True
    searchable &= edge
    if not bool(np.any(searchable)):
        return None

    shape = (camera.height, camera.width)
    pad = params.surround_gain * max(params.scales) + 2
    weighted_integral = _integral(np.where(ice, 0.0, darkness), pad, shape)
    water_integral = _integral((~ice).astype(np.float64), pad, shape)
    best = np.full(image.shape, -np.inf)
    deepest = np.full(image.shape, -np.inf)
    for radius in params.scales:
        response, depth = _response(weighted_integral, water_integral, radius, params, pad, shape)
        np.maximum(best, response, out=best)
        np.maximum(deepest, depth, out=deepest)
    # Significance is read at the scale that maximises it; **depth is read at
    # the scale that maximises depth**, which for a compact hull is the
    # smallest box that fits inside it. Reading depth at the winning-z scale
    # instead measures the hull diluted across a box several times its own
    # size, and dilution is exactly what a broad shallow shadow does not
    # suffer — so the two would come out alike and the floor would separate
    # nothing.
    searchable &= deepest >= params.min_depth_sigma
    # And the same floor in counts. ``deepest`` is in sigmas, and ``sigma`` is
    # the per-row scale those sigmas were measured in, so the product is the
    # hull's depth in 8-bit counts. The sigma floor above cannot tell a deep
    # hull from a shallow one on a row whose noise was measured too low, and
    # on JPEG frames that is most rows; this floor is the one gate in the
    # chain that a collapsing denominator cannot open. ``-inf`` times a
    # positive scale is still ``-inf``, so pixels that formed no box stay out.
    searchable &= deepest * sigma >= params.min_depth_counts
    best[~searchable] = -np.inf

    flat = int(np.argmax(best))
    row, column = divmod(flat, camera.width)
    peak = float(best[row, column])
    if not math.isfinite(peak) or peak < params.min_peak_sigma:
        return None

    y0, y1 = max(0, row - params.exclusion_px), row + params.exclusion_px + 1
    x0, x1 = max(0, column - params.exclusion_px), column + params.exclusion_px + 1
    blanked = best[y0:y1, x0:x1].copy()
    best[y0:y1, x0:x1] = -np.inf
    runner_up = float(np.max(best))
    best[y0:y1, x0:x1] = blanked
    if not math.isfinite(runner_up):
        runner_up = 0.0

    contrast_term = _clamp01(
        (peak - params.min_peak_sigma) / (params.confident_sigma - params.min_peak_sigma)
    )
    margin_term = _clamp01((peak - runner_up) / peak / params.margin_fraction)
    confidence = contrast_term * margin_term
    if confidence < params.min_confidence:
        return None

    scale = _deepest_scale(weighted_integral, water_integral, params, pad, shape, row, column)
    if scale is None:
        # No scale the scorer would accept survives at this pixel, so there is no
        # box to take a centroid from. Reporting one anyway is how a lock beside
        # a floe becomes a confident wrong lat/lon.
        return None
    px, py, area = _centroid(darkness, ice, row, column, scale)

    detection = VesselDetection(
        asset_id=asset_id,
        seq=seq,
        t=float(t),
        px=px,
        py=py,
        confidence=confidence,
        peak_sigma=peak,
        depth_sigma=float(deepest[row, column]),
        runner_up_sigma=runner_up,
        area_px=area,
        scale_px=scale,
        camera=camera,
        pose=pose,
        ground_alt_m=float(ground_alt_m),
        max_range_m=float(max_range_m),
        sync=sync,
    )
    try:
        detection.to_ground()
    except ProjectionError:
        # The geometry refuses this pixel. Silence beats a lat/lon nobody can
        # stand behind: ARENA.md section 5 scores accuracy.
        return None
    return detection
