"""A synthetic Bellot Strait camera frame, for measuring the detector offline.

**This is a stand-in for arena imagery, and it is not evidence about arena
imagery.** No frame from Dominion Dynamics' render exists in this repository
and there is no route to the arena from the machine this was built on, so
every number the detector's scoring harness prints is a number about *these*
frames. Measuring against real ones is issue #63 plus a human with a
recording; :mod:`whiteout.vision.imagery` is shaped so that it is running
``scripts/score_detector.py`` at a directory, not rewriting anything.

What the generator does model, from ``hackathon/ARENA.md`` §2 and §4 and the
scene as issue #66 describes it:

``dark vessel on dark water``
    The hull is *darker* than the water it sits on, by a small margin. It is
    the only thing in the frame with that polarity at that size, which is the
    whole reason a classical detector is enough here.
``bright ice floes``
    Bright ellipses of widely varying size, and — the part that matters for
    false positives — **dark leads between them**. A crack of open water
    between two floes is the one thing in this scene that looks like a hull:
    small, dark, on a light surround.
``rocky shores and ridges``
    Dark, and at the top of a pitched-down frame. Large and connected, which
    is what tells them from a hull, and mostly above the detector's range cut.
``fog is rasterised into the sensor``
    ``ARENA.md`` §4: weather genuinely degrades what an asset sees, it is not
    a separate scalar to multiply a detection probability by. So it is applied
    here the way it reaches a camera — as a haze that mixes into the image and
    grows with range, i.e. toward the horizon, compressing contrast rather
    than adding noise.
``the published intrinsics``
    Frame size comes from a :class:`~whiteout.vision.camera.CameraModel`, so
    the three fleet cameras render at their own resolutions.

Everything stochastic takes an explicit :class:`numpy.random.Generator`
(``SPEC.md`` §5): the same seed renders the same bytes, which is what lets a
measured false-positive rate be re-derived by anyone reading the PR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from whiteout.vision.camera import CameraModel, VisionError

__all__ = [
    "CLEAR",
    "FOGGY",
    "HEAVY_FOG",
    "RenderedScene",
    "SceneParams",
    "at_fog",
    "render_scene",
]

Image = NDArray[np.float64]

#: How many times a vessel placement is redrawn before it is accepted wherever
#: it landed. Small: at the floe densities here, a water draw comes up almost
#: at once, and giving up rather than looping keeps the render's cost bounded.
_PLACEMENT_TRIES = 20


@dataclass(frozen=True, slots=True)
class SceneParams:
    """What is in the water, and what the air is doing to it.

    Luminances are 8-bit, 0..255, and are the values *before* fog, noise and
    the optical blur are applied.

    :param floes: how many ice floes to scatter.
    :param floe_radius_px: smallest and largest floe semi-axis, pixels.
    :param water_luma: the open water's level. Dark, as Arctic water in this
        light is.
    :param water_swell: peak-to-peak of the slow brightness roll across the
        water, from swell and from the sun's angle.
    :param ice_luma: the floes' range. Bright.
    :param lead_luma: the open water seen through a crack in a floe. Deep,
        and bounded by ice on both flanks — which is the only thing that tells
        it from a hull in one frame.
    :param shadow_contrast: how far below open water a floe's shadow falls,
        8-bit counts. **This, against :attr:`vessel_contrast`, is the single
        assumption the measured false-positive rate is most sensitive to**,
        and nothing in ``ARENA.md`` pins either. A shadow as deep as a hull
        would make the scene unsolvable from one frame, and the honest
        response to that is the temporal cue issue #66 also names — the
        vessel is the only thing in the world that moves — not a lower
        threshold here.
    :param glints: how many one- or two-pixel specular highlights to sprinkle
        on the water. Bright, so a dark-polarity detector should ignore them
        entirely; they are here to make sure it does.
    :param shore_rows: how many rows at the top of the frame are rocky shore
        rather than water, as a fraction of the frame height. Zero for a
        nadir or near-nadir view.
    :param shore_luma: the rock's level. Darker than water, and large.
    :param fog: 0 for clear, 1 for the horizon being solid haze.
    :param haze_luma: the level fog mixes toward.
    :param noise_sigma: sensor noise, 8-bit counts, Gaussian.
    :param blur_px: radius of the separable box blur standing in for the
        optics and for JPEG's own smoothing. 0 disables it.
    :param vessel_contrast: how far *below* the local water level the hull
        sits, 8-bit counts. The single most important number here: it is the
        signal, and the honest way to make the scene harder is to lower it.
    :param vessel_length_px: smallest and largest hull length, pixels. The
        range covers a vessel a few hundred metres away down to one at the
        far side of the 2 km channel.
    :param vessel_aspect: hull length over hull beam.
    """

    floes: int = 70
    floe_radius_px: tuple[float, float] = (2.5, 26.0)
    water_luma: float = 42.0
    water_swell: float = 9.0
    ice_luma: tuple[float, float] = (150.0, 244.0)
    lead_luma: float = 24.0
    shadow_contrast: float = 7.0
    glints: int = 120
    shore_rows: float = 0.0
    shore_luma: float = 26.0
    fog: float = 0.15
    haze_luma: float = 170.0
    noise_sigma: float = 2.5
    blur_px: int = 1
    vessel_contrast: float = 17.0
    vessel_length_px: tuple[float, float] = (5.0, 22.0)
    vessel_aspect: float = 2.6

    def __post_init__(self) -> None:
        if self.floes < 0 or self.glints < 0:
            raise VisionError(f"floes and glints must be non-negative, got {self!r}")
        if not 0.0 <= self.fog <= 1.0:
            raise VisionError(f"fog must lie in [0, 1], got {self.fog!r}")
        if not 0.0 <= self.shore_rows < 1.0:
            raise VisionError(f"shore_rows must lie in [0, 1), got {self.shore_rows!r}")
        if self.blur_px < 0:
            raise VisionError(f"blur_px must be non-negative, got {self.blur_px!r}")
        if self.noise_sigma < 0.0:
            raise VisionError(f"noise_sigma must be non-negative, got {self.noise_sigma!r}")
        for name, pair in (
            ("floe_radius_px", self.floe_radius_px),
            ("ice_luma", self.ice_luma),
            ("vessel_length_px", self.vessel_length_px),
        ):
            low, high = pair
            if not 0.0 < low <= high:
                raise VisionError(f"{name} must be an increasing positive pair, got {pair!r}")


#: A good day: thin haze, ordinary sea state.
CLEAR = SceneParams(fog=0.1)

#: Weather the fleet will actually fly in.
FOGGY = SceneParams(fog=0.55, noise_sigma=3.0)

#: Past the point where the hull's contrast survives the haze. The detector is
#: expected to emit **nothing** here, not a guess; see
#: ``tests/test_vision_detect.py``.
HEAVY_FOG = SceneParams(fog=0.92, noise_sigma=3.5)


@dataclass(frozen=True, slots=True)
class RenderedScene:
    """One frame, and the truth about it.

    :param luma: the frame, ``uint8``, shape ``(height, width)``.
    :param vessel_px: the hull's centre as ``(px, py)`` in the pixel
        convention of :mod:`whiteout.vision.camera` — origin at the frame's
        top-left corner, so the centre of index ``(i, j)`` is ``(i + 0.5,
        j + 0.5)``. ``None`` when the frame has no vessel in it.
    :param vessel_length_px: the hull's long axis, pixels; 0 with no vessel.
    """

    luma: NDArray[np.uint8]
    vessel_px: tuple[float, float] | None
    vessel_length_px: float


def _ellipse(
    image: Image,
    centre: tuple[float, float],
    semi_axes: tuple[float, float],
    angle_deg: float,
    value: float,
    *,
    softness: float = 1.0,
) -> None:
    """Paint a soft-edged ellipse into ``image`` in place, over its bounding box.

    Soft-edged rather than hard: a one-pixel step is a texture no camera ever
    produces, and a detector tuned against hard edges would be tuned against
    an artefact of the generator.
    """
    cx, cy = centre
    a, b = semi_axes
    height, width = image.shape
    reach = math.ceil(max(a, b) + softness + 1.0)
    x0, x1 = max(0, int(cx) - reach), min(width, int(cx) + reach + 1)
    y0, y1 = max(0, int(cy) - reach), min(height, int(cy) + reach + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    theta = math.radians(angle_deg)
    dx = (xs + 0.5) - cx
    dy = (ys + 0.5) - cy
    u = dx * math.cos(theta) + dy * math.sin(theta)
    v = -dx * math.sin(theta) + dy * math.cos(theta)
    radius = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    edge = softness / max(a, b)
    alpha = np.clip((1.0 + edge - radius) / (2.0 * edge), 0.0, 1.0)
    patch = image[y0:y1, x0:x1]
    image[y0:y1, x0:x1] = patch * (1.0 - alpha) + value * alpha


def _box_blur(image: Image, radius: int) -> Image:
    """Separable box blur with edge replication, in numpy alone.

    A box blur and not a Gaussian because the point is only to stop pixels
    being independent; SciPy's ``ndimage`` would do it better but ships no
    type stubs, and ``mypy --strict`` runs over this package.
    """
    if radius <= 0:
        return image
    window = 2 * radius + 1
    padded = np.pad(image, radius, mode="edge")
    cumulative = np.cumsum(padded, axis=1)
    rows = cumulative[:, window - 1 :].copy()
    rows[:, 1:] -= cumulative[:, :-window]
    cumulative = np.cumsum(rows, axis=0)
    out = cumulative[window - 1 :, :].copy()
    out[1:, :] -= cumulative[:-window, :]
    return out / float(window * window)


def _upsample(coarse: Image, height: int, width: int) -> Image:
    """Bilinear resize of a small field up to the frame.

    Bilinear and not nearest: nearest leaves the coarse grid's own cell edges
    in the image as hard steps, and a hard-edged dark cell is exactly the
    thing the detector is being asked to find. Measuring against that would be
    measuring against a rendering artefact.
    """
    rows = np.linspace(0.0, coarse.shape[0] - 1.0, height)
    columns = np.linspace(0.0, coarse.shape[1] - 1.0, width)
    r0 = np.floor(rows).astype(np.intp)
    r1 = np.minimum(r0 + 1, coarse.shape[0] - 1)
    c0 = np.floor(columns).astype(np.intp)
    c1 = np.minimum(c0 + 1, coarse.shape[1] - 1)
    wr = (rows - r0)[:, None]
    wc = (columns - c0)[None, :]
    top = coarse[r0][:, c0] * (1.0 - wc) + coarse[r0][:, c1] * wc
    bottom = coarse[r1][:, c0] * (1.0 - wc) + coarse[r1][:, c1] * wc
    blended: Image = top * (1.0 - wr) + bottom * wr
    return blended


def _water(camera: CameraModel, params: SceneParams, rng: np.random.Generator) -> Image:
    """The empty channel: swell-modulated dark water, with a shore band on top."""
    height, width = camera.height, camera.width
    coarse = rng.normal(0.0, 1.0, size=(max(2, height // 24), max(2, width // 24)))
    swell = _box_blur(coarse, 1)
    upsampled = _upsample(swell, height, width)
    spread = float(np.std(upsampled)) or 1.0
    image = params.water_luma + (params.water_swell / 2.0) * (upsampled / spread)

    shore_height = int(round(params.shore_rows * height))
    if shore_height > 0:
        ridge = params.shore_luma + rng.normal(0.0, 4.0, size=(shore_height, width))
        image[:shore_height, :] = ridge
    return image


def _local_water(image: Image, centre: tuple[float, float], reach: int) -> float:
    """The water level near ``centre``, ignoring whatever ice is nearby.

    The lower quartile and not the median: a window that happens to be more
    than half floe has an ice-bright median, and painting a shadow or a hull
    at "ice minus seventeen counts" would put a *bright* object in the water
    and quietly make the frame unsolvable. The quartile still reads water at
    up to three-quarters ice cover, which is as far as this needs to hold.
    """
    height, width = image.shape
    cx, cy = int(centre[0]), int(centre[1])
    patch = image[
        max(0, cy - reach) : min(height, cy + reach + 1),
        max(0, cx - reach) : min(width, cx + reach + 1),
    ]
    if patch.size == 0:
        return 0.0
    return float(np.percentile(patch, 25.0))


def _scatter_floes(
    image: Image,
    open_water: Image,
    camera: CameraModel,
    params: SceneParams,
    rng: np.random.Generator,
    top: int,
) -> None:
    """Bright floes, each with the two dark things a floe actually produces.

    Both are drawn deliberately, because between them they are every dark,
    hull-sized object the channel contains that is not the hull:

    **A shadow on the water beside the floe**, cast the same way for every
    floe in a frame because there is one sun. It lies on open water, so its
    surround is water, and it is therefore the harder of the two — nothing
    about its context gives it away, only its depth. It is shallow
    (:attr:`SceneParams.shadow_contrast`) because a shadow on water this dark
    has little room left to darken.

    **A crack through the floe**, which is deep — it is open water seen
    between ice — but is bounded by ice on both sides. That is what
    :func:`whiteout.vision.detect._response`'s surround-ice test is for, and
    it is drawn after the floe so that it genuinely sits inside it.
    """
    low, high = params.floe_radius_px
    sun_deg = float(rng.uniform(0.0, 360.0))
    sun = (math.cos(math.radians(sun_deg)), math.sin(math.radians(sun_deg)))
    for _ in range(params.floes):
        a = float(rng.uniform(low, high))
        b = a * float(rng.uniform(0.55, 1.0))
        cx = float(rng.uniform(0.0, camera.width))
        cy = float(rng.uniform(top, camera.height))
        angle = float(rng.uniform(0.0, 180.0))
        if rng.random() < 0.75:
            reach = a * float(rng.uniform(0.9, 1.4))
            centre = (cx + reach * sun[0], cy + reach * sun[1])
            _ellipse(
                image,
                centre,
                (max(1.2, a * 0.7), max(1.0, b * 0.45)),
                sun_deg,
                _local_water(open_water, centre, 24) - params.shadow_contrast,
                softness=1.6,
            )
        _ellipse(
            image,
            (cx, cy),
            (a, b),
            angle,
            float(rng.uniform(*params.ice_luma)),
            softness=1.2,
        )
        if a > 6.0 and rng.random() < 0.5:
            _ellipse(
                image,
                (cx, cy),
                (a * 0.75, max(0.7, b * 0.1)),
                float(rng.uniform(0.0, 180.0)),
                params.lead_luma,
                softness=0.8,
            )


def _fog(image: Image, params: SceneParams) -> Image:
    """Mix haze in, more of it toward the top of the frame.

    ``alpha(row) = fog * (0.35 + 0.65 * (1 - row/height))`` — never zero even
    at the bottom of the frame, because air between the lens and the water is
    still air, and strongest at the horizon, where the path is longest.
    """
    if params.fog <= 0.0:
        return image
    height = image.shape[0]
    depth = 1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height
    alpha = (params.fog * (0.35 + 0.65 * depth))[:, None]
    hazed: Image = image * (1.0 - alpha) + params.haze_luma * alpha
    return hazed


def render_scene(
    camera: CameraModel,
    rng: np.random.Generator,
    *,
    params: SceneParams = CLEAR,
    with_vessel: bool = True,
    vessel_px: tuple[float, float] | None = None,
) -> RenderedScene:
    """Render one frame of the channel at ``camera``'s resolution.

    :param camera: the asset's published intrinsics; only the frame size is
        used, but taking the model keeps the three resolutions honest.
    :param rng: the episode's generator. The same generator state renders the
        same bytes.
    :param params: what is in the water and what the air is doing; see
        :class:`SceneParams`.
    :param with_vessel: whether to put a hull in the frame. ``False`` renders
        the negative case, which is what a false-positive rate is measured on.
    :param vessel_px: force the hull's position, in the
        :mod:`~whiteout.vision.camera` pixel convention. Omitted, it is drawn
        uniformly from the water region.
    :raises VisionError: if ``vessel_px`` is outside the frame.

    The order is the order light arrives in: water, then ice, then the hull,
    then glints on the surface, then fog through the whole path, then the
    optics' blur, then the sensor's noise. Fog **before** blur and noise is
    the point of ``ARENA.md`` §4 — it degrades the scene the camera sees, so
    noise is added to an already-flattened image and the signal-to-noise ratio
    genuinely falls.
    """
    top = int(round(params.shore_rows * camera.height))
    open_water = _water(camera, params, rng)
    image = open_water.copy()
    _scatter_floes(image, open_water, camera, params, rng, top)

    centre: tuple[float, float] | None = None
    length = 0.0
    if with_vessel:
        length = float(rng.uniform(*params.vessel_length_px))
        if vessel_px is None:
            # The vessel is in the water, so a draw that lands on a floe is
            # redrawn. Without this the hull is sometimes painted at "floe
            # minus seventeen counts", which is a bright object on ice — not
            # a harder frame, an impossible one, and it would show up in the
            # measurement as a detector failure rather than a generator one.
            margin = length
            centre = (0.0, 0.0)
            for _ in range(_PLACEMENT_TRIES):
                centre = (
                    float(rng.uniform(margin, camera.width - margin)),
                    float(rng.uniform(top + margin, camera.height - margin)),
                )
                if _local_water(image, centre, 8) <= params.water_luma + params.water_swell:
                    break
        else:
            centre = (float(vessel_px[0]), float(vessel_px[1]))
            if not (0.0 <= centre[0] <= camera.width and 0.0 <= centre[1] <= camera.height):
                raise VisionError(
                    f"vessel_px {vessel_px!r} is outside {camera.name}'s "
                    f"{camera.width}x{camera.height} frame"
                )
        local_water = _local_water(open_water, centre, 24)
        _ellipse(
            image,
            centre,
            (length / 2.0, length / (2.0 * params.vessel_aspect)),
            float(rng.uniform(0.0, 180.0)),
            local_water - params.vessel_contrast,
            softness=0.9,
        )

    for _ in range(params.glints):
        _ellipse(
            image,
            (float(rng.uniform(0.0, camera.width)), float(rng.uniform(top, camera.height))),
            (float(rng.uniform(0.6, 1.6)), float(rng.uniform(0.6, 1.2))),
            float(rng.uniform(0.0, 180.0)),
            float(rng.uniform(180.0, 255.0)),
            softness=0.7,
        )

    image = _fog(image, params)
    image = _box_blur(image, params.blur_px)
    if params.noise_sigma > 0.0:
        image = image + rng.normal(0.0, params.noise_sigma, size=image.shape)
    luma = np.clip(np.rint(image), 0.0, 255.0).astype(np.uint8)
    return RenderedScene(luma=luma, vessel_px=centre, vessel_length_px=length)


def at_fog(params: SceneParams, fog: float) -> SceneParams:
    """``params`` with a different amount of fog and nothing else changed."""
    return replace(params, fog=fog)
