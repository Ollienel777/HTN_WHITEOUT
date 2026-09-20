"""Looking somewhere and seeing nothing is evidence. This is what it is worth.

Issue #13. A camera swept an area and reported no vessel. That is not the
absence of data — it is a measurement, and the belief field owes it an update:
belief must fall where the camera could have seen, **in proportion to how well
it could have seen**, and never to zero.

Why this is the cheapest accuracy left
--------------------------------------

Without it, the field is a fixed point of its own diffusion. Nothing erodes
it, so it stays uniform for the whole episode however long the episode is —
which is exactly what the committed demo showed before this module existed:
entropy constant to six decimal places across 1200 ticks. The fleet flew
sweeps and the sweeps meant nothing. *Coverage* and *search efficiency* are
two of ``ARENA.md`` §5's seven criteria, and until a non-detection changes
something they are a heatmap of where we flew rather than a statement about
where the vessel is not.

The geometry, which is exact rather than assumed
------------------------------------------------

The original ticket specified this against assumed sensor footprints with an
assumed detection probability. The real assets are better than that. Each
carries a camera with a **published** field of view — quadcopter 114.6°/99.4°,
fixed-wing 69.0°/42.6°, towers 60.0°/36.1° — and ``ATTITUDE`` gives yaw, pitch
and roll for all four. So "swept here, saw nothing" is not an estimate: it is
a cone, from a known point, at a known attitude, intersected with the water.

A cell is *looked at* when the ray from the camera to it lands inside the
frame. :func:`~whiteout.vision.projection.camera_basis` gives the camera's
ENU axes, the cell's offset is projected onto them, and the result is a pixel.
Off the frame, or behind the camera, and the likelihood is exactly ``1.0`` —
**no evidence, not weak evidence**. That distinction is the whole reason this
module can be trusted with the field: a sweep says nothing whatever about
water it did not cover, and an update that quietly nudged those cells would
accumulate into a confident wrong answer over four hundred ticks.

Two things follow from #111 and are worth stating, because they surprise
people. Hovering **over** a contact does not see it: the quadcopter's camera
is fixed to the airframe and looks at the horizon, so directly below is
outside the frame. And a tower on a cliff covers a long way but a narrow
angle. Both fall out of the frame test above without a special case.

How much a non-detection is worth, and why it falls off
--------------------------------------------------------

This is the part that must not be tuned to taste, so it is derived from the
detector we actually built.

:mod:`whiteout.vision.detect` finds the hull by a centre-against-surround
statistic whose significance grows as the square root of the number of pixels
carrying the hull's contrast. So the question "how well could this camera have
seen a vessel there" reduces to "how many pixels would it have been".

For a target of length :math:`L` at slant range :math:`r` from a camera at
height :math:`h` above the water, with focal lengths :math:`f_x, f_y` in
pixels:

.. math::

    n(r) \\;=\\; \\underbrace{\\frac{f_x L}{r}}_{\\text{across the line of sight}}
                 \\;\\times\\;
                 \\underbrace{\\frac{f_y L}{r}\\,\\sin\\alpha}_{\\text{along it}}
          \\;=\\; \\frac{f_x f_y L^2 h}{r^3}

The second factor carries :math:`\\sin\\alpha = h/r`, the depression angle:
water seen obliquely is foreshortened, and a vessel at a shallow angle spends
far fewer rows than its length suggests. **The pixel count therefore falls as
the cube of range, not the square**, and that single fact is the falloff. It
is geometry, not a knob.

Detection probability then saturates in the pixel count,

.. math::

    p(r) \\;=\\; p_\\text{max}\\,\\frac{n(r)}{n(r) + n_\\text{half}}

and the likelihood of having seen nothing is :math:`1 - p(r)`. The form is
chosen for three properties rather than for fit: it is monotone in :math:`n`,
it goes to zero as the target becomes sub-pixel, and it approaches
:math:`p_\\text{max}` and no further however close the camera gets.

Why it can never reach certainty
---------------------------------

:attr:`SweepParams.p_max` is strictly below one, and that is load-bearing
rather than cautious. A detector with a false-negative rate — and
``whiteout/vision/detect.py`` has five separate paths that turn a real vessel
into ``None``, by design — that was allowed to drive belief to zero would
empty the field of the water the vessel is actually in, on one empty frame.
The fleet would then never look there again, and nothing downstream could
detect the mistake, because there is no ground truth. ``SPEC.md`` §4 names
this as the error class the whole build cannot catch.

So the worst a single sweep can do to a cell is multiply it by
``1 - p_max``. Belief there falls; it does not vanish. Seeing nothing twenty
times running is what should empty a cell, and twenty multiplications is how.

What this does not model
-------------------------

**Terrain occlusion.** There is no terrain model of ours, so a ridge between
the camera and the water is invisible to this and the cell reads as swept.
That is the one direction this module errs *against* safety, and it is
recorded here rather than hidden: the mitigation is that ``p_max`` is well
below one, so an occluded cell recovers as soon as anything else looks at it.

**Fog, sea state and ice cover.** All three change the detector's real
performance and none of them is observable from a pose.
``whiteout/vision/scene.py`` says light fog compresses contrast by about a
third; that lands in ``p_max``'s margin rather than in a model.

**The gimbal.** The quadcopter's mount is not commandable and publishes no
status, so the airframe's attitude is the camera's. #99.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from whiteout.belief.field import BeliefError, Likelihood
from whiteout.geo import GeoPoint, geodetic_to_local
from whiteout.vision.camera import CameraModel
from whiteout.vision.projection import CameraPose, camera_basis, max_flat_plane_range_m

__all__ = [
    "CLASS_CAMERAS",
    "DEFAULT_SWEEP",
    "SweepParams",
    "camera_for",
    "non_detection_likelihood",
]

#: Which published camera each vehicle class carries, ``ARENA.md`` §4. The
#: mapping lives here rather than on the transport because it is a fact about
#: the arena's hardware, and the seam forbids an adapter naming a belief
#: concept — but nothing stops belief naming hardware.
CLASS_CAMERAS: dict[str, str] = {
    "quad": "quadcopter",
    "fixedwing": "fixed-wing",
    "tower": "tower",
}


@dataclass(frozen=True, slots=True)
class SweepParams:
    """Everything the non-detection update can be argued about, in one place.

    :param vessel_length_m: the hull's long axis. ``ARENA.md`` §2 says a boat
        and gives no figure, so this is the one genuinely assumed number here.
        It enters :math:`n` squared, so halving it quarters the pixel count
        and pushes the half-detection range in by about 26% — the update gets
        *weaker*, which is the safe direction for a guess to be wrong in.
    :param half_pixels: the pixel count at which a sweep is half as
        informative as it can ever be. ``detect.py``'s hull spans roughly 5 to
        22 pixels across, so an area of ``8 * 8`` sits at the low end of what
        that detector was measured on: below it the statistic is running out
        of samples.
    :param p_max: the most a single perfect look may claim, **over one
        reference interval**. Strictly below one, and the module docstring
        explains why that is the load-bearing parameter rather than the
        careful one.
    :param reference_s: how long a camera must hold water in frame for that
        to count as **one independent look**. Evidence accrues per second and
        not per tick: see the module docstring. One second is a modelling
        choice, but the invariant it buys — that the same episode gives the
        same field whatever the loop frequency — is not.
    :param max_looks: the most independent looks one tick may be worth,
        however long the gap before it. A stalled loop that resumes after a
        minute is holding a pose from *after* the gap, so crediting sixty
        seconds of staring at the current footprint would claim evidence
        about water the asset had already flown past. Ten seconds is roughly
        how long the fixed-wing takes to cross its own footprint at 26 m/s.
    :param max_range_m: **ground** range beyond which a cell is treated as
        unlooked-at whatever the frame says, because a flat water plane stops
        meaning anything at range. ``None`` defers to
        :func:`~whiteout.vision.projection.max_flat_plane_range_m` for the
        camera's height, which is where that plane's own error reaches 10%.
        Inert on the strait as it stands — the farthest water cell is 5.2 km
        against a 7.1 km bound from the lowest asset — and implemented
        anyway, because a documented bound that does not exist is worse than
        no bound: it is a safety property a reader will rely on.
    """

    vessel_length_m: float = 12.0
    half_pixels: float = 64.0
    p_max: float = 0.7
    reference_s: float = 1.0
    max_looks: float = 10.0
    max_range_m: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.vessel_length_m) and self.vessel_length_m > 0.0):
            raise BeliefError(f"vessel_length_m must be positive, got {self.vessel_length_m!r}")
        if not (math.isfinite(self.half_pixels) and self.half_pixels > 0.0):
            raise BeliefError(f"half_pixels must be positive, got {self.half_pixels!r}")
        if not (math.isfinite(self.reference_s) and self.reference_s > 0.0):
            raise BeliefError(f"reference_s must be positive, got {self.reference_s!r}")
        if not (math.isfinite(self.max_looks) and self.max_looks > 0.0):
            raise BeliefError(f"max_looks must be positive, got {self.max_looks!r}")
        if not 0.0 < self.p_max < 1.0:
            # Not `<= 1.0`. A p_max of exactly one is the failure this whole
            # module is arranged to prevent, and accepting it "because the
            # caller asked" would make every other safeguard here decorative.
            raise BeliefError(
                f"p_max must lie strictly in (0, 1), got {self.p_max!r}; a sweep that can "
                f"be certain empties the field of water the vessel may be in"
            )
        if self.max_range_m is not None and not (
            math.isfinite(self.max_range_m) and self.max_range_m > 0.0
        ):
            raise BeliefError(f"max_range_m must be positive, got {self.max_range_m!r}")


DEFAULT_SWEEP = SweepParams()


def camera_for(cls: str) -> str:
    """The camera a vehicle class carries, or raise naming what is known."""
    try:
        return CLASS_CAMERAS[cls]
    except KeyError:
        raise BeliefError(
            f"no camera is published for vehicle class {cls!r}; "
            f"known classes are {sorted(CLASS_CAMERAS)}"
        ) from None


def non_detection_likelihood(
    camera: CameraModel,
    pose: CameraPose,
    *,
    ground_alt_m: float = 0.0,
    elapsed_s: float | None = None,
    params: SweepParams = DEFAULT_SWEEP,
) -> Likelihood:
    """``P(saw nothing | vessel here)``, as a function of position.

    The returned callable is what
    :meth:`~whiteout.belief.grid.ChannelBeliefGrid.update_likelihood` expects:
    it takes a cell's centre and returns a weight in
    ``[1 - params.p_max, 1.0]``.

    :param camera: the asset's published intrinsics.
    :param pose: where it was and where it pointed **for the frame that saw
        nothing**. A pose fetched later is a different pose, and at 26 m/s the
        fixed-wing moves a frame's width in a few seconds.
    :param ground_alt_m: the water plane in ``pose.alt_m``'s datum. Issue #76.
    :param elapsed_s: how long this sweep held the frame. ``None`` means one
        :attr:`SweepParams.reference_s`, which is one independent look and is
        what a caller with no clock should pass. **Evidence is per second,
        not per call**: a loop running at 4 Hz makes four times as many calls
        about the same water as one at 1 Hz, and without this the field it
        settles to would be a property of the loop frequency rather than of
        the world. Measured, before this existed: 100 s of the same episode
        gave a peak probability of 0.0033, 0.0044 or 0.0058 depending only on
        the tick rate.
    :raises BeliefError: if the camera is not above the water plane, or
        ``elapsed_s`` is negative. The first is a malformed input rather than
        an empty sweep: a camera at or below the surface has no footprint to
        speak of, and silently returning "no evidence everywhere" would hide
        a datum mistake as a quiet episode.
    """
    height = pose.alt_m - ground_alt_m
    if height <= 0.0:
        raise BeliefError(
            f"{camera.name}: camera is not above the water plane — altitude "
            f"{pose.alt_m!r} m against a plane at {ground_alt_m!r} m"
        )
    if elapsed_s is not None and not (math.isfinite(elapsed_s) and elapsed_s >= 0.0):
        raise BeliefError(f"elapsed_s must be finite and non-negative, got {elapsed_s!r}")
    exposure = params.reference_s if elapsed_s is None else elapsed_s
    looks = min(exposure / params.reference_s, params.max_looks)
    right, down, forward = camera_basis(pose)
    origin = GeoPoint(pose.lat_deg, pose.lon_deg)
    # The bound is on **ground** range, because that is what
    # `max_flat_plane_range_m` returns — it is the range at which the flat
    # water plane's own error reaches 10%, measured along the water and not
    # along the ray. Comparing it to a slant range would reject a band of
    # cells the bound does not name, by an amount that grows with altitude.
    limit = max_flat_plane_range_m(height) if params.max_range_m is None else params.max_range_m
    floor = 1.0 - params.p_max
    # n = fx * fy * L^2 * h / r^3, gathered once: only r varies per cell.
    gain = camera.fx * camera.fy * params.vessel_length_m**2 * height

    def likelihood(lat_deg: float, lon_deg: float) -> float:
        offset = geodetic_to_local(origin, GeoPoint(lat_deg, lon_deg))
        east, north, up = offset.east_m, offset.north_m, -height
        depth = east * forward[0] + north * forward[1] + up * forward[2]
        if depth <= 0.0:
            return 1.0  # behind the camera: no evidence, not weak evidence
        across = east * right[0] + north * right[1] + up * right[2]
        below = east * down[0] + north * down[1] + up * down[2]
        px = camera.cx + camera.fx * across / depth
        py = camera.cy + camera.fy * below / depth
        if not (0.0 <= px <= camera.width and 0.0 <= py <= camera.height):
            return 1.0  # outside the frame, which includes straight down
        if math.sqrt(east * east + north * north) > limit:
            return 1.0
        slant = math.sqrt(east * east + north * north + height * height)
        pixels = gain / slant**3
        detected = params.p_max * pixels / (pixels + params.half_pixels)
        # Clamped rather than trusted: `gain` is a product of four caller-
        # supplied numbers and a subnormal `slant` would otherwise return a
        # weight below the floor a single look promises.
        single: float = max(floor, 1.0 - detected)
        # `looks` independent looks at the same water, which for `looks == 1`
        # is exactly `single` and costs one pow to say so.
        return float(single**looks)

    return likelihood
