"""The vessel detector, against issue #66's four acceptance criteria.

Each criterion has at least one test that would go red if it stopped holding,
and the mapping is spelled out because a reviewer should not have to guess it:

``emit a pixel where the vessel is, and nothing where it is not``
    :func:`test_a_frame_with_the_vessel_yields_a_pixel_on_it`,
    :func:`test_open_water_with_nothing_in_it_yields_nothing`.
``a false-positive rate low enough not to poison the belief``
    :func:`test_the_false_positive_rate_on_empty_frames_stays_low` and
    :func:`test_ice_and_its_shadows_are_almost_never_a_vessel`. **On synthetic
    frames**: see :mod:`whiteout.vision.scene`. The real-frame measurement is
    outstanding and is a human action.
``runs fast enough to keep up``
    :func:`test_four_cameras_are_kept_up_with`, marked ``slow`` because it
    asserts on wall-clock time — ``SPEC.md`` §6 requires that.
``degrades honestly: no detection beats a confident wrong one``
    the whole middle section — fog turns detections *off* rather than moving
    them, two candidate hulls produce silence rather than a coin flip, the sky
    is never searched, and nothing is ever emitted that will not project.

The last of those is the reason for
:func:`test_every_detection_projects_to_a_finite_ground_point`: a detection that
cannot become a lat/lon is a detection the tracks API would score us down for,
so the type cannot be allowed to exist.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import pytest

from whiteout.vision.camera import FIXED_WING_CAMERA, QUADCOPTER_CAMERA, TOWER_CAMERA, VisionError
from whiteout.vision.detect import (
    DEFAULT_PARAMS,
    DetectorParams,
    VesselDetection,
    detect_vessel,
)
from whiteout.vision.projection import CameraPose, ProjectionError
from whiteout.vision.scene import CLEAR, HEAVY_FOG, SceneParams, at_fog, render_scene

#: Poses that put the whole frame on water inside the projection's range
#: bound, so that a missed vessel is the detector's doing and not the
#: geometry's. The sky mask gets its own test below.
TOWER_POSE = CameraPose(71.985, -94.40, 60.0, 75.0, -22.0)
QUAD_POSE = CameraPose(71.990, -94.50, 120.0, 90.0, -55.0)
WING_POSE = CameraPose(71.990, -94.52, 200.0, 90.0, -35.0)

FLEET = (
    ("tower-1", TOWER_CAMERA, TOWER_POSE),
    ("quadcopter", QUADCOPTER_CAMERA, QUAD_POSE),
    ("fixed-wing", FIXED_WING_CAMERA, WING_POSE),
)

#: A hull with contrast the detector should have no trouble with. The
#: borderline case is measured by ``scripts/score_detector.py``, not asserted
#: here: a test that sits on the knee is a test that flakes.
PLAIN = SceneParams(fog=0.05, vessel_contrast=30.0, vessel_length_px=(14.0, 16.0))

#: The same hull on open water. Tests that *force* the hull's pixel use this:
#: ``render_scene`` only keeps a vessel off the ice when it chooses the
#: position itself, so a forced pixel in a floe field can land on a floe, and a
#: hull painted on ice is not a frame the detector is being asked about.
PLAIN_OPEN = replace(PLAIN, floes=0, glints=20)


def detect_on(
    scene_params: SceneParams,
    seed: int,
    *,
    asset: tuple[str, object, CameraPose] = FLEET[0],
    with_vessel: bool = True,
    vessel_px: tuple[float, float] | None = None,
    params: DetectorParams = DEFAULT_PARAMS,
) -> tuple[VesselDetection | None, tuple[float, float] | None]:
    """Render one frame and run the detector over it. Returns the pair."""
    asset_id, camera, pose = asset
    scene = render_scene(
        camera,  # type: ignore[arg-type]
        np.random.default_rng(seed),
        params=scene_params,
        with_vessel=with_vessel,
        vessel_px=vessel_px,
    )
    found = detect_vessel(
        scene.luma,
        camera,  # type: ignore[arg-type]
        pose,
        asset_id=asset_id,
        seq=seed,
        t=seed / 10.0,
        ground_alt_m=0.0,
        params=params,
    )
    return found, scene.vessel_px


# --------------------------------------------------------------------------
# "emit a pixel; given a frame without, emit nothing"
# --------------------------------------------------------------------------


@pytest.mark.parametrize("asset", FLEET, ids=lambda a: str(a[0]))
def test_a_frame_with_the_vessel_yields_a_pixel_on_it(
    asset: tuple[str, object, CameraPose],
) -> None:
    hits = 0
    for seed in range(6):
        found, truth = detect_on(PLAIN, 100 + seed, asset=asset)
        if found is None or truth is None:
            continue
        assert math.hypot(found.px - truth[0], found.py - truth[1]) <= 10.0
        hits += 1
    assert hits >= 4, f"{asset[0]} found the hull in only {hits} of 6 clear frames"


def test_open_water_with_nothing_in_it_yields_nothing() -> None:
    """The floor of the criterion, with no clutter to argue about.

    Water, swell, glint, sensor noise, no hull. Every seed, every camera,
    silence. The cluttered version of this — floes, their shadows and the
    cracks through them — is a *rate* rather than a zero, and is
    :func:`test_the_false_positive_rate_on_empty_frames_stays_low`.
    """
    for asset in FLEET:
        for seed in range(8):
            found, truth = detect_on(PLAIN_OPEN, 300 + seed, asset=asset, with_vessel=False)
            assert truth is None
            assert found is None, f"{asset[0]} seed {seed}: {found!r} in empty water"


def test_a_detection_carries_the_pixel_the_confidence_and_the_pose() -> None:
    """Issue #66: a detection has to be enough to make a lat/lon out of."""
    found, truth = detect_on(PLAIN_OPEN, 101, vessel_px=(300.0, 240.0))
    assert found is not None and truth is not None
    assert found.asset_id == "tower-1"
    assert found.seq == 101
    assert found.t == pytest.approx(10.1)
    assert 0.0 <= found.confidence <= 1.0
    assert found.pose is TOWER_POSE
    assert found.camera is TOWER_CAMERA
    assert found.ground_alt_m == 0.0
    assert 0.0 <= found.px <= TOWER_CAMERA.width
    assert 0.0 <= found.py <= TOWER_CAMERA.height


def test_a_lively_sea_does_not_read_as_ice() -> None:
    """Swell is structure, not floe, and the two are measured differently.

    The ice cut is a threshold on how bright a pixel is against the water's
    own *appearance* — swell, the sun across the channel, everything smooth.
    Measured instead against the sensor's noise, which is several times
    smaller, a sea state like this one puts a quarter of the open water on the
    wrong side of the cut; the surround's ice-fraction test then refuses every
    candidate, and the detector goes blind on a frame with no ice in it at
    all. It is silence rather than a wrong answer, so nothing else in this
    file notices.
    """
    lively = replace(PLAIN_OPEN, water_swell=24.0)
    hits = 0
    for seed in range(6):
        found, truth = detect_on(lively, 130 + seed, vessel_px=(300.0, 240.0))
        assert truth is not None
        if found is None:
            continue
        assert math.hypot(found.px - truth[0], found.py - truth[1]) <= 12.0
        hits += 1
    assert hits >= 5, f"a lively sea cost {6 - hits} of 6 detections"


def test_the_same_frame_twice_gives_the_same_detection() -> None:
    first, _ = detect_on(PLAIN, 102)
    second, _ = detect_on(PLAIN, 102)
    assert first == second


# --------------------------------------------------------------------------
# "a false-positive rate low enough that the belief update is not poisoned"
# --------------------------------------------------------------------------


def test_the_false_positive_rate_on_empty_frames_stays_low() -> None:
    """The acceptance criterion, **on synthetic frames**.

    ``scripts/score_detector.py`` is where the number is measured properly and
    over a corpus worth quoting; this is the tripwire that keeps a change from
    quietly trading the rate away. The bound is loose against the measured
    rate on purpose — it is there to catch a regression of the kind that
    doubles it, not to pin the third decimal.
    """
    empty = 0
    alarms = 0
    for asset in FLEET:
        for seed in range(14):
            found, _ = detect_on(CLEAR, 500 + seed, asset=asset, with_vessel=False)
            empty += 1
            alarms += found is not None
    assert alarms / empty <= 0.05, f"{alarms} false alarms in {empty} empty frames"


def test_ice_and_its_shadows_are_almost_never_a_vessel() -> None:
    """Half again the ordinary floe cover, no hull. Every dark thing is ice's.

    The density here is the top of the range the detector is trustworthy over,
    and it is a measured edge rather than a guessed one. Against
    ``whiteout.vision.scene``, 32 empty tower frames per rung:

    ====================  ==========  ===========
    floes                 ice cover   false alarms
    ====================  ==========  ===========
    70 (the default)      0.22        0 / 32
    100 (**this test**)   0.30        0 / 32
    140                   0.39        5 / 32
    180                   0.46        12 / 32
    220                   0.53        16 / 32
    ====================  ==========  ===========

    Past about a third cover the channel stops being water with ice in it and
    starts being a cracked sheet, and a crack in a sheet differs from a hull
    in one frame only by a water surround it does not have.
    :mod:`whiteout.vision.detect`'s docstring says why no threshold in the
    module fixes that, and what would.
    """
    crowded = SceneParams(floes=100, floe_radius_px=(3.0, 30.0), fog=0.05)
    for seed in range(12):
        found, _ = detect_on(crowded, 700 + seed, with_vessel=False)
        assert found is None, f"seed {seed}: a floe field produced {found!r}"


def test_a_bright_object_of_hull_size_is_never_a_detection() -> None:
    """Polarity, stated as sharply as it can be.

    The same generator, the same size, the same place — the sign of the
    contrast reversed. A detector keying on "stands out from the water"
    rather than "is darker than the water" passes every other test in this
    file and fails this one, and in the arena it would spend the run tracking
    ice.
    """
    bright = replace(PLAIN_OPEN, vessel_contrast=-35.0)
    for seed in range(6):
        found, truth = detect_on(bright, 800 + seed, vessel_px=(300.0, 240.0))
        assert truth is not None
        if found is None:
            continue
        assert math.hypot(found.px - truth[0], found.py - truth[1]) > 20.0, (
            f"seed {seed}: a bright object was detected as the hull"
        )


# --------------------------------------------------------------------------
# "degrades honestly: no detection is better than a confident wrong one"
# --------------------------------------------------------------------------


def test_fog_turns_detection_off_rather_than_moving_it() -> None:
    """As haze eats the contrast, the detector must go quiet, not guess.

    The assertion is in two halves, and the second is the one that matters:
    detections must *stop*, and every detection that still comes out must
    still be on the hull. A detector that answered a fogged frame with its
    best remaining guess would keep the first half and break the second.
    """
    seen: list[bool] = []
    for fog in (0.0, 0.3, 0.6, 0.95):
        params = at_fog(replace(PLAIN_OPEN, vessel_contrast=17.0, noise_sigma=3.0), fog)
        detected = 0
        for seed in range(6):
            found, truth = detect_on(params, 900 + seed, vessel_px=(320.0, 250.0))
            if found is None:
                continue
            assert truth is not None
            assert math.hypot(found.px - truth[0], found.py - truth[1]) <= 12.0, (
                f"fog {fog}: a confident wrong answer at {found.px, found.py}"
            )
            detected += 1
        seen.append(detected > 0)
    assert seen[0], "the clear case should detect at all"
    assert not seen[-1], "heaviest fog should produce silence"


def test_heavy_fog_is_silence_and_not_a_guess() -> None:
    for asset in FLEET:
        for seed in range(5):
            found, _ = detect_on(HEAVY_FOG, 950 + seed, asset=asset)
            assert found is None, f"{asset[0]} answered {found!r} through {HEAVY_FOG.fog} fog"


#: Water with nothing happening on it: no floes, no glints, no swell, no fog
#: and — the point — **no sensor noise**. The per-row scale has nothing left to
#: measure, so it sits on ``min_sigma`` and every statistic quoted in sigmas is
#: measured against a floor rather than against the frame.
NO_NOISE = SceneParams(
    floes=0,
    glints=0,
    water_swell=0.0,
    fog=0.0,
    noise_sigma=0.0,
    blur_px=0,
    vessel_length_px=(14.0, 16.0),
)


def test_a_collapsed_noise_scale_cannot_manufacture_depth() -> None:
    """A shallow hull on noiseless water clears the sigma floor, and is refused anyway.

    Every other gate in this detector is a ratio to the per-row noise, so all
    of them are opened together by that noise being measured too small — and
    it is measured too small on any imagery that has been through a low-pass
    filter, which is to say on JPEG, which is what the arena publishes.
    Rendered here as the limiting case: with no noise at all the scale falls
    to ``min_sigma``, and a hull **four counts** deep reads as 4.4 sigmas and
    sails past ``min_depth_sigma``'s 3.6.

    ``min_depth_counts`` is the one gate a collapsing denominator cannot open,
    because it asks the question in the units the hull is actually dark in.
    That is also the quantity fog compresses, which is what makes the module's
    bargain — degrade to ``None``, never to a confident wrong answer — hold on
    imagery the estimator was not calibrated against.
    """
    scene = replace(NO_NOISE, vessel_contrast=4.0)
    ungated = replace(DEFAULT_PARAMS, min_depth_counts=0.0)
    found, _ = detect_on(scene, 5, vessel_px=(320.0, 260.0), params=ungated)
    assert found is not None, "the premise is that the sigma floor alone lets this through"
    assert found.depth_sigma > DEFAULT_PARAMS.min_depth_sigma, found.depth_sigma

    refused, _ = detect_on(scene, 5, vessel_px=(320.0, 260.0))
    assert refused is None, f"four counts of contrast answered {refused!r}"


def test_the_counts_floor_does_not_cost_a_hull_that_is_really_there() -> None:
    """The floor has to sit under a real hull, or it is just a blindfold.

    Same noiseless frame, contrast walked up: the gate refuses what is below
    it and passes what is above, rather than refusing everything once it is
    switched on.
    """
    outcomes = {
        contrast: detect_on(
            replace(NO_NOISE, vessel_contrast=contrast), 5, vessel_px=(320.0, 260.0)
        )[0]
        is not None
        for contrast in (3.0, 4.0, 6.0, 8.0)
    }
    assert outcomes == {3.0: False, 4.0: False, 6.0: True, 8.0: True}, outcomes


def test_confidence_rises_with_contrast() -> None:
    """Confidence has to mean something, or gating on it is theatre."""
    confidences: list[float] = []
    for contrast in (18.0, 26.0, 40.0):
        scene = replace(PLAIN_OPEN, vessel_contrast=contrast)
        found, _ = detect_on(scene, 111, vessel_px=(300.0, 240.0))
        assert found is not None, f"contrast {contrast} produced nothing"
        confidences.append(found.confidence)
    assert confidences == sorted(confidences), confidences


def test_two_equally_good_hulls_produce_silence() -> None:
    """There is exactly one vessel in the world (``ARENA.md`` §3).

    A frame with two candidates is a frame this detector does not understand,
    and the honest answer to it is nothing at all. Rendering the second hull
    from the same generator state as the first keeps them alike.
    """
    params = replace(PLAIN_OPEN, glints=0)
    one = render_scene(
        TOWER_CAMERA, np.random.default_rng(5), params=params, vessel_px=(200.0, 240.0)
    )
    other = render_scene(
        TOWER_CAMERA, np.random.default_rng(5), params=params, vessel_px=(440.0, 240.0)
    )
    alone = detect_vessel(one.luma, TOWER_CAMERA, TOWER_POSE, asset_id="tower-1")
    assert alone is not None, "the single-hull control did not detect"

    both = np.minimum(one.luma, other.luma)
    ambiguous = detect_vessel(both, TOWER_CAMERA, TOWER_POSE, asset_id="tower-1")
    assert ambiguous is None, f"two hulls produced a confident {ambiguous!r}"


def test_the_sky_is_never_searched() -> None:
    """A pixel above the horizon cannot hold a vessel, so it is not a candidate.

    The tower is pitched up, which puts the whole frame above the horizon.
    Every dark thing in it is therefore unprojectable, and the answer must be
    nothing — not the darkest cloud.
    """
    scene = render_scene(TOWER_CAMERA, np.random.default_rng(12), params=PLAIN)
    # The tower's VFOV is 36.1 degrees, so the boresight has to clear the
    # horizon by more than half of that for the whole frame to be sky.
    upward = CameraPose(71.985, -94.40, 60.0, 75.0, 25.0)
    assert detect_vessel(scene.luma, TOWER_CAMERA, upward, asset_id="tower-1") is None


def test_an_unprojectable_distractor_does_not_blind_the_detector() -> None:
    """The mask has to run *before* the peak is picked, not after it.

    Two guards keep an unprojectable pixel out of a detection: the sky and
    over-range mask, and the projection itself, which every candidate is put
    through before it is returned. Either alone gives the right answer on a
    frame whose only dark thing is in the sky, which is why the mask's own
    value needs this frame: a dark blob above the horizon **and** the hull
    below it. Check the projection only at the end and the sky wins the frame,
    fails to project, and the vessel — plainly visible, ten rows down — is
    never reported at all. Masking first finds it.
    """
    shallow = CameraPose(71.985, -94.40, 60.0, 75.0, -5.0)  # horizon near row 132
    hull = render_scene(
        TOWER_CAMERA, np.random.default_rng(21), params=PLAIN_OPEN, vessel_px=(300.0, 280.0)
    )
    in_the_sky = render_scene(
        TOWER_CAMERA,
        np.random.default_rng(21),
        params=replace(PLAIN_OPEN, vessel_contrast=60.0, vessel_length_px=(26.0, 28.0)),
        vessel_px=(200.0, 50.0),
    )
    both = np.minimum(hull.luma, in_the_sky.luma)

    found = detect_vessel(both, TOWER_CAMERA, shallow, asset_id="tower-1")
    assert found is not None, "the distractor above the horizon blinded the detector"
    assert math.hypot(found.px - 300.0, found.py - 280.0) <= 12.0, (found.px, found.py)


def test_a_range_bound_tighter_than_the_channel_is_respected() -> None:
    """Bellot Strait is 2 km across, and a caller may say so.

    With the bound pulled in to a few metres, nothing in the frame projects
    inside it, so nothing may be emitted — the detector may not quietly return
    a pixel whose lat/lon it has already been told is out of range.
    """
    found, _ = detect_on(PLAIN_OPEN, 113, vessel_px=(300.0, 250.0))
    assert found is not None
    tight = replace(DEFAULT_PARAMS, max_range_m=20.0)
    assert detect_on(PLAIN_OPEN, 113, vessel_px=(300.0, 250.0), params=tight)[0] is None


def test_every_detection_projects_to_a_finite_ground_point() -> None:
    """``to_ground`` cannot raise on a detection this module emitted."""
    checked = 0
    for asset in FLEET:
        for seed in range(6):
            found, _ = detect_on(PLAIN, 1100 + seed, asset=asset)
            if found is None:
                continue
            point = found.to_ground()
            assert math.isfinite(point.lat_deg) and math.isfinite(point.lon_deg)
            assert 70.0 < point.lat_deg < 74.0, point
            checked += 1
    assert checked >= 6, "too few detections to say anything about projection"


def test_a_detection_that_would_not_project_is_not_returned() -> None:
    """Constructed directly, the same pixel *does* raise. The detector's
    contract is that it never hands one out, not that projection is lenient."""
    found, _ = detect_on(PLAIN_OPEN, 114, vessel_px=(300.0, 250.0))
    assert found is not None
    unprojectable = replace(found, max_range_m=1.0)
    with pytest.raises(ProjectionError):
        unprojectable.to_ground()


# --------------------------------------------------------------------------
# malformed input is an error, not an empty frame
# --------------------------------------------------------------------------


def test_a_frame_of_the_wrong_size_is_refused() -> None:
    wrong = np.zeros((TOWER_CAMERA.height, TOWER_CAMERA.width - 1), dtype=np.uint8)
    with pytest.raises(VisionError, match="published frame"):
        detect_vessel(wrong, TOWER_CAMERA, TOWER_POSE, asset_id="tower-1")


def test_a_frame_that_is_not_two_dimensional_is_refused() -> None:
    colour = np.zeros((TOWER_CAMERA.height, TOWER_CAMERA.width, 3), dtype=np.uint8)
    with pytest.raises(VisionError, match="2-D"):
        detect_vessel(colour, TOWER_CAMERA, TOWER_POSE, asset_id="tower-1")


def test_a_frame_with_non_finite_samples_is_refused() -> None:
    broken = np.zeros((TOWER_CAMERA.height, TOWER_CAMERA.width), dtype=np.float64)
    broken[10, 10] = np.nan
    with pytest.raises(VisionError, match="non-finite"):
        detect_vessel(broken, TOWER_CAMERA, TOWER_POSE, asset_id="tower-1")


def test_a_camera_under_the_water_is_refused() -> None:
    sunk = CameraPose(71.985, -94.40, 5.0, 75.0, -22.0)
    frame = np.zeros((TOWER_CAMERA.height, TOWER_CAMERA.width), dtype=np.uint8)
    with pytest.raises(VisionError, match="not above the water plane"):
        detect_vessel(frame, TOWER_CAMERA, sunk, asset_id="tower-1", ground_alt_m=10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scales": ()},
        {"scales": (0, 2)},
        {"surround_gain": 1},
        {"min_sigma": 0.0},
        {"confident_sigma": 1.0},
        {"margin_fraction": 0.0},
        {"margin_fraction": 1.5},
        {"min_peak_sigma": 0.0},
        {"min_confidence": 1.5},
        {"max_centre_ice": -0.1},
        {"max_surround_ice": 2.0},
        {"max_range_m": 0.0},
        {"max_range_m": float("inf")},
    ],
)
def test_impossible_detector_parameters_are_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(VisionError):
        DetectorParams(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# "runs fast enough to keep up with the control loop"
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_four_cameras_are_kept_up_with() -> None:
    """A wall-clock floor, so ``SPEC.md`` §6 keeps it out of the gate.

    The fleet is four cameras (``ARENA.md`` §3): one quadcopter frame, one
    fixed-wing frame and two tower frames make one round. The floor is set
    well under the rate measured by ``scripts/score_detector.py`` — this is a
    guard against an order-of-magnitude regression on a shared runner, not a
    performance target.
    """
    rounds = 3
    frames = [
        render_scene(camera, np.random.default_rng(1200 + index), params=CLEAR).luma
        for index, (_, camera, _) in enumerate(FLEET + (FLEET[0],))
    ]
    work = list(zip(frames, FLEET + (FLEET[0],), strict=True))
    for luma, (asset_id, camera, pose) in work:  # warm the caches
        detect_vessel(luma, camera, pose, asset_id=asset_id)
    started = time.perf_counter()
    for _ in range(rounds):
        for luma, (asset_id, camera, pose) in work:
            detect_vessel(luma, camera, pose, asset_id=asset_id)
    elapsed = time.perf_counter() - started
    rate = rounds * len(work) / elapsed
    assert rate >= 8.0, f"{rate:.1f} frames per second across the fleet"
