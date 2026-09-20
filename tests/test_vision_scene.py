"""The synthetic channel, and whether it is the scene issue #66 describes.

These tests are not about the detector. They are about the thing the
detector's numbers are measured against, and they exist because a measurement
is only worth what its corpus is worth: if the hull came out brighter than the
water, or if the same seed rendered different bytes twice, the false-positive
rate in the pull request would be a number about nothing.

So each one pins a claim the generator's docstring makes — dark hull, bright
ice, fog compresses contrast, the same seed gives the same frame — in the form
that would fail if the claim stopped holding.
"""

from __future__ import annotations

import numpy as np
import pytest

from whiteout.vision.camera import FIXED_WING_CAMERA, QUADCOPTER_CAMERA, TOWER_CAMERA, VisionError
from whiteout.vision.scene import (
    CLEAR,
    HEAVY_FOG,
    SceneParams,
    at_fog,
    render_scene,
)

CAMERAS = (QUADCOPTER_CAMERA, FIXED_WING_CAMERA, TOWER_CAMERA)


def test_a_frame_is_the_cameras_published_size() -> None:
    for camera in CAMERAS:
        scene = render_scene(camera, np.random.default_rng(1), params=CLEAR)
        assert scene.luma.shape == (camera.height, camera.width)
        assert scene.luma.dtype == np.uint8


def test_the_same_seed_renders_the_same_bytes() -> None:
    """``SPEC.md`` §5: determinism is the product, and it starts here."""
    first = render_scene(TOWER_CAMERA, np.random.default_rng(11), params=CLEAR)
    second = render_scene(TOWER_CAMERA, np.random.default_rng(11), params=CLEAR)
    assert first.luma.tobytes() == second.luma.tobytes()
    assert first.vessel_px == second.vessel_px


def test_different_seeds_render_different_frames() -> None:
    first = render_scene(TOWER_CAMERA, np.random.default_rng(11), params=CLEAR)
    second = render_scene(TOWER_CAMERA, np.random.default_rng(12), params=CLEAR)
    assert first.luma.tobytes() != second.luma.tobytes()


def test_the_hull_is_darker_than_the_water_around_it() -> None:
    """The one property the whole detector rests on."""
    params = SceneParams(fog=0.0, noise_sigma=0.0, floes=0, glints=0)
    scene = render_scene(
        TOWER_CAMERA, np.random.default_rng(3), params=params, vessel_px=(320.0, 220.0)
    )
    image = scene.luma.astype(float)
    hull = image[218:223, 318:323].mean()
    water = np.median(image[180:260, 260:380])
    assert hull < water - 8.0, (hull, water)


def test_ice_is_brighter_than_water_and_covers_some_of_the_frame() -> None:
    scene = render_scene(
        TOWER_CAMERA, np.random.default_rng(4), params=SceneParams(fog=0.0, noise_sigma=0.0)
    )
    image = scene.luma.astype(float)
    water = float(np.median(image))
    assert float(np.max(image)) > water + 80.0
    assert 0.02 < float((image > water + 60.0).mean()) < 0.7


def test_an_empty_frame_has_no_vessel_truth() -> None:
    scene = render_scene(TOWER_CAMERA, np.random.default_rng(5), params=CLEAR, with_vessel=False)
    assert scene.vessel_px is None
    assert scene.vessel_length_px == 0.0


def test_fog_compresses_contrast_rather_than_adding_noise() -> None:
    """``ARENA.md`` §4: fog is rasterised into the sensor.

    Rendered twice from the same generator state, the fogged frame has less
    spread between its dark and bright quartiles. If fog were modelled as a
    detection-probability multiplier instead, this would not move at all.
    """
    params = SceneParams(noise_sigma=0.0)

    def spread(fog: float) -> float:
        scene = render_scene(
            TOWER_CAMERA, np.random.default_rng(6), params=at_fog(params, fog), with_vessel=False
        )
        low, high = np.percentile(scene.luma.astype(float), (10.0, 90.0))
        return float(high - low)

    assert spread(0.0) > spread(0.4) > spread(0.85)


def test_heavy_fog_leaves_the_hull_under_the_noise() -> None:
    """The scene the detector is expected to answer ``None`` to."""
    clear = render_scene(
        TOWER_CAMERA,
        np.random.default_rng(7),
        params=SceneParams(fog=0.0),
        vessel_px=(320.0, 250.0),
    )
    fogged = render_scene(
        TOWER_CAMERA, np.random.default_rng(7), params=HEAVY_FOG, vessel_px=(320.0, 250.0)
    )

    def depth(frame: np.ndarray) -> float:
        image = frame.astype(float)
        hull = image[248:253, 318:323].mean()
        return float(np.median(image[220:280, 280:360]) - hull)

    assert depth(fogged.luma) < 0.5 * depth(clear.luma)


def test_a_forced_vessel_pixel_is_where_the_truth_says() -> None:
    scene = render_scene(
        QUADCOPTER_CAMERA, np.random.default_rng(8), params=CLEAR, vessel_px=(101.0, 202.0)
    )
    assert scene.vessel_px == (101.0, 202.0)


def test_a_vessel_outside_the_frame_is_refused() -> None:
    with pytest.raises(VisionError, match="outside"):
        render_scene(TOWER_CAMERA, np.random.default_rng(9), vessel_px=(-4.0, 100.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fog": 1.4},
        {"fog": -0.1},
        {"floes": -1},
        {"glints": -1},
        {"shore_rows": 1.0},
        {"blur_px": -1},
        {"noise_sigma": -1.0},
        {"floe_radius_px": (10.0, 2.0)},
        {"ice_luma": (0.0, 200.0)},
        {"vessel_length_px": (0.0, 10.0)},
    ],
)
def test_impossible_scene_parameters_are_refused(kwargs: dict[str, object]) -> None:
    with pytest.raises(VisionError):
        SceneParams(**kwargs)  # type: ignore[arg-type]
