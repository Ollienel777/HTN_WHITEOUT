"""Issue #13's acceptance criteria, one test each — and one of them is wrong.

The ticket asks for five properties. Four hold and are asserted below. The
first, *"belief entropy is non-increasing under non-detections alone"*, is
**false in general**, and
:func:`test_entropy_can_rise_when_a_sweep_contradicts_the_peak` is the
counterexample rather than a skip. The criterion as written holds only when
the prior is uniform, which is asserted separately, and the reason it fails
otherwise is that it describes an information-theoretic intuition Bayes does
not obey: looking where you believed the vessel was and seeing nothing makes
you *less* certain, and correctly so.
"""

from __future__ import annotations

import numpy as np
import pytest

from whiteout.belief.field import BeliefError
from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint
from whiteout.belief.grid import ChannelBeliefGrid
from whiteout.belief.negative import (
    CLASS_CAMERAS,
    DEFAULT_SWEEP,
    SweepParams,
    camera_for,
    non_detection_likelihood,
)
from whiteout.geo import ARENA_ORIGIN, LocalPoint, local_to_geodetic
from whiteout.types import VEHICLE_CLASSES
from whiteout.vision.camera import CAMERAS
from whiteout.vision.projection import CameraPose
from whiteout.vision.standoff import standoff_range_m

SEEDS = range(60)


def _pose(*, alt_m: float = 120.0, yaw_deg: float = 90.0, pitch_deg: float = 0.0) -> CameraPose:
    return CameraPose(
        lat_deg=ARENA_ORIGIN.lat_deg,
        lon_deg=ARENA_ORIGIN.lon_deg,
        alt_m=alt_m,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=0.0,
    )


def _at(east_m: float, north_m: float = 0.0) -> tuple[float, float]:
    point = local_to_geodetic(ARENA_ORIGIN, LocalPoint(east_m, north_m))
    return point.lat_deg, point.lon_deg


def _sweep_grid(grid: ChannelBeliefGrid, camera_name: str, pose: CameraPose) -> list[float]:
    """Apply one sweep and return the weight each water cell received."""
    likelihood = non_detection_likelihood(CAMERAS[camera_name], pose)
    weights: list[float] = []

    def recording(lat_deg: float, lon_deg: float) -> float:
        weight = likelihood(lat_deg, lon_deg)
        weights.append(weight)
        return weight

    grid.update_likelihood(recording)
    return weights


# --- A1: the entropy criterion, which does not hold as written -------------


def test_entropy_can_rise_when_a_sweep_contradicts_the_peak() -> None:
    """#13's first acceptance criterion is false, and this is why.

    The criterion asks for entropy to be non-increasing under non-detections.
    That is an intuition about gaining information, and Bayes does not honour
    it: a sweep over the *most likely* water, returning nothing, moves mass
    out of the peak and into everywhere else. The field becomes flatter, its
    entropy rises, and that is the correct posterior — we looked at our best
    guess and it was wrong, so we know less about where the vessel is than we
    thought we did.

    Asserting the rise rather than skipping the criterion, because a silent
    skip is how a wrong acceptance survives to be cited by the next ticket.
    """
    grid = ChannelBeliefGrid(DEFAULT_STRAIT)
    middle = DEFAULT_STRAIT.to_position(ChannelPoint(s_m=0.5 * DEFAULT_STRAIT.length_m, w_m=0.0))
    grid.update_detection(middle[0], middle[1], sigma_m=400.0)
    before = grid.entropy()

    # Stand off to the west and look east, straight down the peak.
    west = DEFAULT_STRAIT.to_position(
        ChannelPoint(s_m=0.5 * DEFAULT_STRAIT.length_m - 900.0, w_m=0.0)
    )
    pose = CameraPose(
        lat_deg=west[0], lon_deg=west[1], alt_m=120.0, yaw_deg=90.0, pitch_deg=0.0, roll_deg=0.0
    )
    grid.update_likelihood(non_detection_likelihood(CAMERAS["fixed-wing"], pose))

    assert grid.entropy() > before, (
        "a sweep over the peak that saw nothing should make the field less certain, "
        "not more; #13's first acceptance criterion describes the opposite"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_entropy_is_non_increasing_from_a_uniform_prior(seed: int) -> None:
    """The special case #13's criterion was reaching for, over 60 seeds.

    From a uniform prior there is no peak for a sweep to contradict, so every
    non-detection can only concentrate the field. This is the claim worth
    keeping, and it is the state the episode actually starts in.
    """
    rng = np.random.default_rng(seed)
    grid = ChannelBeliefGrid(DEFAULT_STRAIT)
    before = grid.entropy()
    station = DEFAULT_STRAIT.to_position(
        ChannelPoint(s_m=float(rng.uniform(0.1, 0.9)) * DEFAULT_STRAIT.length_m, w_m=0.0)
    )
    pose = CameraPose(
        lat_deg=station[0],
        lon_deg=station[1],
        alt_m=120.0,
        yaw_deg=float(rng.uniform(0.0, 360.0)),
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    grid.update_likelihood(non_detection_likelihood(CAMERAS["fixed-wing"], pose))
    assert grid.entropy() <= before + 1e-12


# --- A2: the posterior stays a probability distribution ---------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_the_posterior_stays_normalised_and_in_range(seed: int) -> None:
    rng = np.random.default_rng(seed)
    grid = ChannelBeliefGrid(DEFAULT_STRAIT)
    for _ in range(5):
        station = DEFAULT_STRAIT.to_position(
            ChannelPoint(s_m=float(rng.uniform(0.0, 1.0)) * DEFAULT_STRAIT.length_m, w_m=0.0)
        )
        pose = CameraPose(
            lat_deg=station[0],
            lon_deg=station[1],
            alt_m=float(rng.uniform(40.0, 230.0)),
            yaw_deg=float(rng.uniform(0.0, 360.0)),
            pitch_deg=0.0,
            roll_deg=0.0,
        )
        grid.update_likelihood(non_detection_likelihood(CAMERAS["tower"], pose))
        assert grid.mass() == pytest.approx(1.0)
        probabilities = grid.probabilities()
        assert float(probabilities.min()) >= 0.0
        assert float(probabilities.max()) <= 1.0


# --- A3: falloff with range, and why ---------------------------------------


def test_a_far_sweep_says_less_than_a_near_one_and_falls_off_as_the_cube() -> None:
    """The likelihood's whole shape, checked against its stated derivation.

    Pixels on target go as ``h / r**3`` — one power from the angular width,
    two from the depression angle foreshortening the along-range axis — so
    once the target is well below ``half_pixels`` the detection probability
    is proportional to the pixel count and the *shortfall* from 1.0 must fall
    by a factor of 8 for every doubling of range.
    """
    likelihood = non_detection_likelihood(CAMERAS["fixed-wing"], _pose())
    shortfalls = [1.0 - likelihood(*_at(r)) for r in (800.0, 1600.0, 3200.0, 6400.0)]

    # Monotone everywhere: a further sweep always says less.
    assert all(near > far for near, far in zip(shortfalls, shortfalls[1:], strict=False))

    # The cube law is the *asymptote*, so it is asserted where the asymptote
    # applies. At 800 m the hull is still about 7 px against a half-point of
    # 64, so the saturating term has not yet vanished and the ratio is 7.13
    # rather than 8 — which is the model behaving correctly, not drifting.
    assert shortfalls[0] / shortfalls[1] == pytest.approx(7.13, rel=0.02)
    assert shortfalls[1] / shortfalls[2] == pytest.approx(7.85, rel=0.02)
    assert shortfalls[2] / shortfalls[3] == pytest.approx(7.98, rel=0.02)


def test_the_near_edge_of_the_frame_is_where_111_said_it_was() -> None:
    """Straight down is not looked at, and that is geometry rather than a case.

    #111: the quadcopter's camera is fixed to the airframe and looks at the
    horizon, so a contact directly beneath it is outside the frame. The near
    edge for a level camera is ``h / tan(vfov/2)``; inside it the likelihood
    must be exactly 1.0, because no evidence is not weak evidence.
    """
    camera = CAMERAS["quadcopter"]
    height = 120.0
    likelihood = non_detection_likelihood(camera, _pose(alt_m=height))
    # `standoff_range_m` at fill=1.0 *is* the near edge, `h / tan(vfov/2)`.
    # Calling #111's own function rather than restating its formula: two
    # spellings of one piece of geometry is how they come to disagree, and
    # the trigonometry guard in tests/test_geo.py refuses the second one.
    near_edge = standoff_range_m(camera, height, fill=1.0)
    assert near_edge == pytest.approx(102.0, abs=2.0)

    assert likelihood(*_at(1.0)) == 1.0, "directly below is not looked at"
    assert likelihood(*_at(near_edge * 0.5)) == 1.0
    assert likelihood(*_at(near_edge * 1.5)) < 1.0, "just outside the near edge is looked at"


def test_water_outside_the_frame_receives_exactly_no_evidence() -> None:
    """The invariant the field's trustworthiness rests on.

    A sweep says nothing whatever about water it did not cover. Not *almost*
    nothing — exactly 1.0, so that four hundred ticks of sweeps elsewhere
    cannot accumulate into a claim about this cell.
    """
    likelihood = non_detection_likelihood(CAMERAS["tower"], _pose(yaw_deg=90.0))
    assert likelihood(*_at(-2000.0)) == 1.0, "behind the camera"
    assert likelihood(*_at(1000.0, 4000.0)) == 1.0, "far off the frame's side"


# --- A4: a closed-form case, with the working -------------------------------


def test_one_camera_at_one_range_matches_the_hand_computed_weight() -> None:
    """The formula, arithmetic done by hand, against the implementation.

    Fixed-wing camera: 69.0 deg horizontal over 640 px, 42.6 deg vertical over
    360 px, level, at h = 120 m. Take a cell 1000 m due east, the way it
    points.

        fx = (640 / 2) / tan(69.0 deg / 2) = 320 / 0.687281 = 465.603
        fy = (360 / 2) / tan(42.6 deg / 2) = 180 / 0.389888 = 461.676

        slant r = sqrt(1000^2 + 120^2) = sqrt(1_014_400) = 1007.174 m
        r^3                            = 1.0216697e9

        n = fx * fy * L^2 * h / r^3
          = 465.603 * 461.676 * 12^2 * 120 / 1.0216697e9
          = 214_962.6 * 144 * 120 / 1.0216697e9
          = 3.7145537e9 / 1.0216697e9
          = 3.63565 px

        p = 0.7 * 3.63565 / (3.63565 + 64)
          = 0.7 * 0.0537536
          = 0.0376275
        weight = 1 - p = 0.9623725

    Three pixels on a hull at a kilometre is the right order: ``detect.py``
    says the hull spans 5 to 22 pixels across the fleet's fields of view, and
    a sweep that can barely resolve it should barely move the field.
    """
    camera = CAMERAS["fixed-wing"]
    assert camera.fx == pytest.approx(465.603, abs=0.001)
    assert camera.fy == pytest.approx(461.676, abs=0.001)

    weight = non_detection_likelihood(camera, _pose())(*_at(1000.0))
    assert weight == pytest.approx(0.9623725, abs=1e-6)


# --- A5: one empty frame may never empty the field --------------------------


def test_a_single_sweep_cannot_drive_any_cell_to_zero() -> None:
    """The load-bearing safeguard, stated in ``ARENA.md``'s own terms.

    ``detect.py`` has five separate paths that turn a real vessel into
    ``None``. If a non-detection could reach certainty, one of those on one
    frame would empty the field of the water the vessel is in, the fleet
    would never look there again, and nothing downstream could notice —
    ``SPEC.md`` §4 records that there is no ground truth to notice it with.
    """
    grid = ChannelBeliefGrid(DEFAULT_STRAIT)
    before = grid.probabilities().copy()
    # Point a tower straight at the channel from directly above the middle,
    # which is the most informative single look the geometry allows.
    weights = _sweep_grid(grid, "quadcopter", _pose(alt_m=200.0))
    assert min(weights) >= 1.0 - DEFAULT_SWEEP.p_max
    after = grid.probabilities()
    assert float(after[after > 0.0].min()) > 0.0
    assert np.count_nonzero(after) == np.count_nonzero(before), "a cell was emptied"


def test_repeated_sweeps_are_what_empty_a_cell() -> None:
    """Belief falls geometrically, so twenty empty looks mean what one does not."""
    grid = ChannelBeliefGrid(DEFAULT_STRAIT)
    # On the centreline, so it is water: `_at` measures from ARENA_ORIGIN,
    # which is not on the channel, and a cell that is land starts at zero and
    # would pass every assertion below for the wrong reason.
    station = DEFAULT_STRAIT.to_position(ChannelPoint(s_m=0.5 * DEFAULT_STRAIT.length_m, w_m=0.0))
    target = DEFAULT_STRAIT.to_position(
        ChannelPoint(s_m=0.5 * DEFAULT_STRAIT.length_m + 700.0, w_m=0.0)
    )
    # An *unswept* cell to measure against. The field is renormalised after
    # every update, so a swept cell's absolute probability understates the
    # erosion: what falls geometrically is its odds against water nobody
    # looked at, and that ratio is what the claim is actually about.
    elsewhere = DEFAULT_STRAIT.to_position(
        ChannelPoint(s_m=0.02 * DEFAULT_STRAIT.length_m, w_m=0.0)
    )
    pose = CameraPose(
        lat_deg=station[0],
        lon_deg=station[1],
        alt_m=200.0,
        yaw_deg=90.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    weight = non_detection_likelihood(CAMERAS["quadcopter"], pose)(*target)
    assert weight < 1.0, "the test target is not being looked at"
    assert grid.probability_at(*target) > 0.0, "the test target is not water"
    assert non_detection_likelihood(CAMERAS["quadcopter"], pose)(*elsewhere) == 1.0

    def odds() -> float:
        return grid.probability_at(*target) / grid.probability_at(*elsewhere)

    def sweep(times: int) -> None:
        for _ in range(times):
            grid.update_likelihood(non_detection_likelihood(CAMERAS["quadcopter"], pose))

    start = odds()
    sweep(10)
    ten = odds()
    sweep(10)
    twenty = odds()

    # Geometric: each look multiplies the odds by the same factor, so the
    # second ten sweeps cost exactly what the first ten did. Asserted as a
    # ratio of ratios rather than against ``weight**20``, because
    # ``update_likelihood`` evaluates at the **cell centre** and ``weight``
    # was taken at the cell's nominal position -- close, and not the same
    # number. The invariant does not depend on which of the two it is.
    assert twenty / ten == pytest.approx(ten / start, rel=1e-9)
    assert twenty < 0.6 * start, "twenty empty looks barely moved the odds"
    assert grid.probability_at(*target) > 0.0, "twenty looks emptied a cell"
    assert grid.mass() == pytest.approx(1.0)


def test_a_sweep_that_could_be_certain_is_refused_at_construction() -> None:
    """``p_max`` of exactly 1.0 is the failure this module exists to prevent."""
    with pytest.raises(BeliefError, match="strictly in"):
        SweepParams(p_max=1.0)
    with pytest.raises(BeliefError, match="strictly in"):
        SweepParams(p_max=0.0)


def test_a_camera_below_the_water_plane_is_an_error_and_not_an_empty_sweep() -> None:
    """A datum mistake must not read as a quiet episode. Issue #76."""
    with pytest.raises(BeliefError, match="not above the water plane"):
        non_detection_likelihood(CAMERAS["tower"], _pose(alt_m=5.0), ground_alt_m=10.0)


# --- the class-to-camera mapping -------------------------------------------


def test_every_vehicle_class_that_flies_has_a_published_camera() -> None:
    for cls, name in CLASS_CAMERAS.items():
        assert name in CAMERAS, f"{cls} maps to {name!r}, which is not published"
        assert camera_for(cls) == name


def test_an_unknown_class_names_the_ones_that_are_known() -> None:
    with pytest.raises(BeliefError, match="fixedwing.*quad.*tower"):
        camera_for("submarine")


def test_the_arena_has_no_rover_so_no_camera_is_published_for_one() -> None:
    """``ARENA.md`` §3. ``rover`` is a valid seam class with no arena hardware."""
    assert "rover" in VEHICLE_CLASSES
    assert "rover" not in CLASS_CAMERAS
