"""Issue #12's acceptance criteria, one group of tests each.

The criterion most likely to pass vacuously is "mass stays on water": a water
mask that covers the whole grid satisfies it and proves nothing. So every
shoreline test here is paired with a **control geometry** that is identical
except that the shore is not there, and the control asserts that the same
diffusion, from the same seeded cell, really would have crossed. A test that
only checks ``p[land] == 0`` cannot tell a working boundary from a boundary
that is never reached.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import whiteout.belief.grid as grid_module
from whiteout.belief import (
    DEFAULT_STRAIT,
    BeliefError,
    BeliefField,
    ChannelBeliefGrid,
    ChannelPoint,
    ChannelVertex,
    GeometryError,
    StraitGeometry,
)
from whiteout.belief.geometry import enu_from_geodetic, geodetic_from_enu
from whiteout.belief.grid import (
    DEFAULT_FALSE_ALARM_RATE,
    DEFAULT_HEADING_PERSISTENCE_S,
    DEFAULT_SPEED_MPS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Mass is conserved by construction, not by renormalising, so the tolerance
#: is floating-point noise over a few thousand cells and nothing else.
MASS_TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# Geometries used by the tests. Each is deliberate: `straight` isolates the
# diffusion from the shoreline, `choked` puts a shoreline exactly where
# diffusion wants to go, and `open` is `choked` with the shore removed.
# --------------------------------------------------------------------------


def straight_channel(half_width_m: float = 5000.0) -> StraitGeometry:
    """A wide, straight, East-West channel: diffusion with no wall in reach."""
    return StraitGeometry(
        vertices=(
            ChannelVertex(lat_deg=71.99, lon_deg=-95.20, half_width_m=half_width_m),
            ChannelVertex(lat_deg=71.99, lon_deg=-94.48, half_width_m=half_width_m),
        )
    )


def choked_channel(narrow_half_width_m: float = 200.0) -> StraitGeometry:
    """A 1 km-wide channel that narrows abruptly to ``narrow_half_width_m``.

    The step is three metres long, so at 100 m resolution the shoreline lands
    between two adjacent rows of cells: the row before the step has water out
    to 1000 m, the row after it does not.
    """
    return StraitGeometry(
        vertices=(
            ChannelVertex(lat_deg=71.99, lon_deg=-95.0000, half_width_m=1000.0),
            ChannelVertex(lat_deg=71.99, lon_deg=-94.9700, half_width_m=1000.0),
            ChannelVertex(lat_deg=71.99, lon_deg=-94.9699, half_width_m=narrow_half_width_m),
            ChannelVertex(lat_deg=71.99, lon_deg=-94.9400, half_width_m=narrow_half_width_m),
        )
    )


def seed(grid: ChannelBeliefGrid, row: int, column: int) -> None:
    """Put the whole of the belief in one cell. Test scaffolding, not API."""
    field = np.zeros(grid.shape, dtype=np.float64)
    field[row, column] = 1.0
    # Reaching past the interface on purpose: seeding a single cell is not
    # something the policy may do, and it is exactly what a shoreline test needs.
    grid._p = field


def shoreline_face(grid: ChannelBeliefGrid) -> tuple[int, int]:
    """A water cell whose along-channel neighbour is land.

    Raises if there is none, which is the point: a test that seeds mass here
    is a test that diffusion has somewhere to leak to.
    """
    mask = grid.water_mask()
    for row in range(grid.shape[0] - 1):
        for column in range(grid.shape[1]):
            if mask[row, column] and not mask[row + 1, column]:
                return row, column
    raise AssertionError("this geometry has no along-channel shoreline to test against")


def cell_position(grid: ChannelBeliefGrid, row: int, column: int) -> tuple[float, float]:
    """The lat/lon of one cell's centre, through public accessors only."""
    point = ChannelPoint(
        s_m=float(grid.along_centres_m()[row]),
        w_m=float(grid.across_centres_m()[column]),
    )
    return grid.geometry.to_geodetic(point)


def beyond_the_mouth(geometry: StraitGeometry, metres: float) -> tuple[float, float]:
    """A position ``metres`` off one end of the centreline, on its axis.

    Negative ``metres`` walks west off vertex 0, positive walks east off the
    last vertex. This is the construction ``to_channel`` cannot distinguish
    from the mouth itself: it clamps ``s`` to the segment and measures ``w``
    against the segment's infinite line, so every one of these comes back as
    ``w ≈ 0`` at ``s = 0`` or ``s = length_m``, however far out it is.
    """
    vertices = geometry.vertices
    if metres < 0.0:
        first, second = vertices[1], vertices[0]
    else:
        first, second = vertices[-2], vertices[-1]
    ax, ay = enu_from_geodetic(first.lat_deg, first.lon_deg)
    bx, by = enu_from_geodetic(second.lat_deg, second.lon_deg)
    span = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / span, (by - ay) / span
    distance = abs(metres)
    return geodetic_from_enu(bx + distance * ux, by + distance * uy)


def along_marginal(grid: ChannelBeliefGrid) -> np.ndarray:
    return np.asarray(grid.probabilities().sum(axis=1))


def along_std_m(grid: ChannelBeliefGrid) -> float:
    """Standard deviation of the belief along the channel, metres."""
    marginal = along_marginal(grid)
    centres = grid.along_centres_m()
    mean = float((marginal * centres).sum())
    return math.sqrt(float((marginal * (centres - mean) ** 2).sum()))


# --------------------------------------------------------------------------
# The geometry: the ribbon the field is defined over.
# --------------------------------------------------------------------------


def test_default_strait_matches_the_arena_briefing() -> None:
    """25 km long, about 2 km wide, at roughly 71.99 N (``ARENA.md`` §2)."""
    assert 24_000.0 < DEFAULT_STRAIT.length_m < 26_000.0
    assert DEFAULT_STRAIT.max_half_width_m == pytest.approx(1000.0)
    middle = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=DEFAULT_STRAIT.length_m / 2.0, w_m=0.0))
    assert middle[0] == pytest.approx(71.99, abs=0.02)
    assert middle[1] == pytest.approx(-94.84, abs=0.05)


def test_the_tracks_api_example_position_is_on_water() -> None:
    """``ARENA.md`` §5 posts ``71.9965, -94.8448``; the ribbon must contain it."""
    assert DEFAULT_STRAIT.is_water(71.9965, -94.8448)
    offset = DEFAULT_STRAIT.to_channel(71.9965, -94.8448)
    assert abs(offset.w_m) < 200.0


def test_geodetic_round_trip_at_the_straits_latitude() -> None:
    """Channel coordinates and lat/lon are inverse at 71.99 N, where lon converges."""
    geometry = DEFAULT_STRAIT
    for fraction in (0.05, 0.25, 0.5, 0.75, 0.95):
        for across in (-400.0, 0.0, 400.0):
            point = ChannelPoint(s_m=fraction * geometry.length_m, w_m=across)
            lat_deg, lon_deg = geometry.to_geodetic(point)
            back = geometry.to_channel(lat_deg, lon_deg)
            assert back.s_m == pytest.approx(point.s_m, abs=1.0)
            assert back.w_m == pytest.approx(point.w_m, abs=1.0)


def test_the_shoreline_is_an_inequality_on_the_across_coordinate() -> None:
    geometry = choked_channel()
    narrow_s = geometry.length_m - 500.0
    lat_in, lon_in = geometry.to_geodetic(ChannelPoint(s_m=narrow_s, w_m=150.0))
    lat_out, lon_out = geometry.to_geodetic(ChannelPoint(s_m=narrow_s, w_m=600.0))
    assert geometry.is_water(lat_in, lon_in)
    assert not geometry.is_water(lat_out, lon_out)


def test_half_width_is_interpolated_between_vertices() -> None:
    """The shore is a taper, not a staircase: the mouth is 1000 m, the next vertex 900 m."""
    geometry = DEFAULT_STRAIT
    first = geometry.vertices[0].half_width_m
    second = geometry.vertices[1].half_width_m
    assert first == 1000.0
    assert second == 900.0
    segment_end = geometry.to_channel(
        geometry.vertices[1].lat_deg, geometry.vertices[1].lon_deg
    ).s_m
    assert geometry.half_width_at(0.5 * segment_end) == pytest.approx(
        0.5 * (first + second), abs=1.0
    )
    assert geometry.half_width_at(0.25 * segment_end) == pytest.approx(975.0, abs=1.0)


def test_half_width_is_clamped_outside_the_channel() -> None:
    geometry = choked_channel()
    assert geometry.half_width_at(-1.0) == pytest.approx(1000.0)
    assert geometry.half_width_at(geometry.length_m + 1000.0) == pytest.approx(200.0)


def test_a_degenerate_polyline_is_refused() -> None:
    with pytest.raises(GeometryError):
        StraitGeometry(vertices=(ChannelVertex(71.99, -94.84, 500.0),))
    with pytest.raises(GeometryError):
        StraitGeometry(
            vertices=(
                ChannelVertex(71.99, -94.84, 500.0),
                ChannelVertex(71.99, -94.84, 500.0),
            )
        )
    with pytest.raises(GeometryError):
        ChannelVertex(71.99, -94.84, 0.0)


# --------------------------------------------------------------------------
# Criterion: the field sums to 1.0 within tolerance after every update.
# --------------------------------------------------------------------------


def test_the_field_starts_uniform_over_the_water_and_sums_to_one() -> None:
    grid = ChannelBeliefGrid()
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)
    values = grid.probabilities()[grid.water_mask()]
    assert values.min() == pytest.approx(values.max())
    # A random spawn over the water is the correct prior (ARENA.md §1 Q2), so
    # a uniform field has maximal entropy: log of the number of water cells.
    assert grid.entropy() == pytest.approx(math.log(grid.water_cells))


def test_mass_is_one_after_every_kind_of_update() -> None:
    grid = ChannelBeliefGrid()
    lat_deg, lon_deg = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=8_000.0, w_m=120.0))
    steps = (
        lambda: grid.diffuse(1.0),
        lambda: grid.update_detection(lat_deg, lon_deg, sigma_m=150.0),
        lambda: grid.diffuse(30.0),
        lambda: grid.update_likelihood(lambda lat, lon: 1.0 + abs(lat - 71.99)),
        lambda: grid.diffuse(600.0),
        lambda: grid.update_detection(lat_deg, lon_deg, sigma_m=40.0),
    )
    for step in steps:
        step()
        assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)
        assert grid.probabilities().min() >= 0.0


def test_diffusion_conserves_mass_without_renormalising() -> None:
    """The operator itself must not leak: 5000 ticks, no correction applied.

    **The seed has to be off-centre**, and that is not a detail. On an
    all-water channel the along-axis fluxes in a column telescope to
    ``p_first - p_last``, so a seed at exactly ``shape[0] // 2`` of a
    symmetric grid makes the net flux identically zero — and a leak
    proportional to it stays zero too. Seeded in the middle, this test passed
    under a 1 % flux leak (mass 1.0000000000000004 after 5000 leaky ticks):
    the module's headline test was blind to the exact failure it exists to
    catch. The choked channel is here for the same reason from the other
    direction — an asymmetric shoreline gives the telescoping nothing to
    cancel against.
    """
    for geometry, row_divisor, column_divisor in (
        (straight_channel(), 3, 3),
        (choked_channel(), 4, 3),
    ):
        grid = ChannelBeliefGrid(geometry)
        row, column = grid.shape[0] // row_divisor, grid.shape[1] // column_divisor
        assert grid.water_mask()[row, column]
        assert row != grid.shape[0] // 2
        seed(grid, row, column)
        for _ in range(5_000):
            grid.diffuse(1.0)
        assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)


# --------------------------------------------------------------------------
# Criterion: mass stays on water. Each test is paired with a control in which
# the shore is removed, so it cannot pass by never reaching the shore.
# --------------------------------------------------------------------------


def test_the_default_mask_has_land_in_it() -> None:
    """Guard against the vacuous version of every test below."""
    grid = ChannelBeliefGrid()
    assert grid.water_cells < grid.shape[0] * grid.shape[1]
    assert grid.water_cells > 0


def test_diffusion_does_not_cross_an_along_channel_shoreline() -> None:
    grid = ChannelBeliefGrid(choked_channel())
    row, column = shoreline_face(grid)
    seed(grid, row, column)
    grid.diffuse(60.0)

    land = ~grid.water_mask()
    assert land.any()
    assert float(grid.probabilities()[land].max()) == 0.0
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)

    # The control: the same grid with the narrows widened away, so the cell
    # that was land is water. If diffusion would not have crossed anyway, the
    # assertion above proves nothing -- so assert that it would have.
    control = ChannelBeliefGrid(choked_channel(narrow_half_width_m=1000.0))
    assert control.shape == grid.shape
    seed(control, row, column)
    control.diffuse(60.0)
    assert float(control.probabilities()[row + 1, column]) > 0.05


def test_diffusion_does_not_cross_the_lateral_shoreline() -> None:
    """The same thing across the channel, where the wall is always in reach."""
    geometry = choked_channel()
    grid = ChannelBeliefGrid(geometry)
    mask = grid.water_mask()
    narrow_row = grid.shape[0] - 1
    columns = np.nonzero(mask[narrow_row])[0]
    outermost = int(columns.max())
    assert not mask[narrow_row, outermost + 1]
    seed(grid, narrow_row, outermost)
    grid.diffuse(60.0)
    assert float(grid.probabilities()[~mask].max()) == 0.0
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)

    control = ChannelBeliefGrid(choked_channel(narrow_half_width_m=1000.0))
    seed(control, narrow_row, outermost)
    control.diffuse(60.0)
    assert float(control.probabilities()[narrow_row, outermost + 1]) > 0.05


def test_the_ends_of_the_strait_reflect_rather_than_absorb() -> None:
    """Mass staying at 1.0 is not evidence about the ends.

    This test used to assert only that, which
    ``test_diffusion_conserves_mass_without_renormalising`` already asserts:
    deleting the end faces entirely would not have failed it unless the total
    also moved. So assert the behaviour a no-flux boundary actually has —
    belief **piles up** against the wall. A free Gaussian of spread ``σ`` puts
    ``cell / (sqrt(2 π) σ)`` in the cell at its centre; reflection at a wall
    one half-cell away folds the missing half back on top of it, so the row
    against the end carries twice that.
    """
    grid = ChannelBeliefGrid(straight_channel())
    column = grid.shape[1] // 2
    seed(grid, 0, column)
    seconds = 200 * 10.0
    for _ in range(200):
        grid.diffuse(10.0)
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)

    sigma = math.sqrt(DEFAULT_SPEED_MPS**2 * DEFAULT_HEADING_PERSISTENCE_S * seconds)
    step = float(grid.along_centres_m()[1] - grid.along_centres_m()[0])
    assert sigma > 5.0 * step  # the wall is well inside the spread
    free_peak = step / (math.sqrt(2.0 * math.pi) * sigma)
    assert float(along_marginal(grid)[0]) == pytest.approx(2.0 * free_peak, rel=0.05)


def test_a_detection_never_puts_mass_on_land() -> None:
    """A fix reported *on the shore* still leaves the field entirely on water."""
    geometry = choked_channel()
    grid = ChannelBeliefGrid(geometry)
    on_land = geometry.to_geodetic(ChannelPoint(s_m=geometry.length_m - 400.0, w_m=800.0))
    assert not geometry.is_water(*on_land)
    grid.update_detection(on_land[0], on_land[1], sigma_m=150.0)
    assert float(grid.probabilities()[~grid.water_mask()].max()) == 0.0
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)


# --------------------------------------------------------------------------
# Criterion: diffusion is matched to a plausible vessel speed over the tick.
# --------------------------------------------------------------------------


def test_the_stated_diffusion_number() -> None:
    """σ = 21.9 m per axis per 1 s tick, from v = 4.0 m/s and tau = 30 s.

    The module docstring states that number; this is the arithmetic behind it,
    so changing either constant without restating the number fails here.
    """
    assert DEFAULT_SPEED_MPS == 4.0
    assert DEFAULT_HEADING_PERSISTENCE_S == 30.0
    variance_per_tick = DEFAULT_SPEED_MPS**2 * DEFAULT_HEADING_PERSISTENCE_S * 1.0
    assert math.sqrt(variance_per_tick) == pytest.approx(21.9, abs=0.05)


def test_every_diffusion_figure_the_docstring_states() -> None:
    """The acceptance criterion is "with the number stated", so check the words.

    ``grid.py``'s docstring states four spreads — 21.9 m at 1 s, 170 m at a
    minute, 537 m at ten minutes, 1.3 km at an hour. Three of them had a test
    behind them and the one-minute figure did not; it read 69 m, which is
    ``sqrt(480 × 60)`` = 169.7 m with the leading 1 dropped, and a reader
    sizing a search box off it sized it 2.5× too small.

    So this asserts the arithmetic *and* that the prose still says it. The
    second half is the part that catches the next dropped digit: the numbers
    live in exactly one place, and it is a place no test could previously see.
    """
    docstring = grid_module.__doc__
    assert docstring is not None

    def sigma(seconds: float) -> float:
        return math.sqrt(DEFAULT_SPEED_MPS**2 * DEFAULT_HEADING_PERSISTENCE_S * seconds)

    assert sigma(1.0) == pytest.approx(21.9, abs=0.05)
    assert sigma(60.0) == pytest.approx(169.7, abs=0.05)
    assert sigma(600.0) == pytest.approx(536.7, abs=0.05)
    assert sigma(3600.0) == pytest.approx(1314.5, abs=0.05)

    for stated in (
        "σ = 21.9 m per axis per 1 s tick",
        "170 m\nafter a minute",
        "537 m after ten minutes",
        "1.3 km after an hour",
    ):
        assert stated in docstring, stated

    # and the operator reaches the one-minute figure, not just the algebra.
    grid = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0)
    seed(grid, grid.shape[0] // 2, grid.shape[1] // 2)
    grid.diffuse(60.0)
    assert along_std_m(grid) == pytest.approx(169.7, rel=0.03)


def test_the_diffusion_reaches_its_analytic_spread_after_one_persistence_time() -> None:
    """What this is, and what it is not.

    It is a check that the implemented operator reaches the analytic variance
    ``v² tau t`` at ``t = tau``: 30 one-second ticks of the real flux-form
    kernel land within 5 % of 120 m.

    It is **not** evidence for ``tau`` = 30 s, though the docstring used to
    claim it was. ``sigma(t) = sqrt(v² tau t)`` makes ``sigma(tau) = v tau``
    an identity at every ``tau`` — 1 s, 30 s, 3000 s all "agree with the
    vessel's travel" — so the equality cannot distinguish the chosen value
    from any other. ``tau`` is a stated judgement call; see ``grid.py``.
    """
    grid = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0)
    seed(grid, grid.shape[0] // 2, grid.shape[1] // 2)
    for _ in range(int(DEFAULT_HEADING_PERSISTENCE_S)):
        grid.diffuse(1.0)
    travelled = DEFAULT_SPEED_MPS * DEFAULT_HEADING_PERSISTENCE_S
    assert along_std_m(grid) == pytest.approx(travelled, rel=0.05)


def test_spread_grows_as_the_square_root_of_time() -> None:
    grid = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0)
    seed(grid, grid.shape[0] // 2, grid.shape[1] // 2)
    grid.diffuse(60.0)
    after_one = along_std_m(grid)
    grid.diffuse(180.0)
    after_four = along_std_m(grid)
    assert after_one == pytest.approx(math.sqrt(DEFAULT_SPEED_MPS**2 * 30.0 * 60.0), rel=0.03)
    assert after_four == pytest.approx(2.0 * after_one, rel=0.03)


def test_a_long_tick_is_sub_stepped_rather_than_refused() -> None:
    """One 30 s tick and thirty 1 s ticks must agree, and neither may go negative."""
    one_step = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0)
    many_steps = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0)
    row, column = one_step.shape[0] // 2, one_step.shape[1] // 2
    seed(one_step, row, column)
    seed(many_steps, row, column)
    one_step.diffuse(30.0)
    for _ in range(30):
        many_steps.diffuse(1.0)
    assert along_std_m(one_step) == pytest.approx(along_std_m(many_steps), rel=0.02)

    enormous = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0)
    seed(enormous, row, column)
    enormous.diffuse(86_400.0)
    assert enormous.probabilities().min() >= 0.0
    assert enormous.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)


def test_diffusion_never_sharpens_the_field() -> None:
    """Entropy is non-decreasing under diffusion alone, which is what "decaying" means."""
    grid = ChannelBeliefGrid()
    lat_deg, lon_deg = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=12_000.0, w_m=0.0))
    grid.update_detection(lat_deg, lon_deg, sigma_m=100.0)
    previous = grid.entropy()
    for _ in range(50):
        grid.diffuse(10.0)
        current = grid.entropy()
        assert current >= previous - 1e-12
        previous = current


def test_a_faster_vessel_diffuses_further() -> None:
    slow = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0, speed_mps=2.0)
    fast = ChannelBeliefGrid(straight_channel(), along_m=100.0, across_m=200.0, speed_mps=8.0)
    row, column = slow.shape[0] // 2, slow.shape[1] // 2
    seed(slow, row, column)
    seed(fast, row, column)
    slow.diffuse(60.0)
    fast.diffuse(60.0)
    assert along_std_m(fast) == pytest.approx(4.0 * along_std_m(slow), rel=0.05)


def test_a_bad_tick_length_is_refused() -> None:
    grid = ChannelBeliefGrid()
    before = grid.probabilities()
    grid.diffuse(0.0)
    assert np.array_equal(grid.probabilities(), before)
    with pytest.raises(BeliefError):
        grid.diffuse(-1.0)
    with pytest.raises(BeliefError):
        grid.diffuse(math.nan)


# --------------------------------------------------------------------------
# Criterion: a positive detection at a known lat/lon concentrates belief there.
# --------------------------------------------------------------------------


def test_a_detection_concentrates_belief_at_the_reported_position() -> None:
    grid = ChannelBeliefGrid()
    lat_deg, lon_deg = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=17_500.0, w_m=100.0))
    before = grid.probability_at(lat_deg, lon_deg)
    entropy_before = grid.entropy()
    grid.update_detection(lat_deg, lon_deg, sigma_m=150.0)

    assert grid.probability_at(lat_deg, lon_deg) > 30.0 * before
    assert grid.entropy() < entropy_before
    peak = grid.peak()
    east, north = enu_from_geodetic(lat_deg, lon_deg)
    peak_east, peak_north = enu_from_geodetic(peak.lat_deg, peak.lon_deg)
    assert math.hypot(peak_east - east, peak_north - north) < 150.0


def test_a_second_detection_elsewhere_moves_the_peak() -> None:
    """A detection is a measurement, not a truth: the field must be able to move."""
    grid = ChannelBeliefGrid()
    first = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=3_000.0, w_m=0.0))
    second = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=21_000.0, w_m=0.0))
    grid.update_detection(first[0], first[1], sigma_m=150.0)
    assert DEFAULT_STRAIT.to_channel(*_peak_position(grid)).s_m == pytest.approx(3_000.0, abs=200.0)
    for _ in range(6):
        grid.update_detection(second[0], second[1], sigma_m=150.0)
    assert DEFAULT_STRAIT.to_channel(*_peak_position(grid)).s_m == pytest.approx(
        21_000.0, abs=200.0
    )


def _peak_position(grid: ChannelBeliefGrid) -> tuple[float, float]:
    peak = grid.peak()
    return peak.lat_deg, peak.lon_deg


def test_the_detection_likelihood_matches_the_documented_formula() -> None:
    """Hand-worked: from a uniform prior the posterior ratio *is* the likelihood ratio.

    ``L(d) = (1 - q) exp(-d² / 2σ²) / (2 π σ²) + q / A``, with ``q`` = 0.02.
    The detection is put at a cell centre, so there ``d`` = 0; σ is set to one
    and a half cells, so the cell three along the channel sits at exactly
    ``d`` = 2σ. Both cells are on the centreline of a straight channel, so the
    separation is exactly three cell steps and nothing is rounded into it.

    With σ = 149.74 m the Gaussian branch is ``0.98 / 140877`` = 6.956e-6 at
    the fix and ``exp(-2)`` of that, 9.4147e-7, three cells away; the uniform
    branch is ``0.02 / 2.4856e8`` = 8.046e-11 at both. So

        ratio = (9.4147e-7 + 8.046e-11) / (6.956e-6 + 8.046e-11) = 0.1353453

    which is ``exp(-2)`` = 0.1353353 lifted by one part in ten thousand by the
    floor — the floor is what stops the far cell going to zero, and at 2σ it
    is all it does.
    """
    geometry = straight_channel()
    grid = ChannelBeliefGrid(geometry, along_m=100.0, across_m=200.0)
    step = float(grid.along_centres_m()[1] - grid.along_centres_m()[0])
    column = grid.shape[1] // 2
    row = grid.shape[0] // 2
    near = cell_position(grid, row, column)
    far = cell_position(grid, row + 3, column)

    sigma = 1.5 * step
    assert sigma == pytest.approx(149.74, abs=0.01)
    grid.update_detection(near[0], near[1], sigma_m=sigma)

    floor = DEFAULT_FALSE_ALARM_RATE / grid.water_area_m2
    peak = (1.0 - DEFAULT_FALSE_ALARM_RATE) / (2.0 * math.pi * sigma**2)
    assert floor == pytest.approx(8.046e-11, rel=1e-3)
    assert peak == pytest.approx(6.956e-6, rel=1e-3)
    expected = (peak * math.exp(-2.0) + floor) / (peak + floor)
    assert expected == pytest.approx(0.1353453, abs=1e-6)
    assert expected == pytest.approx(math.exp(-2.0), rel=1e-4)

    ratio = grid.probability_at(*far) / grid.probability_at(*near)
    assert ratio == pytest.approx(expected, rel=1e-9)


def test_the_false_alarm_floor_keeps_its_stated_share_at_any_sigma() -> None:
    """Why both branches of the likelihood are normalised densities.

    ``L(d) = (1 - q) N(d; σ) + q / A`` with both branches integrating to their
    own weight, so the false-alarm branch keeps exactly ``q`` of the posterior
    away from the fix **whatever σ is**. That is what makes ``q`` a mixture
    weight a caller can reason about, and it is the whole reason the
    ``1 / 2 π σ²`` is there.

    Drop the constant and the bump's integral scales as ``σ²``, so the floor's
    share collapses and varies with the fix's precision: measured under
    exactly that mutation, the mass beyond five sigma falls from 0.0199 to
    2e-6 at σ = 150 m and from 0.0190 to 4e-6 at σ = 400 m.

    The test this replaces as M9's evidence,
    ``test_a_tighter_sigma_concentrates_harder``, **passes** under that
    mutation — unnormalised, a tight fix concentrates 57× harder, not less.
    A wide, straight channel is used so the Gaussian is not clipped by a
    shoreline or by the ends, which is what makes the integral come out at 1.
    """
    geometry = straight_channel()
    shares = []
    for sigma in (150.0, 400.0):
        grid = ChannelBeliefGrid(geometry, along_m=100.0, across_m=200.0)
        row, column = grid.shape[0] // 2, grid.shape[1] // 2
        # Straight channel: (s, w) is Cartesian, so this distance is exact.
        s_centres = grid.along_centres_m()
        w_centres = grid.across_centres_m()
        distance = np.hypot(
            (s_centres - s_centres[row])[:, None],
            (w_centres - w_centres[column])[None, :],
        )
        assert distance.max() > 6.0 * sigma  # the far region is not empty
        grid.update_detection(*cell_position(grid, row, column), sigma_m=sigma)
        shares.append(float(grid.probabilities()[distance > 5.0 * sigma].sum()))

    for share in shares:
        assert share == pytest.approx(DEFAULT_FALSE_ALARM_RATE, rel=0.1)
    assert shares[1] == pytest.approx(shares[0], rel=0.1)


def test_a_tighter_sigma_concentrates_harder() -> None:
    loose = ChannelBeliefGrid()
    tight = ChannelBeliefGrid()
    lat_deg, lon_deg = cell_position(loose, loose.shape[0] // 2, loose.shape[1] // 2)
    loose.update_detection(lat_deg, lon_deg, sigma_m=400.0)
    tight.update_detection(lat_deg, lon_deg, sigma_m=50.0)
    assert tight.probability_at(lat_deg, lon_deg) > loose.probability_at(lat_deg, lon_deg)
    assert tight.entropy() < loose.entropy()


def test_one_frame_cannot_empty_the_strait() -> None:
    """The false-alarm floor: every water cell keeps some belief."""
    grid = ChannelBeliefGrid()
    lat_deg, lon_deg = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=1_000.0, w_m=0.0))
    grid.update_detection(lat_deg, lon_deg, sigma_m=50.0)
    values = grid.probabilities()[grid.water_mask()]
    assert float(values.min()) > 0.0


def test_an_impossible_detection_is_refused_rather_than_absorbed() -> None:
    """With the floor switched off, a fix nowhere near the water has no posterior."""
    grid = ChannelBeliefGrid()
    with pytest.raises(BeliefError):
        grid.update_detection(0.0, 0.0, sigma_m=10.0, false_alarm_rate=0.0)
    # and the field is left usable, not half-updated
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)


def test_a_malformed_detection_is_refused() -> None:
    grid = ChannelBeliefGrid()
    with pytest.raises(BeliefError):
        grid.update_detection(71.99, -94.84, sigma_m=0.0)
    with pytest.raises(BeliefError):
        grid.update_detection(71.99, -94.84, sigma_m=150.0, false_alarm_rate=1.0)
    with pytest.raises(BeliefError):
        grid.update_detection(math.nan, -94.84, sigma_m=150.0)


def test_probability_at_is_zero_off_the_water() -> None:
    grid = ChannelBeliefGrid()
    assert grid.probability_at(0.0, 0.0) == 0.0
    assert grid.probability_at(math.nan, -94.84) == 0.0
    on_land = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=12_500.0, w_m=950.0))
    assert not DEFAULT_STRAIT.is_water(*on_land)
    assert grid.probability_at(*on_land) == 0.0


def test_probability_at_is_zero_off_both_ends_of_the_strait() -> None:
    """The case the lat/lon column bound cannot catch.

    ``(0.0, 0.0)`` above is rejected because its ``w`` is about 8000 km, so it
    fails the *column* test and says nothing about the ends. A position due
    west of the mouth is different: ``to_channel`` clamps ``s`` to 0 and
    measures ``w`` against the first segment's infinite line, so it lands in
    row 0 of a water column however far out it is, and before the guard in
    ``_cell_of`` this returned the full uniform cell mass 200 km outside the
    arena. The eastern end passed only because ``s`` clamps to exactly
    ``length_m``; it is asserted here so it stops being luck.

    The control is the two positions just *inside* each mouth: if the guard
    were a blanket rejection rather than an end check, they would be zero too
    and this test would pass for the wrong reason.
    """
    grid = ChannelBeliefGrid()
    uniform_cell_mass = 1.0 / grid.water_cells

    for metres in (-500.0, -5_000.0, -200_000.0, 500.0, 5_000.0, 200_000.0):
        position = beyond_the_mouth(DEFAULT_STRAIT, metres)
        channel = DEFAULT_STRAIT.to_channel(*position)
        # It really does project onto an end of the centreline, on the axis.
        assert abs(channel.w_m) < 1.0
        assert channel.s_m in (0.0, DEFAULT_STRAIT.length_m)
        assert not DEFAULT_STRAIT.is_water(*position)
        assert grid.probability_at(*position) == 0.0

    for s_m in (50.0, DEFAULT_STRAIT.length_m - 50.0):
        inside = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=s_m, w_m=0.0))
        assert DEFAULT_STRAIT.is_water(*inside)
        assert grid.probability_at(*inside) == pytest.approx(uniform_cell_mass)


def test_probability_at_and_is_water_agree_to_within_one_cell() -> None:
    """One definition of "on water", not two — up to the mask's resolution.

    ``probability_at`` is what the policy reaches through the Protocol and
    ``is_water`` is what the geometry says; them disagreeing is how R1-C1 got
    in. They cannot agree *exactly*: the mask is built by testing each cell's
    **centre**, so a cell straddling the shore is water for its whole area and
    a position up to half a cell outside the channel inherits its mass. That
    residue is bounded by half the across-channel cell, and this asserts the
    bound — which is the same thing as asserting that nothing outside it
    survives, including everything off the two ends.
    """
    grid = ChannelBeliefGrid()
    across_step = float(grid.across_centres_m()[1] - grid.across_centres_m()[0])
    tolerance = 0.5 * across_step + 1e-6

    wet = 0
    for latitude in np.linspace(71.95, 72.04, 31):
        for longitude in np.linspace(-95.60, -94.08, 61):
            mass = grid.probability_at(float(latitude), float(longitude))
            if mass == 0.0:
                continue
            wet += 1
            channel = DEFAULT_STRAIT.to_channel(float(latitude), float(longitude))
            assert 0.0 < channel.s_m < DEFAULT_STRAIT.length_m, (latitude, longitude)
            overhang = abs(channel.w_m) - DEFAULT_STRAIT.half_width_at(channel.s_m)
            assert overhang <= tolerance, (latitude, longitude, overhang)
    # and the sweep is not vacuous: it does land on water, repeatedly.
    assert wet > 100


# --------------------------------------------------------------------------
# Criterion: the policy reaches the field through an interface, so the
# internal shape can change without touching the policy.
# --------------------------------------------------------------------------


def test_the_grid_satisfies_the_belief_field_protocol() -> None:
    assert isinstance(ChannelBeliefGrid(), BeliefField)


def test_a_caller_written_against_the_protocol_works_unchanged() -> None:
    """The shape of the policy's dependency: positions and numbers, no cells."""

    def where_to_look(field: BeliefField) -> tuple[float, float]:
        peak = field.peak()
        return peak.lat_deg, peak.lon_deg

    grid = ChannelBeliefGrid()
    lat_deg, lon_deg = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=6_000.0, w_m=0.0))
    grid.update_detection(lat_deg, lon_deg, sigma_m=100.0)
    looked = where_to_look(grid)
    assert math.hypot(*np.subtract(looked, (lat_deg, lon_deg))) < 0.01


def test_the_interface_module_does_not_know_about_the_grid() -> None:
    """A seam with a back-dependency is not a seam.

    ``field.py`` must not import ``grid.py`` or ``geometry.py``: if it did,
    replacing the implementation would mean editing the protocol.
    """
    source = (REPO_ROOT / "whiteout" / "belief" / "field.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert not [name for name in imported if name.startswith("whiteout")], imported


def test_update_likelihood_is_the_general_seam() -> None:
    """#13 will write the negative-information update through this, in lat/lon only."""
    grid = ChannelBeliefGrid()
    equivalent = ChannelBeliefGrid()
    lat_deg, lon_deg = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=9_000.0, w_m=0.0))
    sigma = 200.0
    east, north = enu_from_geodetic(lat_deg, lon_deg)

    def gaussian(cell_lat: float, cell_lon: float) -> float:
        cell_east, cell_north = enu_from_geodetic(cell_lat, cell_lon)
        squared = (cell_east - east) ** 2 + (cell_north - north) ** 2
        density = math.exp(-squared / (2.0 * sigma**2)) / (2.0 * math.pi * sigma**2)
        return (1.0 - DEFAULT_FALSE_ALARM_RATE) * density + (
            DEFAULT_FALSE_ALARM_RATE / grid.water_area_m2
        )

    grid.update_detection(lat_deg, lon_deg, sigma_m=sigma)
    equivalent.update_likelihood(gaussian)
    assert np.allclose(grid.probabilities(), equivalent.probabilities(), atol=1e-15)


def test_a_likelihood_that_erases_the_field_is_refused() -> None:
    grid = ChannelBeliefGrid()
    with pytest.raises(BeliefError):
        grid.update_likelihood(lambda lat, lon: 0.0)
    with pytest.raises(BeliefError):
        grid.update_likelihood(lambda lat, lon: -1.0)
    with pytest.raises(BeliefError):
        grid.update_likelihood(lambda lat, lon: math.nan)
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)


def test_a_partly_negative_likelihood_is_refused_cell_by_cell() -> None:
    """The one case a total-mass check cannot catch.

    A likelihood that is negative on half the strait and positive on the other
    half still sums to something positive, so the field would renormalise
    happily and carry negative probabilities out of the update. The validation
    has to be per cell.
    """
    grid = ChannelBeliefGrid()
    middle = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=DEFAULT_STRAIT.length_m / 2.0, w_m=0.0))

    def half_negative(lat: float, lon: float) -> float:
        return 4.0 if lon > middle[1] else -1.0

    with pytest.raises(BeliefError):
        grid.update_likelihood(half_negative)
    assert grid.probabilities().min() >= 0.0
    assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)


def test_the_resolution_can_change_without_the_callers_noticing() -> None:
    """The point of the interface: same answers, different internals."""
    lat_deg, lon_deg = DEFAULT_STRAIT.to_geodetic(ChannelPoint(s_m=15_000.0, w_m=0.0))
    coarse = ChannelBeliefGrid(along_m=400.0, across_m=400.0)
    fine = ChannelBeliefGrid(along_m=50.0, across_m=50.0)
    assert coarse.shape != fine.shape
    for grid in (coarse, fine):
        grid.update_detection(lat_deg, lon_deg, sigma_m=150.0)
        grid.diffuse(60.0)
        assert grid.mass() == pytest.approx(1.0, abs=MASS_TOLERANCE)
        peak = grid.peak()
        assert abs(peak.lat_deg - lat_deg) < 0.01
        assert abs(peak.lon_deg - lon_deg) < 0.02
