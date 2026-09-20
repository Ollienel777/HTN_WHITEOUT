"""Issue #124's acceptance criteria: the belief field reaches the episode log.

The criterion most likely to pass vacuously is the round trip. A field that
is uniform round-trips through any encoding that preserves a constant, so
every round-trip test here runs on a field that has been **updated with a
detection first** — a field with structure in it, whose peak is many times
its floor — and the uniform case is asserted separately as the thing it is.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

import whiteout.belief.encode as encode_module
from whiteout.belief import ChannelBeliefGrid
from whiteout.belief.encode import (
    belief_frame,
    belief_geometry,
    decode_cells,
    decode_corners,
    decode_water,
)
from whiteout.cli import main
from whiteout.log import SCHEMA_VERSION, read_episode_log
from whiteout.types import QUANTISATION_STEPS, BeliefFrame, BeliefGeometry, RecordError

#: The middle of the default strait, near enough. A detection here gives the
#: field structure without landing on the shoreline, where a test would be
#: measuring the mask rather than the encoding.
MID_STRAIT_LAT = 71.9965
MID_STRAIT_LON = -94.8448


def _sharp_grid() -> ChannelBeliefGrid:
    """A field with structure in it, so the round trip cannot pass vacuously."""
    grid = ChannelBeliefGrid()
    grid.update_detection(MID_STRAIT_LAT, MID_STRAIT_LON, sigma_m=200.0)
    return grid


# --------------------------------------------------------------------------
# The round trip, and the one quantisation step it promises.
# --------------------------------------------------------------------------


def test_the_round_trip_is_within_one_quantisation_step() -> None:
    grid = _sharp_grid()
    original = grid.probabilities()[grid.water_mask()]
    frame = belief_frame(grid, 0.0, include_geometry=True)

    recovered = decode_cells(frame)

    step = frame.scale / QUANTISATION_STEPS
    assert recovered.shape == original.shape
    assert np.max(np.abs(recovered - original)) <= step


def test_the_round_trip_survives_the_log(tmp_path: Path) -> None:
    """Written, read back as JSON, and still within a step.

    The encoder and the record type could each be right while the JSON
    between them dropped something, so this goes the whole way rather than
    comparing two Python objects.
    """
    grid = _sharp_grid()
    original = grid.probabilities()[grid.water_mask()]
    frame = belief_frame(grid, 3.0, include_geometry=True)

    parsed = BeliefFrame.from_dict(json.loads(json.dumps(frame.to_dict())))

    assert parsed == frame
    assert np.max(np.abs(decode_cells(parsed) - original)) <= frame.scale / QUANTISATION_STEPS


def test_the_field_carries_structure_and_not_just_a_constant() -> None:
    """The guard on the tests above: the field being round-tripped is not flat."""
    frame = belief_frame(_sharp_grid(), 0.0, include_geometry=False)

    codes = np.frombuffer(base64.b64decode(frame.cells), dtype=np.uint8)

    assert codes.max() == QUANTISATION_STEPS
    assert codes.min() == 0
    assert len(set(codes.tolist())) > 20


def test_a_uniform_field_round_trips_exactly() -> None:
    """The prior is uniform, so every code is 255 and the error is zero.

    Stated separately because it is the case the round-trip tests above are
    built to avoid: it would pass under an encoder that discarded the field
    and wrote a constant.
    """
    grid = ChannelBeliefGrid()
    frame = belief_frame(grid, 0.0, include_geometry=False)

    recovered = decode_cells(frame)

    assert np.allclose(recovered, grid.probabilities()[grid.water_mask()])


def test_the_scale_is_the_peak_so_the_codes_span_the_range() -> None:
    """Re-scaling per frame is what keeps the resolution; assert it happens.

    A field over 850 cells peaks near 0.01, so quantising over a fixed
    ``[0, 1]`` would put every cell in code 0 or 1. This is the test that
    fails if someone replaces ``scale`` with a constant.
    """
    grid = _sharp_grid()

    frame = belief_frame(grid, 0.0, include_geometry=False)

    assert frame.scale == pytest.approx(grid.peak().probability)
    assert frame.scale < 0.2


# --------------------------------------------------------------------------
# Cell -> lat/lon, recoverable without a second converter.
# --------------------------------------------------------------------------


def test_the_mask_and_the_frame_agree_on_which_cells_are_water() -> None:
    grid = _sharp_grid()
    frame = belief_frame(grid, 0.0, include_geometry=True)
    assert frame.geometry is not None

    mask = decode_water(frame.geometry)

    assert mask.shape == grid.shape
    assert np.array_equal(mask, grid.water_mask())
    assert int(mask.sum()) == frame.water_cells == grid.water_cells


def test_each_cell_centre_falls_inside_its_own_corners() -> None:
    """The mapping the viewer will use, checked against the grid's own centres.

    Byte *i* is the *i*-th set bit of the mask, which is cell
    ``(row, column)``, which is the quadrilateral on corners ``(row, column)``
    through ``(row + 1, column + 1)``. If any step of that is off by one, a
    centre lands outside its own quad and this fails.
    """
    grid = _sharp_grid()
    frame = belief_frame(grid, 0.0, include_geometry=True)
    assert frame.geometry is not None
    lat, lon = decode_corners(frame.geometry)
    rows, columns = np.nonzero(grid.water_mask())

    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        corner_lats = lat[row : row + 2, column : column + 2]
        corner_lons = lon[row : row + 2, column : column + 2]
        centre_lat = float(np.mean(corner_lats))
        centre_lon = float(np.mean(corner_lons))
        probability = grid.probability_at(centre_lat, centre_lon)
        assert probability == pytest.approx(float(grid.probabilities()[row, column]), rel=1e-9), (
            f"the quad at ({row}, {column}) does not contain the cell it is drawn for"
        )


def test_the_corner_lattice_is_one_larger_than_the_grid_in_both_axes() -> None:
    grid = ChannelBeliefGrid()
    along, across = grid.shape

    lat, lon = decode_corners(belief_geometry(grid))

    assert lat.shape == (along + 1, across + 1)
    assert lon.shape == (along + 1, across + 1)


def test_the_corners_resolve_the_arena_to_about_a_metre() -> None:
    """``float32`` is the wire type; this is what it costs.

    Stated in degrees rather than metres on purpose: converting one to the
    other is :mod:`whiteout.geo`'s job and no other file's, and
    ``tests/test_geo.py`` walks this one looking for a second converter. At
    Bellot Strait 1e-5 degrees is a metre or so either way, against cells
    100 m across, so the rounding is invisible to anything that draws this.
    """
    grid = ChannelBeliefGrid()
    exact_lat, exact_lon = grid.corner_positions()

    lat, lon = decode_corners(belief_geometry(grid))

    assert float(np.abs(lat - exact_lat).max()) < 1e-5
    assert float(np.abs(lon - exact_lon).max()) < 1e-5


# --------------------------------------------------------------------------
# The bytes: deterministic, and within the budget the ticket set.
# --------------------------------------------------------------------------


def test_the_same_field_encodes_to_the_same_bytes() -> None:
    """The gate compares two runs of a seed byte for byte."""
    first = belief_frame(_sharp_grid(), 1.5, include_geometry=True)
    second = belief_frame(_sharp_grid(), 1.5, include_geometry=True)

    assert first == second


def test_the_corners_are_big_endian_whatever_the_machine_is() -> None:
    """Pinned numerically, because a native-order dtype would pass on x86.

    The first corner's latitude is the first four bytes, and big-endian is
    the only reading of them that gives a latitude in the Arctic.
    """
    grid = ChannelBeliefGrid()
    geometry = belief_geometry(grid)

    raw = base64.b64decode(geometry.corners)[:4]

    assert float(np.frombuffer(raw, dtype=">f4")[0]) == pytest.approx(71.99, abs=0.1)


def test_the_episode_fits_the_committed_budget() -> None:
    """#43 caps committed episodes at 25 MB. This is the measurement.

    Every figure in :mod:`whiteout.belief.encode`'s docstring is re-measured
    here, so the prose and the code cannot drift apart.
    """
    grid = _sharp_grid()
    compact = (",", ":")
    kilobyte = 1_000

    with_geometry = len(
        json.dumps(belief_frame(grid, 0.0, include_geometry=True).to_dict(), separators=compact)
    )
    without = len(
        json.dumps(belief_frame(grid, 0.0, include_geometry=False).to_dict(), separators=compact)
    )
    as_floats = len(
        json.dumps(grid.probabilities()[grid.water_mask()].tolist(), separators=compact)
    )

    assert grid.shape == (63, 16)
    assert grid.water_cells == 850
    assert without < 1.3 * kilobyte
    assert (with_geometry - without) / kilobyte == pytest.approx(11.8, abs=0.4)
    assert without * 400 < 520 * kilobyte
    assert as_floats * 400 > 6_000 * kilobyte


# --------------------------------------------------------------------------
# What the record type refuses.
# --------------------------------------------------------------------------


def test_a_frame_whose_cells_are_the_wrong_length_is_refused() -> None:
    frame = belief_frame(_sharp_grid(), 0.0, include_geometry=False)

    with pytest.raises(RecordError, match="expected 850 bytes"):
        BeliefFrame(
            t=frame.t,
            water_cells=frame.water_cells,
            scale=frame.scale,
            cells=base64.b64encode(b"\x00" * 849).decode("ascii"),
            geometry=None,
        )


def test_a_frame_that_disagrees_with_its_own_geometry_is_refused() -> None:
    grid = ChannelBeliefGrid()
    geometry = belief_geometry(grid)

    with pytest.raises(RecordError, match="marks 850 water cells"):
        BeliefFrame(
            t=0.0,
            water_cells=849,
            scale=1.0,
            cells=base64.b64encode(b"\x00" * 849).decode("ascii"),
            geometry=geometry,
        )


def test_a_field_that_is_not_base64_is_refused_rather_than_silently_shortened() -> None:
    with pytest.raises(RecordError, match="not valid base64"):
        BeliefFrame(t=0.0, water_cells=3, scale=1.0, cells="not base64!!", geometry=None)


def test_a_geometry_whose_lattice_is_the_wrong_size_is_refused() -> None:
    geometry = belief_geometry(ChannelBeliefGrid())

    with pytest.raises(RecordError, match="geometry.corners"):
        BeliefGeometry(shape=geometry.shape, water=geometry.water, corners="")


def test_the_encoder_never_decodes_back_into_a_belief_field() -> None:
    """The rendering channel stays one, enforced rather than asked for.

    ``decode_cells`` exists for the tests above and for a reader checking a
    committed log by hand. Nothing in ``whiteout/`` may build a field from it,
    because the field it came from is the state and this is a rounded picture.
    """
    source = Path(encode_module.__file__).read_text(encoding="utf-8")

    assert "ChannelBeliefGrid(" not in source
    assert "update_likelihood" not in source


def test_the_schema_version_moved_with_the_shape() -> None:
    """A required key was added, so a version-5 reader must refuse these logs."""
    assert SCHEMA_VERSION == 6


# --------------------------------------------------------------------------
# End to end: what a real episode log actually carries.
# --------------------------------------------------------------------------


def test_a_real_episode_carries_the_field_every_tick_and_the_layout_once(
    tmp_path: Path,
) -> None:
    """The whole of the ticket, measured on the command the fixture is made with."""
    out = tmp_path / "episode.jsonl"

    assert main(["run", "--seed", "7", "--ticks", "8", "--out", str(out)]) == 0
    records = read_episode_log(out)

    frames = [record.belief_field for record in records]
    assert all(frame is not None for frame in frames)
    assert frames[0] is not None and frames[0].geometry is not None
    assert all(frame is not None and frame.geometry is None for frame in frames[1:])
    for record, frame in zip(records, frames, strict=True):
        assert frame is not None
        assert frame.t == record.t
        along, across = record.belief_digest.grid_shape
        assert 0 < frame.water_cells <= along * across
        assert len(decode_cells(frame)) == frame.water_cells
