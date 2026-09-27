"""Ground elevation from a heightmap, and the check that catches a sea-level site.

Issue #136. ``tower_siting.py`` scored candidates at whatever height it was
handed, warned that "a site at sea level is worth far less than this says",
and then recommended one anyway — twice. These tests are about the measurement
that replaces the warning.

Everything here builds its own heightmap. Nothing needs the arena.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from whiteout.geo import GeoPoint, LocalPoint, local_to_geodetic
from whiteout.terrain import Terrain, TerrainError, load_heightmap

CENTRE_LAT, CENTRE_LON = 71.991960, -94.822428
EXTENT_M, Z_RANGE_M = 6500.0, 252.109


def _terrain(samples: np.ndarray, convergence_deg: float = 0.0) -> Terrain:
    return Terrain(
        samples=samples,
        extent_m=EXTENT_M,
        z_range_m=Z_RANGE_M,
        convergence_deg=convergence_deg,
        centre_lat_deg=CENTRE_LAT,
        centre_lon_deg=CENTRE_LON,
    )


def _ramp(size: int = 65) -> np.ndarray:
    """North high, south low: row 0 is 255, the last row is 0."""
    column = np.linspace(255, 0, size, dtype=np.uint8)
    return np.repeat(column[:, None], size, axis=1)


# -- sampling ----------------------------------------------------------------


def test_the_full_image_range_maps_onto_the_sdf_s_z_range() -> None:
    flat = np.full((9, 9), 255, dtype=np.uint8)
    assert _terrain(flat).elevation_at_grid(0.0, 0.0) == pytest.approx(Z_RANGE_M)
    half = np.full((9, 9), 255, dtype=np.uint8)
    half[4, 4] = 128
    assert _terrain(half).elevation_at_grid(0.0, 0.0) == pytest.approx(
        128 / 255 * Z_RANGE_M, rel=1e-6
    )


def test_row_zero_is_north() -> None:
    """Settled by measurement, not assumed — see the module docstring.

    The four candidate conventions were checked against the two towers already
    standing, whose true elevations MAVLink reports. This is the one that
    reproduced them.
    """
    terrain = _terrain(_ramp())
    north = terrain.elevation_at_grid(0.0, +3000.0)
    south = terrain.elevation_at_grid(0.0, -3000.0)
    assert north > south
    assert north == pytest.approx(Z_RANGE_M, rel=0.05)
    assert south == pytest.approx(0.0, abs=Z_RANGE_M * 0.05)


def test_a_point_outside_the_site_is_refused() -> None:
    with pytest.raises(TerrainError, match="outside"):
        _terrain(_ramp()).elevation_at_grid(EXTENT_M, 0.0)


def test_a_uniformly_zero_heightmap_is_refused_rather_than_divided_by() -> None:
    with pytest.raises(TerrainError, match="uniformly zero"):
        _terrain(np.zeros((9, 9), dtype=np.uint8)).elevation_at_grid(0.0, 0.0)


# -- the grid rotation -------------------------------------------------------


def test_convergence_rotates_the_lookup() -> None:
    """Gazebo's axes are EPSG:3413 grid; grid north is not true north.

    Omitting this rotation is a ~1500 m error, the same one
    ``scripts/truth_probe.py`` carries a measurement for.
    """
    ramp = _ramp()
    north = local_to_geodetic(
        GeoPoint(CENTRE_LAT, CENTRE_LON), LocalPoint(east_m=0.0, north_m=2500.0)
    )
    straight = _terrain(ramp, convergence_deg=0.0).elevation(north.lat_deg, north.lon_deg)
    turned = _terrain(ramp, convergence_deg=-49.8).elevation(north.lat_deg, north.lon_deg)
    assert straight != pytest.approx(turned, abs=1.0)


# -- the check that would have caught both bad recommendations ---------------


def test_a_site_at_the_waterline_reads_as_water() -> None:
    """Both published tower-2 recommendations were here.

    A tower at 0.0 m is *at* the water plane, and
    ``non_detection_likelihood`` refuses the camera outright — not a low
    score, an exception. This is the one-line check that catches it.
    """
    # Land in the north, open water in the southern third: 0 means the water
    # plane, which is exactly what the arena's heightmap encodes.
    samples = _ramp()
    samples[44:, :] = 0
    terrain = _terrain(samples)
    south = local_to_geodetic(
        GeoPoint(CENTRE_LAT, CENTRE_LON), LocalPoint(east_m=0.0, north_m=-2000.0)
    )
    assert terrain.elevation(south.lat_deg, south.lon_deg) < 0.5
    assert terrain.is_water(south.lat_deg, south.lon_deg) is True


def test_high_ground_does_not_read_as_water() -> None:
    terrain = _terrain(_ramp())
    north = local_to_geodetic(
        GeoPoint(CENTRE_LAT, CENTRE_LON), LocalPoint(east_m=0.0, north_m=3100.0)
    )
    assert terrain.is_water(north.lat_deg, north.lon_deg) is False


# -- loading -----------------------------------------------------------------


def test_a_heightmap_round_trips_from_disk(tmp_path) -> None:
    path = tmp_path / "heightmap.png"
    Image.fromarray(_ramp()).save(path)
    terrain = load_heightmap(
        path,
        extent_m=EXTENT_M,
        z_range_m=Z_RANGE_M,
        convergence_deg=0.0,
        centre_lat_deg=CENTRE_LAT,
        centre_lon_deg=CENTRE_LON,
    )
    assert terrain.samples.shape == (65, 65)
    assert terrain.elevation_at_grid(0.0, 3000.0) > terrain.elevation_at_grid(0.0, -3000.0)


def test_a_missing_file_is_diagnosed(tmp_path) -> None:
    with pytest.raises(TerrainError, match="could not read"):
        load_heightmap(
            tmp_path / "absent.png",
            extent_m=EXTENT_M,
            z_range_m=Z_RANGE_M,
            convergence_deg=0.0,
            centre_lat_deg=CENTRE_LAT,
            centre_lon_deg=CENTRE_LON,
        )


def test_a_nonsense_extent_is_refused(tmp_path) -> None:
    path = tmp_path / "heightmap.png"
    Image.fromarray(_ramp()).save(path)
    with pytest.raises(TerrainError, match="must be positive"):
        load_heightmap(
            path,
            extent_m=0.0,
            z_range_m=Z_RANGE_M,
            convergence_deg=0.0,
            centre_lat_deg=CENTRE_LAT,
            centre_lon_deg=CENTRE_LON,
        )
