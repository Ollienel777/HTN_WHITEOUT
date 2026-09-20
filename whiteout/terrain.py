"""Ground elevation, read from the arena's own heightmap.

Issue #136. ``scripts/tower_siting.py`` had no terrain model: it scored every
candidate site at whatever height it was handed, and the height it was handed
was the one the towers happen to stand at today. Its own output warned about
this — *"a site at sea level is worth far less than this says"* — and then
**both** published recommendations put a tower at ``0.0`` m anyway, which is
the waterline, where
:func:`~whiteout.belief.negative.non_detection_likelihood` refuses the camera
outright because it is not above the water plane. Not a poor score: an
exception.

A warning that is correct and ignored twice is a warning that should have been
a measurement.

What this reads
----------------

Gazebo renders the site from a greyscale heightmap, and the arena serves it
next to the viewer:

.. code-block:: text

    assets/terrain_<site>/materials/textures/heightmap.png
    assets/terrain_<site>/model.sdf   ->  <size>6500 6500 252.109</size>

The SDF's three numbers are the footprint in metres and the **full vertical
range** the image's 0..max maps onto. So elevation is
``sample / max * z_range``, and nothing else is needed.

The orientation, settled by measurement
----------------------------------------

Image row 0 is **north**, and the image is not transposed. That was not
assumed: it was chosen from the four candidate conventions by checking each
against the two towers already standing, whose true elevations MAVLink
reports. Row-flipped and un-transposed reproduces 116.8 m and 226.6 m to
**~1 m**; every other convention is out by 5 m to 227 m.

Grid, not true north
---------------------

Gazebo's world axes are EPSG:3413 **grid** axes. Turning a lat/lon into an
``(x, y)`` to sample at therefore needs the site's grid convergence, the same
rotation :mod:`scripts.truth_probe` carries and for the same reason — omitting
it is a ~1500 m error. :class:`whiteout.site.SiteRecord` holds the number the
arena reports.

Network
--------

:func:`load_heightmap` takes a **local path**. Fetching is the caller's job, so
nothing here opens a socket and the gate is unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["Terrain", "TerrainError", "load_heightmap"]


class TerrainError(RuntimeError):
    """The heightmap could not be read, or does not describe this site."""


@dataclass(frozen=True)
class Terrain:
    """A site's ground elevation, sampled from its heightmap.

    :param samples: the greyscale image, rows north-to-south.
    :param extent_m: the square footprint in metres, from the model SDF.
    :param z_range_m: the elevation the image's full range maps onto, also
        from the SDF.
    :param convergence_deg: the site's grid convergence, so a lat/lon can be
        turned into the grid ``(x, y)`` the image is indexed by.
    :param centre_lat_deg: site centre.
    :param centre_lon_deg: site centre.
    """

    samples: np.ndarray
    extent_m: float
    z_range_m: float
    convergence_deg: float
    centre_lat_deg: float
    centre_lon_deg: float

    def elevation_at_grid(self, x_m: float, y_m: float) -> float:
        """Ground elevation at a point in **grid** metres from the centre."""
        rows, columns = self.samples.shape[0], self.samples.shape[1]
        half = self.extent_m / 2.0
        column = int(round((x_m + half) / self.extent_m * (columns - 1)))
        row = int(round((1.0 - (y_m + half) / self.extent_m) * (rows - 1)))
        if not (0 <= row < rows and 0 <= column < columns):
            raise TerrainError(
                f"({x_m:.0f}, {y_m:.0f}) m is outside the {self.extent_m:.0f} m site"
            )
        peak = float(self.samples.max())
        if peak <= 0.0:
            raise TerrainError("the heightmap is uniformly zero")
        return float(self.samples[row, column]) / peak * self.z_range_m

    def elevation(self, lat_deg: float, lon_deg: float) -> float:
        """Ground elevation at a position, in metres above the water plane.

        The water plane is ``z = 0`` in the world, so this is also height
        above the water — which is the number the detector and
        :func:`~whiteout.belief.negative.non_detection_likelihood` want.
        """
        from whiteout.geo import GeoPoint, geodetic_to_local

        offset = geodetic_to_local(
            GeoPoint(self.centre_lat_deg, self.centre_lon_deg),
            GeoPoint(lat_deg, lon_deg),
        )
        angle = math.radians(-self.convergence_deg)
        x_m = offset.east_m * math.cos(angle) + offset.north_m * math.sin(angle)
        y_m = -offset.east_m * math.sin(angle) + offset.north_m * math.cos(angle)
        return self.elevation_at_grid(x_m, y_m)

    def is_water(self, lat_deg: float, lon_deg: float, *, above_m: float = 0.5) -> bool:
        """Whether a position is at or below the water plane.

        A tower here cannot see the water it is standing in, so this is the
        check that would have caught both sea-level recommendations.
        """
        return self.elevation(lat_deg, lon_deg) < above_m


def load_heightmap(
    path: str | Path,
    *,
    extent_m: float,
    z_range_m: float,
    convergence_deg: float,
    centre_lat_deg: float,
    centre_lon_deg: float,
) -> Terrain:
    """Read a greyscale heightmap from disk.

    Pillow and NumPy only, and no network: the caller fetches the file.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(Path(path)) as opened:
            samples = np.asarray(opened.convert("L"))
    except (OSError, ValueError) as error:
        raise TerrainError(f"could not read {path}: {error}") from error
    if samples.ndim != 2 or min(samples.shape) < 2:
        raise TerrainError(f"{path} is not a 2-D heightmap, shape {samples.shape}")
    if extent_m <= 0.0 or z_range_m <= 0.0:
        raise TerrainError(
            f"extent and z range must be positive, got {extent_m!r} and {z_range_m!r}"
        )
    return Terrain(
        samples=samples,
        extent_m=float(extent_m),
        z_range_m=float(z_range_m),
        convergence_deg=float(convergence_deg),
        centre_lat_deg=float(centre_lat_deg),
        centre_lon_deg=float(centre_lon_deg),
    )
