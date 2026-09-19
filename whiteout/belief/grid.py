"""The belief field: where the vessel probably is, on the water of the strait.

Issue #12. A normalised probability field over Bellot Strait, a diffusion step
matched to a plausible vessel speed, and a Bayesian update on a positive
detection. The policy reaches all of it through
:class:`whiteout.belief.field.BeliefField`, never through this class's
internals.

The shape of the field: a ribbon, not a raster
-----------------------------------------------

``ARENA.md`` §2 is blunt about it — "a general 2-D belief grid over open
terrain is the wrong shape". The arena is **25 km long and 2 km wide**, and
the target is a boat, so it is confined to the water.

The field is therefore indexed by the strait's own coordinates
(:mod:`whiteout.belief.geometry`): arc length ``s`` along the centreline, and
signed offset ``w`` across it. Measured against the alternative, on the
default geometry at 100 m resolution:

.. code-block:: text

    grid                                          cells   water   useful
    North/East raster over the bounding box        9072    3992     44 %
    this ribbon, 251 along by 20 across            5020    3992     79 %

Same 39.9 km² of water either way, in 45 % fewer cells — and the gap widens
with every degree the strait bends, because a bend grows the bounding box and
not the ribbon. Cell count is not the main argument, though. These two are:

- **The shoreline is an inequality**, ``abs(w) <= half_width(s)``, not a
  terrain model. ``ARENA.md`` §7 killed the terrain generator; there is no
  heightfield, and this does not need one.
- **The axes are the search's axes.** "Which stretch of the strait is
  unsearched?" is a question about an interval in ``s``, and "the vessel is
  drifting east" is a statement about ``s`` alone. On a North/East raster both
  are diagonal queries; here they are slices. The policy in #14 and the
  negative-information update in #13 both want the along-channel marginal, and
  on this grid it is one sum.

Two consequences fall out of it for free: diffusion along the channel is a
three-point stencil on a line rather than a blur over a plane, and the
across-channel axis can be resolved coarsely on purpose, since there are only
ever twenty cells of it to be wrong about.

The diffusion rate, and the number
-----------------------------------

The vessel spawns at random and walks at random, with no evasion (``ARENA.md``
§1 Q2). That makes diffusion toward uniform the *correct* prior rather than a
convenient one — there is no adversary to model — and it makes the rate the
only thing to choose.

**Speed.** The operator measured the live target at **2.96 m/s on course
304.5°**. That is one observation of one moment, so it sets the scale but not
the value: :data:`DEFAULT_SPEED_MPS` is **4.0 m/s**, about 35 % above it. The
asymmetry is deliberate. Diffusing too fast costs a little sharpness, which
the next detection restores; diffusing too slowly leaves belief concentrated
where the vessel no longer is, the fleet stops looking anywhere else, and
nothing restores it.

**Heading persistence.** A boat is not a Brownian particle: it holds a course
for a while. With speed ``v`` and a heading that decorrelates over a time
``tau`` (:data:`DEFAULT_HEADING_PERSISTENCE_S`, **30 s**), the per-axis
velocity variance is ``v² / 2`` and the per-axis diffusion coefficient is

.. math::

    D = \\frac{v^2 \\tau}{2} = \\frac{4.0^2 \\times 30}{2} = 240\\ \\mathrm{m^2/s}

so the variance a tick of length ``dt`` adds to each axis is

.. math::

    \\sigma^2 = 2 D\\, dt = v^2 \\tau\\, dt
    = 4.0^2 \\times 30 \\times 1.0 = 480\\ \\mathrm{m^2}

**The number, stated: σ = 21.9 m per axis per 1 s tick.** Equivalently 69 m
after a minute, 537 m after ten minutes, and 1.3 km after an hour — at which
point belief is smeared over a tenth of the strait and the field is doing what
it should.

The choice of ``tau`` is what makes that defensible rather than tuned, because
it has a check: after exactly one persistence time the diffused spread is
``sqrt(v² tau × tau) = v tau = 120 m``, which is how far the vessel actually
travels in 30 s while holding a course. The kernel and the ballistic
displacement agree at the timescale the kernel is built for, and
``tests/test_belief_grid.py`` asserts that equality to 5 %.

**What this rate does not model**, stated because it is the way it fails: a
vessel that holds one course for many minutes outruns a diffusion, whose
spread grows as ``sqrt(t)`` while its displacement grows as ``t``. Beyond
about 10 minutes without a detection, this field is optimistic about a
committed runner. The fix is a correlated-walk kernel that carries a velocity,
not a bigger number here, and it is out of this ticket's scope.

Mass, and why it stays on water
--------------------------------

Diffusion is written in **flux form**: for each pair of adjacent cells that
are *both* water, a flux proportional to their difference moves from the
fuller to the emptier, and it is subtracted from one and added to the other in
the same operation. Three properties follow from that structure rather than
from a correction afterwards:

- **Mass is conserved exactly** (to floating point), so :meth:`mass` stays at
  1.0 without renormalising. That matters: a renormalisation after every step
  would make the field sum to 1 even if the operator leaked, which is a test
  that cannot fail.
- **No flux crosses the shoreline**, because a pair with a land cell in it has
  no face. This is a reflecting (no-flux) boundary, which is the physically
  right one — the vessel bounces off the shore, it does not vanish at it.
- **The field stays non-negative**, because a cell gives away at most
  ``4 × MAX_STABLE_RATIO`` of itself per sub-step. The step size is chosen to
  keep that below 1, sub-stepping when a long ``dt`` would break it.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from whiteout.belief.field import BeliefError, BeliefPeak, Likelihood
from whiteout.belief.geometry import (
    DEFAULT_STRAIT,
    ChannelPoint,
    StraitGeometry,
    enu_from_geodetic,
)

__all__ = [
    "DEFAULT_ACROSS_M",
    "DEFAULT_ALONG_M",
    "DEFAULT_DETECTION_SIGMA_M",
    "DEFAULT_FALSE_ALARM_RATE",
    "DEFAULT_HEADING_PERSISTENCE_S",
    "DEFAULT_SPEED_MPS",
    "MAX_STABLE_RATIO",
    "ChannelBeliefGrid",
]

#: Cell size along the channel, metres. 250 cells over the 25 km strait.
DEFAULT_ALONG_M = 100.0

#: Cell size across the channel, metres. 20 cells over the 2 km width.
DEFAULT_ACROSS_M = 100.0

#: Vessel speed scale for the diffusion, m/s. The operator measured the live
#: target at 2.96 m/s on course 304.5°; see the module docstring for why the
#: prior sits above a single observation rather than on it.
DEFAULT_SPEED_MPS = 4.0

#: Heading decorrelation time, seconds. A boat holds a course; this is how
#: long for. Together with the speed it fixes the diffusion coefficient.
DEFAULT_HEADING_PERSISTENCE_S = 30.0

#: One-sigma position error of a camera fix, metres. From PR #72's measured
#: error budget: the flat-water-plane bias is +10 m at 2 km and +164 m at 5 km
#: from a 60 m tower, and the unresolved geoid datum scales range linearly by
#: tens of metres. 150 m is the middle of that, and a caller with a better
#: number for its own asset passes it.
DEFAULT_DETECTION_SIGMA_M = 150.0

#: Weight of the "this detection was a false alarm" branch of the detection
#: likelihood. ``whiteout/types.py`` is explicit that a ``Detection`` is a
#: measurement and not a truth, so no single frame may zero the rest of the
#: strait: with this floor, one detection can concentrate belief but leaves
#: enough elsewhere for the field to recover if it was wrong.
DEFAULT_FALSE_ALARM_RATE = 0.02

#: The largest ``D dt / dx²`` a single diffusion sub-step may use. A cell has
#: at most four water neighbours, so it gives away at most ``4 × this`` of
#: itself per sub-step; below 0.25 the field cannot go negative, and 0.2
#: leaves margin. Longer ticks are sub-stepped rather than refused.
MAX_STABLE_RATIO = 0.2

_FloatArray = npt.NDArray[np.float64]
_BoolArray = npt.NDArray[np.bool_]


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise BeliefError(f"{name} must be finite and positive, got {value!r}")
    return number


class ChannelBeliefGrid:
    """A normalised belief field over the water of a :class:`StraitGeometry`.

    Implements :class:`whiteout.belief.field.BeliefField`. The constructor's
    defaults are the arena's: the strait of ``ARENA.md`` §2 at 100 m
    resolution, diffusing at the rate the module docstring derives.

    The field starts **uniform over the water**, which is the correct prior
    and not a placeholder: Dominion Dynamics confirmed the vessel spawns at a
    random position (``ARENA.md`` §1 Q2).
    """

    def __init__(
        self,
        geometry: StraitGeometry = DEFAULT_STRAIT,
        *,
        along_m: float = DEFAULT_ALONG_M,
        across_m: float = DEFAULT_ACROSS_M,
        speed_mps: float = DEFAULT_SPEED_MPS,
        heading_persistence_s: float = DEFAULT_HEADING_PERSISTENCE_S,
    ) -> None:
        self._geometry = geometry
        self._speed_mps = _positive(speed_mps, "speed_mps")
        self._persistence_s = _positive(heading_persistence_s, "heading_persistence_s")
        _positive(along_m, "along_m")
        _positive(across_m, "across_m")

        length = geometry.length_m
        half_extent = geometry.max_half_width_m
        n_along = max(1, math.ceil(length / along_m))
        n_across = max(1, math.ceil(2.0 * half_extent / across_m))
        self._along_step = length / n_along
        self._across_step = 2.0 * half_extent / n_across
        self._half_extent = half_extent

        self._s_centres: _FloatArray = (np.arange(n_along, dtype=np.float64) + 0.5) * (
            self._along_step
        )
        self._w_centres: _FloatArray = -half_extent + (
            np.arange(n_across, dtype=np.float64) + 0.5
        ) * (self._across_step)

        half_widths = np.array(
            [geometry.half_width_at(float(s)) for s in self._s_centres], dtype=np.float64
        )
        self._water: _BoolArray = np.abs(self._w_centres)[None, :] <= half_widths[:, None]
        if not bool(self._water.any()):
            raise BeliefError("the geometry has no water cells at this resolution")

        rows, columns = np.nonzero(self._water)
        self._rows: npt.NDArray[np.intp] = rows
        self._columns: npt.NDArray[np.intp] = columns

        latitudes: list[float] = []
        longitudes: list[float] = []
        easts: list[float] = []
        norths: list[float] = []
        for row, column in zip(rows, columns, strict=True):
            point = ChannelPoint(
                s_m=float(self._s_centres[row]), w_m=float(self._w_centres[column])
            )
            lat_deg, lon_deg = geometry.to_geodetic(point)
            east, north = enu_from_geodetic(lat_deg, lon_deg)
            latitudes.append(lat_deg)
            longitudes.append(lon_deg)
            easts.append(east)
            norths.append(north)
        self._lat: _FloatArray = np.array(latitudes, dtype=np.float64)
        self._lon: _FloatArray = np.array(longitudes, dtype=np.float64)
        self._east: _FloatArray = np.array(easts, dtype=np.float64)
        self._north: _FloatArray = np.array(norths, dtype=np.float64)

        # Faces, not cells: a flux exists only where both sides are water, and
        # that is what keeps mass off the land.
        self._faces_along: _BoolArray = self._water[:-1, :] & self._water[1:, :]
        self._faces_across: _BoolArray = self._water[:, :-1] & self._water[:, 1:]

        self._p: _FloatArray = np.zeros((n_along, n_across), dtype=np.float64)
        self._p[rows, columns] = 1.0 / float(rows.size)

    # -- shape, for tests, the viewer and nothing in the policy -------------

    @property
    def shape(self) -> tuple[int, int]:
        """Cells along the channel, cells across it."""
        return (self._p.shape[0], self._p.shape[1])

    @property
    def water_cells(self) -> int:
        """How many cells the vessel could be in."""
        return int(self._rows.size)

    @property
    def cell_area_m2(self) -> float:
        """Area of one cell, square metres."""
        return self._along_step * self._across_step

    @property
    def water_area_m2(self) -> float:
        """Total water area the field covers, square metres."""
        return self.cell_area_m2 * float(self._rows.size)

    @property
    def geometry(self) -> StraitGeometry:
        """The channel this field is defined over."""
        return self._geometry

    def probabilities(self) -> _FloatArray:
        """A copy of the raw field, cells along by cells across. Land cells are 0."""
        return self._p.copy()

    def water_mask(self) -> _BoolArray:
        """A copy of the water mask, same shape as :meth:`probabilities`."""
        return self._water.copy()

    def along_centres_m(self) -> _FloatArray:
        """Arc length of each cell centre along the centreline, metres."""
        return self._s_centres.copy()

    def across_centres_m(self) -> _FloatArray:
        """Signed offset of each cell centre across the centreline, metres."""
        return self._w_centres.copy()

    # -- the interface ------------------------------------------------------

    def mass(self) -> float:
        """Total probability mass."""
        return float(self._p.sum())

    def entropy(self) -> float:
        """Shannon entropy, nats. ``log(water_cells)`` when belief is uniform."""
        values = self._p[self._rows, self._columns]
        positive = values[values > 0.0]
        return float(-(positive * np.log(positive)).sum())

    def peak(self) -> BeliefPeak:
        """The most probable cell, as a position and a mass."""
        values = self._p[self._rows, self._columns]
        index = int(np.argmax(values))
        return BeliefPeak(
            lat_deg=float(self._lat[index]),
            lon_deg=float(self._lon[index]),
            probability=float(values[index]),
        )

    def probability_at(self, lat_deg: float, lon_deg: float) -> float:
        """Mass of the cell containing a position; 0.0 on land or off the strait."""
        cell = self._cell_of(lat_deg, lon_deg)
        if cell is None:
            return 0.0
        row, column = cell
        return float(self._p[row, column])

    def diffuse(self, dt_s: float) -> None:
        """Spread belief by ``dt_s`` seconds of the vessel's random walk.

        The per-axis variance added is ``v² tau dt`` — see the module
        docstring for the derivation and the number. ``dt_s`` of zero is a
        no-op; a negative or non-finite ``dt_s`` is an error, because it is
        always a caller bug and its silent effect would be to *sharpen* the
        field.
        """
        seconds = float(dt_s)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise BeliefError(f"dt_s must be finite and non-negative, got {dt_s!r}")
        if seconds == 0.0:
            return
        variance = self._speed_mps * self._speed_mps * self._persistence_s * seconds
        ratio_along = variance / (2.0 * self._along_step * self._along_step)
        ratio_across = variance / (2.0 * self._across_step * self._across_step)
        sub_steps = max(1, math.ceil(max(ratio_along, ratio_across) / MAX_STABLE_RATIO))
        step_along = ratio_along / sub_steps
        step_across = ratio_across / sub_steps
        for _ in range(sub_steps):
            self._diffuse_once(step_along, step_across)

    def update_detection(
        self,
        lat_deg: float,
        lon_deg: float,
        *,
        sigma_m: float = DEFAULT_DETECTION_SIGMA_M,
        false_alarm_rate: float = DEFAULT_FALSE_ALARM_RATE,
    ) -> None:
        """Fold in a positive detection at ``lat_deg``/``lon_deg``.

        The likelihood of *this measurement*, given that the vessel is truly
        in a cell at ground range ``d`` from the reported position, is a
        two-branch mixture of **densities**:

        .. math::

            L(d) = (1 - q)\\, \\frac{e^{-d^2 / 2\\sigma^2}}{2 \\pi \\sigma^2}
                   + \\frac{q}{A}

        The first branch is a real fix, Gaussian about the truth with the
        one-sigma error ``sigma_m``. The second is the detector firing on ice,
        glare or a wave crest with probability ``q``, which says nothing about
        where the vessel is and so is spread uniformly over the water's area
        ``A``.

        **Both branches are normalised, and that is load-bearing**, even
        though only ratios of ``L`` survive the renormalisation. Dropping the
        Gaussian's ``1 / 2 \\pi \\sigma^2`` would make a *tighter* fix carry
        *less* total evidence than a loose one, because the flat branch would
        then outweigh a narrow bump — so a 50 m fix would move the field less
        than a 400 m one, which is backwards.

        Without the flat branch at all, a single frame would drive every cell
        more than a few sigma away to a likelihood of ``1e-300`` and then to
        zero, and the field could never recover from one bad fix.
        """
        sigma = _positive(sigma_m, "sigma_m")
        rate = float(false_alarm_rate)
        if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
            raise BeliefError(f"false_alarm_rate must be in [0, 1), got {false_alarm_rate!r}")
        if not (math.isfinite(float(lat_deg)) and math.isfinite(float(lon_deg))):
            raise BeliefError(f"detection position must be finite, got {lat_deg!r}, {lon_deg!r}")
        east, north = enu_from_geodetic(float(lat_deg), float(lon_deg))
        squared = (self._east - east) ** 2 + (self._north - north) ** 2
        gaussian = np.exp(-squared / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma * sigma)
        weights = (1.0 - rate) * gaussian + rate / self.water_area_m2
        self._multiply(weights)

    def update_likelihood(self, likelihood: Likelihood) -> None:
        """Multiply the field by an arbitrary likelihood over positions.

        ``likelihood`` is called once per water cell with that cell's centre as
        ``(lat_deg, lon_deg)`` and must return a finite, non-negative weight.
        This is the seam issue #13's negative-information update is written
        against: it never has to know that cells exist.
        """
        weights = np.empty(self._rows.size, dtype=np.float64)
        for index in range(self._rows.size):
            value = float(likelihood(float(self._lat[index]), float(self._lon[index])))
            if not math.isfinite(value) or value < 0.0:
                raise BeliefError(
                    f"likelihood must be finite and non-negative, got {value!r} at "
                    f"{self._lat[index]:.5f}, {self._lon[index]:.5f}"
                )
            weights[index] = value
        self._multiply(weights)

    # -- internals ----------------------------------------------------------

    def _diffuse_once(self, ratio_along: float, ratio_across: float) -> None:
        """One explicit, mass-conserving, land-respecting diffusion sub-step.

        Operator splitting: the along-channel pass then the across-channel
        one. Each pass is a flux between adjacent water cells, subtracted from
        one side and added to the other, so the total is unchanged by
        construction rather than by a correction.
        """
        field = self._p
        flux_along = ratio_along * (field[:-1, :] - field[1:, :]) * self._faces_along
        field[:-1, :] -= flux_along
        field[1:, :] += flux_along
        flux_across = ratio_across * (field[:, :-1] - field[:, 1:]) * self._faces_across
        field[:, :-1] -= flux_across
        field[:, 1:] += flux_across

    def _multiply(self, weights: _FloatArray) -> None:
        """Posterior ∝ prior × weights, over the water cells, renormalised.

        Renormalising belongs here and not in :meth:`diffuse`: a Bayesian
        update genuinely changes the total, whereas diffusion must not, and
        collapsing the two would hide a leaking diffusion operator behind a
        field that always sums to 1.
        """
        posterior = self._p[self._rows, self._columns] * weights
        total = float(posterior.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise BeliefError(
                "this evidence leaves the field with no probability mass "
                f"(total {total!r}); discard it rather than believing it"
            )
        self._p[self._rows, self._columns] = posterior / total

    def _cell_of(self, lat_deg: float, lon_deg: float) -> tuple[int, int] | None:
        """Indices of the water cell containing a position, or None."""
        if not (math.isfinite(float(lat_deg)) and math.isfinite(float(lon_deg))):
            return None
        point = self._geometry.to_channel(float(lat_deg), float(lon_deg))
        row = int(math.floor(point.s_m / self._along_step))
        column = int(math.floor((point.w_m + self._half_extent) / self._across_step))
        n_along, n_across = self.shape
        if not (0 <= row < n_along and 0 <= column < n_across):
            return None
        if not bool(self._water[row, column]):
            return None
        return row, column
