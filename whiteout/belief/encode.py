"""The belief field as bytes, so the viewer can draw it.

Issue #124. The viewer reads the episode log and nothing else, so until this
module existed the field's per-cell probability lived only inside the process
and the erosion that makes the fleet's reasoning legible (#13, and #20's
centrepiece) had nothing behind it to render.

This is a **rendering channel and not a state channel.** The quantisation
below is lossy, deliberately, and nothing decodes a frame back into a
:class:`~whiteout.belief.grid.ChannelBeliefGrid`. The decoders here exist for
the tests that hold the round trip to its promise and for a reader who wants
to check a committed episode by hand; a scorer or an estimator that reached
for them would be reasoning about a rounded copy of a number it could have
had exactly.

The size of it, measured on the default grid
---------------------------------------------

``DEFAULT_STRAIT`` at 100 m resolution is 63 × 16 cells, **850 of them
water**. One byte each is 850 B, which base64 carries in 1136 characters —
call it 1.2 kB a tick with the JSON around it, and **about 480 kB over a
400-tick episode**, against ``ARENA.md`` and #43's 25 MB cap on every
committed episode combined. The same 850 cells written as JSON floats is
about 17 kB a tick and **6.8 MB over the episode**, fourteen times over.

The geometry — the water mask and the 64 × 17 cell-corner lattice — is
another 11.8 kB, and it is **episode-constant**, so it is written on the first
frame only. Restating it on all 400 would cost 4.7 MB to say the same thing
400 times. ``tests/test_belief_encode.py`` re-measures every figure in this
paragraph.
"""

from __future__ import annotations

import base64

import numpy as np
import numpy.typing as npt

from whiteout.belief.grid import ChannelBeliefGrid
from whiteout.types import QUANTISATION_STEPS, BeliefFrame, BeliefGeometry

__all__ = [
    "belief_frame",
    "belief_geometry",
    "decode_cells",
    "decode_corners",
    "decode_water",
]

_FloatArray = npt.NDArray[np.float64]
_BoolArray = npt.NDArray[np.bool_]

#: Big-endian ``float32``, so the bytes do not depend on the machine that
#: wrote them. The gate compares two runs of the same seed byte for byte.
_CORNER_DTYPE = np.dtype(">f4")


def belief_geometry(grid: ChannelBeliefGrid) -> BeliefGeometry:
    """The grid's cell layout, as the log's :class:`BeliefGeometry`."""
    water = grid.water_mask()
    lat, lon = grid.corner_positions()
    corners = np.stack((lat, lon), axis=-1).astype(_CORNER_DTYPE)
    return BeliefGeometry(
        shape=grid.shape,
        water=_b64(np.packbits(water).tobytes()),
        corners=_b64(corners.tobytes()),
    )


def belief_frame(grid: ChannelBeliefGrid, t: float, *, include_geometry: bool) -> BeliefFrame:
    """Quantise ``grid``'s water cells into a log frame at time ``t``.

    ``include_geometry`` belongs to the caller because it is a fact about the
    *episode* and not about the field: the layout goes on the first frame of
    the log and nowhere else.
    """
    values = grid.probabilities()[grid.water_mask()]
    scale = float(values.max())
    if scale > 0.0:
        codes = np.rint(values * (QUANTISATION_STEPS / scale)).astype(np.uint8)
    else:
        # A field with no mass anywhere is not reachable through the grid,
        # which normalises after every update. Encoded rather than refused, so
        # that a hand-built field in somebody's test draws as empty water
        # instead of raising out of the log writer.
        codes = np.zeros(values.shape, dtype=np.uint8)
    return BeliefFrame(
        t=t,
        water_cells=int(values.size),
        scale=scale,
        cells=_b64(codes.tobytes()),
        geometry=belief_geometry(grid) if include_geometry else None,
    )


def decode_cells(frame: BeliefFrame) -> _FloatArray:
    """The probability of each water cell, in the frame's own order."""
    codes = np.frombuffer(base64.b64decode(frame.cells, validate=True), dtype=np.uint8)
    return codes.astype(np.float64) * (frame.scale / QUANTISATION_STEPS)


def decode_water(geometry: BeliefGeometry) -> _BoolArray:
    """The water mask, unpacked to the grid's own ``(along, across)`` shape."""
    along, across = geometry.shape
    packed = np.frombuffer(base64.b64decode(geometry.water, validate=True), dtype=np.uint8)
    return np.unpackbits(packed, count=along * across).astype(bool).reshape(along, across)


def decode_corners(geometry: BeliefGeometry) -> tuple[_FloatArray, _FloatArray]:
    """Latitude and longitude of the cell-corner lattice, as two arrays."""
    along, across = geometry.shape
    raw = np.frombuffer(base64.b64decode(geometry.corners, validate=True), dtype=_CORNER_DTYPE)
    pairs = raw.reshape(along + 1, across + 1, 2).astype(np.float64)
    return pairs[..., 0], pairs[..., 1]


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")
