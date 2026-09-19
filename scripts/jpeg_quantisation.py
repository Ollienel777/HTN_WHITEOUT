"""Baseline-JPEG luma quantisation, for measuring a detector against a codec.

**The arena publishes JPEG and nothing in this repository has ever decoded a
frame of it.** ``pyproject.toml`` has no image library (see
:mod:`whiteout.vision.imagery`), and :mod:`whiteout.vision.scene` renders
frames that never meet a codec: it adds Gaussian sensor noise as its *last*
step, so adjacent pixels differ by exactly the white noise the detector's
scale estimator assumes. That is the one property real imagery is guaranteed
not to have, and the detector's every gate is a ratio to it.

So this reproduces the part of the codec that matters — the forward DCT of
each 8x8 block, quantisation by the standard luminance table at a quality, and
the inverse — in thirty lines of NumPy. It is not an encoder: no chroma, no
subsampling, no entropy coding, no file. It is the transform that destroys
pixel-to-pixel variation while leaving a hull-sized blob alone, which is the
only part a detector can tell apart.

**It is a stand-in and it is not evidence about the arena's encoder.** What it
is good for is showing that a number measured on uncompressed frames does not
survive compression at all, which it does show: see issue #90, and
``python scripts/score_detector.py --synthetic --jpeg 75``.

Kept in ``scripts/`` on purpose. It is measurement apparatus, nothing in
``whiteout/`` imports it, and it must never become something the product
depends on.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["quantise"]

#: ITU-T T.81 Annex K's example luminance quantisation table, the one every
#: baseline encoder scales from.
_BASE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def _table(quality: int) -> NDArray[np.float64]:
    """``_BASE`` scaled to a quality, the way libjpeg scales it."""
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((_BASE * scale + 50.0) / 100.0), 1.0, 255.0)


def _dct_matrix() -> NDArray[np.float64]:
    """The orthonormal 8-point DCT-II as a matrix, so a block is ``D @ B @ D.T``."""
    k = np.arange(8)
    matrix: NDArray[np.float64] = np.cos(
        (2 * k[None, :] + 1) * k[:, None] * np.pi / 16.0
    ) * np.sqrt(0.25)
    matrix[0] *= 1.0 / np.sqrt(2.0)
    return matrix


_DCT = _dct_matrix()


def quantise(luma: NDArray[np.generic], quality: int = 75) -> NDArray[np.uint8]:
    """Round-trip 8-bit luma through baseline JPEG's luma quantisation.

    :param luma: a 2-D array of 8-bit samples.
    :param quality: 1 to 100, libjpeg's scale. 75 is a common default for an
        MJPEG camera; 95 is close to lossless and 60 is visibly blocky.
    :returns: the same shape, as ``uint8``.

    Frames whose sides are not multiples of eight are edge-padded to the next
    block and cropped back, which is what an encoder does with the last
    partial block.
    """
    if luma.ndim != 2:
        raise ValueError(f"quantise takes a 2-D frame, got shape {luma.shape!r}")
    if not 1 <= quality <= 100:
        raise ValueError(f"quality is 1..100, got {quality!r}")
    height, width = luma.shape
    padded = np.pad(
        np.asarray(luma, dtype=np.float64),
        ((0, (-height) % 8), (0, (-width) % 8)),
        mode="edge",
    )
    # Level-shift to centre on zero, the way the standard specifies, so that
    # the DC coefficient is a deviation rather than a large constant.
    padded -= 128.0
    blocks = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3)
    table = _table(quality)
    coefficients = _DCT @ blocks @ _DCT.T
    coefficients = np.rint(coefficients / table) * table
    restored = _DCT.T @ coefficients @ _DCT
    restored = restored.transpose(0, 2, 1, 3).reshape(padded.shape)
    return np.clip(np.rint(restored[:height, :width] + 128.0), 0.0, 255.0).astype(np.uint8)
