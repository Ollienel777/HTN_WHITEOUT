"""The published camera intrinsics, as a pinhole model.

``hackathon/ARENA.md`` §4 is the source, and it is the whole source: Dominion
Dynamics publish a horizontal and a vertical field of view per asset, plus a
frame size. Nothing else about the optics is given — no focal length, no
principal point, no distortion coefficients.

| asset | HFOV | VFOV | resolution |
|---|---|---|---|
| quadcopter | 114.6° | 99.4° | 640×480 |
| fixed-wing | 69.0° | 42.6° | 640×360 |
| towers | 60.0° | 36.1° | 640×360 |

A field of view and a frame size determine a pinhole camera exactly, so that
is the model:

.. math::

    f_x = \\frac{W/2}{\\tan(\\mathrm{HFOV}/2)}
    \\qquad
    f_y = \\frac{H/2}{\\tan(\\mathrm{VFOV}/2)}

with the principal point at the exact centre of the frame, :math:`(W/2, H/2)`.

**The two focal lengths are kept separate rather than averaged.** For every
asset the published pair is *nearly* but not exactly consistent with square
pixels — the quadcopter's is 205.2 against 203.8, about 0.7% apart — and
collapsing them to one number would bend the frame edges by roughly a pixel
and a half. The pair as published is the more faithful model, and it costs a
field.

**Distortion is not modelled**, because none is published. If the arena's
renders turn out to be noticeably barrelled, that is a correction applied to
:func:`~whiteout.vision.projection.pixel_ray_enu`'s input and nothing else has
to change.

Pixel coordinates are floats with the origin at the **top-left corner of the
frame**, ``x`` to the right and ``y`` downward, so the frame spans ``[0, W] ×
[0, H]`` and its centre is exactly ``(W/2, H/2)``. A detector reporting an
integer pixel *index* ``(i, j)`` should pass ``(i + 0.5, j + 0.5)``: the centre
of that pixel. Getting this wrong costs half a pixel, which at the tower's
554 px focal length and 2 km range is about 1.8 m — small, but it is free to
get right.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

__all__ = [
    "CAMERAS",
    "FIXED_WING_CAMERA",
    "QUADCOPTER_CAMERA",
    "TOWER_CAMERA",
    "CameraModel",
    "VisionError",
]


class VisionError(ValueError):
    """Base of every error this package raises.

    ``whiteout.vision`` is a leaf: it reads frames and turns a pixel into a
    coordinate. Everything that can go wrong is either a malformed input (a
    field of view of zero, a truncated MJPEG part) or a geometry that has no
    answer (a ray aimed at the sky). Callers upstream — the detector, the
    track-maintenance loop — want to catch the lot with one name, so there is
    one base and the specific classes derive from it.
    """


@dataclass(frozen=True, slots=True)
class CameraModel:
    """A pinhole camera, built from a published field of view and frame size.

    :param name: the asset class this camera belongs to, for diagnostics.
    :param hfov_deg: horizontal field of view, degrees, the full angle.
    :param vfov_deg: vertical field of view, degrees, the full angle.
    :param width: frame width in pixels.
    :param height: frame height in pixels.

    Both fields of view are the *full* angle, edge to edge, as
    ``ARENA.md`` §4 gives them — not the half-angle. So a pixel on the left
    or right edge of the frame sits exactly ``hfov_deg / 2`` off the
    boresight, and that identity is what
    ``tests/test_vision_projection.py`` uses to pin :attr:`fx` without a
    second implementation of it.
    """

    name: str
    hfov_deg: float
    vfov_deg: float
    width: int
    height: int

    def __post_init__(self) -> None:
        for field_name, value in (("hfov_deg", self.hfov_deg), ("vfov_deg", self.vfov_deg)):
            if not 0.0 < value < 180.0:
                raise VisionError(
                    f"{self.name}: {field_name} must lie in (0, 180) degrees, got {value!r}"
                )
        for field_name, size in (("width", self.width), ("height", self.height)):
            if size <= 0:
                raise VisionError(f"{self.name}: {field_name} must be positive, got {size!r}")

    @property
    def fx(self) -> float:
        """Horizontal focal length in pixels, ``(W/2) / tan(HFOV/2)``."""
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        """Vertical focal length in pixels, ``(H/2) / tan(VFOV/2)``."""
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    @property
    def cx(self) -> float:
        """Principal point, horizontal: the exact centre of the frame."""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """Principal point, vertical: the exact centre of the frame."""
        return self.height / 2.0

    def at_frame_size(self, width: int, height: int) -> CameraModel:
        """The same optics, sampled onto a different raster.

        The arena does not serve the resolutions ``ARENA.md`` §4 publishes.
        Measured off the live cameras on 2026-09-20: the quadcopter gimbal
        streams 960x720 where the deck says 640x480, and the fixed-wing and
        both towers stream 1280x720 where it says 640x360. Every one of those
        is a pure upscale — **the aspect ratio is unchanged** — so the field
        of view is still the published one and only the sampling differs.

        That matters because everything derived here scales linearly with the
        raster at a fixed field of view: ``fx``, ``fy``, ``cx`` and ``cy`` are
        all proportional to ``width`` or ``height``. A camera at twice the
        pixels is the same camera, and a pixel at the frame edge is still
        exactly ``hfov_deg / 2`` off the boresight.

        **A change of aspect ratio is refused**, because then the published
        pair of fields of view no longer describes the frame: one axis has
        been cropped or stretched, the focal lengths stop agreeing, and every
        projection off that frame would be quietly wrong rather than loudly
        broken. Silence is the wrong failure here — ``ARENA.md`` §5 scores
        accuracy, and a confidently wrong lat/lon costs more than no lat/lon.
        """
        if width <= 0 or height <= 0:
            raise VisionError(f"{self.name}: frame size must be positive, got {width!r}x{height!r}")
        want = self.width / self.height
        got = width / height
        if abs(want - got) > 1e-3 * want:
            raise VisionError(
                f"{self.name}: {width}x{height} has aspect {got:.4f}, but this camera's "
                f"{self.width}x{self.height} has {want:.4f}. A pure rescale keeps the "
                f"published fields of view; a change of aspect means one axis was cropped "
                f"or stretched and {self.hfov_deg}/{self.vfov_deg} no longer describe it"
            )
        if (width, height) == (self.width, self.height):
            return self
        return replace(self, width=width, height=height)

    def normalised(self, px: float, py: float) -> tuple[float, float]:
        """Return the pixel as a direction in the camera frame at ``z = 1``.

        ``((px - cx) / fx, (py - cy) / fy)``. The camera frame is the usual
        computer-vision one — ``+x`` right, ``+y`` down, ``+z`` along the
        boresight — so the full direction is that pair with a ``1`` appended,
        and :func:`~whiteout.vision.projection.pixel_ray_enu` rotates it into
        the world.
        """
        if not (math.isfinite(px) and math.isfinite(py)):
            raise VisionError(f"{self.name}: pixel must be finite, got ({px!r}, {py!r})")
        return (px - self.cx) / self.fx, (py - self.cy) / self.fy


#: ``ARENA.md`` §4. The widest camera in the fleet, and the one that stares.
QUADCOPTER_CAMERA = CameraModel("quadcopter", 114.6, 99.4, 640, 480)

#: ``ARENA.md`` §4. The sweep asset's FPV camera.
FIXED_WING_CAMERA = CameraModel("fixed-wing", 69.0, 42.6, 640, 360)

#: ``ARENA.md`` §4. Both towers carry the same EO camera.
TOWER_CAMERA = CameraModel("tower", 60.0, 36.1, 640, 360)

#: By asset class. ``tower-1`` and ``tower-2`` share :data:`TOWER_CAMERA`, so
#: the key is the class and not the asset id; the caller holds the mapping
#: from one to the other.
CAMERAS: Mapping[str, CameraModel] = MappingProxyType(
    {
        "quadcopter": QUADCOPTER_CAMERA,
        "fixed-wing": FIXED_WING_CAMERA,
        "tower": TOWER_CAMERA,
    }
)
