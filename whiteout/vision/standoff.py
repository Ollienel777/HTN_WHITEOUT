"""Where to sit so a contact is actually in the frame.

Issue #109. The quadcopter's gimbal is **not commandable** — both
``MAV_CMD_DO_MOUNT_CONFIGURE`` and ``MAV_CMD_DO_MOUNT_CONTROL`` return
``MAV_RESULT_FAILED`` on the arena, and no mount status is published — so its
camera is fixed at the airframe's attitude and looks roughly at the horizon.

That breaks the obvious way to hold a contact. ``whiteout/policy/search.py``
makes the quadcopter the asset that can stop and stare, and hovering directly
over a contact puts it **straight down**, which is exactly where a
horizon-looking camera cannot see. Flown live, the quadcopter ended up on top
of the vessel and saw nothing.

The geometry
-------------

A camera whose boresight sits at a pitch of ``p`` degrees sees from
``p + vfov/2`` down to ``p - vfov/2``. A target on the water at ground range
``r`` from an aircraft at height ``h`` sits at a **depression angle** of
``atan(h / r)`` below horizontal, so it is in frame only while that depression
stays inside the lower half-field:

    atan(h / r)  <=  vfov/2 - p

Solving for ``r`` gives the **minimum** stand-off — any closer and the contact
drops below the bottom of the frame:

    r_min  =  h / tan(vfov/2 - p)

At 140 m on the quadcopter's 99.4° vertical field, hovering level, that is
about 119 m — so "hover over it" is wrong by the width of the frame, not by a
little.

Why the aim point is not the edge
----------------------------------

:func:`standoff_range_m` targets a **fraction** of the available half-field
rather than its limit. A contact at the very bottom edge leaves at 3 m/s and
is gone; one at half the half-field has room to move and to survive the
aircraft rocking. :data:`DEFAULT_FILL` is that fraction and it is a parameter,
because the right value depends on how steady the airframe is and nothing has
measured that yet.

What this does not model
-------------------------

**Roll.** A rolled airframe swings the frame sideways, and the horizontal
field is the wider one on every camera in the fleet, so the vertical edge is
the binding constraint and roll is left out. It stops being safe to ignore if
a hold ever needs a banked orbit.

**Terrain.** The water is flat and at a known datum, which is the whole reason
the projection works; nothing here is valid over ground that rises.
"""

from __future__ import annotations

import math

from whiteout.geo import GeoPoint, LocalPoint, geodetic_to_local, local_to_geodetic
from whiteout.vision.camera import CameraModel, VisionError

__all__ = ["DEFAULT_FILL", "standoff_point", "standoff_range_m"]

#: How much of the available half-field to aim for. Half leaves the contact
#: room to move and the airframe room to rock; 1.0 would put it exactly on
#: the frame edge, where 3 m/s takes it out.
DEFAULT_FILL = 0.5


def standoff_range_m(
    camera: CameraModel,
    height_m: float,
    *,
    boresight_pitch_deg: float = 0.0,
    fill: float = DEFAULT_FILL,
) -> float:
    """Ground range at which a contact sits nicely inside the frame.

    :param camera: the asset's published intrinsics; only the vertical field
        is used.
    :param height_m: the camera's height above the water plane.
    :param boresight_pitch_deg: where the camera points, negative downward.
        The airframe's pitch, since the gimbal cannot be commanded.
    :param fill: fraction of the available half-field to aim for.
    :raises VisionError: if the camera cannot see the water at all — a
        boresight pitched so far **up** that even the bottom edge of the frame
        is above the horizon. No range fixes that; the aircraft has to pitch
        down or the contact cannot be held.

    The returned range is where the contact sits at ``fill`` of the way from
    the boresight to the bottom edge. It shrinks as the aircraft climbs
    *toward* the contact's depression and grows with height.
    """
    if height_m <= 0.0:
        raise VisionError(
            f"a camera at or below the water plane cannot see it: height {height_m!r} m"
        )
    if not 0.0 < fill <= 1.0:
        raise VisionError(f"fill must lie in (0, 1], got {fill!r}")
    # Angle from the boresight down to the bottom edge of the frame.
    half_field = camera.vfov_deg / 2.0
    # Depression of the bottom edge below horizontal. A boresight pitched up
    # (positive) eats into it.
    available = half_field - boresight_pitch_deg
    if available <= 0.0:
        raise VisionError(
            f"{camera.name}: pitched {boresight_pitch_deg:.1f}° with a "
            f"{camera.vfov_deg:.1f}° vertical field, the whole frame is above the "
            f"horizon and no stand-off puts a contact in it"
        )
    depression_deg = available * fill
    return height_m / math.tan(math.radians(depression_deg))


def standoff_point(contact: GeoPoint, observer: GeoPoint, range_m: float) -> GeoPoint:
    """A point ``range_m`` from ``contact``, on the side the observer is already.

    Placing the stand-off on the observer's own bearing means the aircraft
    closes to the ring rather than flying across the contact to reach the far
    side of it — which would overfly the thing it is trying to watch, and lose
    it on the way past.

    An observer already at the contact's position has no bearing to hold, so
    the stand-off is placed due north of it. That is arbitrary and it is
    better than dividing by zero; it only happens when the hold has been
    started from directly overhead, which is the case this whole module
    exists to stop.
    """
    if not math.isfinite(range_m) or range_m <= 0.0:
        raise VisionError(f"stand-off range must be finite and positive, got {range_m!r}")
    offset = geodetic_to_local(contact, observer)
    distance = math.hypot(offset.east_m, offset.north_m)
    if distance < 1.0:
        return local_to_geodetic(contact, LocalPoint(east_m=0.0, north_m=range_m))
    scale = range_m / distance
    return local_to_geodetic(
        contact,
        LocalPoint(east_m=offset.east_m * scale, north_m=offset.north_m * scale),
    )
