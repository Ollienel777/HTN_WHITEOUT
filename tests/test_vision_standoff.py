"""Standing off far enough that a contact is in the frame.

Issue #109. The quadcopter's gimbal cannot be commanded, so its camera looks
where the airframe does — and hovering over a contact puts it straight down,
where a horizon-looking camera cannot see it.
"""

from __future__ import annotations

import math

import pytest

from whiteout.geo import GeoPoint, LocalPoint, geodetic_to_local, local_to_geodetic
from whiteout.vision.camera import CAMERAS, VisionError
from whiteout.vision.standoff import DEFAULT_FILL, standoff_point, standoff_range_m

QUAD = CAMERAS["quadcopter"]
CONTACT = GeoPoint(71.9900, -94.8400)


def _range_between(a: GeoPoint, b: GeoPoint) -> float:
    offset = geodetic_to_local(a, b)
    return math.hypot(offset.east_m, offset.north_m)


def _depression_deg(height_m: float, range_m: float) -> float:
    """How far below horizontal a contact at this range sits."""
    return math.degrees(math.atan2(height_m, range_m))


# -- the geometry ------------------------------------------------------------


def test_the_contact_lands_inside_the_vertical_field() -> None:
    """The whole point: at the returned range it is in frame, not below it."""
    for height in (60.0, 140.0, 250.0):
        range_m = standoff_range_m(QUAD, height)
        depression = _depression_deg(height, range_m)
        assert depression < QUAD.vfov_deg / 2.0, (
            f"at {height} m the contact sits {depression:.1f}° down, past the "
            f"{QUAD.vfov_deg / 2:.1f}° bottom edge"
        )


def test_the_contact_is_not_jammed_against_the_frame_edge() -> None:
    """At the very edge, 3 m/s takes it out of frame; aim for room."""
    height = 140.0
    depression = _depression_deg(height, standoff_range_m(QUAD, height))
    assert depression == pytest.approx(QUAD.vfov_deg / 2.0 * DEFAULT_FILL, abs=0.5)


def test_hovering_overhead_is_exactly_what_this_rejects() -> None:
    """Directly above, the contact is 90° down and no camera here sees it."""
    assert _depression_deg(140.0, 0.001) > QUAD.vfov_deg / 2.0
    assert standoff_range_m(QUAD, 140.0) > 100.0


def test_stand_off_grows_with_height() -> None:
    ranges = [standoff_range_m(QUAD, h) for h in (60.0, 140.0, 250.0)]
    assert ranges == sorted(ranges)


def test_a_narrower_camera_must_stand_further_off() -> None:
    """The fixed-wing's 42.6° field needs far more room than the quad's 99.4°."""
    assert standoff_range_m(CAMERAS["fixed-wing"], 140.0) > standoff_range_m(QUAD, 140.0)


def test_pitching_down_lets_the_aircraft_come_closer() -> None:
    level = standoff_range_m(QUAD, 140.0, boresight_pitch_deg=0.0)
    nose_down = standoff_range_m(QUAD, 140.0, boresight_pitch_deg=-20.0)
    assert nose_down < level


def test_a_camera_pitched_above_its_own_field_is_refused() -> None:
    """No stand-off fixes it: the whole frame is above the horizon."""
    with pytest.raises(VisionError, match="above the horizon"):
        standoff_range_m(QUAD, 140.0, boresight_pitch_deg=60.0)


def test_a_camera_at_the_water_plane_is_refused() -> None:
    with pytest.raises(VisionError):
        standoff_range_m(QUAD, 0.0)


def test_a_nonsense_fill_is_refused() -> None:
    with pytest.raises(VisionError):
        standoff_range_m(QUAD, 140.0, fill=0.0)
    with pytest.raises(VisionError):
        standoff_range_m(QUAD, 140.0, fill=1.5)


# -- where the point goes ----------------------------------------------------


def test_the_stand_off_is_on_the_side_the_observer_already_is() -> None:
    """Otherwise the aircraft flies across the contact to reach the far side.

    That overflies the thing it is trying to watch, and loses it on the way.
    """
    observer = local_to_geodetic(CONTACT, LocalPoint(east_m=800.0, north_m=0.0))
    point = standoff_point(CONTACT, observer, 300.0)
    offset = geodetic_to_local(CONTACT, point)
    assert offset.east_m == pytest.approx(300.0, abs=1.0)
    assert offset.north_m == pytest.approx(0.0, abs=1.0)


def test_the_stand_off_is_at_the_range_asked_for() -> None:
    observer = local_to_geodetic(CONTACT, LocalPoint(east_m=-500.0, north_m=900.0))
    point = standoff_point(CONTACT, observer, 250.0)
    assert _range_between(CONTACT, point) == pytest.approx(250.0, abs=1.0)


def test_an_observer_already_overhead_is_pushed_off_rather_than_divided_by() -> None:
    """Arbitrary direction, and better than a zero divide.

    It only happens when a hold was started from directly overhead, which is
    the case this module exists to stop.
    """
    point = standoff_point(CONTACT, CONTACT, 300.0)
    assert _range_between(CONTACT, point) == pytest.approx(300.0, abs=1.0)


def test_a_nonsense_range_is_refused() -> None:
    with pytest.raises(VisionError):
        standoff_point(CONTACT, CONTACT, 0.0)


# -- the two joined ----------------------------------------------------------


def test_the_computed_point_really_does_put_the_contact_in_frame() -> None:
    """End to end, and against the projection's own idea of a frame edge.

    A stand-off is only worth anything if the contact it was computed for is
    inside the vertical field from the point it names.
    """
    height = 140.0
    observer = local_to_geodetic(CONTACT, LocalPoint(east_m=1500.0, north_m=400.0))
    range_m = standoff_range_m(QUAD, height)
    point = standoff_point(CONTACT, observer, range_m)

    actual = _range_between(CONTACT, point)
    depression = _depression_deg(height, actual)
    assert depression < QUAD.vfov_deg / 2.0
    assert depression > 1.0, "and not so far off that it is a speck on the horizon"
