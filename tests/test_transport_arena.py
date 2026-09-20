"""The arena adapter, exercised without an arena.

Issue #23, and the tower half of #68. The live link needs a WireGuard network
and a running simulator, neither of which CI has, so what is tested here is
everything on our side of the socket: configuration, the refusals, the pose
mapping, and which command shape each vehicle class gets.

The conformance suite (#9) covers the parts that need a link, and skips this
transport by name when it cannot connect — see ``transport_conformance.unavailable``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from whiteout.transport import TRANSPORT_FACTORIES, create_transport
from whiteout.transport.arena import (
    DEFAULT_ROSTER,
    ENDPOINT_ENV,
    ROSTER_ENV,
    ArenaTransport,
    AssetLink,
    roster_from_env,
)
from whiteout.transport.base import TRANSPORT_METHODS, TransportError
from whiteout.types import FleetIntent, WaypointIntent

# -- configuration ----------------------------------------------------------


def test_the_arena_is_registered_and_constructible() -> None:
    assert "arena" in TRANSPORT_FACTORIES
    assert all(hasattr(ArenaTransport, name) for name in TRANSPORT_METHODS)


def test_an_unset_endpoint_refuses_at_connect_not_at_construction() -> None:
    """SPEC.md section 7 makes this an adapter's normal connect-time refusal.

    Raising from the constructor instead would make the arena unconstructible
    on any machine that is not the arena's, which is every machine but one —
    and the conformance suite could then not even name it to skip it.
    """
    transport = ArenaTransport(host=None, roster=DEFAULT_ROSTER)
    with pytest.raises(TransportError) as raised:
        transport.connect()
    assert ENDPOINT_ENV in str(raised.value)


def test_a_seed_is_refused_rather_than_honoured_silently() -> None:
    """The arena is Dominion's world, not one we lay out.

    Accepting a seed would let a run be reported as reproducible when nothing
    about it was.
    """
    with pytest.raises(TransportError) as raised:
        create_transport("arena", seed=7)
    assert "seed" in str(raised.value)


def test_the_default_roster_is_the_observed_one() -> None:
    by_id = {a.asset_id: a for a in DEFAULT_ROSTER}
    assert by_id["quadcopter"].port == 14550
    assert by_id["fixed-wing"].port == 14560
    assert by_id["tower-1"].port == 14580
    assert by_id["tower-2"].port == 14590
    assert {a.system_id for a in DEFAULT_ROSTER} == {1, 2, 4, 5}
    # The boat/rover role runs an idle container; driving it would hang.
    assert "rover" not in by_id and "boat" not in by_id


def test_the_roster_is_data_so_tower_siting_is_not_a_code_change(
    tmp_path: Any,
) -> None:
    """#68: placement is data, not a hardcode."""
    path = tmp_path / "roster.json"
    path.write_text(
        json.dumps([{"asset_id": "tower-1", "cls": "tower", "port": 14580, "system_id": 4}]),
        encoding="utf-8",
    )
    roster = roster_from_env({ROSTER_ENV: str(path)})
    assert roster == (AssetLink("tower-1", "tower", 14580, 4),)


def test_no_roster_file_means_the_default() -> None:
    assert roster_from_env({}) == DEFAULT_ROSTER


def test_a_malformed_roster_raises_rather_than_falling_back(tmp_path: Any) -> None:
    """Falling back would drive the wrong assets and look like it worked."""
    path = tmp_path / "roster.json"
    path.write_text('[{"asset_id": "tower-1"}]', encoding="utf-8")
    with pytest.raises(TransportError) as raised:
        roster_from_env({ROSTER_ENV: str(path)})
    assert "malformed" in str(raised.value)


def test_a_missing_roster_file_raises(tmp_path: Any) -> None:
    with pytest.raises(TransportError):
        roster_from_env({ROSTER_ENV: str(tmp_path / "nope.json")})


# -- the refusals that protect the seam -------------------------------------


def test_observe_and_command_refuse_before_connect() -> None:
    transport = ArenaTransport(host="10.0.0.1", roster=DEFAULT_ROSTER)
    with pytest.raises(TransportError):
        transport.observe()
    with pytest.raises(TransportError):
        transport.command(FleetIntent(t=1.0, intents=()))


def test_close_is_safe_before_connect_and_refuses_reconnect() -> None:
    transport = ArenaTransport(host="10.0.0.1", roster=DEFAULT_ROSTER)
    transport.close()
    transport.close()
    with pytest.raises(TransportError):
        transport.connect()


# -- pose mapping and command dispatch, against a fake link -----------------


class _Message:
    def __init__(self, kind: str, **fields: Any) -> None:
        self._kind = kind
        self.__dict__.update(fields)

    def get_type(self) -> str:
        return self._kind


class _FakeMav:
    """Records what would have gone on the wire."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, tuple[Any, ...]]] = []

    def set_position_target_global_int_send(self, *args: Any) -> None:
        self.sent.append(("position_target", args))

    def command_long_send(self, *args: Any) -> None:
        self.sent.append(("command_long", args))

    def set_mode_send(self, *args: Any) -> None:
        self.sent.append(("set_mode", args))

    def heartbeat_send(self, *args: Any) -> None:
        self.sent.append(("heartbeat", args))


class _FakeConn:
    def __init__(self) -> None:
        self.mav = _FakeMav()

    def recv_match(self, **_: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _wired(cls: str = "quad") -> tuple[ArenaTransport, Any]:
    """A transport with one asset, already 'up', holding a fake connection."""
    asset = AssetLink("quadcopter" if cls != "tower" else "tower-1", cls, 14550, 1)
    transport = ArenaTransport(host="10.0.0.1", roster=(asset,))
    from whiteout.transport.arena import _Link

    link = _Link(asset, "10.0.0.1")
    link.conn = _FakeConn()
    link.position = _Message(
        "GLOBAL_POSITION_INT",
        lat=719958319,
        lon=-948393259,
        relative_alt=60000,
        vx=300,
        vy=400,
        hdg=9000,
    )
    link.hud = _Message("VFR_HUD", groundspeed=5.5)
    transport._links[asset.asset_id] = link
    transport._up = True
    return transport, link


def test_a_pose_carries_degrees_and_an_altitude_in_metres() -> None:
    transport, _ = _wired()
    observation = transport.observe()
    (pose,) = observation.poses
    assert pose.lat == pytest.approx(71.9958319)
    assert pose.lon == pytest.approx(-94.8393259)
    assert pose.z == pytest.approx(60.0)
    assert pose.speed == pytest.approx(5.5)
    assert pose.cls == "quad"


def test_the_tick_clock_is_ours_and_strictly_increases() -> None:
    """Never the autopilot's: two polls of a sampled clock can collide."""
    transport, _ = _wired()
    stamps = [transport.observe().t for _ in range(4)]
    assert stamps == sorted(set(stamps))
    pairs = zip(stamps, stamps[1:], strict=False)
    assert all(later > earlier for earlier, later in pairs)


def test_measured_t_is_none_because_the_offset_is_not_established() -> None:
    """An unconverted time_boot_ms would pass every check and mean nothing."""
    transport, _ = _wired()
    (pose,) = transport.observe().poses
    assert pose.measured_t is None


def test_an_asset_that_has_not_reported_yet_is_silence_not_a_fault() -> None:
    transport, link = _wired()
    link.position = None
    observation = transport.observe()
    assert observation.poses == ()


def test_a_vehicle_gets_a_guided_position_target() -> None:
    transport, link = _wired("quad")
    transport.command(
        FleetIntent(
            t=1.0,
            intents=(
                WaypointIntent(
                    asset_id="quadcopter",
                    t=1.0,
                    target_lat=72.0,
                    target_lon=-94.8,
                    target_z=80.0,
                    speed=5.0,
                    reason="sweep",
                    task_id="task-1",
                ),
            ),
        )
    )
    kinds = [kind for kind, _ in link.conn.mav.sent]
    assert "position_target" in kinds
    (_, args) = next(a for a in link.conn.mav.sent if a[0] == "position_target")
    assert args[5] == int(72.0 * 1e7)
    assert args[6] == int(-94.8 * 1e7)


def test_a_tower_is_aimed_rather_than_sent_a_waypoint() -> None:
    """A tower cannot move, so its intent is a bearing to look along."""
    transport, link = _wired("tower")
    transport.command(
        FleetIntent(
            t=1.0,
            intents=(
                WaypointIntent(
                    asset_id="tower-1",
                    t=1.0,
                    target_lat=72.05,
                    target_lon=-94.80,
                    target_z=0.0,
                    speed=0.0,
                    reason="watch",
                    task_id="task-2",
                ),
            ),
        )
    )
    kinds = [kind for kind, _ in link.conn.mav.sent]
    assert "position_target" not in kinds
    assert "command_long" in kinds


def test_a_tower_refuses_to_launch() -> None:
    transport, _ = _wired("tower")
    with pytest.raises(TransportError) as raised:
        transport.arm_and_launch("tower-1")
    assert "tower" in str(raised.value)


def test_arm_and_takeoff_are_sent_without_a_round_trip_between_them() -> None:
    """Dominion's deck: arming holds for about three seconds.

    Waiting for an ack between the two misses the window often enough to look
    like a broken vehicle, so both commands are on the wire back to back.
    """
    transport, link = _wired("quad")
    transport.arm_and_launch("quadcopter", altitude_m=60.0)
    kinds = [kind for kind, _ in link.conn.mav.sent]
    assert kinds == ["set_mode", "command_long", "command_long"]


def test_scan_is_available_as_the_do_nothing_clever_fallback() -> None:
    transport, link = _wired("tower")
    transport.scan("tower-1")
    assert [kind for kind, _ in link.conn.mav.sent] == ["set_mode"]


def test_servo_one_is_pan_and_servo_two_is_tilt() -> None:
    transport, link = _wired("tower")
    transport.set_servo("tower-1", 2, 1200)
    (_, args) = link.conn.mav.sent[-1]
    assert args[4] == pytest.approx(2.0)
    assert args[5] == pytest.approx(1200.0)


def test_commanding_an_unknown_asset_is_refused_by_name() -> None:
    transport, _ = _wired()
    with pytest.raises(TransportError) as raised:
        transport.set_servo("tower-9", 1, 1500)
    assert "tower-9" in str(raised.value)


# -- the bearing and PWM helpers --------------------------------------------


@pytest.mark.parametrize(
    ("to_lat", "to_lon", "expected"),
    [(73.0, -94.8393259, 0.0), (71.0, -94.8393259, 180.0)],
)
def test_aiming_uses_the_one_converter(to_lat: float, to_lon: float, expected: float) -> None:
    """The adapter does no projection of its own — geo.py owns that (#86)."""
    from whiteout.geo import GeoPoint, bearing_deg

    here = GeoPoint(71.9958319, -94.8393259)
    there = GeoPoint(to_lat, to_lon)
    assert bearing_deg(here, there) == pytest.approx(expected, abs=0.5)


def test_the_pwm_mapping_is_monotonic_and_stays_in_band() -> None:
    """A placeholder calibration, and asserted only as far as it is honest.

    The real map from PWM to pan angle is the tracker's own configuration and
    has to be measured against the camera view (#68). What can be asserted
    without that measurement is that this sweeps the band and never leaves it.
    """
    from whiteout.transport.arena import _pwm_for_bearing

    values = [_pwm_for_bearing(b) for b in range(0, 360, 10)]
    assert values == sorted(values)
    assert min(values) >= 1100
    assert max(values) <= 1900


# -- attitude crosses the seam (#99) ----------------------------------------


def test_a_pose_carries_all_three_angles_so_a_camera_can_be_projected() -> None:
    """``CameraPose`` needs yaw, pitch and roll; a heading alone is not enough.

    Without these the detect-project-post chain has no lat/lon to post, and a
    lat/lon is the only thing the tracks API scores.
    """
    transport, link = _wired()
    link.attitude = _Message("ATTITUDE", yaw=0.0, pitch=-0.1, roll=0.2)
    (pose,) = transport.observe().poses
    assert pose.pitch == pytest.approx(-0.1 * 180.0 / 3.141592653589793, abs=1e-6)
    assert pose.roll == pytest.approx(0.2 * 180.0 / 3.141592653589793, abs=1e-6)


def test_no_attitude_is_none_rather_than_level() -> None:
    """Zero reads as level, which is a measurement we did not take."""
    transport, link = _wired()
    link.attitude = None
    (pose,) = transport.observe().poses
    assert pose.pitch is None
    assert pose.roll is None


def test_the_roster_speaks_the_seam_s_vehicle_classes() -> None:
    """A class outside VEHICLE_CLASSES writes a log that cannot be read back.

    It constructs happily, drives a whole episode, and fails at replay — hours
    after the run that would have to be repeated.
    """
    from whiteout.types import VEHICLE_CLASSES

    for asset in DEFAULT_ROSTER:
        assert asset.cls in VEHICLE_CLASSES, asset


def test_an_arena_pose_survives_the_episode_log_round_trip() -> None:
    from whiteout.types import Pose

    transport, _ = _wired()
    (pose,) = transport.observe().poses
    assert Pose.from_dict(pose.to_dict()) == pose
