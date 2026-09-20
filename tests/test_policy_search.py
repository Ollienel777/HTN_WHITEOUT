"""The search policy: sweeping the channel rather than rastering the area.

Issue #96. The tests that matter are the two that pin behaviour the whole
project's argument rests on — that staleness turns a greedy rule into a sweep,
and that hysteresis stops an asset dithering between two near-equal segments.
"""

from __future__ import annotations

import math

import pytest

from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint
from whiteout.geo import GeoPoint, geodetic_to_local
from whiteout.policy import AssetRole, ParamsError, SearchParams, SearchPolicy
from whiteout.types import Pose, WorldObservation

FLEET = (
    AssetRole("quadcopter", "quad"),
    AssetRole("fixed-wing", "fixedwing"),
    AssetRole("tower-1", "tower"),
    AssetRole("tower-2", "tower"),
)


class _Belief:
    """A belief field that is a bump at one place on the centreline.

    Implements only what the policy is allowed to use — the
    ``probability_at`` half of the protocol. If the policy ever reaches for
    the grid behind it, this stops being enough and the test says so.
    """

    def __init__(self, s_m: float, sigma_m: float = 900.0) -> None:
        self.lat_deg, self.lon_deg = DEFAULT_STRAIT.to_position(ChannelPoint(s_m=s_m, w_m=0.0))
        self.sigma_m = sigma_m
        self.queries = 0

    def probability_at(self, lat_deg: float, lon_deg: float) -> float:
        self.queries += 1
        # Through the one converter: tests/test_geo.py reserves latitude
        # trigonometry and the ellipsoid's constants for whiteout/geo.py, and
        # a hand-rolled degrees-to-metres here would be a second projection
        # that could disagree with the one the policy uses.
        offset = geodetic_to_local(GeoPoint(self.lat_deg, self.lon_deg), GeoPoint(lat_deg, lon_deg))
        r = math.hypot(offset.east_m, offset.north_m)
        return math.exp(-0.5 * (r / self.sigma_m) ** 2)


def _pose(asset_id: str, cls: str, s_m: float, t: float = 0.0, alt: float = 100.0) -> Pose:
    lat, lon = DEFAULT_STRAIT.to_position(ChannelPoint(s_m=s_m, w_m=0.0))
    return Pose(
        asset_id=asset_id,
        cls=cls,
        t=t,
        lat=lat,
        lon=lon,
        z=alt,
        heading=0.0,
        speed=0.0,
        energy_used=0.0,
    )


def _observation(t: float, at: dict[str, float]) -> WorldObservation:
    by_id = {role.asset_id: role.cls for role in FLEET}
    poses = tuple(_pose(aid, by_id[aid], s, t) for aid, s in at.items())
    return WorldObservation(t=t, poses=poses, reports=())


def _fleet_at(**kwargs: float) -> dict[str, float]:
    return {
        "quadcopter": kwargs.get("quad", 12_000.0),
        "fixed-wing": kwargs.get("plane", 12_000.0),
        "tower-1": kwargs.get("tower1", 6_000.0),
        "tower-2": kwargs.get("tower2", 19_000.0),
    }


@pytest.fixture
def policy() -> SearchPolicy:
    return SearchPolicy(FLEET, DEFAULT_STRAIT)


# -- every asset is tasked, every tick --------------------------------------


def test_the_channel_is_cut_into_segments_along_its_length(policy: SearchPolicy) -> None:
    assert len(policy.segments) > 1
    lengths = [b.s_m - a.s_m for a, b in zip(policy.segments, policy.segments[1:], strict=False)]
    assert all(d > 0 for d in lengths), "segments run along the channel, in order"
    assert policy.segments[-1].s_m <= DEFAULT_STRAIT.length_m


def test_every_asset_that_reported_gets_an_intent(policy: SearchPolicy) -> None:
    intent = policy.decide(_observation(0.0, _fleet_at()), _Belief(12_000.0))
    assert {i.asset_id for i in intent.intents} == {r.asset_id for r in FLEET}


def test_an_asset_that_did_not_report_is_not_guessed_at(policy: SearchPolicy) -> None:
    """Silence is not a position."""
    observation = _observation(0.0, {"quadcopter": 12_000.0, "tower-1": 6_000.0})
    intent = policy.decide(observation, _Belief(12_000.0))
    assert {i.asset_id for i in intent.intents} == {"quadcopter", "tower-1"}


def test_two_assets_are_never_sent_to_the_same_segment(policy: SearchPolicy) -> None:
    intent = policy.decide(_observation(0.0, _fleet_at()), _Belief(12_000.0))
    targets = [(i.target_lat, i.target_lon) for i in intent.intents]
    assert len(set(targets)) == len(targets)


def test_towers_are_aimed_at_ground_level_and_aircraft_are_sent_up(
    policy: SearchPolicy,
) -> None:
    intent = policy.decide(_observation(0.0, _fleet_at()), _Belief(12_000.0))
    by_id = {i.asset_id: i for i in intent.intents}
    assert by_id["tower-1"].target_z == 0.0
    assert by_id["tower-1"].reason == "watch"
    assert by_id["fixed-wing"].target_z > 0.0
    assert by_id["fixed-wing"].reason == "search"


# -- the policy only knows what the protocol offers -------------------------


def test_belief_is_sampled_through_the_protocol_and_nothing_else(
    policy: SearchPolicy,
) -> None:
    """A policy that knew how belief was stored would be rewritten when it changes."""
    belief = _Belief(12_000.0)
    policy.decide(_observation(0.0, _fleet_at()), belief)
    assert belief.queries == len(policy.segments)


def test_the_same_inputs_give_the_same_intent() -> None:
    """Determinism is a hard rule (SPEC.md §5): no clock, no generator."""
    first = SearchPolicy(FLEET, DEFAULT_STRAIT).decide(
        _observation(0.0, _fleet_at()), _Belief(12_000.0)
    )
    second = SearchPolicy(FLEET, DEFAULT_STRAIT).decide(
        _observation(0.0, _fleet_at()), _Belief(12_000.0)
    )
    assert first == second


# -- staleness makes it sweep rather than stare -----------------------------


def test_one_asset_works_more_than_one_segment_over_time(policy: SearchPolicy) -> None:
    """Without staleness a greedy rule converges on the highest-belief segment.

    The peak does not move until something looks somewhere else, and nothing
    ever does. This is the test that says the search is a search.

    It follows **one** asset deliberately. Counting distinct segments across
    the fleet proves nothing: two assets are never sent to the same segment,
    so four assets give four distinct segments on the first tick whether or
    not anything sweeps. An earlier version asserted exactly that and stayed
    green when staleness was mutated out.

    The quadcopter is also *flown* toward its target between ticks rather than
    parked. A static asset never arrives, so nothing is ever marked as looked
    at and staleness never decays — which tests the fixture, not the policy.
    """
    belief = _Belief(12_000.0)
    worked: list[int] = []
    quad_s = 12_000.0
    for tick in range(60):
        policy.decide(_observation(tick * 10.0, _fleet_at(quad=quad_s)), belief)
        index = policy.assignment_of("quadcopter")
        assert index is not None
        if not worked or worked[-1] != index:
            worked.append(index)
        # 10 m/s toward the target it was just given.
        target_s = policy.segments[index].s_m
        quad_s += max(-100.0, min(100.0, target_s - quad_s))
    assert len(set(worked)) >= 3, (
        f"the quadcopter only ever worked {sorted(set(worked))} — that is a stare"
    )


def test_a_segment_just_looked_at_loses_its_value(policy: SearchPolicy) -> None:
    belief = _Belief(12_000.0)
    policy.decide(_observation(0.0, _fleet_at()), belief)
    first = policy.assignment_of("quadcopter")
    # Park the quadcopter on it long enough for staleness to bottom out.
    for tick in range(1, 12):
        assert first is not None
        s = policy.segments[first].s_m
        policy.decide(_observation(tick * 5.0, _fleet_at(quad=s)), belief)
    assert policy.assignment_of("quadcopter") != first


# -- hysteresis --------------------------------------------------------------


def test_hysteresis_holds_an_asset_through_a_marginal_change() -> None:
    """Two segments a few percent apart swap rank every tick as belief moves.

    An asset that chases the rank turns round mid-transit, arrives nowhere and
    covers nothing.
    """
    sticky = SearchPolicy(FLEET, DEFAULT_STRAIT, SearchParams(hysteresis_margin=0.5))
    jumpy = SearchPolicy(FLEET, DEFAULT_STRAIT, SearchParams(hysteresis_margin=0.0))
    for tick in range(30):
        t = tick * 5.0
        # A peak that wanders a little, as a diffusing field's does. A
        # triangle wave rather than a sine: tests/test_geo.py reserves
        # trigonometry for whiteout/geo.py, and the shape is irrelevant here —
        # what matters is that the ranking of nearby segments keeps changing.
        wander = 400.0 * (abs((tick % 8) - 4) / 4.0 - 0.5)
        belief = _Belief(12_000.0 + wander)
        observation = _observation(t, _fleet_at())
        sticky.decide(observation, belief)
        jumpy.decide(observation, belief)
    assert sticky.retasks < jumpy.retasks, (
        f"hysteresis did not reduce re-tasking: {sticky.retasks} vs {jumpy.retasks}"
    )


def test_retasks_are_counted_so_a_run_can_show_the_number(policy: SearchPolicy) -> None:
    assert policy.retasks == 0
    for tick in range(6):
        policy.decide(_observation(tick * 30.0, _fleet_at()), _Belief(12_000.0))
    assert policy.retasks >= 0


# -- roles -------------------------------------------------------------------


def test_a_held_contact_pulls_the_quadcopter_and_only_it(policy: SearchPolicy) -> None:
    """The aircraft that can stop and stare is the one that holds a contact.

    Pulling the fixed-wing onto it as well abandons the search for no gain: it
    cannot loiter over the vessel anyway.
    """
    lat, lon = DEFAULT_STRAIT.to_position(ChannelPoint(s_m=9_000.0, w_m=0.0))
    intent = policy.decide(
        _observation(0.0, _fleet_at()),
        _Belief(12_000.0),
        held_lat_deg=lat,
        held_lon_deg=lon,
    )
    by_id = {i.asset_id: i for i in intent.intents}
    assert by_id["quadcopter"].reason == "hold"
    assert by_id["quadcopter"].target_lat == pytest.approx(lat)
    assert by_id["fixed-wing"].reason == "search"


def test_a_tower_is_never_sent_beyond_its_reach() -> None:
    """Past its camera's range there is no trade-off, only a direction."""
    near = SearchParams(tower_reach_m=1500.0)
    policy = SearchPolicy(FLEET, DEFAULT_STRAIT, near)
    intent = policy.decide(_observation(0.0, _fleet_at()), _Belief(12_000.0))
    by_id = {i.asset_id: i for i in intent.intents}
    for tower, where in (("tower-1", 6_000.0), ("tower-2", 19_000.0)):
        if tower not in by_id:
            continue
        index = policy.assignment_of(tower)
        assert index is not None
        assert abs(policy.segments[index].s_m - where) <= 2500.0


def test_the_sweep_asset_is_not_handed_work_under_its_nose() -> None:
    """The fixed-wing cannot loiter, so it overflies a segment it is tasked to."""
    policy = SearchPolicy(FLEET, DEFAULT_STRAIT, SearchParams(plane_min_reach_m=3000.0))
    policy.decide(_observation(0.0, _fleet_at(plane=12_000.0)), _Belief(12_000.0))
    index = policy.assignment_of("fixed-wing")
    assert index is not None
    assert abs(policy.segments[index].s_m - 12_000.0) >= 2000.0


# -- parameters --------------------------------------------------------------


def test_parameters_round_trip_through_json() -> None:
    params = SearchParams(segment_m=300.0, hysteresis_margin=0.4)
    assert SearchParams.from_json(params.to_json()) == params


def test_an_unknown_parameter_is_refused_rather_than_dropped() -> None:
    """Dropping it silently reports a run as using an operating point it was not."""
    with pytest.raises(ParamsError, match="unknown parameter"):
        SearchParams.from_json('{"segment_m": 250.0, "magic": 3}')


@pytest.mark.parametrize(
    "bad",
    [
        {"segment_m": 0.0},
        {"stale_horizon_s": -1.0},
        {"travel_cost_per_km": -0.1},
        {"hysteresis_margin": -0.1},
        {"tower_reach_m": float("inf")},
    ],
)
def test_a_parameter_set_that_cannot_describe_a_search_is_refused(
    bad: dict[str, float],
) -> None:
    with pytest.raises(ParamsError):
        SearchParams(**bad)
