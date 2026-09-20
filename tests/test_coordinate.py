"""The run loop, driven against a list of observations.

Issue #96's follow-on: every part worked and nothing joined them. These tests
are about the joining — the order of a tick, and the two places where the
track's state changes what the fleet is told to do.
"""

from __future__ import annotations

import pytest

from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint
from whiteout.coordinate import Coordinator, NoSightings, TickOutcome
from whiteout.policy import AssetRole
from whiteout.tracks.client import TrackPoster, TracksClient
from whiteout.tracks.maintain import Sighting, TrackHold
from whiteout.tracks.stub import StubTracksServer
from whiteout.types import Pose, WorldObservation

FLEET = (
    AssetRole("quadcopter", "quad"),
    AssetRole("fixed-wing", "fixedwing"),
    AssetRole("tower-1", "tower"),
    AssetRole("tower-2", "tower"),
)
MID = DEFAULT_STRAIT.length_m / 2.0


def _at(s_m: float) -> tuple[float, float]:
    return DEFAULT_STRAIT.to_position(ChannelPoint(s_m=s_m, w_m=0.0))


def _observation(t: float) -> WorldObservation:
    poses = []
    for role, s_m in zip(FLEET, (MID, MID, MID * 0.4, MID * 1.5), strict=True):
        lat, lon = _at(min(s_m, DEFAULT_STRAIT.length_m))
        poses.append(
            Pose(
                asset_id=role.asset_id,
                cls=role.cls,
                t=t,
                lat=lat,
                lon=lon,
                z=100.0,
                heading=0.0,
                speed=0.0,
                energy_used=0.0,
            )
        )
    return WorldObservation(t=t, poses=tuple(poses), reports=())


class _SawItAt:
    """A sighting source that reports the vessel at fixed ticks."""

    def __init__(self, ticks: set[float], s_m: float = MID) -> None:
        self.ticks = ticks
        self.lat, self.lon = _at(s_m)

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        if observation.t not in self.ticks:
            return ()
        return (Sighting(observation.t, self.lat, self.lon, "tower-1"),)


@pytest.fixture
def hold_and_stub():
    with StubTracksServer() as stub:
        poster = TrackPoster(TracksClient(stub.endpoint, backoff_s=0.0))
        try:
            yield TrackHold("Sierra One", poster, post_interval_s=2.0, coast_s=10.0), stub
        finally:
            poster.close()


# -- the shape of a tick ----------------------------------------------------


def test_every_asset_is_tasked_every_tick() -> None:
    coordinator = Coordinator(FLEET)
    outcome = coordinator.tick(_observation(1.0))
    assert isinstance(outcome, TickOutcome)
    assert {i.asset_id for i in outcome.intent.intents} == {r.asset_id for r in FLEET}


def test_the_digest_describes_the_field_the_loop_is_using() -> None:
    coordinator = Coordinator(FLEET)
    digest = coordinator.tick(_observation(1.0)).digest
    assert digest.mass == pytest.approx(1.0, abs=1e-6)
    assert digest.grid_shape[0] > 1
    assert 0.0 <= digest.covered_fraction <= 1.0


def test_coverage_grows_as_the_fleet_sweeps() -> None:
    coordinator = Coordinator(FLEET)
    first = coordinator.tick(_observation(1.0)).digest.covered_fraction
    for tick in range(2, 20):
        last = coordinator.tick(_observation(float(tick))).digest.covered_fraction
    assert last > first


def test_the_same_observations_give_the_same_outcomes() -> None:
    """Determinism is a hard rule: no clock and no generator in the loop."""
    a = Coordinator(FLEET)
    b = Coordinator(FLEET)
    for tick in range(1, 6):
        assert a.tick(_observation(float(tick))) == b.tick(_observation(float(tick)))


def test_dt_is_the_gap_between_ticks_and_not_the_clock_s_reading() -> None:
    """A run whose first observation lands at t=400 must not start flat.

    Two coordinators see the same sightings and the same gaps; only their
    absolute times differ. Their fields must end identical.

    What this does **not** catch, and it was written believing it would: the
    first tick's ``_last_t is None`` guard. A uniform field is a fixed point
    of diffusion, and the field is always uniform on tick one, so ageing it by
    the clock instead of by the gap is unobservable — mutating that guard away
    leaves every test here green. The guard is kept because it is clearer and
    cheaper than relying on that coincidence, not because anything depends on
    it. What is asserted here is the weaker, true thing: the loop is invariant
    to a time shift.
    """
    early = Coordinator(FLEET, sightings=_SawItAt({1.0}))
    late = Coordinator(FLEET, sightings=_SawItAt({401.0}))
    for offset in (0.0, 1.0, 2.0):
        early_entropy = early.tick(_observation(1.0 + offset)).digest.entropy
        late_entropy = late.tick(_observation(401.0 + offset)).digest.entropy
    assert early_entropy == pytest.approx(late_entropy, rel=1e-9)


# -- nothing seen: a pure search --------------------------------------------


def test_with_nothing_seen_nothing_is_posted(hold_and_stub) -> None:
    """A tick with no sighting posts nothing at all — not a stale repeat."""
    hold, stub = hold_and_stub
    coordinator = Coordinator(FLEET, hold=hold, sightings=NoSightings())
    for tick in range(1, 10):
        outcome = coordinator.tick(_observation(float(tick)))
        assert outcome.posted is False
        assert outcome.track_state == "unseen"
        assert outcome.contacts == ()
    assert stub.tracks() == []


def test_with_nothing_seen_every_asset_still_searches() -> None:
    coordinator = Coordinator(FLEET)
    outcome = coordinator.tick(_observation(1.0))
    assert {i.reason for i in outcome.intent.intents} <= {"search", "watch"}
    assert "hold" not in {i.reason for i in outcome.intent.intents}


# -- something seen: the chain runs -----------------------------------------


def test_a_sighting_is_posted_and_pulls_the_quadcopter(hold_and_stub) -> None:
    hold, stub = hold_and_stub
    coordinator = Coordinator(FLEET, hold=hold, sightings=_SawItAt({3.0}))
    seen = [coordinator.tick(_observation(float(t))) for t in range(1, 6)]

    posted = [o for o in seen if o.posted]
    assert len(posted) == 1, "one sighting, one fix"
    assert posted[0].track_state == "held"

    by_id = {i.asset_id: i for i in posted[0].intent.intents}
    assert by_id["quadcopter"].reason == "hold"
    assert by_id["fixed-wing"].reason == "search", "the sweep asset keeps sweeping"

    hold._poster.close()
    (row,) = stub.tracks()
    assert row["name"] == "Sierra One"
    assert row["fixes"] == 1


def test_a_sighting_sharpens_the_belief() -> None:
    coordinator = Coordinator(FLEET, sightings=_SawItAt({2.0}))
    before = coordinator.tick(_observation(1.0)).digest.entropy
    after = coordinator.tick(_observation(2.0)).digest.entropy
    assert after < before, "a detection must make the field more certain"


def test_a_contact_is_recorded_for_the_log(hold_and_stub) -> None:
    hold, _ = hold_and_stub
    coordinator = Coordinator(FLEET, hold=hold, sightings=_SawItAt({2.0}))
    coordinator.tick(_observation(1.0))
    outcome = coordinator.tick(_observation(2.0))
    (contact,) = outcome.contacts
    assert contact.state == "tracked"
    assert contact.assigned_asset_id == "tower-1"
    assert contact.classification == "vessel"


# -- the two state changes that move assets ---------------------------------


def test_a_coasting_track_still_holds_the_quadcopter(hold_and_stub) -> None:
    """The last fix is still where to look, even with nothing new to post."""
    hold, _ = hold_and_stub
    coordinator = Coordinator(FLEET, hold=hold, sightings=_SawItAt({2.0}))
    for tick in range(1, 8):
        outcome = coordinator.tick(_observation(float(tick)))
    assert outcome.track_state == "coasting"
    assert outcome.posted is False, "coasting posts nothing"
    by_id = {i.asset_id: i for i in outcome.intent.intents}
    assert by_id["quadcopter"].reason == "hold"


def test_a_lost_track_releases_the_quadcopter_back_to_the_search(
    hold_and_stub,
) -> None:
    """A silent coast that never ends is the failure that looks like success."""
    hold, _ = hold_and_stub
    coordinator = Coordinator(FLEET, hold=hold, sightings=_SawItAt({2.0}))
    for tick in range(1, 30):
        outcome = coordinator.tick(_observation(float(tick)))
    assert outcome.track_state == "lost"
    by_id = {i.asset_id: i for i in outcome.intent.intents}
    assert by_id["quadcopter"].reason == "search"
    assert outcome.contacts[0].state == "lost"


# -- the seam ---------------------------------------------------------------


def test_the_coordinator_never_reaches_for_a_transport() -> None:
    """The caller does the I/O, so the policy cannot reach a socket by accident.

    It is also what lets the whole loop be tested with no arena, no network
    and no clock — which is what every test in this file does.
    """
    import ast
    import pathlib

    source = pathlib.Path("whiteout/coordinate.py").read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith("whiteout.transport") for name in imported)
    assert "socket" not in imported
