"""The run loop, driven against a list of observations.

Issue #96's follow-on: every part worked and nothing joined them. These tests
are about the joining — the order of a tick, and the two places where the
track's state changes what the fleet is told to do.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import types

import pytest

import whiteout
from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint
from whiteout.coordinate import Coordinator, NoSightings, TickOutcome
from whiteout.policy import AssetRole
from whiteout.tracks.client import TrackPoster, TracksClient
from whiteout.tracks.maintain import Sighting, TrackHold
from whiteout.tracks.stub import StubTracksServer
from whiteout.types import Pose, PoseSync, SightingRefusal, WorldObservation
from whiteout.vision.motion import MotionGate

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

    def __init__(self, ticks: set[float], s_m: float = MID, sync: PoseSync | None = None) -> None:
        self.ticks = ticks
        self.lat, self.lon = _at(s_m)
        self.sync = sync

    def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
        if observation.t not in self.ticks:
            return ()
        return (Sighting(observation.t, self.lat, self.lon, "tower-1", sync=self.sync),)


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
    assert contact.sync is None, "a source that established nothing invents nothing"


def test_what_the_source_refused_reaches_the_outcome() -> None:
    """A refusal that stops at the sighting source is a refusal nobody can see.

    The coordinator is the only path from a camera to the episode log, so a
    source that declined a fix reports it here or not at all.
    """

    class _RefusedEverything:
        def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
            return ()

        def refusals(self) -> tuple[SightingRefusal, ...]:
            return (
                SightingRefusal(
                    asset_id="fixed-wing",
                    t=1.0,
                    sync=PoseSync(status="telemetry_stale", skew_s=0.4),
                ),
            )

    outcome = Coordinator(FLEET, sightings=_RefusedEverything()).tick(_observation(1.0))
    (refusal,) = outcome.refusals
    assert refusal.asset_id == "fixed-wing"
    assert refusal.sync.status == "telemetry_stale"


def test_a_source_that_refuses_nothing_reports_nothing() -> None:
    outcome = Coordinator(FLEET, sightings=NoSightings()).tick(_observation(1.0))
    assert outcome.refusals == ()


def test_a_source_predating_the_method_still_runs() -> None:
    """The `getattr` is tolerance for a hand-written double, not a contract.

    Shipped sources are held to `SightingSource` by the test below; a one-method
    double in somebody's test must not crash the loop.
    """

    class _OldStyle:
        def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
            return ()

    assert Coordinator(FLEET, sightings=_OldStyle()).tick(_observation(1.0)).refusals == ()


def test_a_wrapped_source_s_refusals_survive_the_wrapper() -> None:
    """The composition the camera path is actually headed for.

    `MotionGate` exists so arena ice stops reading as a hull, so the real source
    is `MotionGate(VisionSightings(...))`. A gate that answered only `sightings`
    would swallow every refusal, `EpisodeRecord.refusals` would be empty for the
    whole run, and a camera dark for a known reason would read as an empty sea —
    with nothing failing anywhere.
    """
    refusal = SightingRefusal(
        asset_id="fixed-wing", t=1.0, sync=PoseSync(status="telemetry_stale", skew_s=0.4)
    )

    class _RefusedEverything:
        def sightings(self, observation: WorldObservation) -> tuple[Sighting, ...]:
            return ()

        def refusals(self) -> tuple[SightingRefusal, ...]:
            return (refusal,)

    gated = MotionGate(source=_RefusedEverything())
    assert gated.sightings(_observation(1.0)) == ()
    assert gated.refusals() == (refusal,), "the gate swallowed the refusal"
    outcome = Coordinator(FLEET, sightings=gated).tick(_observation(1.0))
    assert outcome.refusals == (refusal,), "the refusal did not survive the run loop"


def test_every_shipped_sighting_source_answers_both_halves() -> None:
    """A new decorator that forgets `refusals` fails here, not silently in a log.

    The coordinator discovers the method on the object, which is what let
    `MotionGate` drop refusals with no symptom. This is the signal that shape
    otherwise lacks: anything in `whiteout/` that reports sightings must also
    say what it refused, even if the answer is always empty.
    """
    package = pathlib.Path(whiteout.__file__).parent
    sources: list[type] = []
    for path in sorted(package.rglob("*.py")):
        name = ".".join(path.relative_to(package.parent).with_suffix("").parts)
        module = importlib.import_module(name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__ or obj in sources:
                continue
            # Protocols describe a shape rather than implement one, and
            # `motion.Sightable` is deliberately the narrower of the two: the
            # gate wraps anything that reports sightings and forwards whatever
            # it can. The rule is for the classes that *are* sources.
            if getattr(obj, "_is_protocol", False):
                continue
            if isinstance(getattr(obj, "sightings", None), types.FunctionType):
                sources.append(obj)
    assert sources, "no sighting sources found — this guard has stopped guarding"
    missing = [obj.__name__ for obj in sources if not callable(getattr(obj, "refusals", None))]
    assert not missing, f"{', '.join(missing)} reports sightings but cannot say what it refused"


def test_a_contact_carries_its_last_fix_s_synchronisation(hold_and_stub) -> None:
    """The contact's position *is* the last sighting's, so its caveat is too.

    Without this the log would show a lat/lon with every field populated and no
    way to see that the frame and the attitude behind it were a quarter-second
    apart — 5.5 m at the fixed-wing's cruise.
    """
    hold, _ = hold_and_stub
    sync = PoseSync(status="telemetry_missing", skew_s=None)
    coordinator = Coordinator(FLEET, hold=hold, sightings=_SawItAt({2.0}, sync=sync))
    coordinator.tick(_observation(1.0))
    (contact,) = coordinator.tick(_observation(2.0)).contacts
    assert contact.sync == sync


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


# -- the hold stands off, it does not hover overhead (#109) ------------------


def test_the_hold_point_is_a_stand_off_and_not_the_contact_itself(hold_and_stub) -> None:
    """The quadcopter's camera looks where the airframe does (#109).

    Hovering over a contact puts it straight down, where a horizon-looking
    camera cannot see it. Flown live, the quadcopter ended up on top of the
    vessel and saw nothing.
    """
    import math

    from whiteout.geo import GeoPoint, geodetic_to_local

    hold_at = _at(MID)
    hold, _ = hold_and_stub
    coordinator = Coordinator(FLEET, hold=hold, sightings=_SawItAt({2.0}, s_m=MID))
    coordinator.tick(_observation(1.0))
    outcome = coordinator.tick(_observation(2.0))
    quad = next(i for i in outcome.intent.intents if i.asset_id == "quadcopter")
    assert quad.reason == "hold"

    offset = geodetic_to_local(
        GeoPoint(hold_at[0], hold_at[1]), GeoPoint(quad.target_lat, quad.target_lon)
    )
    stand_off = math.hypot(offset.east_m, offset.north_m)
    assert stand_off > 50.0, "sitting on the contact is the bug this fixes"


def test_the_stand_off_follows_the_camera_and_the_altitude(hold_and_stub) -> None:
    """Higher means further out, because the depression angle is the constraint."""
    import math

    from whiteout.geo import GeoPoint, geodetic_to_local
    from whiteout.vision.camera import CAMERAS
    from whiteout.vision.standoff import standoff_range_m

    base_hold, stub = hold_and_stub

    def make_hold():
        return TrackHold("Sierra One", base_hold._poster, post_interval_s=2.0)

    def stand_off_for(alt: float) -> float:
        poses = tuple(
            Pose(
                asset_id=r.asset_id,
                cls=r.cls,
                t=1.0,
                lat=_at(MID * 0.5)[0],
                lon=_at(MID * 0.5)[1],
                z=alt,
                heading=0.0,
                speed=0.0,
                energy_used=0.0,
                pitch=0.0,
                roll=0.0,
            )
            for r in FLEET
        )
        c = Coordinator(FLEET, hold=make_hold(), sightings=_SawItAt({1.0}, s_m=MID))
        out = c.tick(WorldObservation(t=1.0, poses=poses, reports=()))
        quad = next(i for i in out.intent.intents if i.asset_id == "quadcopter")
        contact = _at(MID)
        offset = geodetic_to_local(
            GeoPoint(contact[0], contact[1]), GeoPoint(quad.target_lat, quad.target_lon)
        )
        return math.hypot(offset.east_m, offset.north_m)

    low, high = stand_off_for(80.0), stand_off_for(240.0)
    assert high > low
    assert low == pytest.approx(standoff_range_m(CAMERAS["quadcopter"], 80.0), rel=0.02)


def test_a_hold_without_the_holding_asset_s_pose_falls_back_to_the_contact(hold_and_stub) -> None:
    """A poor place to watch from, and still better than no instruction."""
    hold, _ = hold_and_stub
    coordinator = Coordinator(FLEET, hold=hold, sightings=_SawItAt({1.0}, s_m=MID))
    without_quad = WorldObservation(
        t=1.0,
        poses=tuple(p for p in _observation(1.0).poses if p.asset_id != "quadcopter"),
        reports=(),
    )
    outcome = coordinator.tick(without_quad)
    assert "quadcopter" not in {i.asset_id for i in outcome.intent.intents}
