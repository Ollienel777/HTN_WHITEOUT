"""The shadow vessel, and the thing that sees it.

The claim under test is not "the fake is realistic" — it is not, and
``whiteout/shadow.py`` says so at length. It is that **the loop's plumbing
works end to end**: a target exists, an asset that could see it does, a
sighting becomes a contact, and a contact is held. Before this module every
run answered "not detected" on two of five scored axes.
"""

from __future__ import annotations

import pytest

from whiteout.belief.geometry import DEFAULT_STRAIT
from whiteout.coordinate import DEFAULT_TRACK_NAME, Coordinator
from whiteout.geo import GeoPoint, LocalPoint, geodetic_to_local, local_to_geodetic
from whiteout.policy import AssetRole
from whiteout.shadow import (
    DEFAULT_FIX_SIGMA_M,
    DEFAULT_SPEED_MPS,
    ShadowSightings,
    ShadowVessel,
)
from whiteout.tracks.maintain import TrackHold
from whiteout.transport.kinematic import FLEET, KinematicTransport

SEEDS = range(12)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_vessel_never_leaves_the_water(seed: int) -> None:
    """``ARENA.md`` §2 confines the target to the channel, so the fake must too.

    A vessel that wandered onto land would be findable only by a belief field
    that also believed in land, and every accuracy number taken against it
    would be measured off a target that could not exist.
    """
    vessel = ShadowVessel(seed=seed)
    for _ in range(1200):
        vessel.advance(0.5)
        point = vessel.position()
        assert DEFAULT_STRAIT.is_water(point.lat_deg, point.lon_deg), (
            f"seed {seed} put the vessel ashore at {point}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_the_vessel_moves_at_the_speed_the_arena_reports(seed: int) -> None:
    """``SHIP_SPEED=3.0`` in the arena's own `.env`, not a guess.

    Measured as the distance actually covered per second along the channel,
    which is the quantity the belief field's diffusion is set against.
    """
    vessel = ShadowVessel(seed=seed)
    before = DEFAULT_STRAIT.to_channel(*_pair(vessel))
    for _ in range(200):
        vessel.advance(0.5)
    after = DEFAULT_STRAIT.to_channel(*_pair(vessel))
    travelled = abs(after.s_m - before.s_m)
    # Along-channel only: the vessel also drifts across, so the along
    # component is at most the full speed and, with a drift, a little under.
    assert 0.0 < travelled <= DEFAULT_SPEED_MPS * 100.0 + 1e-6


def _pair(vessel: ShadowVessel) -> tuple[float, float]:
    point = vessel.position()
    return point.lat_deg, point.lon_deg


def test_one_seed_gives_one_episode() -> None:
    """The gate compares two runs byte for byte, so the target cannot wander."""
    first = ShadowVessel(seed=5)
    second = ShadowVessel(seed=5)
    for _ in range(80):
        first.advance(0.5)
        second.advance(0.5)
    assert _pair(first) == _pair(second)
    assert _pair(ShadowVessel(seed=6)) != _pair(ShadowVessel(seed=5))


def test_watching_the_vessel_does_not_move_it() -> None:
    """The sighting source draws from its own generator, and this is why.

    If whether a look succeeded consumed the vessel's randomness, the target's
    path would depend on who happened to be watching it — so an episode with
    a better search would have a *different* vessel, and the two runs could
    not be compared.
    """
    watched = ShadowVessel(seed=3)
    alone = ShadowVessel(seed=3)
    source = ShadowSightings(watched, seed=3)
    transport = KinematicTransport(seed=3)
    transport.connect()
    for _ in range(40):
        source.sightings(transport.observe())
    transport.close()
    # Advanced in the same increments, not in one jump: a random walk of 39
    # steps is not one step of 39 times the length, and comparing them would
    # test arithmetic this module does not claim.
    for _ in range(39):
        alone.advance(0.5)
    assert _pair(watched) == pytest.approx(_pair(alone))


def test_the_source_refuses_nothing_because_it_has_no_frame_to_distrust() -> None:
    source = ShadowSightings(ShadowVessel(seed=1))
    assert source.refusals() == ()


def test_an_asset_with_no_attitude_cannot_see() -> None:
    """The same guard the belief update has, for the same reason.

    A pose with no pitch cannot be turned into a footprint, and assuming level
    would have the fake report sightings from a camera pointed at the horizon.
    """
    from whiteout.types import Pose, WorldObservation

    vessel = ShadowVessel(seed=2)
    at = vessel.position()
    blind = Pose(
        asset_id="quadcopter",
        cls="quad",
        t=0.0,
        lat=at.lat_deg,
        lon=at.lon_deg,
        z=200.0,
        heading=0.0,
        speed=0.0,
        energy_used=0.0,
    )
    source = ShadowSightings(vessel, seed=2)
    assert source.sightings(WorldObservation(t=0.0, poses=(blind,), reports=())) == ()


def test_the_fleet_finds_and_holds_the_vessel() -> None:
    """The regression that matters: two scored axes stop saying "not detected".

    Asserted at the loop level rather than on the source, because every part
    of this chain — detector, projection, hold, poster — was individually
    correct and merged for a day while the run found nothing, simply because
    nothing was there to find.
    """
    transport = KinematicTransport(seed=7)
    transport.connect()
    # The hold is what turns sightings into contacts, so a coordinator built
    # without one reports an empty `contacts` however much it can see. That is
    # `cmd_run`'s own wiring, and leaving it out here made this test fail
    # against a loop that works.
    coordinator = Coordinator(
        tuple(AssetRole(asset.asset_id, asset.cls) for asset in FLEET),
        hold=TrackHold(DEFAULT_TRACK_NAME, None),
        sightings=ShadowSightings(ShadowVessel(seed=7), seed=7),
    )
    contacts = 0
    first_at: float | None = None
    for _ in range(400):
        observation = transport.observe()
        outcome = coordinator.tick(observation)
        transport.command(outcome.intent)
        if outcome.contacts:
            contacts += 1
            if first_at is None:
                first_at = observation.t
    transport.close()

    assert first_at is not None, "400 ticks and the fleet never saw the vessel"
    assert first_at < 120.0, f"first contact at {first_at} s is too late to be a search"
    assert contacts > 100, f"only {contacts} of 400 ticks carried a contact"


def test_the_fix_error_is_the_same_size_on_both_axes() -> None:
    """A sighting is a measurement, and it is as wrong East-West as North-South.

    This is the test whose absence let the two axes disagree by a factor of
    ten. The offsets are degrees, and a degree of longitude at 72 N is under a
    third of a degree of latitude, so an offset converted with the other
    axis's scale looks plausible and is out by the square of the ratio. The
    first version of this module reported 5.7 m East-West against 60 m
    North-South -- along the channel, which is the axis a track's course is
    measured over, so the track hold looked better than it had earned.

    Two things this test does deliberately, because the obvious versions of
    both are what let the defect through:

    - It measures the **positions ``sightings`` reports**, not ``_offset``'s
      return. A test of the helper passes while the call sites hand it the
      wrong axis's scale, which is exactly what happened.
    - It measures in **metres, through ``whiteout.geo``**. Comparing the raw
      degree offsets is the mistake being guarded against, so a test that
      compared them would share it.

    The clock is held still, so the vessel does not move between draws and
    the whole spread is the fix error.
    """
    from whiteout.types import Pose, WorldObservation

    vessel = ShadowVessel(seed=5)
    truth = vessel.position()
    source = ShadowSightings(vessel, seed=5)

    # Due south of the vessel and low enough to have it in frame: a quadcopter
    # at 120 m, 150 m back, sees it about half the looks.
    stand = local_to_geodetic(truth, LocalPoint(0.0, -150.0))
    watcher = Pose(
        asset_id="quadcopter",
        cls="quad",
        t=0.0,
        lat=stand.lat_deg,
        lon=stand.lon_deg,
        z=120.0,
        heading=0.0,
        speed=0.0,
        energy_used=0.0,
        pitch=0.0,
        roll=0.0,
    )
    observation = WorldObservation(t=0.0, poses=(watcher,), reports=())

    east: list[float] = []
    north: list[float] = []
    for _ in range(6000):
        for sighting in source.sightings(observation):
            offset = geodetic_to_local(truth, GeoPoint(sighting.lat_deg, sighting.lon_deg))
            east.append(offset.east_m)
            north.append(offset.north_m)

    assert len(east) > 500, f"too few sightings to measure a spread: {len(east)}"
    sigma_east = _stdev(east)
    sigma_north = _stdev(north)
    assert sigma_north == pytest.approx(DEFAULT_FIX_SIGMA_M, rel=0.15)
    assert sigma_east == pytest.approx(DEFAULT_FIX_SIGMA_M, rel=0.15)
    assert sigma_east / sigma_north == pytest.approx(1.0, rel=0.15)


def _stdev(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
