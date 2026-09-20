"""The loop, against the thing it exists to fix.

Until ``whiteout/episode.py`` existed, every component in this tree was
reachable only from its own test file: the detector from
``test_vision_detect.py``, the belief field from ``test_belief_grid.py``, the
tracks client from ``test_tracks.py``, and ``whiteout.cli run`` wrote four
parked assets with ``contacts=[]`` and a placeholder digest. So the first test
here is the one that would have caught that — a run has to *produce* something
— and the rest pin the properties that make the composition correct rather
than merely non-empty.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from whiteout.belief import ChannelBeliefGrid
from whiteout.belief.geometry import DEFAULT_STRAIT
from whiteout.cli import main
from whiteout.episode import EpisodeRunner, roles_of, run_episode, sweep_likelihood
from whiteout.geo import GeoPoint, geodetic_to_local
from whiteout.log import SCHEMA_VERSION
from whiteout.policy import SearchPolicy
from whiteout.sim import SENSORS, SimTransport, footprint_range_m
from whiteout.tracks import Sighting, TrackHold
from whiteout.types import FleetIntent, SensorFootprint, Truth, WorldObservation


class _Silent:
    """A poster that accepts fixes and sends nothing."""

    def submit(self, fix: object) -> None:
        pass


def _run(ticks: int = 120, seed: int = 3) -> tuple[list, EpisodeRunner]:
    transport = SimTransport(seed=seed)
    transport.connect()
    try:
        records, runner = run_episode(
            transport,
            ticks=ticks,
            hold=TrackHold("t", _Silent()),
            schema_version=SCHEMA_VERSION,
        )
    finally:
        transport.close()
    assert runner is not None
    return records, runner


def test_a_run_produces_detections_contacts_and_intents() -> None:
    """The regression that motivates the module: an episode that is not empty.

    Each of these was an empty tuple or a constant before the loop was wired,
    and every one of them passed the log's own validation, so nothing anywhere
    went red while the product did nothing.
    """
    records, runner = _run()
    assert runner.detections > 0, "no sensor ever reported the vessel"
    assert runner.first_detection_t is not None
    assert any(record.contacts for record in records), "no contact was ever held"
    assert any(record.intent.intents for record in records), "the fleet was never tasked"
    assert any(report.detections for record in records for report in record.observation.reports), (
        "no report carried a detection"
    )
    assert any(record.truth.targets for record in records), "the log carries no ground truth"


def test_the_belief_sharpens_when_the_vessel_is_seen() -> None:
    """Entropy has to fall on evidence, or the field is decoration.

    Measured against the field's own starting entropy rather than a constant:
    the number depends on how finely the channel is resolved (see
    ``BeliefPeak``), so a fixed threshold here would be a threshold about the
    grid size.
    """
    records, _ = _run()
    start = records[0].belief_digest.entropy
    best = min(record.belief_digest.entropy for record in records)
    assert best < start / 2.0, f"entropy only fell from {start} to {best}"
    assert max(record.belief_digest.peak_p for record in records) > 0.5


def test_the_reported_contact_is_near_the_vessel_it_reports() -> None:
    """A contact that is not where the vessel is, is the failure that matters.

    ``ARENA.md`` §5 scores tracking accuracy, and a confident wrong position is
    worse than silence. The bound is loose on purpose -- it is a property, not
    a tuning target -- but a broken projection or a lat/lon swap moves this by
    kilometres, not metres.
    """
    errors = []
    for record in _run()[0]:
        if not record.contacts or not record.truth.targets:
            continue
        contact, target = record.contacts[0], record.truth.targets[0]
        # Through `whiteout.geo`, not a hand-rolled degrees-to-metres: the
        # ellipsoid and trigonometry guards in `tests/test_geo.py` fail any
        # second converter, and a test that spells one is still a second one.
        offset = geodetic_to_local(
            GeoPoint(target.lat, target.lon), GeoPoint(contact.lat, contact.lon)
        )
        errors.append(math.hypot(offset.east_m, offset.north_m))
    assert errors, "no tick had both a contact and a truth to compare it with"
    errors.sort()
    assert errors[len(errors) // 2] < 200.0, f"median error {errors[len(errors) // 2]:.0f} m"


def test_two_seeds_are_two_episodes() -> None:
    """The seed has to move the world, not only the detection coins.

    It did not: the vessel spawned at a fixed point and only the coins were
    drawn, so two seeds produced byte-identical logs over a short run and the
    CLI's own ``test_run_differs_across_seeds`` went red. ``ARENA.md`` §3 says
    the spawn is random, and an episode whose target always starts in the same
    place measures one geometry rather than a policy.
    """
    first, _ = _run(ticks=40, seed=1)
    second, _ = _run(ticks=40, seed=2)
    assert [r.truth.targets[0].lat for r in first] != [r.truth.targets[0].lat for r in second]


def test_the_same_seed_is_the_same_episode() -> None:
    """The other half: nothing here reads a clock or an unseeded generator."""
    first, _ = _run(ticks=40, seed=5)
    second, _ = _run(ticks=40, seed=5)
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


def test_a_sweep_that_saw_nothing_never_rules_a_cell_out() -> None:
    """The negative update is a multiplication, and it must never be by zero.

    A sensor that sweeps open water and misses is the normal case. A hard cut
    would delete the cell the vessel is actually in on the first miss, and no
    later evidence could bring it back -- belief that reaches zero stays there
    under a product of likelihoods.
    """
    sensor = SENSORS["tower"]
    footprint = SensorFootprint(
        kind="circle", lat=72.0, lon=-94.8, radius=sensor.reach_m, heading=0.0, half_angle=0.0
    )
    likelihood = sweep_likelihood(footprint, sensor)
    assert likelihood(72.0, -94.8) > 0.0, "a miss at zero range still leaves belief"
    assert likelihood(72.0, -94.8) < 1.0, "a miss at zero range has to cost something"


def test_a_sweep_leaves_everything_outside_its_footprint_alone() -> None:
    """Exactly 1.0 outside, or a sweep quietly erodes water it never looked at."""
    sensor = SENSORS["fixedwing"]
    footprint = SensorFootprint(
        kind="cone",
        lat=72.0,
        lon=-94.8,
        radius=sensor.reach_m,
        heading=0.0,
        half_angle=sensor.half_angle_rad,
    )
    likelihood = sweep_likelihood(footprint, sensor)
    behind = likelihood(71.98, -94.8)
    assert behind == 1.0, f"a cell behind the aircraft was touched ({behind})"
    assert footprint_range_m(footprint, 71.98, -94.8) is None
    assert likelihood(72.0, -94.8) < 1.0, "a cell dead ahead was not touched"


def test_the_footprint_test_is_the_one_the_sim_uses() -> None:
    """One answer to "could this sensor have seen that", not two.

    The sim asks it to decide whether the vessel is visible and the runner asks
    it to decide what a fruitless sweep rules out. Two nearly-identical copies
    of the cone arithmetic is how a belief field ends up eroding water no
    sensor pointed at, so the guard in ``tests/test_geo.py`` -- which fails any
    trigonometry outside its allowlist -- is what keeps the second copy from
    being written.
    """
    sensor = SENSORS["quad"]
    footprint = SensorFootprint(
        kind="cone",
        lat=72.0,
        lon=-94.8,
        radius=sensor.reach_m,
        heading=0.0,
        half_angle=sensor.half_angle_rad,
    )
    inside = footprint_range_m(footprint, 72.005, -94.8)
    assert inside is not None and inside == pytest.approx(556.0, abs=30.0)
    assert footprint_range_m(footprint, 72.5, -94.8) is None, "beyond the reach"


def test_truth_is_empty_when_the_transport_has_none() -> None:
    """An arena run has no ground truth and must not invent any.

    Empty targets is the honest value, and the scorer reads it as "not
    measurable" rather than as "the vessel was at the origin" -- which is what
    a default-constructed position would say, in the Gulf of Guinea.
    """

    class _NoTruth:
        def __init__(self) -> None:
            self._inner = SimTransport(seed=0)
            self._inner.connect()

        def observe(self) -> WorldObservation:
            return self._inner.observe()

        def command(self, intent: FleetIntent) -> None:
            self._inner.command(intent)

    records, _ = run_episode(
        _NoTruth(), ticks=5, hold=TrackHold("t", _Silent()), schema_version=SCHEMA_VERSION
    )
    assert all(record.truth == Truth(t=record.t, targets=()) for record in records)


def test_roles_follow_the_fleet_that_reported() -> None:
    """Not a constant: the arena's roster is configuration and differs."""
    transport = SimTransport(seed=0)
    transport.connect()
    try:
        roles = roles_of(transport.observe())
    finally:
        transport.close()
    assert {role.asset_id for role in roles} == {"fw-1", "quad-1", "rover-1", "tower-1"}
    assert {role.cls for role in roles} == {"fixedwing", "quad", "rover", "tower"}


def test_a_handoff_is_counted_when_the_holding_asset_changes() -> None:
    """Collaboration is a scored axis, so a handoff is counted, not inferred."""
    hold = TrackHold("t", _Silent(), post_interval_s=0.0)
    runner = EpisodeRunner(
        belief=ChannelBeliefGrid(DEFAULT_STRAIT),
        policy=SearchPolicy((), DEFAULT_STRAIT),
        hold=hold,
    )
    for index, asset in enumerate(("tower-1", "tower-1", "fw-1", "fw-1", "tower-1")):
        # Sighting the hold directly rather than through a report: this is
        # about which asset is holding, and routing it through the sim would
        # make the test's subject the sensor geometry instead.
        hold.sight(Sighting(t=float(index), lat_deg=71.99, lon_deg=-94.84, asset_id=asset))
        runner.note_sighting()
        runner.tick(WorldObservation(t=float(index), poses=(), reports=()))
    assert runner.handoffs == 2, "two changes of holder over five sightings"


def test_the_cli_writes_an_episode_with_contacts_in_it(tmp_path: Path) -> None:
    """End to end through the real command line, which is what the gate runs."""
    out = tmp_path / "episode.jsonl"
    assert main(["run", "--seed", "7", "--ticks", "60", "--out", str(out)]) == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 60
    assert any(record["contacts"] for record in records)
    assert any(record["truth"]["targets"] for record in records)
    assert {record["belief_digest"]["grid_shape"][0] for record in records} != {1}


def test_the_cli_opens_no_socket_without_an_endpoint(tmp_path: Path) -> None:
    """The gate runs ``run`` twice per commit and must never touch the network.

    Posting is opt-in through ``--tracks-endpoint`` for exactly that reason, so
    the default path has to be reachable with sockets denied.
    """
    import socket

    real = socket.socket

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("run opened a socket with no --tracks-endpoint")

    socket.socket = refuse  # type: ignore[assignment,misc]
    try:
        assert main(["run", "--seed", "1", "--ticks", "5", "--out", str(tmp_path / "e.jsonl")]) == 0
    finally:
        socket.socket = real  # type: ignore[misc]
