"""Holding a track across gaps, handoffs and losses.

Issue #67. One test per acceptance criterion, and the criterion that matters
most is the one about *not* posting: a padded track is worse than a short one,
because the sponsor holds the ground truth and accuracy is scored.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from whiteout.tracks import StubTracksServer, TrackPoster, TracksClient
from whiteout.tracks.maintain import Sighting, TrackHold


@pytest.fixture
def stub() -> Iterator[StubTracksServer]:
    with StubTracksServer() as server:
        yield server


@pytest.fixture
def hold(stub: StubTracksServer) -> Iterator[TrackHold]:
    poster = TrackPoster(TracksClient(stub.endpoint, backoff_s=0.0))
    try:
        yield TrackHold("Sierra One", poster, post_interval_s=2.0, coast_s=12.0)
    finally:
        poster.close()


def _walk(hold: TrackHold, start: float, count: int, step_s: float = 1.0) -> None:
    """Sight a vessel heading roughly north at about 3 m/s."""
    for i in range(count):
        t = start + i * step_s
        hold.sight(Sighting(t, 71.99 + i * 2.7e-5, -94.84, "quadcopter"))
        hold.tick(t)


# -- posts at a steady interval, with heading and speed ---------------------


def test_a_held_track_is_posted_at_the_interval_not_every_tick(
    hold: TrackHold,
) -> None:
    _walk(hold, 0.0, 11)
    # Eleven seconds of sightings at a two-second interval.
    assert 5 <= hold.posts <= 6
    assert hold.state == "held"


def test_heading_and_speed_come_from_successive_fixes(hold: TrackHold) -> None:
    _walk(hold, 0.0, 11)
    heading, speed = hold.course()
    assert heading == pytest.approx(0.0, abs=5.0)
    assert speed == pytest.approx(3.0, abs=0.5)


def test_a_course_is_not_reported_from_two_fixes_in_the_same_place(
    hold: TrackHold,
) -> None:
    """Two fixes metres apart give a bearing that is noise, not a course.

    The API would store it as fact, so it is better to send nothing.
    """
    hold.sight(Sighting(0.0, 71.99, -94.84, "tower-1"))
    hold.sight(Sighting(1.0, 71.990001, -94.84, "tower-1"))
    assert hold.course() == (None, None)


def test_the_first_fix_carries_no_course(hold: TrackHold) -> None:
    hold.sight(Sighting(0.0, 71.99, -94.84, "tower-1"))
    assert hold.course() == (None, None)


def test_what_reaches_the_api_is_one_track_being_updated(
    hold: TrackHold, stub: StubTracksServer
) -> None:
    _walk(hold, 0.0, 11)
    hold._poster.close()
    (row,) = stub.tracks()
    assert row["name"] == "Sierra One"
    assert row["fixes"] >= 1
    assert hold.posts >= 5


def test_the_api_ends_holding_the_newest_position(hold: TrackHold, stub: StubTracksServer) -> None:
    """Handing over more fixes than the endpoint drains is not a loss.

    ``TrackPoster`` is latest-wins per name (#64), so a burst of submits
    collapses to the newest. In a run they are two seconds apart and the
    worker drains in milliseconds, so nothing collapses; when something does,
    what survives is the freshest fix, which is the one worth having.
    """
    _walk(hold, 0.0, 11)
    hold._poster.close()
    (row,) = stub.tracks()
    assert row["lat"] == pytest.approx(hold.last_sighting.lat_deg)
    assert row["fixes"] <= hold.posts


# -- a gap does not drop the track, and is not papered over -----------------


def test_a_lost_frame_does_not_drop_the_track(hold: TrackHold) -> None:
    _walk(hold, 0.0, 5)
    for t in range(5, 11):
        hold.tick(float(t))
    assert hold.state == "coasting"
    assert hold.last_sighting is not None


def test_nothing_is_posted_while_coasting(hold: TrackHold) -> None:
    """The criterion: never post a stale position as if it were fresh.

    Re-posting the last fix every interval would inflate the fix count and
    look like a long healthy track, while being a lie about where the vessel
    is. The sponsor holds the ground truth and accuracy is scored, so a
    padded track is worse than a short one.
    """
    _walk(hold, 0.0, 5)
    before = hold.posts
    for t in range(5, 12):
        assert hold.tick(float(t)) is False
    assert hold.posts == before


def test_the_track_resumes_when_the_vessel_is_seen_again(hold: TrackHold) -> None:
    _walk(hold, 0.0, 5)
    for t in range(5, 11):
        hold.tick(float(t))
    hold.sight(Sighting(11.0, 71.9903, -94.84, "fixed-wing"))
    assert hold.tick(11.0) is True
    assert hold.state == "held"


def test_a_fix_arriving_out_of_order_does_not_walk_the_track_backwards(
    hold: TrackHold,
) -> None:
    """Two assets reporting in one tick can arrive in either order."""
    hold.sight(Sighting(5.0, 71.995, -94.84, "quadcopter"))
    hold.sight(Sighting(3.0, 71.991, -94.84, "tower-1"))
    assert hold.last_sighting is not None
    assert hold.last_sighting.t == 5.0


# -- handoff, without a gap in posting --------------------------------------


def test_a_handoff_changes_the_holder_and_nothing_else(hold: TrackHold) -> None:
    """The API keys on the name, so there is no handoff protocol to get wrong.

    A quad taking over from a fixed-wing is simply the next sighting under the
    same name: no gap, no re-creation, and the track never notices.
    """
    hold.sight(Sighting(0.0, 71.99, -94.84, "fixed-wing"))
    hold.tick(0.0)
    assert hold.holder == "fixed-wing"

    hold.sight(Sighting(2.0, 71.9901, -94.84, "quadcopter"))
    assert hold.tick(2.0) is True
    assert hold.holder == "quadcopter"


def test_a_handoff_does_not_re_create_the_track(hold: TrackHold, stub: StubTracksServer) -> None:
    for i, asset in enumerate(("tower-1", "fixed-wing", "quadcopter", "tower-2")):
        t = i * 3.0
        hold.sight(Sighting(t, 71.99 + i * 9e-5, -94.84, asset))
        hold.tick(t)
    hold._poster.close()
    rows = stub.tracks()
    assert len(rows) == 1
    assert hold.posts == 4


# -- saying so when the vessel is genuinely lost ----------------------------


def test_a_long_silence_is_called_lost_rather_than_coasted_forever(
    hold: TrackHold,
) -> None:
    """A silent coast that never ends is the failure that looks like success."""
    _walk(hold, 0.0, 3)
    hold.tick(2.0 + hold.coast_s + 1.0)
    assert hold.state == "lost"


def test_a_lost_track_still_does_not_post(hold: TrackHold) -> None:
    _walk(hold, 0.0, 3)
    before = hold.posts
    assert hold.tick(100.0) is False
    assert hold.posts == before


def test_an_unseen_track_says_so_and_posts_nothing(hold: TrackHold) -> None:
    assert hold.state == "unseen"
    assert hold.tick(5.0) is False
    assert hold.posts == 0
    assert hold.holder is None
