"""The tracks client, the poster, and the stub they are built against.

Issue #64. One test per acceptance criterion, plus the two failure modes that
cost scoring seconds rather than raising: a dropped update, and a stalled
control loop.
"""

from __future__ import annotations

import threading
import time

import pytest

from whiteout.tracks import (
    ENDPOINT_ENV,
    StubTracksServer,
    TrackFix,
    TrackPoster,
    TracksClient,
    TracksError,
    endpoint_from_env,
    open_stub_tracks_server,
)


@pytest.fixture
def stub() -> object:
    with open_stub_tracks_server() as server:
        yield server


@pytest.fixture
def client(stub: StubTracksServer) -> TracksClient:
    return TracksClient(stub.endpoint, backoff_s=0.0)


# -- the stub reproduces created:true then created:false --------------------


def test_first_post_creates_and_mints_a_uuid(client: TracksClient) -> None:
    ack = client.post(TrackFix("Sierra One", 71.9965, -94.8448))
    assert ack.ok
    assert ack.created is True
    assert ack.name == "Sierra One"
    assert ack.uuid is not None
    assert ack.uuid.startswith("entity-")


def test_a_matching_name_updates_rather_than_creating(client: TracksClient) -> None:
    """The identity key is the name, and this is the whole of tracking.

    A run that only ever sees ``created: true`` is minting one track per fix.
    That scores as many one-fix tracks instead of one long one, and *tracking
    duration* is one of the seven criteria.
    """
    first = client.post(TrackFix("Sierra One", 71.9965, -94.8448))
    second = client.post(
        TrackFix("Sierra One", 71.9975, -94.8450, heading_deg=315.0, speed_mps=6.5)
    )
    assert first.created is True
    assert second.created is False
    assert second.uuid == first.uuid


def test_fixes_counts_updates_so_duration_is_visible(client: TracksClient) -> None:
    for _ in range(5):
        client.post(TrackFix("Sierra One", 71.99, -94.84))
    (row,) = client.list()
    assert row.fixes == 5


def test_a_body_without_a_name_is_refused(client: TracksClient) -> None:
    """400 is our bug, so it raises immediately rather than being retried."""
    with pytest.raises(TracksError) as raised:
        client.post(TrackFix("   ", 71.99, -94.84))
    assert "400" in str(raised.value)


# -- create, update-by-name and list ----------------------------------------


def test_listing_returns_every_track_with_its_optional_fields(
    client: TracksClient,
) -> None:
    client.post(TrackFix("Sierra One", 71.99, -94.84, heading_deg=315.0, speed_mps=3.0))
    client.post(TrackFix("Sierra Two", 72.01, -94.80))

    rows = {row.name: row for row in client.list()}
    assert set(rows) == {"Sierra One", "Sierra Two"}
    assert rows["Sierra One"].heading_deg == pytest.approx(315.0)
    assert rows["Sierra One"].speed_mps == pytest.approx(3.0)
    # Omitted is not the same as null-on-purpose, and read-back shows it.
    assert rows["Sierra Two"].heading_deg is None


def test_optional_fields_are_omitted_not_nulled() -> None:
    assert TrackFix("S", 1.0, 2.0).payload() == {"name": "S", "lat": 1.0, "lon": 2.0}
    full = TrackFix("S", 1.0, 2.0, heading_deg=90.0, speed_mps=3.0).payload()
    assert full["heading"] == 90.0
    assert full["speed"] == 3.0


# -- the address is configuration, never hardcoded --------------------------


def test_the_endpoint_comes_from_the_environment() -> None:
    assert endpoint_from_env({ENDPOINT_ENV: "http://10.99.4.1:8010/"}) == ("http://10.99.4.1:8010")


def test_an_unset_endpoint_raises_rather_than_defaulting() -> None:
    """There is no default on purpose.

    The arena's address is handed out per team. An endpoint quietly pointing
    at nothing posts nothing and scores nothing without ever failing, which is
    the one failure mode that looks exactly like success.
    """
    with pytest.raises(TracksError) as raised:
        endpoint_from_env({})
    assert ENDPOINT_ENV in str(raised.value)


# -- a failed post is retried and never silently dropped --------------------


class _FlakyClient(TracksClient):
    """Fails ``failures`` times, then succeeds. Counts every attempt.

    The failures are real refused connections against a closed port, not a
    raised sentinel, so the test exercises the client's own classification of
    what is worth retrying rather than trusting a stand-in for it.
    """

    _DEAD = "http://127.0.0.1:1"

    def __init__(self, endpoint: str, failures: int) -> None:
        super().__init__(endpoint, attempts=4, backoff_s=0.0, timeout_s=0.25)
        self._live = endpoint.rstrip("/")
        self._remaining = failures
        self.attempts_made = 0

    def _request(self, method: str, path: str, body: bytes | None) -> bytes:
        self.attempts_made += 1
        if self._remaining > 0:
            self._remaining -= 1
            self._endpoint = self._DEAD
        else:
            self._endpoint = self._live
        return super()._request(method, path, body)


def test_a_transient_failure_is_retried_until_it_lands(
    stub: StubTracksServer,
) -> None:
    flaky = _FlakyClient(stub.endpoint, failures=2)
    ack = flaky.post(TrackFix("Sierra One", 71.99, -94.84))
    assert ack.created is True
    assert flaky.attempts_made == 3
    assert len(stub.tracks()) == 1


def test_exhausted_retries_raise_rather_than_returning_quietly(
    stub: StubTracksServer,
) -> None:
    """A lost update is a lost scoring second, so it is never swallowed."""
    doomed = _FlakyClient(stub.endpoint, failures=99)
    with pytest.raises(TracksError) as raised:
        doomed.post(TrackFix("Sierra One", 71.99, -94.84))
    assert "4 attempts" in str(raised.value)
    assert stub.tracks() == []


def test_retries_back_off_instead_of_hammering(stub: StubTracksServer) -> None:
    waits: list[float] = []
    flaky = TracksClient(
        "http://127.0.0.1:1",
        attempts=4,
        backoff_s=0.25,
        timeout_s=0.25,
        sleep=waits.append,
    )
    with pytest.raises(TracksError):
        flaky.post(TrackFix("Sierra One", 71.99, -94.84))
    assert waits == [0.25, 0.5, 1.0]


# -- posting is non-blocking ------------------------------------------------


def test_submit_does_not_block_on_a_dead_endpoint() -> None:
    """The control loop must run at tick rate whatever the endpoint is doing.

    The endpoint here is a closed port with a long timeout and four attempts,
    so a blocking post would take seconds. Sixty submits must still return in
    well under one.
    """
    dead = TracksClient("http://127.0.0.1:1", timeout_s=2.0, attempts=4, backoff_s=1.0)
    poster = TrackPoster(dead)
    try:
        started = time.monotonic()
        for i in range(60):
            poster.submit(TrackFix("Sierra One", 71.99 + i * 1e-5, -94.84))
        elapsed = time.monotonic() - started
    finally:
        poster.close(timeout_s=0.1)
    assert elapsed < 0.5


def test_a_slow_endpoint_costs_intermediate_fixes_not_freshness(
    stub: StubTracksServer,
) -> None:
    """Latest-wins: a superseded fix is replaced, never queued behind.

    Posting a two-second-old position behind five older ones scores worse than
    posting the newest position once, so the queue holds one slot per name.
    """
    gate = threading.Event()

    class _Held(TracksClient):
        def _request(self, method: str, path: str, body: bytes | None) -> bytes:
            gate.wait(2.0)
            return super()._request(method, path, body)

    poster = TrackPoster(_Held(stub.endpoint, backoff_s=0.0))
    try:
        for i in range(20):
            poster.submit(TrackFix("Sierra One", 71.99 + i * 1e-4, -94.84))
        assert poster.superseded > 0
        gate.set()
    finally:
        poster.close()

    # Whatever was dropped, the final position is the one that landed.
    (row,) = stub.tracks()
    assert row["lat"] == pytest.approx(71.99 + 19 * 1e-4)
    assert row["fixes"] < 20


def test_close_drains_so_a_run_does_not_lose_its_last_fix(
    stub: StubTracksServer,
) -> None:
    poster = TrackPoster(TracksClient(stub.endpoint, backoff_s=0.0))
    poster.submit(TrackFix("Sierra One", 71.99, -94.84))
    poster.close()
    assert poster.posted == 1
    assert len(stub.tracks()) == 1


def test_the_poster_counts_what_it_gave_up_on() -> None:
    poster = TrackPoster(
        TracksClient("http://127.0.0.1:1", timeout_s=0.05, attempts=1, backoff_s=0.0)
    )
    poster.submit(TrackFix("Sierra One", 71.99, -94.84))
    poster.close()
    assert poster.failed == 1
    assert poster.posted == 0


def test_separate_names_are_not_superseded_by_each_other(
    stub: StubTracksServer,
) -> None:
    with TrackPoster(TracksClient(stub.endpoint, backoff_s=0.0)) as poster:
        poster.submit(TrackFix("Sierra One", 71.99, -94.84))
        poster.submit(TrackFix("Sierra Two", 72.01, -94.80))
    assert {row["name"] for row in stub.tracks()} == {"Sierra One", "Sierra Two"}
