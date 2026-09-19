"""The tracks API: the only artifact Dominion Dynamics actually scores.

``hackathon/ARENA.md`` §5. Every detection the fleet makes is worth nothing
until it arrives here, and three of the seven scored criteria — *detection
speed*, *tracking duration* and *accuracy* — are measured on what this posts
and how long it keeps posting.
"""

from whiteout.tracks.client import (
    ENDPOINT_ENV,
    TrackAck,
    TrackFix,
    TrackPoster,
    TrackRecord,
    TracksClient,
    TracksError,
    endpoint_from_env,
)
from whiteout.tracks.stub import StubTracksServer, open_stub_tracks_server

__all__ = [
    "ENDPOINT_ENV",
    "StubTracksServer",
    "TrackAck",
    "TrackFix",
    "TrackPoster",
    "TrackRecord",
    "TracksClient",
    "TracksError",
    "endpoint_from_env",
    "open_stub_tracks_server",
]
