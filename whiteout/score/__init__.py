"""The scorer: a pure function of an episode log.

``hackathon/SPEC.md`` §5 sites it here, and §4 makes the episode log the only
artifact it reads. Everything in this package takes a sequence of
:class:`~whiteout.types.EpisodeRecord` and returns numbers; nothing here opens
a socket, reads the clock or touches the policy.

:mod:`whiteout.score.episode` holds the axes and what each one is derived
from. This is the demo-sized subset of the scoring model, not the full one
(issue #26): it answers only for what an episode log already carries, and says
in words where it cannot answer at all.
"""

from __future__ import annotations

from whiteout.score.episode import (
    AXES,
    AxisScore,
    EpisodeScore,
    ScoreError,
    score_episode,
    unscorable_criteria,
)

__all__ = [
    "AXES",
    "AxisScore",
    "EpisodeScore",
    "ScoreError",
    "score_episode",
    "unscorable_criteria",
]
