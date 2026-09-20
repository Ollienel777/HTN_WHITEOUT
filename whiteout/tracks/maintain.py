"""Holding a track: what happens after the first detection.

Issue #67. *Tracking duration* and *accuracy* are two of the seven criteria
(``hackathon/ARENA.md`` §5) and both are about the rest of the run, not the
moment of finding. Detection is one instant; this is the hours.

The rule this module is built around
-------------------------------------

**Never post a position as current when it is not.** Re-posting the last known
fix every two seconds while nothing can see the vessel would inflate ``fixes``
and look like a long, healthy track. It would also be a lie about where the
vessel is, and the sponsor holds the ground truth to compare it against —
*accuracy* is scored, so a padded track is not merely dishonest but actively
worse than a short one.

So a fix is posted when it is **new**, rate-limited to the posting interval.
A gap in sightings produces a gap in posts, and the track is still *held*:
losing sight of the vessel for six seconds is a lost frame, not a lost vessel.

Three states, and the difference between the last two matters
--------------------------------------------------------------

``unseen``
    Never sighted. Nothing has been posted and there is no track.

``held``
    Sighted within :attr:`TrackHold.coast_s`. Fixes are flowing and being
    posted.

``coasting``
    Not sighted recently, but not long enough to give up. The track stays
    alive, its identity is kept, and **nothing is posted** — there is nothing
    new to say.

``lost``
    No sighting for longer than ``coast_s``. Said out loud, so a policy can
    re-task the fleet to search rather than continue to believe it is
    tracking. A silent coast that never ends is the failure mode that looks
    like success.

Handoff is free, and that is the point
---------------------------------------

The tracks API keys on ``name``, not on which asset saw the vessel, so a quad
taking over from a fixed-wing simply produces the next sighting under the same
name. There is no handoff protocol, no gap, and no re-creation of the track —
:attr:`TrackHold.holder` changes and nothing else does. *Collaboration* is one
of the seven criteria and this is where it stops being a slogan: the chain is
visible in which asset supplied each fix, and the track never notices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from whiteout.geo import GeoPoint, bearing_deg, geodetic_to_local
from whiteout.tracks.client import TrackFix, TrackPoster
from whiteout.types import PoseSync

__all__ = [
    "DEFAULT_ASSOCIATION_SLACK_M",
    "DEFAULT_COAST_S",
    "DEFAULT_MAX_SPEED_MPS",
    "DEFAULT_POST_INTERVAL_S",
    "Sighting",
    "TrackHold",
]

#: How often a held track is posted, at most. Dominion's workshop guidance
#: was "every few seconds"; faster spends bandwidth for a vessel that moves
#: 3 m/s and is not more accurate, since the fix is only as fresh as the
#: frame it came from.
DEFAULT_POST_INTERVAL_S = 2.0

#: How long a track survives with no sighting before it is called lost. At
#: 3 m/s the vessel travels 36 m in twelve seconds, so a fix this old is
#: still useful to a policy deciding where to look — and that is the point of
#: coasting, not of posting.
DEFAULT_COAST_S = 12.0

#: Fastest the held thing could plausibly be going, m/s. The vessel does 3.0
#: (``SHIP_SPEED``), and this is the same upper bound the motion gate uses:
#: above it, a fix is not the thing we were holding.
DEFAULT_MAX_SPEED_MPS = 8.0

#: Slack added to the travel allowance, metres, for projection error. Starts
#: at the detection sigma the coordinator folds into belief, because that is
#: our standing estimate of how far a fix can be wrong.
DEFAULT_ASSOCIATION_SLACK_M = 150.0

#: Below this, two fixes are the same place and the bearing between them is
#: noise rather than a course. 15 m is five seconds of travel at 3 m/s.
#:
#: This is a distance and not a tick count on purpose. At 1 Hz sightings the
#: *consecutive* pair is only ever ~3 m apart, so a course taken from the
#: previous fix alone would be dominated by fix noise and, with any threshold
#: worth having, would never be reported at all. :meth:`TrackHold.course`
#: therefore reaches back through the history for the most recent fix far
#: enough away to carry a bearing.
_MIN_COURSE_BASELINE_M = 15.0

#: And the baseline must also span this many seconds.
#:
#: The distance threshold alone is not enough, and the reason is that 15 m was
#: sized against a fix error of about 3 m. Nothing has measured the real one.
#: The projection's flat-water model is 10% wrong at its own range bound --
#: 200 m at 2 km -- so a fix error comfortably larger than 15 m is the
#: expected case at range, not the pessimistic one.
#:
#: When it is, *every* adjacent pair clears 15 m on noise alone. ``course``
#: then takes the most recent qualifying pair, which is the noisiest one
#: available, and divides that noise by a one-second gap. Measured against a
#: vessel moving at a known 3.0 m/s, that reported speeds of 10.7, 8.4 and
#: **81.2 m/s** -- twenty-seven times the truth, on a field ``ARENA.md`` §5
#: carries in its own example payload.
#:
#: Ten seconds is thirty metres of travel at 3 m/s, so the signal beats a
#: two-sigma fix error rather than merely exceeding a floor.
_MIN_COURSE_BASELINE_S = 10.0

#: How much sighting history to keep for that baseline. Sixty seconds is
#: 180 m of travel — far more than the baseline needs, and still small.
_HISTORY_S = 60.0


@dataclass(frozen=True)
class Sighting:
    """One asset seeing the vessel at one time.

    ``t`` is the tick the sighting belongs to, in the transport's timebase.
    ``asset_id`` is kept so that a handoff is visible afterwards rather than
    inferred.

    ``sync`` is how the frame this came from stood in time against the pose it
    was projected with (:class:`~whiteout.types.PoseSync`), or ``None`` when
    the source did not establish it. It is carried through rather than
    inspected here: this module decides *when* to post, and the question of how
    good a fix's geometry is belongs to whatever reads the fix. It reaches the
    episode log on :class:`~whiteout.types.Contact`.
    """

    t: float
    lat_deg: float
    lon_deg: float
    asset_id: str
    sync: PoseSync | None = None


class TrackHold:
    """Maintains one named track from a stream of sightings.

    Feed it :meth:`sight` whenever an asset sees the vessel and :meth:`tick`
    every control cycle. It decides what to post and when, and reports its own
    state honestly.

    ``poster`` is optional, and that is what separates *holding* a track from
    *submitting* it. With no poster the hold does everything else it does —
    associates sightings, keeps its identity through a handoff, names its own
    state, and reports the fixes it would have sent — and sends nothing. A
    rehearsal or an offline demo therefore still tracks, still tasks the fleet
    and still logs contacts; only the outbound POST is missing.
    """

    def __init__(
        self,
        name: str,
        poster: TrackPoster | None = None,
        *,
        post_interval_s: float = DEFAULT_POST_INTERVAL_S,
        coast_s: float = DEFAULT_COAST_S,
        max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
        association_slack_m: float = DEFAULT_ASSOCIATION_SLACK_M,
    ) -> None:
        self._name = name
        self._poster = poster
        self._post_interval_s = float(post_interval_s)
        self._coast_s = float(coast_s)
        self._max_speed_mps = float(max_speed_mps)
        self._association_slack_m = float(association_slack_m)
        self._unassociated = 0
        self._last: Sighting | None = None
        self._history: list[Sighting] = []
        self._posted_t: float | None = None
        self._last_post_at: float | None = None
        self._posts = 0
        self._now = 0.0

    @property
    def name(self) -> str:
        """The track's name — the tracks API's identity key."""
        return self._name

    @property
    def coast_s(self) -> float:
        """Seconds without a sighting before the track is called lost."""
        return self._coast_s

    @property
    def state(self) -> str:
        """``unseen``, ``held``, ``coasting`` or ``lost``."""
        if self._last is None:
            return "unseen"
        age = self._now - self._last.t
        if age > self._coast_s:
            return "lost"
        if age > self._post_interval_s:
            return "coasting"
        return "held"

    @property
    def holder(self) -> str | None:
        """Which asset supplied the most recent fix, if any."""
        return None if self._last is None else self._last.asset_id

    @property
    def unassociated(self) -> int:
        """Sightings refused because the held thing could not have got there.

        Rising steadily means the detector is producing scattered false
        positives, which is worth seeing rather than silently absorbing.
        """
        return self._unassociated

    @property
    def posts(self) -> int:
        """How many new fixes the hold released — handed to the poster, if any."""
        return self._posts

    @property
    def last_sighting(self) -> Sighting | None:
        """The most recent sighting, or ``None``."""
        return self._last

    def course(self) -> tuple[float | None, float | None]:
        """``(heading_deg, speed_mps)`` over the longest useful baseline.

        Not the last two sightings: at 1 Hz those are ~3 m apart and the
        bearing between them is fix noise. This walks back through the
        history for the most recent fix at least
        :data:`_MIN_COURSE_BASELINE_M` away and measures against that.

        The pair must also span :data:`_MIN_COURSE_BASELINE_S`. The distance
        bound alone fails exactly when fix error is large: then every adjacent
        pair clears it on noise, the walk stops at the *newest* such pair, and
        the noise is divided by a one-second gap. Against a vessel moving at a
        known 3.0 m/s that reported 81 m/s.

        ``(None, None)`` when no such pair exists — on the first fix, and
        whenever the vessel has been sitting still. Reporting a course from
        two fixes 2 m apart would hand the API sensor noise, and it would
        store it as fact.
        """
        second = self._last
        if second is None or len(self._history) < 2:
            return None, None
        end = GeoPoint(second.lat_deg, second.lon_deg)
        # Walk back to the most recent fix far enough away to carry a bearing.
        for first in reversed(self._history[:-1]):
            dt = second.t - first.t
            if dt <= 0.0:
                continue
            start = GeoPoint(first.lat_deg, first.lon_deg)
            offset = geodetic_to_local(start, end)
            distance_m = math.hypot(offset.east_m, offset.north_m)
            # Both, and the time bound is what stops fix noise being read as
            # speed: a pair far enough apart in metres may be so only because
            # both fixes were wrong in opposite directions.
            if distance_m >= _MIN_COURSE_BASELINE_M and dt >= _MIN_COURSE_BASELINE_S:
                return bearing_deg(start, end), distance_m / dt
        return None, None

    def sight(self, sighting: Sighting) -> None:
        """Record that an asset has seen the vessel.

        A sighting older than the one already held is ignored: fixes can
        arrive out of order when two assets report in the same tick, and
        letting an older one win would walk the track backwards.
        """
        if self._last is not None and sighting.t < self._last.t:
            return
        if not self._could_have_got_there(sighting):
            self._unassociated += 1
            return
        self._last = sighting
        self._history.append(sighting)
        cutoff = sighting.t - _HISTORY_S
        self._history = [s for s in self._history if s.t >= cutoff]
        self._now = max(self._now, sighting.t)

    def _could_have_got_there(self, sighting: Sighting) -> bool:
        """Could the thing we are holding have reached this fix by now?

        A sighting further from the current fix than the vessel could have
        travelled since the last one is a **different object**, and taking it
        teleports the track. Flown live before this existed, four unrelated
        false positives were stitched into one track reporting 19.8 m/s — the
        course between two of them, not a boat.

        The allowance grows with the gap, so a long coast does not reject the
        re-acquisition that ends it.

        Two sightings are always accepted: the first, which has nothing to be
        inconsistent with, and any sighting once the track is ``lost``. A lost
        track has no position worth defending, and without that escape one bad
        initial lock would poison the whole run with nothing able to recover.
        """
        last = self._last
        if last is None or self.state == "lost":
            return True
        elapsed = max(0.0, sighting.t - last.t)
        allowed = self._max_speed_mps * elapsed + self._association_slack_m
        offset = geodetic_to_local(
            GeoPoint(last.lat_deg, last.lon_deg),
            GeoPoint(sighting.lat_deg, sighting.lon_deg),
        )
        return math.hypot(offset.east_m, offset.north_m) <= allowed

    def tick(self, t: float) -> bool:
        """Advance the clock to ``t`` and post if there is something new.

        Returns whether there was a new fix to post — which is also whether
        one was handed to the poster, when there is a poster. Posting is skipped
        when there is no sighting, when the newest sighting has already been
        posted, and when the interval has not elapsed — never because the
        vessel is hard to see, and never with a position we did not just
        observe.
        """
        self._now = max(self._now, float(t))
        last = self._last
        if last is None:
            return False
        if self._posted_t is not None and last.t <= self._posted_t:
            # Nothing new to say. Re-posting here is exactly the lie this
            # module exists to avoid.
            return False
        if (
            self._last_post_at is not None
            and self._now - self._last_post_at < self._post_interval_s
        ):
            return False
        heading, speed = self.course()
        if self._poster is not None:
            self._poster.submit(
                TrackFix(
                    name=self._name,
                    lat_deg=last.lat_deg,
                    lon_deg=last.lon_deg,
                    heading_deg=heading,
                    speed_mps=speed,
                )
            )
        self._posted_t = last.t
        self._last_post_at = self._now
        self._posts += 1
        return True
