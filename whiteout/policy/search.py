"""Where every asset goes next, decided along the channel rather than over it.

Issue #96. The whole project rests on one observation, and this module is the
first place it pays.

Why the search is one-dimensional
----------------------------------

The vessel is not wandering. Dominion's own simulator plans its course with a
widest-path search whose step cost is ``1 + 3×(1 − clearance/max)`` — one times
mid-channel, four times against the bank — so the route **provably hugs the
centreline**. Every waypoint clears the shore by at least 120 m. The plugin
then moves it along that polyline at a constant 3.0 m/s and reverses at each
end.

So the vessel's support is not 6.5 km of water. It is a **curve**, and the
question "where should we look?" is a question about an **interval in arc
length**. A policy that rasters the area is searching a space two dimensions
larger than the one the target lives in.

Everything below is therefore written in channel coordinates: the strait is
cut into segments of :attr:`~whiteout.policy.params.SearchParams.segment_m`
along ``s``, each segment is scored, and assets are sent to segments.

What a segment is worth
------------------------

Three terms, multiplied and then discounted:

**Belief.** Sampled from the field through the
:class:`~whiteout.belief.field.BeliefField` protocol at the segment's
centreline point. Sampling rather than reaching into the grid is deliberate —
the policy is not allowed to know how belief is stored, and a policy that did
would have to be rewritten when the field is.

**Staleness.** Seconds since anything looked at that segment, saturating at
:attr:`~whiteout.policy.params.SearchParams.stale_horizon_s`. Without it the
fleet converges on the single highest-belief segment and stares at it: the
peak does not move until something looks somewhere else, and nothing ever
does. Staleness is what turns a greedy rule into a sweep.

**Travel.** A segment is discounted by the distance the asset must fly to
reach it. This is not a tie-breaker — *efficiency* is scored, and a fleet that
crosses the strait for a marginally better segment burns the run's endurance
for nothing.

Re-tasking hysteresis
----------------------

An asset keeps its current target unless a new one beats it by
:attr:`~whiteout.policy.params.SearchParams.hysteresis_margin`. Two segments a
few percent apart will swap rank every tick as belief diffuses, and an asset
that chases the rank turns round mid-transit, arrives nowhere, and covers
nothing. The margin is the difference between a search and a dither.

What this module does not do
-----------------------------

It does not detect, it does not update belief, and it does not post tracks. It
reads a :class:`~whiteout.types.WorldObservation` and a belief field, and
returns a :class:`~whiteout.types.FleetIntent`. That is the whole contract,
and it is what lets the same policy drive the fake and the arena without
knowing which it is on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from whiteout.belief.field import BeliefField
from whiteout.belief.geometry import ChannelPoint, StraitGeometry
from whiteout.geo import GeoPoint, geodetic_to_local
from whiteout.policy.params import DEFAULT_SEARCH_PARAMS, SearchParams
from whiteout.types import FleetIntent, Pose, WaypointIntent, WorldObservation

__all__ = ["AssetRole", "SearchPolicy", "Segment"]

#: Vehicle classes this policy knows how to task. Anything else is left alone
#: rather than sent somewhere a guess says it might go.
_AIRCRAFT = frozenset({"quad", "fixedwing"})
_TOWER = "tower"


@dataclass(frozen=True)
class AssetRole:
    """One asset and the job its class gives it.

    ``cls`` is the seam's vehicle-class string: ``copter``, ``plane`` or
    ``tower``. The roles are not configuration — they follow from what each
    airframe can physically do, and ``ARENA.md`` §3 is the source:

    - ``fixedwing`` covers ground fastest and cannot loiter, so it takes the
      distant work.
    - ``quad`` is slow and can hover, so it holds contacts and otherwise takes
      near work.
    - ``tower`` cannot move at all, so it is aimed rather than sent.

    The spellings are :data:`whiteout.types.VEHICLE_CLASSES`, because they
    arrive on a :class:`~whiteout.types.Pose` and nothing translates them.
    """

    asset_id: str
    cls: str


@dataclass(frozen=True)
class Segment:
    """One stretch of the channel, and where its middle is."""

    index: int
    s_m: float
    lat_deg: float
    lon_deg: float


@dataclass
class _Assignment:
    """What an asset is currently working, so hysteresis has something to hold."""

    segment: int
    score: float = 0.0
    reason: str = "search"
    task_id: str = ""


@dataclass
class _Coverage:
    """When each segment was last looked at."""

    last_seen_t: dict[int, float] = field(default_factory=dict)

    def staleness(self, index: int, now: float, horizon: float) -> float:
        """0.0 for a segment just looked at, 1.0 for one never looked at."""
        seen = self.last_seen_t.get(index)
        if seen is None:
            return 1.0
        return min(1.0, max(0.0, (now - seen) / horizon))


class SearchPolicy:
    """Assigns every asset a waypoint, each tick, in channel coordinates.

    Stateful only in the two things a search must remember: which segments
    have been looked at, and what each asset is currently working. Both are
    functions of the observations it has been given, so the same sequence of
    observations always produces the same intents — there is no clock and no
    generator anywhere in it.
    """

    def __init__(
        self,
        roles: tuple[AssetRole, ...],
        geometry: StraitGeometry,
        params: SearchParams = DEFAULT_SEARCH_PARAMS,
    ) -> None:
        self._roles = roles
        self._geometry = geometry
        self._params = params
        self._segments = _cut(geometry, params.segment_m)
        self._coverage = _Coverage()
        self._assigned: dict[str, _Assignment] = {}
        self._decisions = 0
        self._retasks = 0

    @property
    def segments(self) -> tuple[Segment, ...]:
        """The channel, cut into the intervals this policy reasons about."""
        return self._segments

    @property
    def retasks(self) -> int:
        """How many times an asset has been moved off a target it had.

        The number hysteresis exists to hold down, exposed so that a run can
        show it rather than claim it.
        """
        return self._retasks

    def assignment_of(self, asset_id: str) -> int | None:
        """Which segment an asset is working, or ``None``."""
        held = self._assigned.get(asset_id)
        return None if held is None else held.segment

    def decide(
        self,
        observation: WorldObservation,
        belief: BeliefField,
        *,
        held_lat_deg: float | None = None,
        held_lon_deg: float | None = None,
    ) -> FleetIntent:
        """One tick's tasking for the whole fleet.

        ``held_lat_deg``/``held_lon_deg`` are a contact currently being
        tracked, when there is one. It pulls the quadcopter and nothing else:
        the aircraft that can stop and stare is the one that should be holding
        a contact, and pulling the fixed-wing onto it as well would abandon
        the search with no gain, since it cannot loiter over the vessel
        anyway.
        """
        self._decisions += 1
        now = observation.t
        poses = {pose.asset_id: pose for pose in observation.poses}
        self._mark_looked_at(poses, now)

        values = self._segment_values(belief, now)
        taken: set[int] = set()
        intents: list[WaypointIntent] = []

        holding = held_lat_deg is not None and held_lon_deg is not None
        for role in self._roles:
            pose = poses.get(role.asset_id)
            if pose is None:
                # Silence from an asset is not a reason to guess where it is.
                continue
            if role.cls == "quad" and holding:
                assert held_lat_deg is not None and held_lon_deg is not None
                intents.append(
                    self._waypoint(
                        role,
                        held_lat_deg,
                        held_lon_deg,
                        self._params.hold_alt_m,
                        reason="hold",
                        task_id=f"hold-{self._decisions}",
                    )
                )
                continue
            chosen = self._choose(role, pose, values, taken)
            if chosen is None:
                continue
            taken.add(chosen)
            segment = self._segments[chosen]
            altitude = 0.0 if role.cls == _TOWER else self._params.search_alt_m
            reason = "watch" if role.cls == _TOWER else "search"
            intents.append(
                self._waypoint(
                    role,
                    segment.lat_deg,
                    segment.lon_deg,
                    altitude,
                    reason=reason,
                    task_id=f"{reason}-s{segment.index}",
                )
            )
        return FleetIntent(t=now, intents=tuple(intents))

    # -- scoring ------------------------------------------------------------

    def _segment_values(self, belief: BeliefField, now: float) -> list[float]:
        horizon = self._params.stale_horizon_s
        values = []
        for segment in self._segments:
            probability = belief.probability_at(segment.lat_deg, segment.lon_deg)
            stale = self._coverage.staleness(segment.index, now, horizon)
            values.append(probability * stale)
        return values

    def _choose(
        self,
        role: AssetRole,
        pose: Pose,
        values: list[float],
        taken: set[int],
    ) -> int | None:
        """The best segment for this asset, subject to hysteresis."""
        best: int | None = None
        best_score = -math.inf
        for segment in self._segments:
            if segment.index in taken:
                continue
            score = self._score_for(role, pose, segment, values[segment.index])
            if score is None:
                continue
            if score > best_score:
                best, best_score = segment.index, score
        if best is None:
            return None

        held = self._assigned.get(role.asset_id)
        previous = held.segment if held is not None else None
        if held is not None and previous not in taken:
            current = self._score_for(
                role, pose, self._segments[held.segment], values[held.segment]
            )
            if current is not None and best_score <= current * (
                1.0 + self._params.hysteresis_margin
            ):
                # Not enough better to be worth turning round for.
                held.score = current
                return held.segment
        # Counted only when the asset actually changes target. Re-deriving the
        # same segment is not a re-task, and neither is losing a contended one
        # to an asset earlier in the roster — counting either would inflate the
        # very number hysteresis exists to hold down, and this number is
        # reported rather than merely tested.
        if previous is not None and previous != best:
            self._retasks += 1
        self._assigned[role.asset_id] = _Assignment(segment=best, score=best_score)
        return best

    def _score_for(
        self, role: AssetRole, pose: Pose, segment: Segment, value: float
    ) -> float | None:
        """A segment's worth to this asset, or ``None`` if it is out of its job."""
        range_m = _range_m(pose.lat, pose.lon, segment.lat_deg, segment.lon_deg)
        if role.cls == _TOWER:
            # A tower is aimed, not sent, so distance is a hard reach rather
            # than a cost: past its camera's useful range there is nothing to
            # trade off, only a direction it cannot resolve anything in.
            if range_m > self._params.tower_reach_m:
                return None
            return value
        if role.cls not in _AIRCRAFT:
            return None
        if role.cls == "fixedwing" and range_m < self._params.plane_min_reach_m:
            # The sweep asset is not given work it is already on top of; it
            # cannot loiter, so a segment under its nose is a segment it will
            # overfly before it can be tasked to it.
            return None
        return value - self._params.travel_cost_per_km * (range_m / 1000.0) * value

    # -- coverage -----------------------------------------------------------

    def _mark_looked_at(self, poses: dict[str, Pose], now: float) -> None:
        radius = self._params.look_radius_m
        for role in self._roles:
            pose = poses.get(role.asset_id)
            if pose is None:
                continue
            reach = self._params.tower_reach_m if role.cls == _TOWER else radius
            held = self._assigned.get(role.asset_id)
            if role.cls == _TOWER:
                # A tower sees only where it is pointed, so only the segment it
                # was aimed at counts. Crediting its whole reach would mark the
                # channel searched from a camera looking one way down it.
                if held is not None:
                    segment = self._segments[held.segment]
                    if _range_m(pose.lat, pose.lon, segment.lat_deg, segment.lon_deg) <= reach:
                        self._coverage.last_seen_t[held.segment] = now
                continue
            for segment in self._segments:
                if _range_m(pose.lat, pose.lon, segment.lat_deg, segment.lon_deg) <= reach:
                    self._coverage.last_seen_t[segment.index] = now

    def _waypoint(
        self,
        role: AssetRole,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        *,
        reason: str,
        task_id: str,
    ) -> WaypointIntent:
        return WaypointIntent(
            asset_id=role.asset_id,
            t=0.0,
            target_lat=lat_deg,
            target_lon=lon_deg,
            target_z=alt_m,
            speed=0.0,
            reason=reason,
            task_id=task_id,
        )


def _cut(geometry: StraitGeometry, segment_m: float) -> tuple[Segment, ...]:
    """Cut the channel into segments, each stamped with its centreline point."""
    count = max(1, int(math.ceil(geometry.length_m / segment_m)))
    segments = []
    for index in range(count):
        s_m = min(geometry.length_m, (index + 0.5) * segment_m)
        lat_deg, lon_deg = geometry.to_position(ChannelPoint(s_m=s_m, w_m=0.0))
        segments.append(Segment(index=index, s_m=s_m, lat_deg=lat_deg, lon_deg=lon_deg))
    return tuple(segments)


def _range_m(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
    """Ground range in metres, through the one converter."""
    offset = geodetic_to_local(GeoPoint(from_lat, from_lon), GeoPoint(to_lat, to_lon))
    return math.hypot(offset.east_m, offset.north_m)
