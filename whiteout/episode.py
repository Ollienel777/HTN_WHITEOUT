"""The loop: observe, believe, decide, track, post — one tick at a time.

``hackathon/SPEC.md`` §4. Everything this module does is composition. The
belief field, the search policy, the track hold and the transports each arrived
with their own ticket and their own tests; until this existed, every one of
them was reachable only from its own test file, and ``whiteout.cli run`` wrote
a log of parked assets with empty contacts and a placeholder digest. This is
the file that makes them one system.

The order inside a tick, and why it is that order
-------------------------------------------------

1. **Observe.** The transport hands over poses and sensor reports.
2. **Diffuse**, by the tick's own elapsed time, *before* folding in this tick's
   evidence. The field has to be advanced to the instant the measurement was
   taken or the measurement lands on a belief that is one tick stale.
3. **Fold in every report**: positive detections sharpen the field, and a
   footprint that was swept and saw nothing erodes it. Both, because a sweep
   that found nothing is evidence and dropping it is how a search re-checks
   water it has already cleared.
4. **Track.** A detection is a sighting; the hold decides what is worth posting
   and when, and it is the only thing here that talks to the network.
5. **Decide.** The policy sees the updated field, and the contact if one is
   held, and tasks the fleet.
6. **Command**, and record.

Deciding *after* the update rather than before is the difference between a
fleet that reacts this tick and one that reacts next tick; over a 400-tick
episode it is most of the coverage score.

Nothing here is wall-clock and nothing here draws a random number, so an
episode is a pure function of its seed, its transport and its parameters —
which is what the gate's determinism step checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whiteout.belief import BeliefField, ChannelBeliefGrid
from whiteout.belief.geometry import DEFAULT_STRAIT, StraitGeometry
from whiteout.policy import DEFAULT_SEARCH_PARAMS, AssetRole, SearchParams, SearchPolicy
from whiteout.sim import SENSORS, SensorModel, detection_sigma_m, footprint_range_m
from whiteout.tracks import Sighting, TrackHold
from whiteout.types import (
    BeliefDigest,
    Contact,
    EpisodeRecord,
    FleetIntent,
    SensorFootprint,
    SensorReport,
    Truth,
    WorldObservation,
)

__all__ = ["EpisodeRunner", "roles_of", "run_episode", "sweep_likelihood"]

#: The track name this run reports under. The tracks API keys on it, so it is
#: one name for the whole episode: a run that renamed its track mid-episode
#: would read at the far end as two vessels, one of which stopped existing.
DEFAULT_TRACK_NAME = "whiteout-1"


def sweep_likelihood(
    footprint: SensorFootprint, sensor: SensorModel
) -> Callable[[float, float], float]:
    """The likelihood of *seeing nothing* here, as a function of position.

    A swept footprint that reported no detection is evidence, and the shape of
    that evidence is ``1 - P(detect | the vessel is at this cell)``. Inside the
    footprint that is less than one and the cell loses belief; outside it is
    exactly one and the cell is untouched, which is what makes this a
    multiplication the field can take without knowing anything about sensors.

    **It is never zero**, because a sensor that sweeps water and misses is the
    normal case, not the impossible one. A hard cut -- "the vessel is not here,
    probability zero" -- is unrecoverable: one missed detection would delete
    the cell the vessel is actually in and no later evidence could bring it
    back.

    Whether a cell is inside the footprint at all is
    :func:`~whiteout.sim.footprint_range_m`'s answer, not a second copy of the
    cone arithmetic here. The two have to agree exactly, or a sweep erodes
    water the sensor never pointed at.
    """

    def likelihood(lat_deg: float, lon_deg: float) -> float:
        range_m = footprint_range_m(footprint, lat_deg, lon_deg)
        if range_m is None:
            return 1.0
        return 1.0 - sensor.p_detect_at(range_m)

    return likelihood


@dataclass(frozen=True, slots=True)
class _Tick:
    """One tick's derived state, before it becomes a record."""

    digest: BeliefDigest
    contacts: tuple[Contact, ...]
    intent: FleetIntent


class EpisodeRunner:
    """Drives one episode and yields its records.

    Built from the pieces rather than building them, so a test can hand it a
    stub transport, a fake belief field or a hold with no network behind it.
    """

    def __init__(
        self,
        *,
        belief: BeliefField,
        policy: SearchPolicy,
        hold: TrackHold,
        geometry: StraitGeometry = DEFAULT_STRAIT,
    ) -> None:
        self._belief = belief
        self._policy = policy
        self._hold = hold
        self._geometry = geometry
        self._last_t: float | None = None
        self._first_detection_t: float | None = None
        self._detections = 0
        self._sightings = 0
        self._holder: str | None = None
        self._handoffs = 0

    @property
    def first_detection_t(self) -> float | None:
        """When the fleet first saw the vessel. ``None`` if it never did.

        Detection speed is one of ``ARENA.md`` §5's scored criteria, so this
        is reported rather than recomputed from the log by eye.
        """
        return self._first_detection_t

    @property
    def detections(self) -> int:
        """How many detections were folded into the belief this episode."""
        return self._detections

    @property
    def sightings(self) -> int:
        """How many detections became sightings on the track."""
        return self._sightings

    @property
    def handoffs(self) -> int:
        """How many times the asset holding the contact changed.

        ``ARENA.md`` §5 scores collaboration, and a handoff between two assets
        is the thing that word means here — one aircraft picking a contact up
        as another loses it. Counted rather than inferred from the log.
        """
        return self._handoffs

    def note_sighting(self) -> None:
        """Count a sighting that reached the hold without passing through a report.

        Only a test does this. It exists so that the handoff counter can be
        exercised without building a sensor geometry that would make the test
        about ranges rather than about who is holding the contact.
        """
        self._sightings += 1

    def tick(self, observation: WorldObservation) -> _Tick:
        """Fold one observation in and produce this tick's decision."""
        now = observation.t
        elapsed = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if elapsed > 0.0:
            self._belief.diffuse(elapsed)

        for report in observation.reports:
            self._fold(report, now)

        self._hold.tick(now)
        contacts = self._contacts(now)
        held = self._hold.last_sighting
        lit = held is not None and self._hold.state in ("held", "coasting")
        intent = self._policy.decide(
            observation,
            self._belief,
            held_lat_deg=held.lat_deg if lit and held is not None else None,
            held_lon_deg=held.lon_deg if lit and held is not None else None,
        )
        return _Tick(digest=self._digest(now), contacts=contacts, intent=intent)

    # -- internals ----------------------------------------------------------

    def _fold(self, report: SensorReport, now: float) -> None:
        sensor = SENSORS.get(_class_of(report.asset_id))
        for detection in report.detections:
            self._detections += 1
            if self._first_detection_t is None:
                self._first_detection_t = now
            sigma = detection_sigma_m(report.footprint.radius * 0.5)
            self._belief.update_detection(detection.lat, detection.lon, sigma_m=sigma)
            self._sightings += 1
            self._hold.sight(
                Sighting(
                    t=report.t,
                    lat_deg=detection.lat,
                    lon_deg=detection.lon,
                    asset_id=report.asset_id,
                )
            )
        if report.negative and sensor is not None:
            self._belief.update_likelihood(sweep_likelihood(report.footprint, sensor))

    def _contacts(self, now: float) -> tuple[Contact, ...]:
        """This tick's contact, in the log's own lifecycle vocabulary.

        :data:`~whiteout.types.CONTACT_STATES` is what the viewer paints and
        what the scorer's collaboration axis reads, and it is not the same
        vocabulary :class:`~whiteout.tracks.TrackHold` reports — the hold is
        answering "should I post?" and this is answering "what does the
        operator see?". The mapping:

        ``confirming``
            one sighting so far. A single fix is a candidate, not a track, and
            calling it tracked is how a false alarm becomes a confident line
            on the map.
        ``handed_off``
            the asset supplying fixes changed since the last tick. Held for
            exactly the tick it happens on, because a handoff is an event; the
            next tick is ``tracked`` again.
        ``tracked``
            fixes are arriving and one asset is holding it.
        ``lost``
            nothing for longer than the hold's coast time.
        """
        last = self._hold.last_sighting
        if last is None or self._hold.state == "unseen":
            self._holder = None
            return ()
        holder = self._hold.holder
        if self._hold.state == "lost":
            state = "lost"
        elif self._sightings <= 1:
            state = "confirming"
        elif holder != self._holder and self._holder is not None:
            state = "handed_off"
            self._handoffs += 1
        else:
            state = "tracked"
        self._holder = holder
        contact = Contact(
            contact_id=self._hold.name,
            t=now,
            state=state,
            lat=last.lat_deg,
            lon=last.lon_deg,
            confidence=self._belief.probability_at(last.lat_deg, last.lon_deg),
            classification="vessel",
            assigned_asset_id=holder,
        )
        return (contact,)

    def _digest(self, now: float) -> BeliefDigest:
        peak = self._belief.peak()
        shape = getattr(self._belief, "shape", (1, 1))
        return BeliefDigest(
            t=now,
            entropy=self._belief.entropy(),
            mass=self._belief.mass(),
            peak_lat=peak.lat_deg,
            peak_lon=peak.lon_deg,
            peak_p=peak.probability,
            covered_fraction=self._policy.covered_fraction(now),
            grid_shape=shape,
        )


def _class_of(asset_id: str) -> str:
    """The vehicle class an asset id implies, for the sensor lookup.

    The report does not carry a class — :class:`~whiteout.types.SensorReport`
    is keyed by ``asset_id`` alone — and the pose that does is a separate
    collection. The ids the fleet uses are prefixed by class, so this reads
    the prefix rather than building a second index every tick.
    """
    head = asset_id.split("-", 1)[0]
    return {"fw": "fixedwing", "quad": "quad", "tower": "tower", "rover": "rover"}.get(head, head)


def roles_of(observation: WorldObservation) -> tuple[AssetRole, ...]:
    """One :class:`~whiteout.policy.AssetRole` per asset in an observation.

    Roles are read off the fleet that reported rather than from a constant, so
    the policy tasks whatever the selected transport actually has. The arena's
    roster is configuration; the offline world's is not the same list, and
    hard-coding either would silently drop an asset from the other.
    """
    return tuple(AssetRole(asset_id=pose.asset_id, cls=pose.cls) for pose in observation.poses)


def run_episode(
    transport: object,
    *,
    ticks: int,
    hold: TrackHold,
    geometry: StraitGeometry = DEFAULT_STRAIT,
    params: SearchParams = DEFAULT_SEARCH_PARAMS,
    schema_version: int,
) -> tuple[list[EpisodeRecord], EpisodeRunner | None]:
    """Run ``ticks`` ticks against ``transport`` and return the log and the runner.

    ``transport`` is duck-typed rather than annotated as
    :class:`~whiteout.transport.base.Transport` so that the optional ``truth()``
    the offline world offers, and the arena does not, stays off the seam.

    The runner is built on the **first** observation, because that is the first
    moment the fleet is known, and that observation is then processed like any
    other — peeking a tick to read the roster and throwing it away would cost a
    tick of world time on a link where ticks are seconds.

    Returns ``None`` for the runner when ``ticks`` is zero, which is the only
    case where no observation was ever taken.
    """
    runner: EpisodeRunner | None = None
    records: list[EpisodeRecord] = []
    for _ in range(ticks):
        observation = transport.observe()  # type: ignore[attr-defined]
        if runner is None:
            runner = EpisodeRunner(
                belief=ChannelBeliefGrid(geometry),
                policy=SearchPolicy(roles_of(observation), geometry, params),
                hold=hold,
                geometry=geometry,
            )
        outcome = runner.tick(observation)
        transport.command(outcome.intent)  # type: ignore[attr-defined]
        records.append(
            EpisodeRecord(
                schema_version=schema_version,
                t=observation.t,
                observation=observation,
                intent=outcome.intent,
                belief_digest=outcome.digest,
                contacts=outcome.contacts,
                truth=_truth_of(transport, observation.t),
            )
        )
    return records, runner


def _truth_of(transport: object, t: float) -> Truth:
    """``transport.truth()`` when it has one, and empty truth when it does not.

    An arena run has no ground truth and must not pretend to. Empty targets is
    the honest value there, and the scorer's accuracy axis reads it as "not
    measurable" rather than as "the vessel was at the origin".
    """
    getter = getattr(transport, "truth", None)
    if getter is None:
        return Truth(t=t, targets=())
    value = getter()
    return value if isinstance(value, Truth) else Truth(t=t, targets=())
