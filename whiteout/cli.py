"""Command line entry point.

Subcommands: ``run``, ``replay``, ``score``, ``sweep``, ``ablate``, ``serve``.

Only ``run``, ``score`` and ``serve`` do anything yet, and only enough for
the gate's smoke and determinism steps to exercise the real command lines from
``hackathon/SPEC.md`` §6. ``run`` drives the selected transport (§4's seam)
one tick at a time and writes a real, validated episode log; the belief
digest, the contacts and the truth on each record are placeholders, because
the belief field, the estimator and the sim land with their own tickets, and
the episode loop that fills them is issue #25. ``score`` reports four finite
zeros. ``serve`` hands the source tree out to a browser so that the viewer in
``viz/`` can read an episode log beside it, on ``PORT`` or an ephemeral port
(``SPEC.md`` §6). The other subcommands are stubs that refuse loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from whiteout.coordinate import DEFAULT_TRACK_NAME, Coordinator, SightingSource
from whiteout.geo import ARENA_ORIGIN
from whiteout.log import SCHEMA_VERSION, EpisodeLogError, write_episode_log
from whiteout.policy import AssetRole
from whiteout.serve import ServeError, open_viewer_server, resolve_port, viewer_url
from whiteout.tracks.client import TrackPoster, TracksClient, TracksError, endpoint_from_env
from whiteout.tracks.maintain import TrackHold
from whiteout.transport import TransportError, create_transport, selected_transport_name
from whiteout.transport.arena import host_from_env
from whiteout.types import (
    BeliefDigest,
    BeliefFrame,
    Contact,
    EpisodeRecord,
    FleetIntent,
    SightingRefusal,
    Truth,
)
from whiteout.vision.camera import VisionError
from whiteout.vision.motion import MotionGate
from whiteout.vision.sightings import VisionSightings, arena_feeds

#: Ordered scoring axes. The sponsor scores on exactly these four.
AXES: tuple[str, str, str, str] = (
    "coverage",
    "collaboration",
    "efficiency",
    "tracking_accuracy",
)

_NOT_YET = "not implemented yet"

#: The water plane's altitude in the datum the poses are reported in, metres
#: (#76). The arena adapter reports MSL and ``ARENA.md``'s water plane is
#: ``z = 0`` in the world, so zero is the number here. It is threaded from
#: this one place into both the coordinator's negative-information update and
#: the camera path's projection: two defaults that happened to agree would be
#: two numbers to change, and the day they disagreed the belief and the
#: sightings would be measuring heights above different surfaces.
GROUND_ALT_M = 0.0


def _resolve_seed(explicit: int | None) -> int | None:
    """Episode seed: ``--seed`` wins, else ``WHITEOUT_SEED``, else 0.

    Returns ``None`` when the environment holds something that is not an
    integer, so the caller diagnoses it instead of raising. This resolution is
    deliberately deferred to ``cmd_run`` rather than used as an argparse
    default: as a default it ran at parser-construction time, so a bad
    ``WHITEOUT_SEED`` took down every subcommand, including the five that
    accept no seed at all, and ``--help``.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get("WHITEOUT_SEED")
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return None


def _placeholder_digest(t: float) -> BeliefDigest:
    """A belief digest for a build with no belief field yet.

    Every field is finite, and zero-valued apart from the peak, over a 1×1
    grid, so the record validates and the viewer has something to parse. The
    belief ticket replaces this; nothing may read these numbers as meaning
    anything.

    The peak is :data:`~whiteout.geo.ARENA_ORIGIN` rather than ``(0, 0)``: the
    frame of record is lat/lon (``SPEC.md`` §5), where ``(0, 0)`` is a real
    place in the Gulf of Guinea and reads as a plausible datum rather than as
    the placeholder it is.
    """
    return BeliefDigest(
        t=t,
        entropy=0.0,
        mass=0.0,
        peak_lat=ARENA_ORIGIN.lat_deg,
        peak_lon=ARENA_ORIGIN.lon_deg,
        peak_p=0.0,
        covered_fraction=0.0,
        grid_shape=(1, 1),
    )


def _coordinator_for(
    transport: object,
    poster: TrackPoster | None = None,
    sightings: SightingSource | None = None,
) -> Coordinator | None:
    """A coordinator over whatever assets this transport rosters.

    ``None`` when the transport does not publish a roster. The kinematic fake
    does not, so ``run`` against it stays exactly the observed-and-logged
    episode it was — this is additive, and the gate's smoke and determinism
    steps are unaffected by it.
    """
    assets = getattr(transport, "assets", None)
    if not assets:
        return None
    roles = tuple(AssetRole(asset.asset_id, asset.cls) for asset in assets)
    hold: TrackHold | None = None
    if poster is not None:
        hold = TrackHold(DEFAULT_TRACK_NAME, poster)
    return Coordinator(roles, hold=hold, sightings=sightings, ground_alt_m=GROUND_ALT_M)


def _arena_camera() -> VisionSightings | None:
    """The arena's cameras as a sighting source, or ``None`` if there are none.

    Arena-only, and built here rather than inside :func:`_coordinator_for` so
    that the run loop keeps something to open and close. The host is the
    transport's own — :func:`~whiteout.transport.arena.host_from_env` — so the
    cameras and the MAVLink links can never be pointed at different arenas.

    **A camera that will not open does not end the run.** Each feed is drained
    by its own thread which records a stream failure instead of raising
    (:attr:`~whiteout.vision.sightings.CameraFeed.error`), so four assets over
    a network degrade one at a time to "no sightings from that asset". Only a
    :class:`~whiteout.vision.camera.VisionError` while *building* the source
    is caught here, and it too degrades rather than aborting: an arena episode
    is a live run that cannot be repeated, and half a fleet watching beats
    none.
    """
    try:
        feeds = arena_feeds(host_from_env())
        if not feeds:
            print("run: no cameras in the arena roster; running without sightings")
            return None
        return VisionSightings(feeds=feeds, ground_alt_m=GROUND_ALT_M)
    except VisionError as exc:
        print(f"run: no camera sightings ({exc}); running blind", file=sys.stderr)
        return None


def _arena_poster() -> TrackPoster | None:
    """A poster for the tracks API, or ``None`` when no endpoint is configured.

    ``WHITEOUT_TRACKS_ENDPOINT`` is the gate, and it is the client's own
    (:func:`~whiteout.tracks.client.endpoint_from_env`): unset, it raises
    rather than inventing an address, and this turns that into "do not submit"
    rather than into a failed run. So a run posts to a live endpoint exactly
    when that variable names one, and never by default.
    """
    try:
        endpoint = endpoint_from_env()
    except TracksError as exc:
        print(f"run: not submitting tracks ({exc})")
        return None
    print(f"run: submitting held tracks to {endpoint}")
    return TrackPoster(TracksClient(endpoint))


def cmd_run(args: argparse.Namespace) -> int:
    """Drive the selected transport for ``--ticks`` ticks and write its log."""
    seed = _resolve_seed(args.seed)
    if seed is None:
        raw = os.environ.get("WHITEOUT_SEED")
        print(f"run: WHITEOUT_SEED={raw} is not an integer", file=sys.stderr)
        return 1
    if seed < 0:
        # `np.random.default_rng` rejects a negative seed with an eight-frame
        # numpy traceback. Diagnosed here, alongside the non-integer case, for
        # the same reason: a bad seed is the caller's mistake, not a crash.
        # Normalising it (`abs`) would silently merge two distinct episodes.
        print(f"run: seed {seed} is negative; episode seeds are non-negative", file=sys.stderr)
        return 1
    try:
        name = selected_transport_name()
        transport = create_transport(name, seed=seed, pose_age_seconds=args.pose_age)
    except TransportError as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 1
    records: list[EpisodeRecord] = []
    camera: VisionSightings | None = None
    poster: TrackPoster | None = None
    # `connect` is inside both the `try/except` and the `try/finally`: a
    # transport that refuses to start — `SPEC.md` §7 makes that sitl's normal
    # path — must produce the diagnostic `base.TransportError` promises, and a
    # connect that fails partway must still be `close()`d.
    try:
        try:
            transport.connect()
            if name == "arena":
                # Arena-only, deliberately. The gate's determinism step runs
                # the smoke episode twice on the default kinematic transport
                # and compares bytes; a camera or a poster on that path would
                # put a network and a wall clock inside a run that has to be
                # reproducible, so neither is ever built for it.
                camera = _arena_camera()
                poster = _arena_poster()
            sightings: SightingSource | None = None
            if camera is not None:
                camera.open()
                # The gate keeps ice out of the belief (#107). It is a
                # decorator over the source, so the loop below cannot tell it
                # is there.
                sightings = MotionGate(camera)
            coordinator = _coordinator_for(transport, poster=poster, sightings=sightings)
            for _tick in range(args.ticks):
                observation = transport.observe()
                if coordinator is None:
                    # No roster to build roles from, so there is nothing to
                    # coordinate. The episode is still recorded: an observed
                    # run with no tasking is a valid, and readable, log.
                    intent = FleetIntent(t=observation.t, intents=())
                    digest = _placeholder_digest(observation.t)
                    contacts: tuple[Contact, ...] = ()
                    refusals: tuple[SightingRefusal, ...] = ()
                    field: BeliefFrame | None = None
                else:
                    outcome = coordinator.tick(observation)
                    intent, digest, contacts, refusals = (
                        outcome.intent,
                        outcome.digest,
                        outcome.contacts,
                        outcome.refusals,
                    )
                    field = outcome.field
                if not args.dry_run:
                    transport.command(intent)
                records.append(
                    EpisodeRecord(
                        schema_version=SCHEMA_VERSION,
                        t=observation.t,
                        observation=observation,
                        intent=intent,
                        belief_digest=digest,
                        contacts=contacts,
                        truth=Truth(t=observation.t, targets=()),
                        refusals=refusals,
                        belief_field=field,
                    )
                )
        finally:
            transport.close()
            # Both outlive the tick loop on threads of their own, so both are
            # brought down here: a run that failed partway must not leave four
            # camera threads draining sockets, and the poster drains what is
            # pending on the way out so the last held fix is not lost.
            if camera is not None:
                camera.close()
            if poster is not None:
                poster.close()
    except TransportError as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    try:
        written = write_episode_log(out, records)
    except EpisodeLogError as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 1
    print(f"run: wrote {written} records to {out} (transport={name})")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Print the four scoring axes for an episode log."""
    log = Path(args.log)
    if not log.is_file():
        print(f"score: no such episode log: {log}", file=sys.stderr)
        return 1
    weights = {axis: 0.25 for axis in AXES}
    if args.weights is not None:
        weights_path = Path(args.weights)
        if not weights_path.is_file():
            print(f"score: no such weights file: {weights_path}", file=sys.stderr)
            return 1
        try:
            loaded = json.loads(weights_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"score: weights file is not valid JSON: {exc}", file=sys.stderr)
            return 1
        if not isinstance(loaded, dict):
            print("score: weights file is not a JSON object", file=sys.stderr)
            return 1
        missing = [axis for axis in AXES if axis not in loaded]
        if missing:
            print(f"score: weights file is missing axes: {missing}", file=sys.stderr)
            return 1
        try:
            weights = {axis: float(loaded[axis]) for axis in AXES}
        except (TypeError, ValueError) as exc:
            print(f"score: weights file has a non-numeric axis: {exc}", file=sys.stderr)
            return 1
    records = sum(1 for line in log.read_text(encoding="utf-8").splitlines() if line)
    scores = {axis: 0.0 for axis in AXES}
    total = sum(weights[axis] * scores[axis] for axis in AXES)
    for axis in AXES:
        print(f"{axis}: {scores[axis]:.4f}")
    print(f"total: {total:.4f} ({records} records)")
    return 0


def cmd_serve(_args: argparse.Namespace) -> int:
    """Serve the viewer until interrupted.

    No default port and no reuse: ``PORT`` decides, or the kernel does. Two
    worktrees each running this command therefore never answer for each
    other, which is what ``SPEC.md`` §6 asks of it.
    """
    try:
        server = open_viewer_server(resolve_port())
    except ServeError as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 1
    print(f"serve: {viewer_url(server)}")
    print("serve: ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nserve: stopped")
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _cmd_stub(name: str) -> int:
    print(f"{name}: {_NOT_YET}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whiteout", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an episode and write its log")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="decide and log, but send no waypoint to the fleet",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--ticks", type=int, default=400)
    run.add_argument("--out", default="artifacts/episode.jsonl")
    run.add_argument(
        "--pose-age",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "report every pose as a fix taken SECONDS before its tick "
            "(measured_t = t - SECONDS); omitted, no measurement time is reported"
        ),
    )
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="score an episode log")
    score.add_argument("log")
    score.add_argument("--weights", default=None)
    score.set_defaults(func=cmd_score)

    serve = sub.add_parser("serve", help="serve the viewer on PORT, or an ephemeral port")
    serve.set_defaults(func=cmd_serve)

    for name, help_text in (
        ("replay", "replay an episode log"),
        ("sweep", "sweep policy parameters"),
        ("ablate", "run the ablation table"),
    ):
        stub = sub.add_parser(name, help=help_text)
        stub.set_defaults(func=lambda _args, _name=name: _cmd_stub(_name))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
