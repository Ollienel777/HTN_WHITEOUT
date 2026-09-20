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
from typing import Any

from whiteout.episode import DEFAULT_TRACK_NAME, run_episode
from whiteout.log import SCHEMA_VERSION, EpisodeLogError, write_episode_log
from whiteout.serve import ServeError, open_viewer_server, resolve_port, viewer_url
from whiteout.sim import SimTransport
from whiteout.tracks import TrackFix, TrackHold, TrackPoster, TracksClient, TracksError
from whiteout.transport import TransportError, create_transport, selected_transport_name
from whiteout.types import EpisodeRecord

#: Ordered scoring axes. The sponsor scores on exactly these four.
AXES: tuple[str, str, str, str] = (
    "coverage",
    "collaboration",
    "efficiency",
    "tracking_accuracy",
)

_NOT_YET = "not implemented yet"


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


def cmd_run(args: argparse.Namespace) -> int:
    """Run one episode end to end and write its log.

    Observe, believe, decide, track, record — :mod:`whiteout.episode` owns the
    order and this owns the wiring: which transport, which roles, and whether
    anything is posted to the tracks API. Posting is **off unless an endpoint
    is given**, because the gate runs this command twice on every commit and
    must never open a socket (``SPEC.md`` §6).
    """
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
        transport = _build_transport(name, seed=seed, pose_age_seconds=args.pose_age)
    except TransportError as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 1

    try:
        poster, closing = _open_poster(args.tracks_endpoint)
    except TracksError as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 1

    records: list[EpisodeRecord] = []
    runner = None
    # `connect` is inside both the `try/except` and the `try/finally`: a
    # transport that refuses to start — `SPEC.md` §7 makes that sitl's normal
    # path — must produce the diagnostic `base.TransportError` promises, and a
    # connect that fails partway must still be `close()`d.
    try:
        try:
            transport.connect()
            hold = TrackHold(args.track_name, poster)
            records, runner = run_episode(
                transport,
                ticks=args.ticks,
                hold=hold,
                schema_version=SCHEMA_VERSION,
            )
        finally:
            transport.close()
            if closing is not None:
                closing.close()
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
    if runner is not None:
        first = runner.first_detection_t
        seen = "never" if first is None else f"t={first:.1f}s"
        print(
            f"run: {runner.detections} detections, first {seen}; "
            f"track {hold.state} after {hold.posts} posts"
        )
    return 0


def _build_transport(name: str, *, seed: int, pose_age_seconds: float | None) -> Any:
    """The far side of the seam for this run.

    ``kinematic`` — ``SPEC.md`` §7's fake, and the default — is served by
    :class:`whiteout.sim.SimTransport`: a channel with a vessel in it and a
    fleet that flies the waypoints it is given, which is what an episode needs
    to be an episode. It is built **here**, in the composition root, rather
    than by :func:`~whiteout.transport.create_transport`, because the transport
    package is guarded against importing :mod:`whiteout.sim` at all
    (``tests/test_transport.py``) — an adapter to a real vehicle must not know
    what a simulated one would do. Every other name goes to the registry,
    which is the one place that turns the environment variable into an object.
    """
    if name == "kinematic":
        return SimTransport(seed=seed, pose_age_seconds=pose_age_seconds)
    return create_transport(name, seed=seed, pose_age_seconds=pose_age_seconds)


class _NullPoster:
    """A poster that accepts fixes and sends nothing, for a run with no endpoint.

    The alternative is threading an optional through
    :class:`~whiteout.tracks.TrackHold`, which would put "are we posting?" into
    the one class whose job is deciding *what* to post. A hold with this behind
    it still counts its posts, so an offline run reports the same numbers a
    live one would.
    """

    def submit(self, fix: TrackFix) -> None:
        """Accept and drop."""


def _open_poster(endpoint: str | None) -> tuple[Any, Any]:
    """The thing fixes go to, and the thing that has to be closed afterwards.

    ``None`` for the endpoint means an offline run: nothing opens, nothing is
    sent, and the second element is ``None`` so the caller's ``finally`` has
    nothing to do.
    """
    if endpoint is None:
        return _NullPoster(), None
    poster = TrackPoster(TracksClient(endpoint))
    return poster, poster


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
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--ticks", type=int, default=400)
    run.add_argument("--out", default="artifacts/episode.jsonl")
    run.add_argument(
        "--tracks-endpoint",
        default=None,
        help=(
            "post the episode's track to this tracks API base URL. Omitted, the run "
            "is entirely offline and opens no socket, which is what the gate needs"
        ),
    )
    run.add_argument(
        "--track-name",
        default=DEFAULT_TRACK_NAME,
        help="the name this episode reports its one track under",
    )
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
