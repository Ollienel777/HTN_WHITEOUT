"""Command line entry point.

Subcommands: ``run``, ``replay``, ``score``, ``sweep``, ``ablate``, ``serve``.

Only ``run`` and ``score`` do anything yet, and only enough for the gate's
smoke and determinism steps to exercise the real command lines from
``hackathon/SPEC.md`` §6. The episode record set and the scorer land with
their own tickets; until then ``run`` emits a placeholder log that is a pure
function of ``--seed`` and ``--ticks``, and ``score`` reports four finite
zeros. The other subcommands are stubs that refuse loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

#: Ordered scoring axes. The sponsor scores on exactly these four.
AXES: tuple[str, str, str, str] = (
    "coverage",
    "collaboration",
    "efficiency",
    "tracking_accuracy",
)

_NOT_YET = "not implemented yet"


def _transport() -> str:
    """Selected transport. ``WHITEOUT_TRANSPORT`` unset means ``kinematic``."""
    return os.environ.get("WHITEOUT_TRANSPORT") or "kinematic"


def _default_seed() -> int:
    """Episode seed. ``WHITEOUT_SEED`` absent means 0, never "random"."""
    raw = os.environ.get("WHITEOUT_SEED")
    return int(raw) if raw else 0


def cmd_run(args: argparse.Namespace) -> int:
    """Write a placeholder episode log that depends only on seed and ticks."""
    out = Path(args.out)
    if out.parent != Path():
        out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "type": "header",
                "seed": args.seed,
                "ticks": args.ticks,
                "transport": _transport(),
                "schema": "placeholder",
            },
            sort_keys=True,
        )
    ]
    for tick in range(args.ticks):
        lines.append(
            json.dumps(
                {"type": "tick", "tick": tick, "seed": args.seed},
                sort_keys=True,
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"run: wrote {len(lines)} records to {out} (transport={_transport()})")
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
        loaded = json.loads(weights_path.read_text(encoding="utf-8"))
        missing = [axis for axis in AXES if axis not in loaded]
        if missing:
            print(f"score: weights file is missing axes: {missing}", file=sys.stderr)
            return 1
        weights = {axis: float(loaded[axis]) for axis in AXES}
    records = sum(1 for line in log.read_text(encoding="utf-8").splitlines() if line)
    scores = {axis: 0.0 for axis in AXES}
    total = sum(weights[axis] * scores[axis] for axis in AXES)
    for axis in AXES:
        print(f"{axis}: {scores[axis]:.4f}")
    print(f"total: {total:.4f} ({records} records)")
    return 0


def _cmd_stub(name: str) -> int:
    print(f"{name}: {_NOT_YET}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whiteout", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an episode and write its log")
    run.add_argument("--seed", type=int, default=_default_seed())
    run.add_argument("--ticks", type=int, default=400)
    run.add_argument("--out", default="artifacts/episode.jsonl")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="score an episode log")
    score.add_argument("log")
    score.add_argument("--weights", default=None)
    score.set_defaults(func=cmd_score)

    for name, help_text in (
        ("replay", "replay an episode log"),
        ("sweep", "sweep policy parameters"),
        ("ablate", "run the ablation table"),
        ("serve", "serve the viewer"),
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
