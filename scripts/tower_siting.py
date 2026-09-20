#!/usr/bin/env python
"""Where the two towers should stand, and what the current answer costs.

Issue #68. ``ARENA.md`` §3: the aircraft start near-optimal and should be left
alone, but **tower lat/lon is the one `.env` edit Dominion Dynamics
encouraged**. It is therefore the only placement decision in the run that is
ours to make, and it was going unmade — the towers sit where the stock
`.env.example` puts them, which is 1.6 km and 1.7 km off the channel.

What this measures
------------------

For a tower at a given lat/lon and height, sweep its camera through a full
360° and count the channel's water cells it could detect a vessel in. "Could
detect" is :func:`whiteout.belief.negative.non_detection_likelihood` — the
same model the belief field erodes with, so this is not a second opinion about
the sensor, it is the one we already act on.

The threshold is a **detection probability**, not a range: a cell counts when
one look would find a vessel there with probability above ``--threshold``. A
range cut-off would have to pick a number; this inherits the pixel-count
physics the detector's own statistic implies, where a target's pixels fall as
:math:`1/r^3` once obliquity is included.

What it is not
--------------

**These percentages are our model's estimate and not a measurement.** They
rest on ``SweepParams.vessel_length_m`` (assumed — ``ARENA.md`` says "a boat"
and gives no figure) and ``half_pixels`` (read off the detector's own working
range). Neither has been checked against the arena.

**The comparison between two sitings is the robust part**, because it is
driven by the :math:`r^3` geometry rather than by those constants: a tower
twice as far from the water sees an eighth as well whatever a hull's length
turns out to be. Quote the ratio to a judge, not the absolute.

**There is no terrain model here.** The height passed in is the height used,
so comparing two sites assumes you know the ground elevation at both. Read it
off gzweb by clicking the point; that is the one input this script cannot
supply, and the one the answer is second-most sensitive to.

What review corrected
---------------------

Both worth keeping, because both are the kind of error that reads as a result.

**It did not score the placement we already have.** The siting in
``whiteout/transport/kinematic.py``'s ``_TOWER_STATIONS`` beats anything this
script generates with a setback, and the first version recommended
coordinates 190 m away and 3.6 points worse without mentioning it. See
:func:`incumbent`.

**It held the camera level.** Tilt is commanded — ``servo set 2`` — and a
level tower is blind inside ``h / tan(vfov/2)``, which grows with height. So
the height column was confounded: sweeping tilt takes the 226.5 m site from
18.7% to 26.1% and the 116.8 m one from 14.5% to 15.8%. ``--level``
reproduces the old behaviour, deliberately kept.

Usage
-----

::

    python scripts/tower_siting.py                    # the three-way comparison
    python scripts/tower_siting.py --survey           # candidates along the channel
    python scripts/tower_siting.py --at 72.0016 -94.8812 116.8
    python scripts/tower_siting.py --inland 0         # what the setback costs
    python scripts/tower_siting.py --level            # the confounded version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Run from anywhere, and against *this* tree: an ambient editable install
    # can point at another worktree (scripts/gate.py has the long version).
    sys.path.insert(0, str(REPO_ROOT))

from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint, StraitGeometry  # noqa: E402
from whiteout.belief.grid import ChannelBeliefGrid  # noqa: E402
from whiteout.belief.negative import (  # noqa: E402
    DEFAULT_SWEEP,
    SweepParams,
    non_detection_likelihood,
)
from whiteout.vision.camera import CAMERAS  # noqa: E402
from whiteout.vision.projection import CameraPose  # noqa: E402

#: The stock `.env.example` siting, and the heights #108 measured off gzweb.
#: This is what the run uses today.
AS_SITED: tuple[tuple[str, float, float, float], ...] = (
    ("tower-1", 71.980671, -94.853711, 116.8),
    ("tower-2", 72.011778, -94.804721, 226.5),
)

#: How far inland of the water's edge a generated candidate sits, and it is
#: **not free**: measured, 150 m costs 2.9 points of coverage at 116.8 m and
#: 1.8 at 226.5 m. An earlier version of this comment claimed "under a
#: percentage point" without checking, and that setback is the entire reason
#: the generated sites lose to :func:`incumbent` below.
#:
#: Kept as a default rather than deleted, because the alternative is siting a
#: mast exactly on a waterline our own channel model draws from seven
#: vertices. ``--inland 0`` makes the trade visible.
INLAND_M = 150.0

#: Bearings the sweep samples. Ten degrees against a 60° horizontal field
#: oversamples by six, which is enough that a cell cannot fall between looks.
SWEEP_STEP_DEG = 10

#: Tilts the sweep samples, degrees, negative pointing down at the water.
#:
#: **A tower is not a fixed camera.** ``servo set 2`` drives tilt over
#: :data:`whiteout.vision.tower.TOWER_PAN_TILT`'s range, and
#: ``whiteout/vision/tower.py`` already models it. The first version of this
#: script pinned pitch at zero, which blinds a tower inside ``h / tan(vfov/2)``
#: — 358 m at 116.8 m of height and **695 m at 226.5 m** — and so charged the
#: taller mast for a near field it can simply tilt down into. That understated
#: the 226.5 m bank site by 6.8 points, and because the penalty grows with
#: height it understated the value of height itself.
SWEEP_TILTS_DEG: tuple[float, ...] = (0.0, -5.0, -10.0, -20.0, -35.0, -55.0, -80.0)


def incumbent() -> tuple[tuple[str, float, float, float], ...]:
    """The siting already committed in ``whiteout/transport/kinematic.py``.

    Chosen in #118 by a rule of thumb — 18% and 82% along the channel, on the
    water's edge, alternating banks — and it **beats** anything this script
    generates with a setback.

    It is here because the first version of this script did not have it, and
    so recommended a siting 190 m away and 3.6 points worse than one already
    in the tree. A script that compares placements has to compare the
    placement we already have. Imported rather than copied, so the two cannot
    drift apart.
    """
    from whiteout.transport.kinematic import _TOWER_STATIONS

    heights = [height for _, _, _, height in AS_SITED]
    return tuple(
        (f"tower-{index + 1}", lat, lon, height)
        for index, ((lat, lon, _), height) in enumerate(zip(_TOWER_STATIONS, heights, strict=True))
    )


def water_cells(geometry: StraitGeometry) -> list[tuple[float, float]]:
    """Every water cell's centre, taken from the belief grid itself.

    Asked of :class:`ChannelBeliefGrid` rather than recomputed, so that this
    script and the field it is reasoning about cannot disagree about which
    water exists.
    """
    cells: list[tuple[float, float]] = []

    def collect(lat_deg: float, lon_deg: float) -> float:
        cells.append((lat_deg, lon_deg))
        return 1.0

    ChannelBeliefGrid(geometry).update_likelihood(collect)
    return cells


def swept(
    lat_deg: float,
    lon_deg: float,
    height_m: float,
    cells: list[tuple[float, float]],
    *,
    threshold: float,
    tilts: tuple[float, ...] = SWEEP_TILTS_DEG,
    params: SweepParams = DEFAULT_SWEEP,
) -> set[int]:
    """Cells this tower could detect a vessel in, sweeping pan **and tilt**.

    Both, because both are commanded: ``servo set 1`` is pan and ``servo set
    2`` is tilt. Sweeping yaw alone holds the camera level and blinds the
    tower inside ``h / tan(vfov/2)`` — 695 m from the taller mast — which is
    a large piece of channel it can reach by tilting down, charged against it
    for nothing. See :data:`SWEEP_TILTS_DEG`.
    """
    camera = CAMERAS["tower"]
    seen: set[int] = set()
    for bearing in range(0, 360, SWEEP_STEP_DEG):
        for tilt in tilts:
            likelihood = non_detection_likelihood(
                camera,
                CameraPose(
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    alt_m=height_m,
                    yaw_deg=float(bearing),
                    pitch_deg=float(tilt),
                    roll_deg=0.0,
                ),
                params=params,
            )
            for index, (cell_lat, cell_lon) in enumerate(cells):
                if index in seen:
                    continue
                if 1.0 - likelihood(cell_lat, cell_lon) > threshold:
                    seen.add(index)
    return seen


def bank_site(
    geometry: StraitGeometry, along: float, bank: int, *, inland_m: float = INLAND_M
) -> tuple[float, float]:
    """A point ``inland_m`` beyond the water's edge, ``along`` the channel."""
    s_m = along * geometry.length_m
    offset = bank * (geometry.half_width_at(s_m) + inland_m)
    return geometry.to_position(ChannelPoint(s_m=s_m, w_m=offset))


def _report(
    label: str,
    roster: tuple[tuple[str, float, float, float], ...],
    cells: list[tuple[float, float]],
    threshold: float,
    tilts: tuple[float, ...],
) -> float:
    total = len(cells)
    sets = [
        swept(lat, lon, alt, cells, threshold=threshold, tilts=tilts) for _, lat, lon, alt in roster
    ]
    union: set[int] = set().union(*sets) if sets else set()
    overlap = set.intersection(*sets) if len(sets) > 1 else set()
    print(f"\n{label}")
    for (name, lat, lon, alt), seen in zip(roster, sets, strict=True):
        print(f"  {name:8s} {lat:10.6f},{lon:11.6f}  {alt:6.1f} m   {len(seen) / total:6.1%}")
    share = len(union) / total
    both = len(overlap) / total
    print(f"  {'union':8s} {'':32s}{share:6.1%}   overlap {both:.1%}")
    return share


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="detection probability a cell must reach to count as covered",
    )
    parser.add_argument(
        "--at",
        nargs=3,
        type=float,
        metavar=("LAT", "LON", "HEIGHT_M"),
        help="score one site instead of the comparison",
    )
    parser.add_argument(
        "--survey",
        action="store_true",
        help="score candidate bank sites along the channel, both banks",
    )
    parser.add_argument(
        "--inland",
        type=float,
        default=INLAND_M,
        help="metres beyond the water's edge a generated candidate sits",
    )
    parser.add_argument(
        "--level",
        action="store_true",
        help="hold the camera level instead of sweeping tilt — what the first "
        "version of this script did, kept so the error stays reproducible",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold < 1.0:
        # Unvalidated, a negative threshold counted every cell as covered,
        # including ones no ray reaches, and printed 100% for a tower facing
        # away from the channel.
        parser.error(
            f"--threshold is a detection probability and must lie in (0, 1), got {args.threshold!r}"
        )
    tilts = (0.0,) if args.level else SWEEP_TILTS_DEG

    geometry = DEFAULT_STRAIT
    cells = water_cells(geometry)
    print(f"{len(cells)} water cells over {geometry.length_m:.0f} m of channel")
    print(f"covered = one look detects a vessel there with p > {args.threshold:.0%}")

    if args.at is not None:
        lat, lon, height = args.at
        seen = swept(lat, lon, height, cells, threshold=args.threshold, tilts=tilts)
        channel = geometry.to_channel(lat, lon)
        print(
            f"\n{lat:.6f},{lon:.6f} at {height:.1f} m: {len(seen) / len(cells):.1%}"
            f"   ({channel.s_m:.0f} m along, {channel.w_m:+.0f} m across the centreline)"
        )
        return 0

    if args.survey:
        print("\nCandidate bank sites, at each of the two heights the arena reports:")
        for along in (0.20, 0.35, 0.50, 0.65, 0.80):
            for bank, side in ((1, "N"), (-1, "S")):
                lat, lon = bank_site(geometry, along, bank, inland_m=args.inland)
                for height in (AS_SITED[0][3], AS_SITED[1][3]):
                    seen = swept(lat, lon, height, cells, threshold=args.threshold, tilts=tilts)
                    print(
                        f"  s={along:.0%} {side} {height:6.1f} m  {len(seen) / len(cells):6.1%}"
                        f"   {lat:.6f},{lon:.6f}"
                    )
        return 0

    north = bank_site(geometry, 0.20, 1, inland_m=args.inland)
    south = bank_site(geometry, 0.80, -1, inland_m=args.inland)
    generated = (
        ("tower-1", north[0], north[1], AS_SITED[0][3]),
        ("tower-2", south[0], south[1], AS_SITED[1][3]),
    )

    scored = [
        (label, roster, _report(label, roster, cells, args.threshold, tilts))
        for label, roster in (
            ("As sited today (stock .env.example):", AS_SITED),
            ("_TOWER_STATIONS, already committed in kinematic.py:", incumbent()),
            (f"Generated bank sites, {args.inland:.0f} m inland:", generated),
        )
    ]
    best_label, best_roster, best_share = max(scored, key=lambda row: row[2])
    print(f"\nBest of the three: {best_label.rstrip(':')} at {best_share:.1%}\n")

    print(
        "The heights are the ones the towers have where they stand now. There"
        "\nis no terrain model here, so read the ground elevation at any"
        "\nproposed point off gzweb and re-score it with --at before trusting a"
        "\nrow: reach scales with height, and a site at sea level is worth far"
        "\nless than this says.\n"
    )
    if best_roster is AS_SITED:
        print("  Nothing to change: the current siting already wins.")
        return 0
    for index, (name, lat, lon, _) in enumerate(best_roster):
        print(f"  ASSET_{'2' if index == 0 else '5'}=tower,{name},{lat:.6f},{lon:.6f}")
    print("\nNothing else in .env - DD warned the rest breaks the sim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
