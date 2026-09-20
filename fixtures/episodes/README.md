# Committed episodes

Real seeded runs, not invented data. The viewer loads `demo.jsonl` when it is
opened with no log, so this is the first thing a judge sees if anything goes
wrong with a live run — which means every number in it has to have been
computed rather than typed.

## Regenerating

One command, one named seed:

```bash
python -m whiteout.cli run --seed 7 --ticks 400 --out fixtures/episodes/demo.jsonl
```

`WHITEOUT_TRANSPORT` unset selects `kinematic` (`SPEC.md` §7), which never
opens a socket, so this reproduces anywhere. The run is a pure function of its
seed: regenerating at seed 7 gives a byte-identical file, and the gate's
determinism step depends on that staying true.

## What `demo.jsonl` is

400 ticks at 0.5 s, so 200 s of world time over the four arena assets —
`quadcopter`, `fixed-wing`, `tower-1`, `tower-2`, the same `asset_id`s the
arena adapter drives. The coordinator tasks the fleet every tick, the aircraft
fly toward what they were told, and the towers stand where they were sited.

## What it does not show yet, and why that matters

**The belief field is uniform for the whole episode, and stays uniform.**
Nothing in this run erodes it: there is no sighting source behind the
kinematic transport, and — more to the point — the negative-information update
(#13) is not merged. A field nobody has looked away from is a fixed point of
diffusion, so `entropy` is identical at t=199.5 and t=100.0.

That is honest, and it is also the gap between an episode that shows a fleet
moving and one that shows a fleet *thinking*. The sweeps are visible; the
reason for them is not. Anyone regenerating this after #13 lands should expect
the digest to change, and should expect that to be the point.

## Size

`ARENA.md` and #43 cap every committed episode log here at 25 MB combined,
including #48's SITL recording when it arrives. `demo.jsonl` is about 730 KB.
