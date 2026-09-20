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

Schema 5 (#117), so every pose carries `pitch`, `roll` and `measured_t`. This
transport reports no measurement time by default — its fixes are computed, not
received — so `measured_t` is `null` throughout, which is the honest answer
rather than a zero-age fix nobody took. Pass `--pose-age` to exercise the
staleness path.

**Regenerate it, never hand-merge it.** It is 400 records of generated output,
so a conflict in it has no correct manual resolution: take either side, run the
command above, and commit what comes out. #117 and this file's own history are
both examples.

## What the episode shows

The belief field is worked, not decorative. Every tick, each asset that looked
and saw nothing erodes the water it could have seen, in proportion to how well
it could have seen it (#13):

| quantity | tick 0 | tick 400 |
|---|---|---|
| `entropy` | 6.7429 | **5.9286** |
| `peak_p` | 0.00122 | **0.00429** (3.5x) |
| distinct `entropy` values across the file | — | **400** |
| quadcopter altitude | 40 m | 120 m, commanded, reached at t=27 s |

The drop is spread across the run rather than front-loaded: 47.7% of it by
tick 100, 78.4% by tick 200. Before #13 that table read `6.745236` in both
columns and **one** distinct value across the whole file. The fleet flew the
same sweeps; they just meant nothing. A 1200-tick probe at the time showed the
same flat digest, so length was never the missing piece — erosion was.

**These numbers are properties of the model, not of the tick rate.** Evidence
accrues per second of exposure rather than per call, so halving
`DEFAULT_TICK_SECONDS` no longer doubles the erosion. It did until review
caught it: the same 100 s of episode gave a peak probability of 0.0033, 0.0044
or 0.0058 depending only on how often the loop ran.

## Size

`ARENA.md` and #43 cap every committed episode log here at 25 MB combined,
including #48's SITL recording when it arrives. `demo.jsonl` is about 730 KB.
