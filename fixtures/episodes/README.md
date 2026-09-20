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

## Where the episode stops being interesting, measured

The fleet reaches steady state early, and the honest numbers are these:

| quantity | behaviour |
|---|---|
| `covered_fraction` | 0.12 at tick 0, **1.0 by tick 57**, and 1.0 for the remaining 343 |
| `entropy` | one value, `6.745236`, for every tick of the episode |
| quadcopter altitude | climbs 40 m → 120 m over the first 27 s, then holds |
| re-taskings | roughly two per tick, at a **constant rate**, start to finish |

So a before/after comparing tick 0 with tick 400 overstates it: almost all of
the visible change happens in the first minute.

**Making the episode longer does not fix this, and it was worth checking.** The
obvious suspicion is that `SearchParams.stale_horizon_s` is 240 s against a
200 s episode, so a segment can never go stale and be re-swept. A 1200-tick
probe — 600 s, two and a half times the horizon — says otherwise: `entropy` is
still constant to six decimal places, `covered_fraction` sits at 1.0 from tick
57 with one brief dip to 0.96, and the re-tasking rate never changes. The
horizon is not the binding constraint.

## What is actually missing

**Nothing erodes the belief field.** There is no sighting source behind the
kinematic transport, and the negative-information update (#13) is not merged.
A field nobody has looked away from is a fixed point of diffusion, so it stays
uniform however long the run is — and with the field flat, the policy's
staleness model is the only thing left varying, and it saturates at tick 57.

That is the gap between an episode that shows a fleet moving and one that
shows a fleet *thinking*. The sweeps are visible; the reason for them is not.
Anyone regenerating this after #13 lands should expect these numbers to change,
should expect the back half to stop being flat, and should revisit whether 400
ticks is still the right length — at that point a longer run probably does earn
its bytes.

## Size

`ARENA.md` and #43 cap every committed episode log here at 25 MB combined,
including #48's SITL recording when it arrives. `demo.jsonl` is about 730 KB.
