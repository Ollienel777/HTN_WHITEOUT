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
seed — two runs on one machine are byte-identical, which is what the gate's
determinism step asserts.

**Across machines it is identical to about 1e-15, not to the byte.** The
committed file was regenerated on a different numpy build from the one that
wrote the previous commit's, and the two diverge in the last bit of `entropy`
from record 16 on — 1.8e-15 at worst over the whole run, which moves no
rendered pixel and no scored number. Worth knowing before you `cmp` this file
against your own run and conclude something is wrong: the gate compares two
fresh runs to *each other*, never to these committed bytes, so a diff here is
not a determinism failure.

## What `demo.jsonl` is

400 ticks at 0.5 s, so 200 s of world time over the four arena assets —
`quadcopter`, `fixed-wing`, `tower-1`, `tower-2`, the same `asset_id`s the
arena adapter drives. The coordinator tasks the fleet every tick, the aircraft
fly toward what they were told, and the towers stand where they were sited.

Schema 6 (#124), so every pose carries `pitch`, `roll` and `measured_t`, and
every record carries `belief_field` — the belief grid's 850 water cells,
quantised to a byte each, with the cell-corner lattice on the first record
only. That is what the viewer draws the field from; it is a rendering channel
and nothing reads it back. This transport reports no measurement time by
default — its fixes are computed, not received — so `measured_t` is `null`
throughout, which is the honest answer rather than a zero-age fix nobody took.
Pass `--pose-age` to exercise the staleness path.

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

**And since #124 the field is in the log, so this is drawable rather than
merely true.** Every record carries its 850 water cells, and **400 of the 400
cell payloads are distinct** — the bytes change every tick because the fleet
is eroding them. An earlier draft of this file, written against a build that
had #124 but not #13, said the opposite in as many words: "every tick's 850
bytes are the same 850 bytes". That was accurate when it was written and is
the reason this section is measured rather than remembered. If you regenerate
and find one distinct payload, the belief field is not being worked and the
episode is showing a fleet that moves without thinking.

## Size

`ARENA.md` and #43 cap every committed episode log here at 25 MB combined,
including #48's SITL recording when it arrives. `demo.jsonl` is 1.24 MB, of
which 504 kB is #124's belief field — 492 kB of per-tick cells and the 11.8 kB
corner lattice that rides the first record.
