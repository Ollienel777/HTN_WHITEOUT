# Backlog draft

Local IDs `D1…D46`. These become GitHub issues on the plan lap that follows the
spec PR merge; dependencies are rewritten from `Dn` to `#n` at filing time.

**Spikes:** `S1` = **D44**, `S2` = **D45** (as referenced in `SPEC.md` §10).

**`area:` labels to create before filing:** `area:sim`, `area:score`,
`area:belief`, `area:policy`, `area:transport`, `area:tune`, `area:viz`,
`area:infra`, `area:submit`.

**Only D31 and D32 carry `needs-decision`.** Everything else gets `loop-ok`.
The sim, the scorer and the seam are invariant to all five open questions —
that is the point of the ordering.

---

## M0 — Skeleton (one ticket at a time)

### D1 — chore: scaffold the package, the gate script and CI

**Milestone:** M0 · **Demo beat:** — · **Size:** M
**Labels:** `chore` `P0` `area:infra` `loop-ok`

## What
Create the Python package skeleton and the one gate command, and add the CI
workflow that runs it.

- `pyproject.toml`: package `whiteout`, Python 3.11, deps `numpy`, `scipy`,
  `networkx`; `[dev]` extra `ruff`, `mypy`, `pytest`, `build`.
- `whiteout/` package with `cli.py` exposing `run|replay|score|sweep|ablate|serve`
  as stubs, `viz/`, `tests/`, `fixtures/`, `scripts/`, `artifacts/`.
- `scripts/gate.py` running the eight steps in `SPEC.md` §6 in order, failing
  fast, with the smoke and determinism steps present but allowed to be trivial
  until D2/D3 land. Copy the test step's command exactly, including
  `-m "not slow"`.
- `.github/workflows/ci.yml`: **on every pull request with no path filters**,
  and on every push to `main`. One job: checkout, `setup-python@v5` (3.11),
  `pip install -e ".[dev]"`, `python scripts/gate.py`.
- Exclude `.claude/**` from ruff, mypy and pytest via `pyproject.toml`, and
  register the `slow` marker in `[tool.pytest.ini_options] markers`.
- `.env.example` with the names in `SPEC.md` §7 and no values.
- **Delete `hackathon/backlog-draft.md`.**

## Why
Nothing else can be reviewed or merged until a PR produces check runs. There is
currently no `.github/workflows/` directory at all.

## Acceptance
- [ ] `python scripts/gate.py` passes from a clean venv
- [ ] The CI workflow runs on this PR and shows a green check
- [ ] The workflow has no `paths:` or `paths-ignore:` filter
- [ ] `ruff`, `mypy` and `pytest` all exclude `.claude/**`
- [ ] The gate's test step is `pytest -q -m "not slow"`, and `slow` is a
      registered marker
- [ ] `hackathon/backlog-draft.md` is deleted
- [ ] No secret, key or value appears in `.env.example`

## Notes
`SPEC.md` §5, §6, §7. Files: `pyproject.toml`, `scripts/gate.py`,
`.github/workflows/ci.yml`, `whiteout/__init__.py`, `whiteout/cli.py`.

---

### D2 — feat: record types and the episode log schema

**Milestone:** M0 · **Demo beat:** — · **Size:** M
**Labels:** `feat` `P0` `area:infra` `loop-ok`

## What
`whiteout/types.py` with frozen, JSON-serialisable dataclasses: `Pose`,
`SensorReport`, `WaypointIntent`, `WorldObservation`, `FleetIntent`,
`BeliefDigest`, `Contact`, `Truth`, `EpisodeRecord`. Plus
`whiteout/log.py`: a versioned JSONL writer and reader, and a validator that
rejects a malformed or wrong-version log with the failing line number.

## Why
The episode log is the only artifact the scorer, the viewer, the tuner and the
ablation harness read. Everything downstream depends on it being fixed early.

## Acceptance
- [ ] Round-trip test: write N records, read them back, assert equality
- [ ] Validator rejects a truncated line, an unknown field and a wrong
      `schema_version`, naming the line each time
- [ ] `mypy` strict passes on `whiteout/types.py` and `whiteout/log.py`
- [ ] `truth` is a separate top-level field, not nested inside the observation

## Notes
`SPEC.md` §4 "The data model". Depends on D1.

---

### D3 — feat: the Transport protocol and the kinematic stub

**Milestone:** M0 · **Demo beat:** 6 · **Size:** S
**Labels:** `feat` `P0` `area:transport` `loop-ok`

## What
`whiteout/transport/base.py`: a `Transport` `Protocol` with exactly
`connect()`, `observe() -> WorldObservation`, `command(FleetIntent)`,
`close()`. `whiteout/transport/kinematic.py` as a stub that returns static
poses. A factory reading `WHITEOUT_TRANSPORT`, defaulting to `kinematic`.

## Why
The seam is the single most important structural decision in the build
(`DECISION.md`). Establishing it before any policy code exists is what makes it
hold.

## Acceptance
- [ ] The protocol carries no belief, policy or scoring concept
- [ ] `WHITEOUT_TRANSPORT` unset selects `kinematic`
- [ ] An unknown value fails loudly at startup, naming the valid values
- [ ] `python -m whiteout.cli run --ticks 10` produces a valid episode log

## Notes
`SPEC.md` §4 "The seam", §7. Depends on D2.

---

## M1 — Demo path

### D4 — feat: terrain, synthetic generator and occlusion queries

**Milestone:** M1 · **Demo beat:** 1 · **Size:** M
**Labels:** `feat` `P1` `area:sim` `loop-ok`

## What
`whiteout/sim/terrain.py`: a deterministic synthetic Arctic terrain generator
(ridges, drainages, saddles — the structure the flow network needs), a height
array, a hillshade export for the viewer, and `line_of_sight(a, b) -> bool`
plus `occlusion_fraction(footprint)`. A DEM loader hook reading `WHITEOUT_DEM`
but never requiring it.

## Why
Beat 1 and beat 3. Occlusion is what makes negative information non-trivial and
what makes choke points exist.

## Acceptance
- [ ] Same seed produces a byte-identical height array
- [ ] Generated terrain contains at least three distinct saddles at the default
      seed, asserted by a test
- [ ] `WHITEOUT_DEM` unset uses the generator and logs that it did
- [ ] Line-of-sight is symmetric and correct on a hand-checked ridge case

## Notes
`SPEC.md` §4, §7. Depends on D2.

---

### D5 — feat: vehicle kinematics per class, and the ground-truth guard

**Milestone:** M1 · **Demo beat:** 1 · **Size:** M
**Labels:** `feat` `P1` `area:sim` `loop-ok`

## What
`whiteout/sim/vehicles.py`: four classes with genuinely distinct envelopes —
fixed-wing (min turn radius, stall speed, cannot loiter), quadcopter (hover,
slow transit), rover (terrain-limited, cannot cross a ridge), tower (immobile,
free, permanent). Energy accounting per class. Plus the **ground-truth
isolation guard**: `Truth` is written by the sim and readable only by the
scorer and the viewer.

## Why
Heterogeneity is the track's whole problem, and it is a quoted prize clause.
The guard prevents the single easiest way to cheat ourselves into believing a
score.

## Acceptance
- [ ] A fixed-wing cannot hold station; a test asserts its speed never goes
      below stall
- [ ] A rover path is rejected when it crosses a slope above its limit
- [ ] A test asserts no module under `whiteout/policy/` or `whiteout/belief/`
      imports or receives `Truth`
- [ ] A runtime guard raises if a policy is handed a record containing `truth`

## Notes
`SPEC.md` §4, §5 "Conventions", §10. Depends on D4.

---

### D6 — chore: the transport conformance suite

**Milestone:** M1 · **Demo beat:** 6 · **Size:** S
**Labels:** `chore` `P1` `area:transport` `loop-ok`

## What
A shared pytest suite that any `Transport` implementation must pass: ordering
guarantees, tick monotonicity, behaviour when an asset is silent, behaviour on
`close()` mid-episode, and that `observe()` never returns `Truth`. Run against
`kinematic` now; `sitl` and `arena` inherit it.

## Why
The mitigation for "the sponsor's interface is not MAVLink" is the seam, and a
seam is only real if a new adapter has a definition of done.

## Acceptance
- [ ] The suite is importable and parameterised over a transport factory
- [ ] `kinematic` passes it
- [ ] Adding a new transport requires no change to the suite

## Notes
`SPEC.md` §4, §10. Depends on D3.

---

### D7 — feat: the sensor model with terrain occlusion

**Milestone:** M1 · **Demo beat:** 3 · **Size:** M
**Labels:** `feat` `P1` `area:sim` `loop-ok`

## What
`whiteout/sim/sensors.py`: per-class footprints (fixed-wing wide strip, quad
narrow cone, rover short arc, tower fixed sector), a detection probability
falling with range and rising with dwell, occlusion from D4, false positives at
a configurable rate, and **explicit negative reports** — a `SensorReport` with
`negative=True` and the swept footprint, emitted every tick a sensor sees
nothing.

## Why
Negative reports are the input to the cheapest accuracy win available (D12) and
to the wow beat.

## Acceptance
- [ ] Every tick, every sensor emits either detections or a negative report —
      never silence
- [ ] Detection probability is monotonically decreasing in range, asserted
- [ ] An occluded target is never detected, asserted on a hand-built ridge case
- [ ] False-positive rate is a parameter and defaults to non-zero

## Notes
`SPEC.md` §4. Depends on D4, D5.

---

### D8 — feat: target movers

**Milestone:** M1 · **Demo beat:** 2 · **Size:** S
**Labels:** `feat` `P1` `area:sim` `loop-ok`

## What
`whiteout/sim/target.py`: at minimum a scripted waypoint mover and a
drainage-following mover that respects terrain. A `reactive` mover that avoids
observed sensor footprints, behind a flag, defaulting off.

## Why
Beat 2 needs a target. The scripted/reactive split is how we will answer open
question Q2 by measurement once the booth answers it.

## Acceptance
- [ ] Both movers are deterministic under seed
- [ ] The mover is selected by episode config, not by code change
- [ ] The reactive mover reads only what a target could plausibly perceive

## Notes
`SPEC.md` §11 Q2. Depends on D4.

---

### D9 — feat: the episode loop, with a ≥100× real-time budget

**Milestone:** M1 · **Demo beat:** 5 · **Size:** M
**Labels:** `feat` `P1` `area:sim` `loop-ok`

## What
Wire terrain, vehicles, sensors and target into `whiteout/sim/episode.py`,
vectorised with NumPy, writing the episode log. Measure and report the
throughput of a 6-vehicle, 256×256, 400-tick episode against the ≥100×
simulated-wall-clock budget.

**The throughput assertion is marked `-m slow` and is excluded from the gate**
(`SPEC.md` §6). A wall-clock threshold measured on shared CI runners and from
parallel worktrees flakes, and a red gate on an unrelated PR is a PR that
cannot settle. It is asserted once here and re-measured by the tuner (D23).

## Why
The overnight parameter search is the entire strategic bet and it is impossible
without this. Measuring it here, as a `-m slow` test outside the gate that the
tuner re-measures, means we learn at hour 6, not hour 26.

## Acceptance
- [ ] `python -m whiteout.cli run --seed 7 --ticks 400` writes a valid log
- [ ] Two runs at the same seed produce byte-identical logs
- [ ] The throughput number is measured and reported in the PR
- [ ] The ≥100× assertion exists as a `-m slow` test, passes locally, and is
      not run by `scripts/gate.py`
- [ ] No module-level `np.random` anywhere in `whiteout/sim/`

## Notes
`SPEC.md` §5 "Conventions", §6, §10. Depends on D5, D7, D8. Run **D45 (S2)**
first.

---

### D10 — feat: the scorer, four axes, weights as parameters

**Milestone:** M1 · **Demo beat:** 2 · **Size:** M
**Labels:** `feat` `P1` `area:score` `loop-ok`

## What
`whiteout/score/`: our best model of **coverage** (fraction of prior
probability mass observed, weighted by detection quality — not area flown),
**collaboration** (completed cueing chains and cross-class handoffs, from the
contact lifecycle), **efficiency** (distance and energy per unit of information
gained), and **tracking accuracy** (time-in-track and position error while
tracked). Pure function of an episode log. Weights loaded from JSON; fixtures
in `fixtures/weights/` including `equal.json`.

## Why
It is the objective function, the tuning signal and the argument in beat 5.
Weights are parameters so that the booth's answer to Q3 costs minutes.

## Acceptance
- [ ] `score(log, weights)` is pure and returns four finite values in [0,1]
      plus the weighted total
- [ ] No weight is hard-coded anywhere in `whiteout/`
- [ ] Each axis has a hand-computed unit case
- [ ] Re-scoring a cached log with new weights requires no re-simulation

## Notes
`SPEC.md` §1, §3, §11 Q3. Depends on D2, D9.

---

### D11 — feat: the decaying belief grid and the detection update

**Milestone:** M1 · **Demo beat:** 3 · **Size:** M
**Labels:** `feat` `P1` `area:belief` `loop-ok`

## What
`whiteout/belief/grid.py`: a normalised occupancy field over the terrain, a
diffusion/decay step matched to plausible target speed and constrained by
terrain, and a Bayesian update on positive detections. A `BeliefDigest` written
to the episode log each tick for the viewer.

## Why
Beat 3, and the substrate for everything in `policy/`.

## Acceptance
- [ ] The field sums to 1.0 within tolerance after every update
- [ ] Diffusion does not leak mass across impassable terrain
- [ ] The digest is small enough that a 4-minute episode log stays under 8 MB.
      This number is derived from D41's committed-fixture budget (25 MB total
      across every committed episode, including D28's SITL log) — the fixture
      budget is the binding one, in a public repo.

## Notes
`SPEC.md` §4. Depends on D4, D7.

---

### D12 — feat: the negative-information update

**Milestone:** M1 · **Demo beat:** 3 (wow) · **Size:** M
**Labels:** `feat` `P1` `area:belief` `loop-ok`

## What
`whiteout/belief/negative.py`: update the field by the likelihood of a
non-detection given range, terrain occlusion and sensor class. A footprint that
swept and saw nothing must reduce belief there **in proportion to how well it
could have seen**, not to zero. Docstring carries the formula.

## Why
The single cheapest accuracy win available, the thing that makes coverage
quantitative rather than a heatmap of where we flew, and the wow beat. It is
also the highest-severity correctness risk in the build.

## Acceptance
- [ ] Property test over ≥50 seeds: belief entropy is non-increasing under
      non-detections alone
- [ ] Posterior stays in [0,1] and normalised, always
- [ ] A fully occluded footprint reporting nothing changes belief by ~0
- [ ] A single-sensor closed-form case matches a hand-computed value
- [ ] An ablation flag switches it off cleanly, for D22

## Notes
`SPEC.md` §4 "Where the hard part lives", §10. Depends on D11.

---

### D13 — feat: terrain as a flow network, and cut posting stations

**Milestone:** M1 · **Demo beat:** 3 (wow) · **Size:** M
**Labels:** `feat` `P1` `area:belief` `loop-ok`

## What
`whiteout/belief/flow.py`: build a coarse graph over passable terrain with
capacities from traversability, compute min-cuts between high-belief and
low-belief regions with `networkx`, and emit **posting stations** at the cut
edges' midpoints, ranked. Emit the cut chords into the episode log for the
viewer.

## Why
"Guard the gaps" beats "sweep the field" whenever the target must transit, and
it is a genuinely different artifact from a belief raster. Towers are free,
permanent and immobile, which is exactly what a cut wants.

## Acceptance
- [ ] On a hand-built two-basin terrain with one saddle, the top cut is that
      saddle
- [ ] Cut computation is under 50ms at the episode grid size, or is cached
      across ticks with an invalidation rule
- [ ] Stations are ranked and the ranking is in the log
- [ ] An ablation flag switches it off cleanly, for D22

## Notes
**Timeboxed to four hours.** If sane cuts are not appearing, reduce to an
overlay-only feature and let the policy ignore it — say so on the PR rather
than extending. `SPEC.md` §10. Depends on D11.

---

### D14 — feat: the frontier-coverage fallback policy

**Milestone:** M1 · **Demo beat:** 2 · **Size:** M
**Labels:** `feat` `P1` `area:policy` `loop-ok`

## What
`whiteout/policy/frontier.py`: a mode-agnostic systematic-sweep policy —
lawnmower lanes assigned by class, no hypotheses, no information gain.
Selectable by config alongside the main coordinator.

## Why
Information-gain routing **is a bet**. Sent to unobserved ground and wrong, it
scores worse on coverage than a lawnmower sweep, and against a scripted target
the clever stack can lose. This is not a stub: it is the thing we might ship,
and it is beat 2's left panel either way.

## Acceptance
- [ ] Produces full coverage of passable terrain given enough time, asserted
- [ ] Respects every class's kinematic envelope
- [ ] Selectable with `--policy frontier`, no code change
- [ ] Scored by D10 with no special-casing

## Notes
`SPEC.md` §10 (first risk). Depends on D5, D10.

---

### D15 — feat: auction allocation across heterogeneous classes

**Milestone:** M1 · **Demo beat:** 2 · **Size:** M
**Labels:** `feat` `P1` `area:policy` `loop-ok`

## What
`whiteout/policy/auction.py`: tasks are bid on by assets; bids price in the
class's envelope (a fixed-wing bids badly on "hold station", a rover bids
badly on "cross the ridge", a tower bids zero cost on "watch this sector
forever"). Clearing is deterministic under seed.

## Why
"Shared intelligence" and "distributed thinking" are quoted prize clauses, and
heterogeneity is what makes the allocation interesting.

## Acceptance
- [ ] A hold-station task is never won by a fixed-wing
- [ ] Clearing is deterministic and tie-breaks are explicit
- [ ] Unassigned tasks are reported, not silently dropped

## Notes
`SPEC.md` §3, §4. Depends on D5, D11.

---

### D16 — feat: information-gain routing

**Milestone:** M1 · **Demo beat:** 3 · **Size:** M
**Labels:** `feat` `P1` `area:policy` `loop-ok`

## What
`whiteout/policy/infogain.py`: score candidate waypoints by expected reduction
in belief entropy, accounting for travel cost and the sensor's actual footprint
at arrival. Feeds tasks into D15.

## Why
Beat 2 and beat 3, and the half of the field we are racing.

## Acceptance
- [ ] Expected gain accounts for occlusion at the arrival pose, not just range
- [ ] Travel cost uses the class's real envelope
- [ ] An ablation flag switches it off cleanly, for D22

## Notes
Depends on D11, D12, D15.

---

### D17 — feat: the contact lifecycle cueing machine

**Milestone:** M1 · **Demo beat:** 4 · **Size:** M
**Labels:** `feat` `P1` `area:policy` `loop-ok`

## What
`whiteout/policy/contacts.py`: a contact is a first-class object with a state
(`unconfirmed → confirming → tracked → handed off`), an owning asset, and a
transition log. The policy's job is to move contacts along it: tower trips,
fixed-wing diverts to sweep, quad confirms and holds, rover takes persistence.
Transitions and handoffs are written to the episode log.

## Why
**Collaboration is an explicit scored axis and it is the one most teams
under-serve**, because independent agents are easier to write. Scoring the
chain explicitly beats hoping collaboration emerges from an auction. It is also
beat 4.

## Acceptance
- [ ] Every transition is logged with both assets and a reason
- [ ] A contact cannot skip a state
- [ ] A cross-class handoff is distinguishable in the log from a same-class one
- [ ] D10's collaboration axis reads this and nothing else

## Notes
`SPEC.md` §2 beat 4, §3. Depends on D10, D15.

---

### D18 — feat: re-tasking hysteresis

**Milestone:** M1 · **Demo beat:** 5 · **Size:** S
**Labels:** `feat` `P1` `area:policy` `loop-ok`

## What
Do not switch an asset's task unless expected gain exceeds a switching cost.
Switching cost is a tuned parameter in `policy/params.py`, per class.

## Why
Efficiency penalises brute force. Re-tasking every tick burns distance and
battery for marginal gain. A few lines, probably real points, and almost
everyone will miss it.

## Acceptance
- [ ] Task-switch count per episode falls measurably with hysteresis on, at
      equal or better total score, shown in the PR
- [ ] The switching cost is a parameter, not a constant
- [ ] An ablation flag switches it off cleanly, for D22

## Notes
`DECISION.md` "Where we try to actually be better" #4. Depends on D15.

---

### D19 — feat: the coordinator's objective is the scorer

**Milestone:** M1 · **Demo beat:** 5 · **Size:** M
**Labels:** `feat` `P1` `area:policy` `loop-ok`

## What
`whiteout/policy/objective.py`: candidate allocations are evaluated by a short
rollout scored with **the same `whiteout/score` code** the leaderboard model
uses, under the current weights — not by a hand-written proxy that drifts from
it. Plus `policy/params.py`: one dataclass holding every policy parameter,
JSON-serialisable.

## Why
Most teams will build "a good search system" and hope it scores well. Ours
optimises the thing being measured. This ticket is the difference.

## Acceptance
- [ ] `objective.py` imports `whiteout.score` and defines no scoring of its own
- [ ] Changing the weights file changes the policy's behaviour with no code
      change, demonstrated by a test
- [ ] Every tunable number in `whiteout/policy/` lives in `PolicyParams`
- [ ] Rollout depth and count are themselves parameters

## Notes
`SPEC.md` §4 "Where the hard part lives". Depends on D10, D15, D16.

---

### D20 — feat: the viewer shell, tokens and states

**Milestone:** M1 · **Demo beat:** 1 · **Size:** M
**Labels:** `feat` `P1` `area:viz` `loop-ok`

## What
`viz/index.html`, `viz/tokens.css`, `viz/viewer.css`, `viz/viewer.js`: the
three-region shell from `DESIGN.md` — topbar, field area, 320px rail, transport
bar — with every token defined and every empty/loading/error state present.
`whiteout.cli serve` binds `PORT` or an ephemeral port, never a fixed one.
Opens from `file://` too.

## Why
Beat 1, and it is the debugging surface every other M1 ticket is verified
through.

## Acceptance
- [ ] No raw hex value or one-off pixel size anywhere outside `tokens.css`
- [ ] The seven states in `DESIGN.md` all render
- [ ] Loading is a skeleton in the final layout's shape, never a spinner
- [ ] `PORT` is respected; no server reuse
- [ ] No build step, no `package.json`, no `node_modules`

## Notes
`DESIGN.md` throughout; `SPEC.md` §5 "Why the viewer is not a Node front end",
§6. Depends on D2.

---

### D22 — feat: the baseline-versus-policy comparison harness

**Milestone:** M1 · **Demo beat:** 2 · **Size:** M
**Labels:** `feat` `P1` `area:tune` `loop-ok`

## What
`whiteout/tune/compare.py` and `whiteout.cli ablate`: run N seeds against a set
of policy configurations, score each with D10, and emit a table with means and
confidence intervals. Ships with the ablation set: full, negative-information
off, choke-points off, hysteresis off, info-gain off, and frontier baseline.

## Why
This is how we find out whether information-gain routing actually beats a
lawnmower sweep under the booth's weights, instead of guessing. It is the
argument in beat 5 and the evidence in the README.

## Acceptance
- [ ] `whiteout.cli ablate --seeds 50` emits a table to `artifacts/`
- [ ] Every ablation flag from D12, D13, D16, D18 is wired
- [ ] Results are reproducible from the seed list alone
- [ ] The table states plainly when a differentiator does **not** help

## Notes
`SPEC.md` §10 (first risk). Depends on D10, D12, D13, D14, D16, D18.

---

### D24 — feat: the split-screen replay

**Milestone:** M1 · **Demo beat:** 2 · **Size:** S
**Labels:** `feat` `P1` `area:viz` `loop-ok`

## What
Two `FieldCanvas` instances sharing one topbar, one transport bar and one scrub
position, labelled `frontier baseline` and `tuned`. Eight score dials, four per
side, with the tuned side's deltas coloured.

## Why
Beat 2. It is the honest race: same seed, same terrain, same fleet.

## Acceptance
- [ ] Both sides scrub from one control and stay in lock-step
- [ ] Labels use `--t-12` `--n-400` per `DESIGN.md`, not headings
- [ ] Works at 1440×900 without horizontal scroll

## Notes
**First on the cut list** (`SPEC.md` §9 item 8) — two sequential replays tell
the same story. Depends on D34, D35.

---

### D34 — feat: FieldCanvas, the belief field render

**Milestone:** M1 · **Demo beat:** 3 (wow) · **Size:** M
**Labels:** `feat` `P1` `area:viz` `loop-ok`

## What
One `<canvas>`, layered in a single pass: terrain hillshade → belief field →
cut chords → sensor footprints → waypoint intent lines → asset sprites → truth
marker. **The erosion animation** — negative reports visibly carving luminance
out of the field over `--m-slow` — is this ticket's centrepiece. Accent ring on
an asset posted at a cut.

## Why
The signature element and the wow beat. It is also the only way anyone debugs
D12 and D13.

## Acceptance
- [ ] 60fps at 256×256 belief and 6 assets at 20× replay, measured
- [ ] `--truth` red appears nowhere **on the field** but the truth marker
      (it shares a hex with `--danger`, which is legitimate off the field)
- [ ] The beat-3 `TRACKED` badge renders top-left of the field at `--t-48`
      mono in `--ok` with the clock, per `DESIGN.md` "Per-beat hero states"
- [ ] Belief-ramp colours appear nowhere but the belief field
- [ ] `prefers-reduced-motion` snaps between ticks with no animation
- [ ] Nothing glows: no halo shadows, no blur filters, no bloom

## Notes
`DESIGN.md` "The signature element". Depends on D20, D11, D12, D13.

---

### D35 — feat: score dials, fleet rows and the inspector

**Milestone:** M1 · **Demo beat:** 1 · **Size:** S
**Labels:** `feat` `P1` `area:viz` `loop-ok`

## What
`ScoreDial` ×4 in the topbar, `FleetRow` ×N and `Inspector` in the rail, per
`DESIGN.md` "Components". Every number in `--font-mono` with `tabular-nums`.

## Acceptance
- [ ] Each `ScoreDial` renders its value at `--t-24` mono tabular and its
      label at `--t-12` `--n-400`, fitting the 48px topbar
- [ ] Score values do not jitter horizontally as they tick
- [ ] Callsigns are `WING-1`/`QUAD-1`/`ROVR-1`/`TOWR-1` style, never "Vehicle 1"
- [ ] Inspector values share one right edge

## Notes
`DESIGN.md`. Depends on D20, D10.

---

### D36 — feat: the contact ribbon and handoff arcs

**Milestone:** M1 · **Demo beat:** 4 · **Size:** S
**Labels:** `feat` `P1` `area:viz` `loop-ok`

## What
`ContactRibbon` in the rail — four lifecycle segments — and handoff arcs drawn
on the field in the two participating class colours.

## Why
Beat 4. Collaboration is a scored axis and this is the only place it is visible.

## Acceptance
- [ ] The ribbon reads directly from D17's transition log
- [ ] Zero contacts shows the designed empty state, saying what would create one
- [ ] Many contacts scrolls the rail; the field never scrolls

## Notes
`DESIGN.md`. Depends on D17, D34.

---

### D41 — chore: the committed demo episode fixture

**Milestone:** M1 · **Demo beat:** 1 · **Size:** S
**Labels:** `chore` `P2` `area:infra` `loop-ok`

## What
Commit one real seeded episode log per side of beat 2 under `fixtures/episodes/`,
regenerable by a documented command, and load one by default when the viewer is
opened with no log.

## Why
The viewer's empty state should offer the next action, and every beat needs a
fallback that is a real run rather than invented data.

## Acceptance
- [ ] The fixture is regenerated by one documented command at a named seed
- [ ] Every number the viewer shows from it was computed by the sim
- [ ] Combined fixture size stays under 25 MB **across every committed episode
      log in `fixtures/episodes/`, including D28's SITL recording** — not just
      this ticket's two

## Notes
`DESIGN.md` "Data and copy voice". Depends on D9, D14, D19.

---

### D43 — docs: the arena runbook and the workshop checklist

**Milestone:** M1 · **Demo beat:** — · **Size:** S
**Labels:** `docs` `P1` `area:transport` `loop-ok`

## What
`docs/arena.md`: the five open questions as a printable checklist, what each
answer changes, and the exact mapping work each possible interface implies.

## Why
The Dominion Dynamics API Workshop is the highest-value hour in the build. It
should not be improvised — which is why this ticket **depends on nothing and is
M1**. It is the input to the workshop, not an output of it.

## Acceptance
- [ ] All five questions from `SPEC.md` §11 appear with the tickets they affect
- [ ] Each possible answer to Q1 names the files that would change
- [ ] The checklist has a blank line per question for the answer, and prints on
      one side of A4
- [ ] The checklist is finished and its PR open **before 2026-09-19T14:30Z**

## Notes
Printing the checklist and carrying it to the workshop is a human action
(`DECISION.md` "Human actions"). This ticket's job is to have it ready to
print in time.
**Depends on nothing. Hard deadline: the Dominion Dynamics API Workshop,
10:30 local = 2026-09-19T14:30Z, hour 10.5 of 32** (`SPEC.md` §11) — 3.5 hours
before the 18:00Z sponsor selection lock. This ticket is worthless after that
time, so it is scheduled against the clock rather than against its
dependencies. D32 and D31 depend on this checklist coming back filled in.

---

## M2 — Wow

### D46 — chore: obtain ArduPilot SITL as a Docker image

**Milestone:** M2 · **Demo beat:** 6 · **Size:** S
**Labels:** `chore` `P1` `area:transport` `loop-ok`

## What
**Timeboxed to one hour. This is a loop action, not a human action** — it needs
no account, no key and no purchase.

ArduPilot SITL is **not installed on this machine and is not natively supported
on Windows 11**; the normal routes are WSL2 or Docker. Docker 29.4.1 is
installed and working here, so Docker is the route. Two sources, in order:

1. Clone `https://github.com/ArduPilot/ardupilot` with submodules and build its
   own `docker/Dockerfile`. ArduPilot's documentation
   (`https://github.com/ArduPilot/ardupilot_wiki`) states SITL runs "without any
   special hardware".
2. `https://github.com/radarku/sitl-swarm`, which our own research recorded as
   a Docker-based multi-vehicle SITL bring-up shortcut, if (1) overruns.

Smoke test inside the container: `sim_vehicle.py -v ArduCopter --count 2
--auto-sysid --mcast`, and confirm a `pymavlink` connection from the host sees
`HEARTBEAT` from two distinct sysids. Write the exact commands — image name,
build command, run command, published ports/endpoint — into `docs/sitl.md`.

## Why
Three tickets (D44, D21, D28) begin from a running `sim_vehicle.py`, and
`SPEC.md` §3 maps the quoted prize clause "all running ArduPilot and MAVLink"
to the `sitl` transport. Nothing in the backlog obtained SITL. Without this,
that clause has only the MAVLink-only fallback in §3.

## Acceptance
- [ ] An image exists locally, named in `docs/sitl.md`, and rebuilds from the
      documented command
- [ ] Two SITL vehicles heartbeat to a host `pymavlink` connection, with the
      endpoint recorded as the value for `WHITEOUT_SITL_ENDPOINT`
- [ ] `docs/sitl.md` records the wall-clock cost of the build
- [ ] **On overrun:** the ticket closes with "SITL unobtainable in the timebox"
      and the reason. It does **not** silently extend, and it does not block
      any M1 ticket. M2(a) then exits per `SPEC.md` §8.
- [ ] **On overrun, before closing:** stand a host-side `pymavlink` loopback —
      two processes over a UDP socket, no ArduPilot behind it, one sending
      `SET_POSITION_TARGET_GLOBAL_INT` and replying `GLOBAL_POSITION_INT` —
      and record in `docs/sitl.md` the exact command that runs it and the
      output it prints. **This is beat 6's fallback artifact** (`SPEC.md` §2):
      the loopback and its recorded run are the only thing on screen in that
      branch, so nothing else may be named as its evidence. Still no
      production code — the loopback is a script under `scripts/`, not a
      transport under `whiteout/`.

## Notes
`SPEC.md` §3, §4, §8. No production code — nothing under `whiteout/` changes.
Depends on nothing. Blocks D44, D21, D28.

---

### D21 — feat: the SITL transport

**Milestone:** M2 · **Demo beat:** 6 · **Size:** M
**Labels:** `feat` `P1` `area:transport` `loop-ok`

## What
`whiteout/transport/sitl.py`: `pymavlink` against `sim_vehicle.py --count N
--auto-sysid --mcast`, GUIDED mode, `SET_POSITION_TARGET_GLOBAL_INT` for
intents, `GLOBAL_POSITION_INT` for poses. Endpoint from
`WHITEOUT_SITL_ENDPOINT`; absent means refuse to start, loudly. Must pass D6.

## Why
"All running ArduPilot and MAVLink" is a quoted prize clause, and this is the
validation that the coordinator is not simulator-shaped.

## Acceptance
- [ ] Passes the D6 conformance suite unchanged
- [ ] Moves ≥4 real SITL vehicles under GUIDED with no change to `policy/` or
      `belief/`
- [ ] Absent endpoint fails at startup with a message naming the variable
- [ ] Never imported or started by the gate

## Notes
Run **D46** then **D44 (S1)** first — SITL is not installed on this machine and
is not natively supported on Windows (D46 obtains it in Docker), and the CPU
ceiling is real with six instances our untested cut. If D46 reported SITL
unobtainable, this ticket is closed as such, the `§3` MAVLink-only fallback
applies, and beat 6 is demoed as the `pymavlink` loopback D46 stands and
records on overrun, per `SPEC.md` §2 — the prize clause is then met by the
MAVLink half alone. `SPEC.md` §10. Depends on D6, D19, **D46**.

---

### D23 — feat: the parameter sweep and its report

**Milestone:** M2 · **Demo beat:** 5 · **Size:** M
**Labels:** `feat` `P1` `area:tune` `loop-ok`

## What
`whiteout/tune/sweep.py` and `whiteout.cli sweep`: random search over
`PolicyParams` within a declared box, N episodes per point, checkpointed and
resumable, writing results to `artifacts/tuning/`. Emits the best parameter set
as JSON that `policy/params.py` loads directly.

## Why
The strategic bet. With a 100×-real-time sim and a scorer, thousands of runs
give us a coordinator tuned by measurement. No other team in a 32-hour
hackathon will have that, and it is infrastructure rather than insight — which
is what a duo can build while everyone else debugs MAVLink.

## Acceptance
- [ ] Interrupting and restarting resumes without losing completed points
- [ ] Partial results produce a valid report
- [ ] The best parameter set is loadable by the coordinator with no code change
- [ ] Re-scoring cached episodes under new weights needs no re-simulation

## Notes
`SPEC.md` §10. Depends on D10, D19, D22.

---

### D26 — feat: classification in the contact lifecycle

**Milestone:** M2 · **Demo beat:** 4 · **Size:** S
**Labels:** `feat` `P2` `area:policy` `loop-ok`

## What
A classify step between `confirming` and `tracked`: a confidence over target
class from accumulated detections, with a threshold to advance.

## Why
"Detect, classify and track" is a quoted prize clause and classify is currently
implicit.

## Acceptance
- [ ] Classification confidence appears in the contact record and the inspector
- [ ] A contact below threshold stays in `confirming`

## Notes
On the cut list (`SPEC.md` §9 item 6) — collapse to a threshold if time is
short. Depends on D17.

---

### D28 — chore: record a real SITL episode and commit it

**Milestone:** M2 · **Demo beat:** 6 · **Size:** S
**Labels:** `chore` `P1` `area:transport` `loop-ok`

## What
Run the coordinator against live SITL, record the episode log, commit it under
`fixtures/episodes/sitl-*.jsonl`, and confirm the viewer replays it with the
topbar reading `sitl`.

## Why
**The recorded log is the evidence; the live run is the flourish.** Beat 6's
primary path is this file, which is why it is a ticket of its own rather than a
step inside D21.

## Acceptance
- [ ] The committed log replays in the viewer with no special-casing
- [ ] Fleet rows show real MAVLink sysids
- [ ] The README states which commit produced it

## Notes
`SPEC.md` §9 item 4. Depends on D21, and transitively on **D46** for SITL
itself. If D46 reported SITL unobtainable, this ticket is closed as such, the
`§3` MAVLink-only fallback applies, and beat 6 is demoed as the `pymavlink`
loopback D46 stands and records on overrun, per `SPEC.md` §2. **M1 does not
exit on this ticket** — beat 6's evidence is M2 work (`SPEC.md` §8).

---

### D29 — feat: CMA-ES over policy parameters

**Milestone:** M2 · **Demo beat:** 5 · **Size:** S
**Labels:** `feat` `P3` `area:tune` `loop-ok`

## What
Add `cma` as an optional optimiser behind the same sweep interface as D23's
random search.

## Why
Marginal improvement over random search in a well-chosen box.

## Acceptance
- [ ] Selectable with `--optimiser cma`; random search stays the default
- [ ] `cma` absent degrades to random search with a warning, never a crash

## Notes
**Third on the cut list** (`SPEC.md` §9 item 3). Depends on D23.

---

### D31 — feat: the policy-mode filter (stretch)

**Milestone:** M2 · **Demo beat:** — · **Size:** M
**Labels:** `feat` `P3` `area:belief` `needs-decision`

## What
A bank of candidate target behaviour modes (`commit to goal`, `feint then
reverse`, `hug occlusion`, `hold still through a sweep`, `follow the
drainage`), each a short parameterised controller rolled forward into a
trajectory distribution. Observations, **including negative ones**, reweight a
posterior over modes and goal parameters; the position belief is the marginal.

## Why
The one mechanism our ideation found that is genuinely off the converged
default. It lets the fleet act on a hypothesis with no supporting observation,
which a cell-based information-gain searcher structurally cannot do.

## Blocked by
**Open question Q2** (`SPEC.md` §11), answered at the Dominion Dynamics API
Workshop, **2026-09-19T14:30Z** (hour 10.5), via D43's checklist. Until then
this ticket stays `needs-decision`. Is the target adversarial or scripted?
**This only pays against an evading target.** Against a scripted mover it is
strictly worse than the effort spent elsewhere. If the workshop answers
"adversarial", this comes off the cut list immediately and is re-prioritised to
P1.

## Acceptance
- [ ] Posterior over modes is normalised and updated by negative reports too
- [ ] D22 shows it beating the mode-agnostic policy against a reactive target
      and not losing against a scripted one
- [ ] Hold as a stretch: it never becomes a dependency of a P1 ticket

## Notes
**Top of the cut list** (`SPEC.md` §9 item 1). Depends on D12, D22.

---

### D37 — feat: the tuning report page

**Milestone:** M2 · **Demo beat:** 5 · **Size:** M
**Labels:** `feat` `P1` `area:viz` `loop-ok`

## What
A view that replaces the field: small multiples of score against each policy
parameter on a 4-column grid, the chosen operating point ringed in `--accent`,
and the D22 ablation table below. Same tokens, same type scale.

## Why
Beat 5 is the real argument of the whole project — "we did not guess these
numbers" — and it is currently a directory of JSON.

## Acceptance
- [ ] Reads `artifacts/tuning/` directly; no separate data prep
- [ ] Renders correctly from a partial (interrupted) sweep
- [ ] States plainly where a differentiator did not help
- [ ] No belief-ramp colours; neutral scale and the accent only

## Notes
`DESIGN.md` "The hero screen, beat 5". Depends on D22, D23, D20.

---

### D44 — spike S1: how many SITL instances fit on the laptop

**Milestone:** M2 · **Demo beat:** 6 · **Size:** S
**Labels:** `chore` `P1` `area:transport` `loop-ok`

## What
**Timeboxed to one hour.** **Inside the Docker image D46 builds**, start
`sim_vehicle.py --count 4 --auto-sysid --mcast`, connect `pymavlink` from the
host, confirm GUIDED `SET_POSITION_TARGET_GLOBAL_INT` moves a vehicle. Record
CPU and memory. Report the instance ceiling.

## Why
Our own analysis put eleven instances plus a renderer as not fitting, cut it to
six, and never tested six. D21 should not discover this.

## Acceptance
- [ ] A number is reported: how many instances run at real time
- [ ] One vehicle demonstrably moved under GUIDED
- [ ] Findings are a comment on D21, not a document

## Notes
`SPEC.md` §10. No production code. Depends on D3, **D46**. If D46 reported
SITL unobtainable, this spike does not run.

---

### D45 — spike S2: the kinematic tick cost

**Milestone:** M1 · **Demo beat:** 5 · **Size:** S
**Labels:** `chore` `P1` `area:sim` `loop-ok`

## What
**Timeboxed to 45 minutes.** Measure the naive tick cost at 6 vehicles ×
256×256 grid. Answer whether ≥100× real time needs vectorisation or just care,
and which step dominates.

## Acceptance
- [ ] A per-step timing breakdown is reported on D9
- [ ] A yes/no on whether the target is reachable without restructuring

## Notes
Depends on D5, D7.

---

## M3 — Prizes

### D32 — feat: the arena transport adapter

**Milestone:** M3 · **Demo beat:** — · **Size:** M
**Labels:** `feat` `P1` `area:transport` `needs-decision`

## What
`whiteout/transport/arena.py` against the sponsor's harness, whatever it turns
out to be. Until then: a documented stub that passes D6 against a local echo,
plus a runbook naming exactly what has to change.

## Why
If the sponsor supplies the arena, this is the only thing between our tuned
coordinator and a scored run.

## Blocked by
**Open question Q1** (`SPEC.md` §11), answered at the Dominion Dynamics API
Workshop, **2026-09-19T14:30Z** (hour 10.5). Until then this ticket stays
`needs-decision`; after it, the answer is in D43's returned checklist. Do they
provide the harness, and
is the interface plain MAVLink over a socket or a hosted service with its own
schema? `dominiondynamics.online` does not resolve and the Devpost block is the
only public text in existence — this cannot be answered from a desk.

## Acceptance
- [ ] The stub passes the D6 conformance suite
- [ ] The runbook lists every field that must map, with our type on one side
- [ ] Nothing outside `whiteout/transport/` changes when this is implemented

## Notes
`SPEC.md` §4, §11. Depends on D6 and on **D43's checklist coming back from the
14:30Z API workshop with Q1 answered**.

---

### D25 — feat: load the Arctic DEM tile

**Milestone:** M3 · **Demo beat:** 1 · **Size:** S
**Labels:** `feat` `P2` `area:sim` `loop-ok`

## What
Load the committed downsampled public Arctic DEM tile through `WHITEOUT_DEM`,
reprojected and clipped to the episode extent.

## Why
Real terrain is a better picture and a better story for the sponsor. It is not
a better score.

## Acceptance
- [ ] Absent DEM still falls back to the generator, silently and correctly
- [ ] The committed tile is downsampled and under 10 MB
- [ ] Provenance and licence of the tile are in the README

## Notes
**Seventh on the cut list** (`SPEC.md` §9 item 7). Depends on D4.
**Fetching the tile is authorised as a loop action and blocks nothing.** If
`fixtures/dem/` is empty, fetch one public Arctic DEM tile, downsample it and
commit it as part of this ticket. The one-time fetch is a setup step outside
the product, like D46's image build: `SPEC.md` §4 forbids network egress on a
runtime code path in `whiteout/`, which this is not. Nothing waits on it
either way — D7's terrain loader falls back to the deterministic synthetic
generator whenever `WHITEOUT_DEM` is unset (`SPEC.md` §7), so the ticket is
droppable rather than blocking.

---

### D30 — feat: comms degradation in the sim

**Milestone:** M3 · **Demo beat:** — · **Size:** S
**Labels:** `feat` `P3` `area:sim` `loop-ok`

## What
An optional model of inter-agent comms: bandwidth cap, latency and dropout, so
the shared belief map has to be gossiped rather than assumed. Off by default.

## Why
Open question Q4. If bandwidth is modelled by the sponsor, the shared belief
map becomes a much harder and more interesting problem — and we want the option
already in the sim rather than starting then.

## Acceptance
- [ ] Off by default; on, the coordinator still completes an episode
- [ ] D22 can ablate it

## Notes
Not blocked — this is built speculatively and cut if unanswered.
**Second on the cut list** (`SPEC.md` §9 item 2). Depends on D9.

---

### D38 — feat: the weights field and re-score

**Milestone:** M3 · **Demo beat:** 7 · **Size:** S
**Labels:** `feat` `P1` `area:viz` `loop-ok`

## What
Four numeric inputs in the rail and a re-score button. Entering the sponsor's
weights re-scores cached episodes and moves the recommended operating point on
the D37 report.

## Why
Beat 7, and the practical answer to open question Q3: when the booth tells us
the weights, we retune in minutes instead of rearchitecting.

## Acceptance
- [ ] Entering weights and re-scoring completes in under two minutes on cached
      episodes, with no re-simulation
- [ ] Weights persist to `fixtures/weights/` as a named file
- [ ] Inputs are labelled, focusable and at least 40px tall

## Notes
`SPEC.md` §11 Q3. Depends on D10, D37.

---

## M4 — Polish

### D39 — ui: the viewer polish pass against the rubric

**Milestone:** M4 · **Demo beat:** 1–7 · **Size:** M
**Labels:** `ui` `P1` `area:viz` `loop-ok`

## What
Capture every view at 1440×900, 1280×720 and 960px wide, audit against
`DESIGN.md` and the `ui-craft` rubric, fix, capture again. Two passes minimum.

## Acceptance
- [ ] Screenshots taken in-session are attached to the PR, before and after
- [ ] No raw hex or one-off size survives outside `tokens.css`
- [ ] Text contrast is at least AA throughout; focus is visible everywhere
- [ ] No layout shift on first paint; fonts preloaded
- [ ] None of the `DESIGN.md` "Don'ts" appear

## Notes
`ui-craft` skill, `DESIGN.md`. Depends on D34, D35, D36, D37, D38.

---

### D40 — ui: every state, exercised

**Milestone:** M4 · **Demo beat:** 1 · **Size:** S
**Labels:** `ui` `P2` `area:viz` `loop-ok`

## What
Force and screenshot all seven states: no log, loading, malformed log, running,
finished, zero contacts, many contacts.

## Acceptance
- [ ] Each state has a screenshot in the PR
- [ ] The malformed-log state names the failing line and the schema version
- [ ] The empty state offers the bundled fixture as the next action

## Notes
`DESIGN.md`. Depends on D20, D41.

---

## M5 — Submission

### D33 — docs: the README and the four-axis argument

**Milestone:** M5 · **Demo beat:** — · **Size:** M
**Labels:** `docs` `P1` `area:submit` `loop-ok`

## What
The README a sponsor engineer opens: the architecture diagram, the transport
seam, the four differentiators with their formulas linked, the ablation table
from D22, and a plain statement of method — the four axes are the sponsor's own
published criteria, the weights are parameters, and the table shows which
mechanism earned which points.

## Why
Optimising a published objective is engineering, not gaming, and saying so
plainly is cheaper than being asked.

## Acceptance
- [ ] Architecture diagram matches `SPEC.md` §4
- [ ] The ablation table is real output, not illustrative
- [ ] Every claim in it is checkable against a committed artifact
- [ ] No attribution stamps anywhere

## Notes
`SPEC.md` §10 (last risk). Depends on D22, D28.

---

### D42 — submission: Devpost, badge IDs and the repo link

**Milestone:** M5 · **Demo beat:** — · **Size:** S
**Labels:** `submission` `P0` `area:submit` `loop-ok`

## What
Select **Dominion Dynamics WHITEOUT** on Devpost, both badge IDs exactly as
shown under the QR code, the source link, and the design assets.

## Why
**Sponsor selection locks 2026-09-19T18:00Z**, hour 14, before any scored run
is likely to exist. Select WHITEOUT regardless — it is the whole project.

## Acceptance
- [ ] WHITEOUT selected before the lock
- [ ] Both badge IDs present and exact
- [ ] Source link resolves for a logged-out visitor

## Notes
`EVENT.md` "Submission requirements". Badge IDs are a human action.

---

### D27 — submission: the demo video

**Milestone:** M5 · **Demo beat:** 1–7 · **Size:** S
**Labels:** `submission` `P2` `area:submit` `loop-ok`

## What
A three-minute capture of the §2 beats at 1280×720.

## Why
Optional but recommended, and cheap once the beats work.

## Acceptance
- [ ] Under three minutes, 1280×720, audible
- [ ] Beat 3 is legible at video bitrate

## Notes
**Fifth on the cut list** (`SPEC.md` §9 item 5). Depends on D39.

---

## Count

**46 tickets.**

| milestone | count | IDs |
|---|---|---|
| M0 | 3 | D1–D3 |
| M1 | 25 | D4–D20, D22, D24, D34, D35, D36, D41, D43, D45 |
| M2 | 9 | D21, D23, D26, D28, D29, D31, D37, D44, D46 |
| M3 | 4 | D25, D30, D32, D38 |
| M4 | 2 | D39, D40 |
| M5 | 3 | D27, D33, D42 |

Both spikes (D44 = S1, D45 = S2) are listed under the M2 heading for
adjacency, but **D45 is an M1 ticket** and runs before D9.

`needs-decision`: **2** — D31 (Q2) and D32 (Q1). Everything else is
`loop-ok`.
