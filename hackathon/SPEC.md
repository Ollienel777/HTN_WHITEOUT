# SPEC — WHITEOUT

Authority: `hackathon/DECISION.md`. Where this file and that one disagree, that
one wins. Clock and rules: `hackathon/EVENT.md`. Design direction:
`hackathon/DESIGN.md`.

Target: **Dominion Dynamics WHITEOUT** ($2,000 / $1,000 / $500 plus a guaranteed
first-round interview). It is **scored, not judged** — this project is won by a
number, not by a performance.

---

## 1. The product

# Dogwatch

**One-liner.** Four machines keep watch over a two-kilometre strait, and pass
the watch between them until the dark ship is found and held.

**The name.** A *dogwatch* is a deliberately shortened watch at sea, and it
exists for one reason: so the rotation shifts and the same hands do not always
draw the same duty. It is **handover, named** — by sailors, centuries before
this problem. That is the shape of this project. Two towers see nothing until
they are pointed; a fixed-wing is fast with a narrow eye; a quadcopter is slow
with a wide one. The intelligence is not in any one of them. It is in the
passing.

That matters beyond taste: Dominion Dynamics judge **collaboration**, and
defined it at the workshop as *how the robots team up with each other*. The
name states the thesis the run has to demonstrate.

**The user.** Two, and they want different things.

- **The sponsor's scoring harness**, which has no opinions and **seven**
  criteria: search efficiency, coverage, detection speed, tracking duration,
  accuracy, autonomy and collaboration. No weights are published.
- **A judge in a five-minute presentation**, who weighs how we thought about
  the problem, how we decided to implement it, and whether the fleet acts by
  itself. Two of the seven criteria — **autonomy** and **collaboration** — are
  read off behaviour and explanation, not off a scoreboard.

**Superseded.** This section previously claimed one user ("the scoring
harness… it has four numbers") and a product "tuned against a reimplementation
of the sponsor's own scoring function". Both are wrong, and the sections below
have not all been re-planned yet. `hackathon/ARENA.md` is ground truth for
this track and wins wherever it and this file disagree.

**The story.** We do not write flight control. ArduPilot already flies the
vehicles. We write the layer above GUIDED-mode waypoints: where each asset
should be next, why, and who hands a contact to whom. The fleet is
heterogeneous and that is the whole problem — a fixed-wing cannot loiter, a
quad is slow but can stare, a rover is terrain-bound and patient, a tower is
free, permanent and immobile. Good coordination is the difference between four
vehicles sweeping the same valley and four vehicles each doing the thing only
they can do.

**The strategic bet, restated because it orders the backlog.** We measured an
identical architecture (shared decaying belief grid + auction allocation +
particle filter + information-gain routing) appearing in 3 of 3 independent
cold assistant sessions on this track. Roughly half the field will ship it. We
do not try to out-think it. **We out-measure it.** Therefore:

1. **M1 builds the kinematic simulator and the scorer before the coordinator.**
2. **The coordinator's objective function is literally the scorer.**
3. **Parameter search runs overnight**, made possible by (1).
4. **ArduPilot SITL is a validation target, not the dev loop.**

**On demos.** A demo is a secondary hope, not the plan. The belief-map
visualiser in §2 is specified as a **debugging tool first** — it exists because
a belief field and an allocation you cannot see is a belief field and an
allocation you cannot debug. That it also happens to be the most legible
artifact we will own is a bonus we take, not a reason to bend the architecture.
No architectural decision in this spec is made for the stage.

---

## 2. The demo script

Three minutes. Runs entirely on the `kinematic` transport from recorded and
live episode logs, so **no beat depends on SITL, on the sponsor's arena, or on
a network**. The whole thing is a laptop.

Every beat is a view of the **run viewer** (§4), which is the same tool the
team uses to debug. Nothing is built only for this script.

| # | time | beat | needs | fallback |
|---|---|---|---|---|
| 1 | 0:00–0:20 | **The board.** Arctic terrain hillshade, six heterogeneous assets placed, target hidden. Four score dials at zero. "This is the task: find and hold a moving target, scored on coverage, collaboration, efficiency, tracking accuracy." | Terrain render, fleet render, score dials, episode load | Static first-frame PNG from `artifacts/` |
| 2 | 0:20–0:50 | **The race.** Split view, one seed, one terrain, one fleet. Left: mode-agnostic frontier-coverage baseline. Right: our tuned coordinator. Played at 20×. Four scores tick live on both sides. | Split replay, baseline policy, tuned policy, live score readout | Pre-recorded side-by-side MP4 |
| 3 | 0:50–1:25 | **WOW — the map that erodes.** Zoom the belief field. Sensor cones that swept and saw *nothing* visibly carve probability out of the map (negative information). Then the flow-network overlay appears: terrain cuts drawn as bright chords across saddles, and a rover is **posted on a cut instead of sweeping area**. The target walks into it. `TRACKED 0:41`. | Belief field render, negative-information update, choke-point overlay, contact badge | Replay the same seed from a recorded log; if the cut overlay misbehaves, run beat 3 on the belief field alone — the erosion carries it |
| 4 | 1:25–1:50 | **Collaboration, made explicit.** The contact lifecycle ribbon: `unconfirmed → confirming → tracked → handed off`, with handoff arcs drawn between assets. Tower trips, fixed-wing diverts to sweep, quad confirms and holds, rover takes persistence. | Contact lifecycle state machine, handoff arcs, lifecycle ribbon | Ribbon alone, without arcs |
| 5 | 1:50–2:20 | **The real argument.** The tuning report: N episodes run overnight, score against each policy parameter, the chosen operating point marked, and the ablation table (negative information off, choke points off, hysteresis off). "We did not guess these numbers." | Tuning report page, sweep artifacts, ablation table | Committed report from the last completed sweep in `artifacts/tuning/` |
| 6 | 2:20–2:45 | **Validation.** The same coordinator, unmodified, driving ArduPilot SITL vehicles over MAVLink. Only the transport was swapped. | `sitl` transport, `sim_vehicle.py` running | **Primary path is the recorded SITL episode log replayed through the same viewer** (D28) — the log is the evidence, the live run is the flourish. **If D46 closes as SITL unobtainable there is no live run and no recording**: beat 6 becomes the §3 MAVLink-only claim, shown as the `pymavlink` loopback D46 stands and records on overrun (its acceptance owns it), said out loud as "the MAVLink half, against a loopback — we could not obtain ArduPilot" |
| 7 | 2:45–3:00 | **The close.** Type the sponsor's four weights into the weights field, press retune, watch the recommended operating point move. "Your weights are parameters here, not constants." | Weights input, cached sweep re-scored | Show the weights file and the re-scored table from disk |

**The wow beat is beat 3.** It is the only beat that shows something the
converged architecture structurally does not do: a map that gets *more*
informative from seeing nothing, and assets posted at cuts rather than
rastering area.

**Every P1 ticket maps to a beat or to a scored axis**, except three that are
P1 for a deadline or a prize clause instead: D43 (the arena runbook, due
before the 14:30Z workshop), D32 (the `arena` adapter) and D33 (the submission
README). A ticket that maps to no beat, no scored axis and no prize clause is
P3 and probably should not exist.

---

## 3. Prize requirements

### Dominion Dynamics: WHITEOUT — targeted, the whole project

Requirement, quoted from `hackathon/tracks/prizes.md` (source: the Devpost prize
block, which our red team confirmed is the only public text in existence):

> "A live Arctic simulation. Write the shared intelligence coordinating a
> heterogeneous fleet — fixed-wing aircraft, quadcopters, rovers and fixed
> sensor towers, all running ArduPilot and MAVLink — to detect, classify and
> track a moving target across contested terrain. No swarm experience required;
> bring distributed thinking. Scored live on coverage, collaboration, efficiency
> and tracking accuracy."

How each clause is met visibly:

| clause | met by | visible in |
|---|---|---|
| heterogeneous fleet — fixed-wing, quad, rover, tower | Four vehicle classes with distinct kinematic envelopes and distinct sensor models, each with a distinct role in the policy | Beats 1, 4 |
| running ArduPilot and MAVLink | **Primary:** `sitl` transport against real ArduPilot — `sim_vehicle.py --count N --auto-sysid --mcast` inside the Docker image D46 builds, `pymavlink`, GUIDED mode, `SET_POSITION_TARGET_GLOBAL_INT`. **Fallback if D46 fails:** the MAVLink half only — `pymavlink`-encoded `SET_POSITION_TARGET_GLOBAL_INT` / `GLOBAL_POSITION_INT` over a socket, proved by the `pymavlink` loopback D46's overrun path stands and records, with the README stating plainly that the ArduPilot half was not exercised. There is no honest fallback that exercises ArduPilot itself without ArduPilot. | Beat 6 |
| detect, classify, track | Contact lifecycle machine with an explicit classify step | Beat 4 |
| shared intelligence / distributed thinking | One shared belief state, auction allocation across classes, explicit cueing and handoff | Beats 3, 4 |
| coverage, collaboration, efficiency, tracking accuracy | Our reimplementation of all four with weights as parameters; the coordinator optimises them directly | Beats 2, 5, 7 |
| contested terrain | Terrain occlusion in the sensor model; sensor dropout and comms-degradation options in the sim | Beat 3 |

### Solana: Best Badge Hack — **out of scope**

**Decided, not deferred.** The operator has ruled the Solana Best Badge Hack
**out of scope for this repository and this run**: it is being pursued in a
separate session, in a different shape from the one
`hackathon/ideation/BADGE-HACK-BRIEF.md` recorded (local-only —
`hackathon/ideation/` is gitignored, so that path resolves on the build machine
and nowhere else). It shares no code, no time, no submission and no demo slot
with this build. **No ticket in this backlog serves it, and none should be
added.** It is not an open question, and §11 does not name it.

### Nothing else

Each extra sponsor claim is another live demo inside the same 09:45–11:45
window in which this team already owes Dominion a scored run. Adding a prize is
a human decision (§11).

---

## 4. Architecture

> **Re-planned 2026-09-19T17:15Z against `hackathon/ARENA.md`**, which is
> ground truth for this track. The version below replaces one written when no
> public source about the arena existed. Sections not yet re-planned are
> marked where they stand.

### The seam — structural, and it paid off

The coordinator talks to the world through **one interface**. Observations in;
intents out. Nothing else crosses it.

```
                  ┌──────────────────────────────────────┐
                  │             coordinator              │
                  │   belief · contacts · allocation     │
                  └────────▲──────────┬──────────┬───────┘
          WorldObservation │          │FleetIntent│ Track
                  ┌────────┴──────────▼───────┐  │
                  │      Transport (Protocol) │  │
                  └──┬──────────┬─────────────┘  │
                     │          │                │
              ┌──────▼───┐ ┌────▼─────┐   ┌──────▼───────┐
              │kinematic │ │   sitl   │   │  arena       │
              │ fast fake│ │local dev │   │ ArcticSim    │
              │  no deps │ │ harness  │   │ ** the one   │
              │          │ │ ArduPilot│   │  that counts**│
              └──────────┘ └──────────┘   └──────────────┘
```

**The roles have inverted from the original plan, and that is the whole
story.** `arena` was the speculative third implementation against an unknown
interface; it is now the target, and its interface is known. `sitl` was the
validation step; it is now a **local development harness** — ArcticSim's
assets are stock ArduPilot Copter, Plane and AntennaTracker, so a vanilla
local SITL of those three is a near-exact stand-in that costs nothing to run
and does not compete for the one heavyweight Docker stack. `kinematic` remains
the fast fake that CI runs.

The seam earned its keep exactly as intended: the arena turned out **not** to
be a single MAVLink socket — it is MAVLink per asset **plus an HTTP tracks
API** — and the cost of that surprise was an adapter.

**Note the third arrow.** A `Track` leaves the coordinator by a second path,
to the sponsor's tracks API. That is the artifact the judges read, and it does
not travel as a `FleetIntent`.

It is a `Protocol`, not an ABC.

### Components

| component | package | what it owns |
|---|---|---|
| **Transport** | `whiteout/transport/` | The one interface and its implementations. Adapters only: no belief, no policy, no scoring ever lives here. |
| **Vision** | `whiteout/vision/` | **New, and the hard part.** Read camera frames, detect the vessel, and project a pixel to a lat/lon using the asset's pose and its published FOV. |
| **Tracks** | `whiteout/tracks/` | **New.** The client for the sponsor's `/api/tracks`, and the track-maintenance loop that keeps posting once the vessel is held. **The only artifact the judges read.** |
| **Belief** | `whiteout/belief/` | The decaying field **over the water of the strait**, and the **negative-information** update keyed on real camera footprints. |
| **Policy** | `whiteout/policy/` | Allocation, information-gain routing, tower placement, the contact lifecycle machine, re-tasking hysteresis, and the **frontier-coverage fallback**. |
| **Viz** | `viz/` | The run viewer. Static HTML + canvas. Zero build step. Debugging tool *and* presentation material. |
| **CLI** | `whiteout/cli.py` | `run`, `replay`, `score`, `serve`. |

**Gone, and why.** `whiteout/sim/` — ArcticSim supplies terrain, vehicles,
sensing and the target; we were building a second, worse copy.
`whiteout/tune/` — the sweep needed a fast simulator and a faithful scorer,
and has neither. `whiteout/score/` survives only as a **development
instrument** for comparing two policies, rebuilt small against the seven real
criteria; it is not a target and nothing optimises against it.

### The data model

Three record types, and everything else is derived from them.

```python
# whiteout/types.py  —  frozen dataclasses, JSON-serialisable

Pose        : asset_id, cls, t, x, y, z, heading, speed, energy_used
SensorReport: asset_id, t, footprint, detections[], negative: bool
WaypointIntent: asset_id, t, target_xy, target_z, speed, reason, task_id
```

- `WorldObservation` = `{t, poses[], reports[]}` — everything in.
- `FleetIntent` = `{t, intents[]}` — everything out, to the fleet.
- **`Track`** — `{name, lat, lon, heading?, speed?}`, posted to the sponsor's
  API. **This is the scored artifact.** Everything else exists to produce it.
- **`EpisodeLog`** (JSONL, one record per tick) is our own instrument: it
  drives the viewer and lets two runs be compared. Versioned and validated, as
  seven review rounds established — but it is **no longer the project's
  spine**, because the judges never see it.

**There is no ground truth.** The original model had the sim write a `truth`
field read only by the scorer, with a test stopping the coordinator importing
it. **ArcticSim gives us nothing of the kind**: we never learn where the
vessel actually was. The guard is harmless and can stay, but the thing it
guarded against no longer exists — and the consequence is sharper than it
sounds. **We cannot measure our own accuracy.** Nothing tells us whether a
posted lat/lon was right, which is why a confident wrong detection is worse
than none.

### Where the hard part lives

Three places, in order of risk. **All three are different from the original.**

1. **`vision/detect.py`** — find a small dark vessel among bright ice floes in
   a 640×480 or 640×360 frame, with no labelled data and no time to train. It
   was absent from this spec entirely until the workshop, and nothing
   downstream works without it.
2. **`belief/negative.py`** — the non-detection likelihood given range and the
   camera's field of view. Get it wrong and belief erodes confidently in the
   wrong water; it looks plausible for hours, and the fleet never looks there
   again.
3. **`vision/project.py`** — pixel to lat/lon, via pose and FOV. It converts a
   detection into the only number that scores, and *accuracy* is a judged
   criterion. It is also the one piece here that is **fully determined and
   testable before the sim arrives**.

### External services

**The arena is a network service, and this section used to deny that.**
It previously read "there are none at runtime". That is false: at runtime we
speak MAVLink to four ArcticSim assets and **HTTP POST to their tracks API**,
which is the scored path. What remains true is that we hold **no API keys, no
accounts and no cards**, and that nothing leaves the local network.

| service | what | when |
|---|---|---|
| **ArcticSim** | four MAVLink endpoints; asset camera streams | runtime — it is the arena |
| **`/api/tracks`** | `POST` a name, lat and lon; update by name | runtime — the scored path |
| **Mapbox** | a free token for hi-res terrain **inside their sim** | setup only, optional |
| **ArduPilot SITL** | local dev harness, three vehicle types | development only |

**ArcticSim runs on our machine or theirs.** `git clone`, `cp .env.example
.env`, `docker compose up --build`, needing **8 cores (12 recommended), 16 GB
RAM (24 recommended) and 50 GB disk**. If the machine falls short, Dominion
Dynamics issue a **WireGuard config to a dedicated cloud instance** and need
the team roster — a human round-trip, and the first thing to resolve.

**The DEM is gone.** ArcticSim renders Bellot Strait itself; we ship no
terrain and load no tile.

---

## 5. Stack and conventions

| | |
|---|---|
| language | **Python 3.11** (confirmed present: 3.11.9) |
| package manager | **`pip` with `pyproject.toml`**, one venv at `.venv`. One manager, no `uv`/`poetry`/`conda` mixing. |
| core libs | NumPy, SciPy, `pymavlink` (sitl transport only), `networkx` (flow cuts), `cma` (optional, tuner) |
| lint + format | **ruff** (`ruff check`, `ruff format --check`) |
| typecheck | **mypy**, strict on `whiteout/`, `Protocol` for the transport seam |
| test | **pytest**, with `pytest-randomly` disabled — determinism is the product |
| viewer | **static HTML + `<canvas>`, vanilla JS, no build step** |
| layout | `whiteout/` package · `viz/` viewer · `tests/` · `fixtures/` · `artifacts/` (gitignored except committed reports) · `scripts/` |

### Why the viewer is not a Node front end

Node v22 and npm are available, and a Next.js app is the reflex. We are not
taking it, and the reason is the gate, not taste:

- The viewer's job is to **replay a JSONL episode log**. There is no account,
  no routing, no server-side data fetching, no auth — none of the things a
  framework buys.
- A Node front end adds a **second package manager, a second lint and
  typecheck stack, a `node_modules` install and a dev-server port** to a gate
  that must stay fast enough to run on every PR from parallel worktrees.
- The log schema is validated in pytest. One language for the tests.
- A `<canvas>` draws a 256×256 belief field plus sixty sprites at 60fps
  without help.
- It opens from `file://`. At a sponsor booth, on someone else's laptop, with
  no install, that matters more than component libraries do.

If the viewer later needs a framework, the episode log is the seam and nothing
else changes. That is why the choice is cheap to reverse, and why it is being
made rather than deferred.

### Areas (`area:` labels)

One per module, so parallel worktrees do not collide:

`area:sim` · `area:score` · `area:belief` · `area:policy` · `area:transport` ·
`area:tune` · `area:viz` · `area:infra` · `area:submit`

### Conventions

- **Determinism is a hard rule.** Every stochastic call takes an explicit
  `numpy.random.Generator` derived from the episode seed. No module-level
  `np.random`. A test asserts byte-identical episode logs across two runs at the
  same seed, and it is a P0 when it fails.
- Frozen dataclasses for all record types. No dicts crossing module boundaries.
- Policy parameters live in **one** dataclass, `policy/params.py`, serialisable
  to and from JSON. The tuner writes that JSON; nothing else configures policy.
- No file in `whiteout/` imports from `viz/`. No file in `whiteout/policy/`
  imports from `whiteout/sim/`.
- Docstrings on the physics and the likelihoods, with the formula. The four
  differentiators must be readable by a sponsor engineer in five minutes.

---

## 6. The gate

**One command.** CI runs it and so does every implementer, every time, before
opening a PR.

```bash
python scripts/gate.py
```

It runs, in order, failing fast:

| step | command |
|---|---|
| install | `pip install -e ".[dev]"` (done by CI before the gate; the gate asserts imports resolve) |
| lint | `ruff check whiteout tests scripts` |
| format | `ruff format --check whiteout tests scripts` |
| typecheck | `mypy whiteout` |
| test | `pytest -q -m "not slow"` — **the `-m "not slow"` is required**: bare `pytest -q` collects `slow`-marked tests, and the one such test is a wall-clock throughput assertion (D9) that is a coin flip on shared runners and in parallel worktrees. `slow` tests run in the tuner (D23), never in the gate. |
| build | `python -m build --wheel --no-isolation` (proves the package is installable, and it is what the submission links). **`--no-isolation` is required**: the default fetches the build backend from PyPI on every invocation, which breaks §4's no-egress rule and spends venue wifi on every gate run. |
| smoke | `python -m whiteout.cli run --seed 7 --ticks 400 --out artifacts/smoke.jsonl` (no transport flag exists; `WHITEOUT_TRANSPORT` unset selects `kinematic`, per §7 and D3) then `python -m whiteout.cli score artifacts/smoke.jsonl --weights fixtures/weights/equal.json` — must exit 0 and print four finite scores |
| determinism | second smoke run at the same seed; logs must be byte-identical |

The gate must:

- **run in fake mode always.** `WHITEOUT_TRANSPORT` defaults to `kinematic`.
  The gate never starts SITL, never opens a socket, never spends anything.
- **take any port from `PORT`**, with no reuse of a running server. The viewer
  dev server (`whiteout.cli serve`) binds `PORT` or an ephemeral port, never a
  fixed 8000. Parallel worktrees run side by side.
- **exclude `.claude/**` from every lint, format, typecheck and test glob** —
  agent worktrees live there. This goes in `pyproject.toml` (`[tool.ruff]
  exclude`, `[tool.mypy] exclude`, `[tool.pytest.ini_options] norecursedirs`)
  and in `.gitignore`-adjacent globs, not in the workflow file.
- **finish under three minutes** on a laptop. The `slow` mark is the release
  valve: any test that threatens that budget, or that asserts on wall-clock
  time, is marked `slow`, which the test step already deselects, and is run in
  the tuner (D23) instead. `slow` must be registered in
  `[tool.pytest.ini_options] markers` so the deselection cannot silently
  become a no-op.

Changing the gate means editing `scripts/gate.py`, never `.github/`. The
workflow calls the script and nothing else.

### CI

**There is currently no CI workflow at all — `.github/workflows/` does not
exist. The first M0 ticket creates it.** It must:

- run on **every pull request, with no path filters**. A PR with no check runs
  can never settle.
- also run on **every push to `main`**.
- be a single job: checkout, `setup-python@v5` with 3.11, `pip install -e
  ".[dev]"`, `python scripts/gate.py`.
- never install ArduPilot, never touch the network beyond PyPI.

After M0, `.github/` is harness territory and waits for a human merge
(`CLAUDE.md`).

---

## 7. Environment

**No secrets exist in this project.** There is no API key, no token, no
account, no card. `.env.example` carries names only; `.env.local` is gitignored
and, in this build, is expected to stay empty.

| variable | used by | meaning |
|---|---|---|
| `WHITEOUT_TRANSPORT` | everything | `kinematic` (default, the fake) · `sitl` · `arena` |
| `WHITEOUT_SEED` | sim, tuner | Episode seed. Absent means 0, never "random". |
| `WHITEOUT_DEM` | `sim/terrain.py` | Path to the DEM tile. Absent means the synthetic generator — never a failure. |
| `WHITEOUT_SITL_ENDPOINT` | `transport/sitl.py` | `mcast:` or `udp:` endpoint for `sim_vehicle.py`. Absent means the sitl transport refuses to start, loudly. |
| `WHITEOUT_ARENA_ENDPOINT` | `transport/arena.py` | The sponsor's endpoint, once known. |
| `WHITEOUT_ARTIFACTS` | tuner, viz | Output directory. Defaults to `artifacts/`. |
| `PORT` | `whiteout.cli serve` | Viewer server port. Ephemeral when unset. |

**Fake mode is the default and is not a degraded path.** `kinematic` is the dev
loop, the tuning loop, the CI loop and the demo. `sitl` is the validation
target; its absence is normal and never blocks the gate, the demo or a ticket.
`arena` is a stub until the 14:30Z API workshop answers Q1 (§11).

---

## 8. Milestones

| milestone | exit criterion |
|---|---|
| **M0 Skeleton** | Package scaffold, `pyproject.toml`, ruff/mypy/pytest configured with `.claude/**` excluded, **the CI workflow**, `scripts/gate.py`, the `types.py` record set, the episode-log schema and its validator, the `Transport` Protocol with a trivial kinematic stub, `.env.example`, and a smoke test that runs 400 ticks and scores them. The first M0 ticket also **deletes `hackathon/backlog-draft.md`**. |
| **M1 Demo path** | Every beat of §2 works end to end on the `kinematic` transport: sim, scorer, belief with negative information, flow cuts, contact lifecycle, hysteresis, auction allocation, the frontier fallback, and the viewer. **Beats 5, 6 and 7 are out of M1's scope.** Beat 5's evidence is the sweep (D23) and the tuning report page (D37), both M2, and its §2 fallback needs D23 too, so beat 5 exits under M2(b) below. Beat 6's evidence is D28's recording, which is M2 and depends transitively on D46, so M1 cannot exit on it either way. Beat 7's builder is D38, which is M3, and the operating point it moves lives on D37's report, which is M2; its §2 fallback needs the cached sweep (D23) too, so beat 7 exits under M3 below, where §9 item 4 already sites its CLI fallback. M1 exits with beats 5, 6 and 7 standing on whichever §2 fallback is live at the time. |
| **M2 Wow** | (a) The `sitl` transport drives ≥4 ArduPilot vehicles for real **inside the Docker image D46 builds**, and the run is recorded as a committed episode log that the viewer replays everywhere else. If D46 reports SITL unobtainable inside its timebox, M2(a) exits as "not obtainable", recorded with the reason, and the §3 fallback applies. (b) A **completed overnight sweep with a decided operating point and an ablation table**, whichever policy wins. M2(b) does **not** require the tuned set to beat the frontier baseline — if frontier wins, that is the result, and §10's top risk row says we ship frontier. |
| **M3 Prizes** | Every clause of the §3 table is visibly met. The `arena` adapter exists as a documented stub with a runbook, and the sponsor's weights can be entered and applied in under two minutes. |
| **M4 Polish** | The viewer passes the `ui-craft` rubric at 1440×900, with real episode data, and every state handled (no log, log loading, log malformed, episode running, episode finished, no contacts, many contacts). |
| **M5 Submission** | Per `docs/build/SUBMIT.md`: repo, README with the architecture diagram and the four-axis argument, badge IDs, WHITEOUT selected on Devpost before the 18:00Z lock, demo video. |

**M0 runs one ticket at a time.** While any M0 ticket is open, no other
milestone's ticket is eligible.

**The ordering is the bet.** M1 builds the sim and the scorer *before* the
coordinator. An implementer who finds this backwards should read
`DECISION.md` §"The strategic bet" rather than reorder it.

---

## 9. The cut list

Drop in this order. Everything above the line survives to the end.

1. **D31 the policy-mode filter** (inferring evader behaviour modes). Only pays
   against a human or adversarial evader. Stretch, never a foundation.
2. **D30 comms-bandwidth modelling.** Interesting, unscored unless the booth
   says otherwise.
3. **D29 CMA-ES.** Random search over a well-chosen parameter box gets most of
   the win. Cut the optimiser, keep the sweep.
4. **D28 the live SITL run in beat 6.** The recorded log is the evidence; the
   live run is the flourish. Cut the flourish, never the recording — unless
   D46 closes as SITL unobtainable, in which case neither exists and beat 6
   falls back to the MAVLink-only claim in §2 and §3.
5. **D27 the demo video.** Recommended, not required.
6. **D26 classification depth.** Collapse `classify` to a confidence threshold
   on the lifecycle machine rather than a separate classifier.
7. **D25 the DEM tile.** The synthetic terrain generator is deterministic and
   sufficient. The DEM is a prettier picture, not a better score.
8. **D24 the split-screen view** in beat 2. Two sequential replays tell the same
   story; the split is polish.

— cut line —

Never cut: the transport seam, the episode log, the scorer, determinism, the
frontier-coverage fallback, negative information.

### If M1 is late

The eight items above protect M2–M5. M1 is 25 of the 46 tickets and the entire
demo path, so it needs its own degrade order. Drop in this order, and stop as
soon as the schedule is recovered.

**Only items 1, 2 and 3 free M1 hours.** They are the whole of the relief this
list offers — D24 (S), D13 (M, plus a refused four-hour timebox) and D36 (S),
all three M1 — and the hysteresis degrade below. Items 4 and 5 are listed to
keep the demo script coherent once those three are gone; **neither recovers any
M1 time**, and reading this as five items of relief at hour 20 will overstate
how recoverable the schedule is by roughly a factor of two.

Freeing M1 hours:

1. **D24, the split-screen view** (M1, S; also item 8 above). Two sequential
   replays tell the same story.
2. **D13, the flow-network cuts, becomes overlay-only** (M1, M). This is
   already §10's mitigation and its four-hour timebox is the first budget to
   refuse: if M1 is late when D13 comes up, the timebox is not spent at all and
   `belief/flow.py` ships as a render-only layer the policy ignores. Beat 3
   keeps the erosion; it loses the cut chords.
3. **D36's handoff arcs go static** (M1, S). The `ContactRibbon` still shows
   the lifecycle, so beat 4's collaboration evidence survives; only the drawn
   arc is lost.

Freeing no M1 hours — script consequences, recorded so the three-minute run
still hangs together:

4. **Beat 7 collapses to a file.** `WeightsField` becomes "edit
   `fixtures/weights/*.json` and re-score" — the two-minute re-weight claim in
   M3 is met by the CLI rather than the UI. **No M1 relief:** the builder is
   **D38, which is M3**, and §2 already gives beat 7 exactly this fallback.
5. **Beat 4 narrows to the ribbon**, and D26's classification depth collapses
   to a confidence threshold (item 6 above, pulled forward). **No M1 relief:**
   **D26 is M2** and is already on the main cut list.

Of the four differentiators, **negative information (D12) and the contact
lifecycle (D17) are never degraded** — the first is on the never-cut line
above, the second is the only visible evidence on a scored axis. The flow-cut
view degrades first, and re-tasking hysteresis degrades to a fixed constant
rather than a tuned parameter before either of those is touched.

---

## 10. Risks and spikes

**Re-planned 2026-09-19T17:15Z against `hackathon/ARENA.md`.** The original
table's top risks were an unknown sponsor interface, a sweep that might not
finish, and a flow network that might eat a day. The first is answered; the
other two are closed. **The table below is the live one. The superseded table
follows it, struck, so the reasoning is not lost.**

| risk | severity | mitigation | ticket |
|---|---|---|---|
| **Vision does not work.** Find a small dark vessel among bright ice floes in a 640×480 frame, with no labelled data and no time to train. Nothing downstream scores without it, and it was absent from this spec until the workshop. | **highest** | Start with the cheap thing: floes are bright and static, the vessel is dark, and it is **the only moving object in the world**. Background subtraction, contrast against water, motion across frames. Build the pipeline and the evaluation harness **before** the sim arrives; fit the detector when real imagery exists. | #66 |
| **The machine cannot run ArcticSim.** 8 cores, 16 GB, 50 GB. Nothing can be validated until the arena runs somewhere. | **highest** | Check the specs **now**. If short, DD issue a WireGuard config to a cloud instance and need the team roster — a human round-trip that does not get faster by starting it later. | #63 |
| **We never see the vessel at all.** A 25 × 2 km channel, four assets, a random spawn and a random path. A search that never intersects the target scores nothing on every criterion at once. | **high** | Towers posted deliberately across the narrows, the fixed-wing sweeping the long axis, and a belief field that rules out water already swept. This is what *search efficiency* and *coverage* measure. | #68, #12, #13 |
| **The negative-information likelihood is subtly wrong.** It looks plausible for hours, and an over-confident update empties belief from water the vessel is in — after which the fleet never looks there again. | **high** | Property test over many seeds: entropy non-increasing under non-detections, posterior always in [0,1]. A single-camera closed-form case checked by hand. A detector with a known false-negative rate must not drive belief to zero on one empty frame. | #13 |
| **A wrong lat/lon is posted and we cannot tell.** *Accuracy* is judged, and **the arena gives no ground truth** — nothing ever confirms a fix was right. | **high** | Treat a confident wrong detection as worse than none: the detector degrades to silence rather than guessing. Sanity-check every fix against the belief field and the channel — a fix on land is a bug we *can* catch. | #66, #67 |
| **One judged run, on Sunday.** No leaderboard, no retries, no iterating against their scorer. | **high** | Rehearse end to end against their sim, more than once. The original answer to risk was a tuning sweep; the answer now is rehearsal. | #63, #69 |
| **The sim crashes.** DD said so plainly in their own deck. | medium | Reset, and wait five minutes before asking them. Do not edit `.env` beyond tower lat/lon. Everything above the transport is developed against local SITL, so a crash costs minutes. | #45, #63 |
| **Information-gain routing loses to a systematic sweep.** Against a target on a random walk — now **confirmed** non-adversarial — betting an asset on unobserved water and being wrong scores worse than sweeping. | medium | The **frontier-coverage fallback is a first-class, selectable policy**, not a stub. The original mitigation was "the scorer tells us which wins" — **that arbiter is gone**, so choose by rehearsal, and be ready to ship frontier. | #27, #16 |
| **The presentation is unprepared.** Half the judging, and this spec used to say presentation work was worth nothing here. | medium | Draft early, rehearse against a clock. Blocked by nothing. | #69 |
| **Autonomy is overclaimed.** It is judged, and the panel watches the run. | medium | State plainly what is autonomous and what is hand-flown. Overclaiming to people who can see the screen is the worst available move. | #69 |

**Spikes**, timeboxed, run before the work they de-risk:

- **S1 (1h, before #47):** stand up local ArduPilot SITL for **copter, plane
  and antennatracker** — the last especially, since towers are driven by raw
  servo PWM. Confirm `guided <lat> <lon> <alt>` moves a vehicle and `servo
  set` moves a tracker. Answer: can the control layer be built before the
  arena arrives.
- **S2 (1h, immediately after #63):** point one asset's camera at the vessel
  by hand and look at the frames. Answer: **is it visibly detectable at all**,
  and at what range. The earliest possible read on the project's highest risk,
  and it needs no code.

**Dropped:** the eleven-SITL-instance ceiling (ArcticSim runs its own four),
SITL-unobtainable (a dev convenience now, not a deliverable), the overnight
sweep, "teaching to the test reads as gaming" (no scorer to teach to), the
flow-network over-engineering risk (#14 closed), and accidentally reading
ground truth (there is none).

---

### Superseded table, kept for its reasoning

| risk | severity | mitigation | owner ticket |
|---|---|---|---|
| **Information-gain routing loses to a lawnmower sweep.** Against a scripted, non-adversarial target, sending an asset to unobserved ground on a hypothesis and being wrong scores *worse* on coverage than systematic sweeping. This is the red team's standing warning and it is not hypothetical. | **high** | The **mode-agnostic frontier-coverage fallback is a first-class, selectable policy**, not a stub. The scorer tells us which wins under the booth's weights instead of us guessing. If frontier wins, we ship frontier and the tuning report is *still* the argument. | D14, D22 |
| **The negative-information likelihood is subtly wrong.** It looks plausible for hours. It is also the cheapest accuracy win, so it is load-bearing. | **high** | A property test: over many seeds, belief entropy must fall monotonically under non-detections, and the posterior must never assign <0 or >1. Plus a closed-form single-sensor case checked by hand. | D11, D12 |
| **The sponsor's interface is not MAVLink.** `dominiondynamics.online` does not resolve; the Devpost block is the only public text in existence; a search-engine summary calling it "an arena simulation SDK" appears in no sponsor text and is **not treated as sourced**. | **high** | The seam. Only the adapter is lost. Answered at the Dominion Dynamics API Workshop, **2026-09-19T14:30Z** (hour 10.5), with D43's checklist in hand. | D6, D32, D43 |
| **Eleven SITL instances plus a renderer do not fit on one laptop.** Our own analysis put the cut at six, and six was untested. | medium | The kinematic sim exists so this never blocks anything. SITL is validation; the *recorded log* is the deliverable. Spike: time four instances early, before committing to six. | **Spike S1**, D21 |
| **SITL cannot be obtained at all.** It is not installed, and Windows 11 has no native ArduPilot SITL. | medium | **D46** acquires it as a Docker image (Docker 29.4.1 is installed and working) in a one-hour timebox, as a loop action needing no account, key or purchase. On overrun D46 closes as "unobtainable", no M1 ticket is blocked, the §3 MAVLink-only fallback for the ArduPilot clause applies, and **beat 6 falls back to that same claim shown as the `pymavlink` loopback D46's overrun path stands and records** (§2), all of it stated as such in the README. | **D46** |
| **The overnight sweep does not finish, or finds nothing.** | medium | The sweep is checkpointed and resumable; partial results are a valid report. The sim's speed target (≥100× real time on the kinematic transport) is a **measured number, reported in D9 and re-measured by the tuner**, not an aspiration — so we learn at hour 6 rather than hour 26. It is **not** a wall-clock assertion inside the gate: the gate runs on shared CI runners and from parallel worktrees (§6), where a fixed throughput threshold is a coin flip and a red gate on an unrelated PR is a PR that cannot settle. The assertion lives behind `-m slow`, excluded from the gate per §6. | D9, D23 |
| **Teaching to the test reads as gaming.** | low | It is not, and the README says so plainly: the four axes are the sponsor's own published criteria, the weights are parameters, and the ablation table shows which mechanism earned which points. Optimising a published objective is engineering. | D33 |
| **The flow-network cut view is over-engineered.** It is the most interesting idea and therefore the most likely to eat a day. | medium | Timeboxed. `networkx` min-cut on a coarse graph, posting stations as the cut edges' midpoints. If it is not producing sane cuts in four hours, it becomes an overlay-only feature and the policy ignores it. | D13 |
| **We accidentally read ground truth in the coordinator.** | medium | A test asserts the coordinator module never imports the truth field, plus a runtime guard in the transport. | D5 |

**Spikes**, timeboxed, run before the work they de-risk:

- **S1 (1h, after D46, before D21):** inside D46's Docker image, start four
  `sim_vehicle.py` instances with
  `--count 4 --auto-sysid --mcast`, connect `pymavlink`, confirm GUIDED-mode
  `SET_POSITION_TARGET_GLOBAL_INT` moves a vehicle. Record CPU. Answer: how
  many instances fit.
- **S2 (45m, before D9):** measure the naive kinematic tick cost at
  6 vehicles × 256×256 grid. Answer: does ≥100× real time need vectorisation or
  just care.

---

## 11. Open questions for the human

**These do not hold the spec PR.** All five are resolved at the **Dominion
Dynamics API Workshop, 10:30 local = 2026-09-19T14:30Z, hour 10.5 of 32** —
3.5 hours before the 18:00Z sponsor selection lock below. That is a hard time,
not "when the bay opens": **D43's checklist must be printed before 14:30Z**, so
it is an input to the workshop rather than something written during it. **The
sim, the scorer and the transport seam are invariant to all five questions —
that is the entire point of the ordering.** Only two tickets genuinely cannot
start without an answer, and both are `needs-decision` until 14:30Z.

| # | question | blocks |
|---|---|---|
| Q1 | **Do you provide the harness and simulator, or do we stand up ArduPilot SITL ourselves? What interface — plain MAVLink over a socket, or a hosted service with its own schema?** | **D32** (`arena` transport adapter) — `needs-decision` |
| Q2 | **Is the target adversarial (evades) or scripted? Does "contested" mean an active adversary, comms denial, or terrain and weather?** | **D31** (policy-mode filter) — `needs-decision`. If the answer is "adversarial", D31 comes off the cut list immediately and is re-prioritised. |
| Q3 | **The relative weights on the four axes.** Highest-value single answer available. | Nothing. Weights are parameters (`fixtures/weights/`), the sweep is re-scored from cached episodes, and retuning takes minutes. This is why they are parameters. |
| Q4 | **Is comms between agents free, or is bandwidth modelled?** If modelled, the shared belief map becomes a much harder and more interesting problem. | Nothing. D30 builds a comms-degradation *option* in the sim speculatively; it is on the cut list until answered. |
| Q5 | **Continuous leaderboard across the weekend, or one judged run Sunday?** Decides whether we iterate against the scorer or polish one run. | Nothing structural. It changes the hour-24 triage, not the build. |

Also noted, not a build question: **sponsor selection locks
2026-09-19T18:00Z**, hour 14, before any scored run is likely to exist. Select
WHITEOUT regardless — it is the whole project.

**Decisions reserved to the human** (the loop files `needs-decision` rather than
choosing): adding any second sponsor prize, changing the stack after M0,
abandoning the scorer-first ordering, and anything that spends money, needs a
sign-up, or is irreversible and public. The Solana Best Badge Hack is **not**
one of these — it is decided and out of scope (§3), so nothing about it is
escalated.
