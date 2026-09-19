# SPEC — WHITEOUT

Authority: `hackathon/DECISION.md`. Where this file and that one disagree, that
one wins. Clock and rules: `hackathon/EVENT.md`. Design direction:
`hackathon/DESIGN.md`.

Target: **Dominion Dynamics WHITEOUT** ($2,000 / $1,000 / $500 plus a guaranteed
first-round interview). It is **scored, not judged** — this project is won by a
number, not by a performance.

---

## 1. The product

**One-liner.** A coordination layer for a heterogeneous ArduPilot/MAVLink fleet
— fixed-wing, quadcopters, rovers and fixed sensor towers — that detects,
classifies and tracks a moving target across Arctic terrain, and that was
**tuned against a reimplementation of the sponsor's own scoring function**
rather than argued into existence.

**The user.** The sponsor's scoring harness. It is the only user that matters
and it does not have opinions; it has four numbers: **coverage, collaboration,
efficiency, tracking accuracy**. Everything in this spec exists to move those
four numbers, or to let us measure that we moved them.

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
| 6 | 2:20–2:45 | **Validation.** The same coordinator, unmodified, driving ArduPilot SITL vehicles over MAVLink. Only the transport was swapped. | `sitl` transport, `sim_vehicle.py` running | **Primary path is the recorded SITL episode log replayed through the same viewer** — the log is the evidence, the live run is the flourish |
| 7 | 2:45–3:00 | **The close.** Type the sponsor's four weights into the weights field, press retune, watch the recommended operating point move. "Your weights are parameters here, not constants." | Weights input, cached sweep re-scored | Show the weights file and the re-scored table from disk |

**The wow beat is beat 3.** It is the only beat that shows something the
converged architecture structurally does not do: a map that gets *more*
informative from seeing nothing, and assets posted at cuts rather than
rastering area.

**Every P1 ticket maps to a beat.** A ticket that maps to no beat and to no
scored axis is P3 and probably should not exist.

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
| running ArduPilot and MAVLink | **Primary:** `sitl` transport against real ArduPilot — `sim_vehicle.py --count N --auto-sysid --mcast` inside the Docker image D46 builds, `pymavlink`, GUIDED mode, `SET_POSITION_TARGET_GLOBAL_INT`. **Fallback if D46 fails:** the MAVLink half only — `pymavlink`-encoded `SET_POSITION_TARGET_GLOBAL_INT` / `GLOBAL_POSITION_INT` over a socket, proved by the D6 conformance suite against a `pymavlink` loopback, with the README stating plainly that the ArduPilot half was not exercised. There is no honest fallback that exercises ArduPilot itself without ArduPilot. | Beat 6 |
| detect, classify, track | Contact lifecycle machine with an explicit classify step | Beat 4 |
| shared intelligence / distributed thinking | One shared belief state, auction allocation across classes, explicit cueing and handoff | Beats 3, 4 |
| coverage, collaboration, efficiency, tracking accuracy | Our reimplementation of all four with weights as parameters; the coordinator optimises them directly | Beats 2, 5, 7 |
| contested terrain | Terrain occlusion in the sensor model; sensor dropout and comms-degradation options in the sim | Beat 3 |

### Solana: Best Badge Hack — **not in this repo**

`DECISION.md` runs it as an explicitly decoupled side entry per
`hackathon/ideation/BADGE-HACK-BRIEF.md` (local-only — `hackathon/ideation/`
is gitignored, so this path resolves on the build machine and nowhere else).
It shares no code and no time with
this build. **No ticket in this backlog serves it.**

### Nothing else

Each extra sponsor claim is another live demo inside the same 09:45–11:45
window in which this team already owes Dominion a scored run. Adding a prize is
a human decision (§11).

---

## 4. Architecture

### The seam — structural and non-negotiable

The coordinator talks to the world through **one interface**. Poses and sensor
reports in; waypoint intents out. Nothing else crosses it.

```
                        ┌──────────────────────────────────┐
                        │          coordinator             │
                        │  belief · contacts · allocation  │
                        └───────────▲──────────┬───────────┘
                    WorldObservation│          │FleetIntent
                        ┌───────────┴──────────▼───────────┐
                        │          Transport (ABC)         │
                        └──┬──────────────┬──────────────┬─┘
                           │              │              │
                     ┌─────▼────┐   ┌─────▼────┐   ┌─────▼────┐
                     │kinematic │   │   sitl   │   │  arena   │
                     │dev+tuning│   │validation│   │ sponsor  │
                     └──────────┘   └──────────┘   └──────────┘
```

If the sponsor's interface turns out not to be plain MAVLink over a socket,
**only the adapter is lost** — never the estimator, never the policy, never a
single tuned parameter. This is the single most important structural decision
in the build and it is made now.

### Components

| component | package | what it owns |
|---|---|---|
| **Transport** | `whiteout/transport/` | The one interface and its three implementations. Adapters only: no belief, no policy, no scoring ever lives here. |
| **Sim** | `whiteout/sim/` | Terrain, vehicle kinematics per class, sensor footprints and detection models, the target mover, the episode clock. Headless, deterministic under a seed, hundreds of × real time. |
| **Scorer** | `whiteout/score/` | Our model of coverage, collaboration, efficiency, tracking accuracy. **Weights are parameters.** Consumes an episode log; pure function of it. |
| **Belief** | `whiteout/belief/` | The decaying occupancy field, the **negative-information** update, the particle/target estimator, the terrain flow network and its cuts. |
| **Policy** | `whiteout/policy/` | Allocation (auction), information-gain routing, choke-point posting, the contact lifecycle machine, re-tasking hysteresis, and the **frontier-coverage fallback**. Objective = the scorer. |
| **Tune** | `whiteout/tune/` | Batch runner, random search and CMA-ES over policy parameters, sweep artifacts, ablations, reports. |
| **Viz** | `viz/` | The run viewer. Static HTML + canvas over the episode log. Zero build step. |
| **CLI** | `whiteout/cli.py` | `run`, `replay`, `score`, `sweep`, `ablate`, `serve`. |

### The data model

Three record types, and everything else is derived from them.

```python
# whiteout/types.py  —  frozen dataclasses, JSON-serialisable

Pose        : asset_id, cls, t, x, y, z, heading, speed, energy_used
SensorReport: asset_id, t, footprint, detections[], negative: bool
WaypointIntent: asset_id, t, target_xy, target_z, speed, reason, task_id
```

- `WorldObservation` = `{t, poses[], reports[]}` — everything in.
- `FleetIntent` = `{t, intents[]}` — everything out.
- **`EpisodeLog`** (JSONL, one `{t, observation, intent, belief_digest,
  contacts[], truth}` record per tick) is the **only** artifact the scorer, the
  viewer, the tuner and the ablation harness read. It is the project's spine.
  Its schema is versioned and validated in the gate.

**Ground truth (`truth`) is written by the sim and read only by the scorer and
the viewer.** The coordinator never sees it. This is enforced by a test — it is
the single easiest way to accidentally cheat ourselves and believe a number.

### Where the hard part lives

Three places, in order of risk:

1. **`belief/negative.py`** — the non-detection likelihood given range, terrain
   occlusion and sensor class. Get the likelihood wrong and the field erodes
   confidently in the wrong places; it looks plausible for hours.
2. **`belief/flow.py`** — terrain as a flow network, and the min-cut that yields
   posting stations. Genuinely different from a raster, and genuinely the
   thing most likely to be over-engineered.
3. **`policy/objective.py`** — that the coordinator's objective is the scorer,
   evaluated on a rollout, not a hand-written proxy that drifts from it.

### External services

**There are none at runtime.** No network egress on any code path in
`whiteout/`; no API keys anywhere; no account, no card. The only external
dependency is **ArduPilot SITL**, and it sits behind the `Transport` interface
like everything else.

**SITL is not installed on the build machine, and it is not natively supported
on Windows 11.** It is acquired by **D46**, a loop action, as a **Docker
image** — Docker 29.4.1 is installed and working on this machine, and the image
needs no account, no key and no purchase. Two sources, in order: build
ArduPilot's own `docker/Dockerfile` from a clone of `ArduPilot/ardupilot`
(submodules included), or `https://github.com/radarku/sitl-swarm`, which our
own research recorded as a Docker-based multi-vehicle bring-up shortcut.
**Time cost: one hour, timeboxed**, most of it the clone and the waf build
inside the image. The image pull/clone is a one-time setup step outside the
product; it is not a runtime path and the "no network egress" rule above is
unaffected. The `kinematic` transport is its fake, is what CI runs, is
what the tuner runs, and is what the demo runs. See §7.

One data input: a **public Arctic DEM tile**, fetched before the event
(pre-event dataset gathering is permitted by the rules) and committed as a
downsampled fixture. `sim/terrain.py` also ships a deterministic synthetic
terrain generator, so **nothing blocks on the DEM arriving**.

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
| test | `pytest -q` |
| build | `python -m build --wheel --no-isolation` (proves the package is installable, and it is what the submission links). **`--no-isolation` is required**: the default fetches the build backend from PyPI on every invocation, which breaks §4's no-egress rule and spends venue wifi on every gate run. |
| smoke | `python -m whiteout.cli run --transport kinematic --seed 7 --ticks 400 --out artifacts/smoke.jsonl` then `python -m whiteout.cli score artifacts/smoke.jsonl --weights fixtures/weights/equal.json` — must exit 0 and print four finite scores |
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
- **finish under three minutes** on a laptop. If it does not, the slow test is
  moved to a `-m slow` mark excluded from the gate and run in the tuner.

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
`arena` is a stub until the booth answers (§11).

---

## 8. Milestones

| milestone | exit criterion |
|---|---|
| **M0 Skeleton** | Package scaffold, `pyproject.toml`, ruff/mypy/pytest configured with `.claude/**` excluded, **the CI workflow**, `scripts/gate.py`, the `types.py` record set, the episode-log schema and its validator, the `Transport` Protocol with a trivial kinematic stub, `.env.example`, and a smoke test that runs 400 ticks and scores them. The first M0 ticket also **deletes `hackathon/backlog-draft.md`**. |
| **M1 Demo path** | Every beat of §2 works end to end on the `kinematic` transport: sim, scorer, belief with negative information, flow cuts, contact lifecycle, hysteresis, auction allocation, the frontier fallback, and the viewer. Beat 6 is satisfied by a recorded SITL log if SITL is not yet up. |
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
   live run is the flourish. Cut the flourish, never the recording.
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
soon as the schedule is recovered:

1. **D24, the split-screen view** (also item 8 above). Two sequential replays
   tell the same story.
2. **Beat 7 collapses to a file.** `WeightsField` becomes "edit
   `fixtures/weights/*.json` and re-score" — the two-minute re-weight claim in
   M3 is met by the CLI rather than the UI.
3. **D13, the flow-network cuts, becomes overlay-only.** This is already
   §10's mitigation and its four-hour timebox is the first budget to refuse:
   if M1 is late when D13 comes up, the timebox is not spent at all and
   `belief/flow.py` ships as a render-only layer the policy ignores. Beat 3
   keeps the erosion; it loses the cut chords.
4. **D36's handoff arcs go static.** The `ContactRibbon` still shows the
   lifecycle, so beat 4's collaboration evidence survives; only the drawn arc
   is lost.
5. **Beat 4 narrows to the ribbon**, and D26's classification depth collapses
   to a confidence threshold (item 6 above, pulled forward).

Of the four differentiators, **negative information (D12) and the contact
lifecycle (D17) are never degraded** — the first is on the never-cut line
above, the second is the only visible evidence on a scored axis. The flow-cut
view degrades first, and re-tasking hysteresis degrades to a fixed constant
rather than a tuned parameter before either of those is touched.

---

## 10. Risks and spikes

| risk | severity | mitigation | owner ticket |
|---|---|---|---|
| **Information-gain routing loses to a lawnmower sweep.** Against a scripted, non-adversarial target, sending an asset to unobserved ground on a hypothesis and being wrong scores *worse* on coverage than systematic sweeping. This is the red team's standing warning and it is not hypothetical. | **high** | The **mode-agnostic frontier-coverage fallback is a first-class, selectable policy**, not a stub. The scorer tells us which wins under the booth's weights instead of us guessing. If frontier wins, we ship frontier and the tuning report is *still* the argument. | D14, D22 |
| **The negative-information likelihood is subtly wrong.** It looks plausible for hours. It is also the cheapest accuracy win, so it is load-bearing. | **high** | A property test: over many seeds, belief entropy must fall monotonically under non-detections, and the posterior must never assign <0 or >1. Plus a closed-form single-sensor case checked by hand. | D11, D12 |
| **The sponsor's interface is not MAVLink.** `dominiondynamics.online` does not resolve; the Devpost block is the only public text in existence; a search-engine summary calling it "an arena simulation SDK" appears in no sponsor text and is **not treated as sourced**. | **high** | The seam. Only the adapter is lost. Booth visit at hour one of the sponsor bay. | D6, D32 |
| **Eleven SITL instances plus a renderer do not fit on one laptop.** Our own analysis put the cut at six, and six was untested. | medium | The kinematic sim exists so this never blocks anything. SITL is validation; the *recorded log* is the deliverable. Spike: time four instances early, before committing to six. | **Spike S1**, D21 |
| **SITL cannot be obtained at all.** It is not installed, and Windows 11 has no native ArduPilot SITL. | medium | **D46** acquires it as a Docker image (Docker 29.4.1 is installed and working) in a one-hour timebox, as a loop action needing no account, key or purchase. On overrun D46 closes as "unobtainable", no M1 ticket is blocked, and the §3 MAVLink-only fallback for the ArduPilot clause applies, stated as such in the README. | **D46** |
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

**These do not hold the spec PR.** All five are resolved at the Dominion
Dynamics booth when the sponsor bay opens, and **the sim, the scorer and the
transport seam are invariant to all of them — that is the entire point of the
ordering.** Only two tickets genuinely cannot start without an answer.

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
choosing): **whether the Solana Best Badge Hack entry (§3, $2,500) is still
being run alongside and by whom, or is dropped** — `DECISION.md` targets it and
this backlog serves it with no ticket, so it is currently neither scheduled nor
dropped, and it still costs a submission and a slot in the same 09:45–11:45
window; adding any second sponsor prize, changing the stack after M0,
abandoning the scorer-first ordering, and anything that spends money, needs a
sign-up, or is irreversible and public.
