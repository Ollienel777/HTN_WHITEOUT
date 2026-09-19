# Decision — Dominion Dynamics WHITEOUT

**Decided 2026-09-19T05:50Z**, hour ~1.8 of 32. Chosen candidate: **R3-11 Cold
Call**, re-scoped. This file is the authority for the build; the packet's
recommendation (R3-13 Any%) is **not** what we are building.

## Operator notes, verbatim

> lets just zero in on dominion. Start the repo and move onto the build stage
> for dominion. If the field will converge and its optimal that's fine, but
> lets try to think of some out of the box approaches that are actually better
> and zoom out for a sec.

## Why we are here, briefly

Three things moved between the packet and this decision.

1. **The Cloudflare prize does not fit Any%.** The track reads "build an agent
   that remembers context, uses tools, manages state, and **completes useful
   work for a user**," judged on agent capability, Workers usage, technical
   execution, creativity and **usefulness**. Any%'s agent is a build-time config
   generator; the project is not an agent project. The $50K that made Any% the
   pick is effectively off the table.
2. **The kit is not available.** No USB webcam, no printer, no markers. That
   kills Degrees of Freedom (printed ArUco fiducials and an overhead camera are
   both hard-required) and wounds Any%. Zero-kit options only.
3. **WHITEOUT needs nothing but a laptop**, has three cash slots plus a
   guaranteed first-round interview, and is the one track where the field
   converging on a known architecture is survivable, because it is **scored, not
   judged**.

## The project

A coordination layer for a heterogeneous ArduPilot/MAVLink fleet — fixed-wing,
quadcopters, rovers, fixed sensor towers — that detects, classifies and tracks a
moving target across terrain, scored live on **coverage, collaboration,
efficiency and tracking accuracy**.

## The strategic bet, and it is not an algorithm

The field will converge. We measured this before the run: an "Arctic fleet
coordinator assigning sectors by expected information gain, degrading when links
drop, with a live belief/coverage map" appeared in **3 of 3** independent
assistant sessions run cold on these tracks. Shared decaying belief grid +
market/auction allocation + particle filter + info-gain routing is the default,
and roughly half the field will ship it.

**We do not try to out-think that architecture. We out-measure it.**

The bet: on a live-scored track, the winner is whoever has the **fastest
evaluation loop**, not the cleverest policy. So the first thing we build is not
a coordinator — it is a **scoring harness and a simulator fast enough to run
thousands of scenarios**. Then the policy is tuned against the score rather than
argued about.

Four consequences, and they drive the backlog order:

1. **Build the kinematic fake first, from hour one.** Not SITL. A cheap
   headless simulator — vehicles as kinematic points with per-class envelopes
   (fixed-wing min turn radius and stall speed, quad hover-but-slow, rover
   terrain-limited, tower immobile), sensors as footprints with a detection
   probability, and the target as a scripted or reactive mover. It must run
   **hundreds of times faster than real time** and be deterministic under a
   seed. ArduPilot SITL is a *validation target*, not our dev loop.
2. **Reimplement the scorer.** Our own best model of coverage, collaboration,
   efficiency and tracking accuracy, with the four weights as **parameters, not
   constants**. When the booth tells us the real weights, we retune in minutes
   instead of rearchitecting. This is teaching to the test, and for a live-scored
   leaderboard that is the correct strategy, not a cheat.
3. **Make the coordinator's objective function literally the scorer.** Most
   teams will build "a good search system" and hope it scores well. Ours
   optimises the thing being measured.
4. **Run overnight parameter search.** With a 100×-real-time sim and a scorer,
   random search or CMA-ES over the policy's parameters gives us a coordinator
   tuned by thousands of runs. **No other team in a 32-hour hackathon will have
   that**, and it requires infrastructure rather than insight — which is exactly
   what a duo can build while everyone else debugs MAVLink.

## The transport seam — structural, decided now

The coordinator talks to the world through **one interface**: poses and sensor
reports in, waypoint intents out. Three implementations behind it:

| transport | purpose |
|---|---|
| `kinematic` | dev loop and tuning. Hundreds of × real time, deterministic, no MAVLink |
| `sitl` | validation. ArduPilot SITL via `sim_vehicle.py --count N --auto-sysid --mcast`, `pymavlink`, GUIDED mode + `SET_POSITION_TARGET_GLOBAL_INT` |
| `arena` | the sponsor's harness, whatever it turns out to be |

**This is the single most important structural decision in the build.** The
sponsor's interface is unknown (see below). If it is not plain MAVLink over a
socket, only the adapter is lost — never the estimator, never the policy, never
the tuning.

Known CPU ceiling from our own analysis: eleven SITL instances plus a renderer
does not fit on one laptop. Six was our cut, and six was still untested. The
kinematic fake exists so this never blocks us.

## Where we try to actually be better

Beyond the default stack, four angles worth building. Each is cheap and each
targets a scored axis the default under-serves.

1. **Negative information, done rigorously.** Most implementations update belief
   only on detections. A sensor footprint that swept and saw nothing is
   information: update the belief by the likelihood of a non-detection given
   range, terrain and occlusion. This is the single cheapest accuracy win
   available and it makes "coverage" quantitative rather than a heatmap of where
   we flew.
2. **Choke points, not rasters.** The default view is a probability grid. But
   towers are free, permanent, and immobile, and terrain constrains movement —
   so a target transiting from region A to region B must pass through gaps
   between tower footprints. Model the terrain as a **flow network and find the
   cuts**, then post mobile assets at the cuts rather than sweeping area.
   "Guard the gaps" beats "sweep the field" whenever the target must transit,
   and it is a genuinely different artifact from a belief raster.
3. **A contact lifecycle as an explicit cueing machine.** *Collaboration* is an
   explicit scored axis and it is the one most teams under-serve, because
   independent agents are easier to write. So make cueing a first-class object:
   a contact has a state (`unconfirmed → confirming → tracked → handed off`) and
   the policy's job is to move contacts along it — tower trips, fixed-wing
   diverts to sweep, quad confirms and holds, rover takes persistence. Score the
   chain explicitly rather than hoping collaboration emerges from an auction.
4. **Hysteresis on re-tasking.** *Efficiency* penalises brute force. Re-tasking
   every tick burns distance and battery for marginal information gain. Do not
   switch a vehicle's task unless expected gain exceeds a switching cost. A few
   lines, probably real points, and almost everyone will miss it.

**One warning we carry from the red team:** information-gain routing *is a bet*.
Sent to unobserved ground and right, it wins time-to-track; sent and wrong, it
has spent an asset on empty terrain and scores **worse than a lawnmower sweep on
coverage**. Against a scripted, non-adversarial target the clever stack can lose
to systematic sweeping. Hence the scorer-first design: we will be able to
*measure* which wins under their weights instead of guessing. The policy must be
able to run a **mode-agnostic fallback (plain frontier coverage)**, selectable,
so we are never scored on a bet we cannot justify.

## What is unknown, and how we resolve it

Our red team spent dedicated searches across three rounds. **The Devpost prize
block is the only public text in existence.** `dominiondynamics.online` does not
resolve; `defendthedominion.com` does not mention the track. A search-engine
summary rendering it as "an arena simulation SDK" appears in no sponsor text and
is not treated as sourced.

Standing `assumption`, twice re-checked: **scored in an arena Dominion supplies,
against a target Dominion drives.**

**Booth questions, hour one of the sponsor bay opening — these change the build:**

1. Do you provide the harness and simulator, or do we stand up ArduPilot SITL
   ourselves?
2. Is the target adversarial (evades) or scripted? Does "contested" mean an
   active adversary, comms denial, or terrain and weather?
3. **Relative weights on the four axes.** Highest-value single answer.
4. Is comms between our agents free, or is bandwidth modelled? If modelled, the
   shared belief map becomes a much harder and more interesting problem.
5. Continuous leaderboard scoring across the weekend, or one judged run Sunday?
   Decides whether we iterate against the scorer or polish one run.
6. What interface: plain MAVLink over a socket, or a hosted service with its own
   schema?

**Sponsor selection locks 2026-09-19T18:00Z**, hour 14 — before any scored run
is likely to exist. Select WHITEOUT regardless; it is the whole project.

## What we are giving up, stated plainly

- **This is not a finalist-shaped project.** It is software-only with no
  physical rig, won by optimising a score rather than performing a demo. All the
  demo-repeatability, fallback and stranger-testing work in the three shortlist
  briefs is worth nothing here.
- **There is no idea advantage available.** We are knowingly building the 3/3
  default and betting on execution infrastructure.
- **Main award is a secondary hope, not the plan.** If a demo falls out of the
  belief-map visualisation, good; we do not bend the build toward it.
- The one mechanism our run found that is genuinely off-default — a filter over
  *policies* rather than positions, inferring evader behaviour modes — **only
  pays against a human evader**, which the sponsor probably does not provide.
  Hold it as a stretch, not a foundation. If the answer to booth question 2 is
  "adversarial," it comes back on the table immediately.

## Prizes

- **Dominion Dynamics WHITEOUT** — $2,000 / $1,000 / $500, three winners, plus a
  guaranteed first-round interview. The whole project.
- **Solana: Best Badge Hack** — $2,500, explicitly decoupled ("does not need to
  relate to the main project"). Run alongside per
  `hackathon/ideation/BADGE-HACK-BRIEF.md`. Cheapest extra demo available: you
  hand someone a badge.
- **Nothing else.** Each extra claim is another live demo in the 09:45–11:45
  sponsor window.

## Human actions

- **None before building.** No key, no account, no card, no network, no borrowed
  hardware.
- Download one public Arctic DEM tile (pre-event dataset gathering is permitted;
  doing it now is fine).
- **Visit the Dominion Dynamics booth when the sponsor bay opens**, with the six
  questions above. This is the highest-value hour in the build.
