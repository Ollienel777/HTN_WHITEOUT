# ArcticSim — the real arena

**Source:** Dominion Dynamics' `ArcticSim` deck, presented at the API workshop
2026-09-19T14:30Z, plus answers given verbally at that workshop. This file is
**ground truth** and supersedes every `assumption` in `SPEC.md` and
`DECISION.md` about the WHITEOUT track.

`SPEC.md` was written when **no public source about this track existed** — the
red team verified that three times. Where the two disagree, this file wins,
and `SPEC.md` needs re-planning against it.

---

## 1. The workshop answers

| # | question | answer |
|---|---|---|
| **Q1** | Do they provide the harness and simulator? | **Yes.** `arctic-sim`, Docker, run locally or on a cloud instance they provide. |
| **Q2** | Is the target adversarial? | **No.** It is a boat that **spawns at a random position and follows a random movement path**. No evasion. |
| **Q4** | Is inter-agent comms modelled? | **No bandwidth model.** Comms are free — but **you build the mechanism that communicates between assets yourself**; nothing is shared for you. |
| **Q5** | Continuous leaderboard or one judged run? | **Judged Sunday.** One run. No iterating against their scorer. |
| **Q3** | Relative weights on the scoring axes | **Still unknown**, and the axis list itself is different from what `SPEC.md` assumed (below). |
| **Q6** | Interface | **Answered by the deck**: MAVLink per asset, plus an HTTP tracks API for submitting detections. |

## 2. What is actually being simulated

- **Place:** Bellot Strait, near Fort Ross, Northwest Passage.
- **Extent: 25 km × 2 km.** A narrow water channel with ice floes, flanked by
  rocky shores and ridges.
- **Target:** a *shadow vessel* — a boat, **no AIS**, random spawn, random
  movement path. The goal is to **detect and track its movement**.
- A real render of the strait, not a generated heightfield.

**The strait geometry is the single most important fact here.** The search
space is effectively one-dimensional: 25 km long, 2 km wide, and the target is
confined to water. This makes the choke-point idea in `DECISION.md` stronger,
not weaker — but it also means a general 2-D belief grid over open terrain is
the wrong shape.

## 3. The fleet — four assets, not eleven

| asset | ArduPilot vehicle | control |
|---|---|---|
| `quadcopter` | **ArduPilot Copter** | `mode guided` · `arm throttle` · `takeoff <alt>` |
| `fixed-wing` | **ArduPilot Plane** | `mode guided` · `arm throttle` · `mode takeoff` (circles) |
| `tower-1` | **ArduPilot AntennaTracker** | `servo set 1 <pwm>` pan/yaw · `servo set 2 <pwm>` tilt/pitch · `mode scan` |
| `tower-2` | **ArduPilot AntennaTracker** | as above |

Both aircraft fly to a point with `guided <lat> <lon> <alt>`.

**Relative capability:** the fixed-wing is **slightly faster than the
quadcopter and has slightly better sensors**.

**Starting positions are already close to optimal and should mostly be left
alone.** The quadcopter starts over the highest point of the map; the
fixed-wing is placed where it has the flat run it needs to take off. Moving
either is more likely to hurt than help.

**The towers are the exception, and they are a real strategic lever.** Their
lat/lon may be changed deliberately — that is the one `.env` edit DD
encouraged. Everything else in `.env` is constraint-laden and breaks the sim
easily.

**There is exactly one moving target.** No decoys, no second vessel.

Coordinates can be read straight off the sim: click anywhere to get x, y, z
and lat/lon. Each asset has a FOV view and a copyable entry point.

**The sim crashes.** Press `Reset` to restart it and all assets, and give it
**five minutes** before asking DD for help.

**There are no rovers.** `SPEC.md`'s heterogeneous fleet of fixed-wing,
quadcopters, rovers and towers is wrong: it is **two towers, one quad, one
plane**.

**`arm throttle` only holds for 3 seconds** — the takeoff command has to follow
immediately.

Asset placement is set in `.env` as `ASSET_N=role,name,lat,lon`; save and
rebuild restarts all containers. The deck warns explicitly: **do not play with
`.env` too much, you can break the sim.**

## 4. Detection is a computer-vision problem

**This is the largest gap between `SPEC.md` and reality.** Every asset carries
a camera, and finding the boat means finding it **in a video frame**:

| asset | HFOV | VFOV | resolution |
|---|---|---|---|
| quadcopter | 114.6° | 99.4° | 640×480 |
| fixed-wing | 69.0° | 42.6° | 640×360 |
| towers | 60.0° | 36.1° | 640×360 |

`SPEC.md` models sensing as abstract footprints with a detection probability.
The real task is: point a camera, read frames, decide whether a small dark
vessel is present among ice floes and open water, and convert a pixel
detection into a lat/lon.

The known FOVs are a gift for the **negative-information** idea — a camera
pose plus a known FOV gives an exact "nothing was here" cone, which is far
better evidence than an assumed footprint.

## 5. How you are scored, and how you submit

**Seven criteria, not four:** *search efficiency, coverage, detection speed,
tracking duration, accuracy, autonomy & collaboration.*

### Half the judging is qualitative, and `DECISION.md` gets this wrong

There is a **five-minute presentation**, and DD named what they weigh in it:

- **how you thought about the problem**
- **how you decided to implement your solution**
- **autonomy is rated higher than manual control** — a fleet that searches and
  tracks by itself beats one a human flies, even to the same result
- **collaboration means how the robots team up with each other**, not how the
  humans did

`DECISION.md` says in several places that this track is *"scored, not judged —
won by a number, not a performance"*, and treats demo-repeatability,
fallbacks and rehearsal as worth nothing here. **That framing is wrong.** A
live number is part of it, but a presentation that explains the reasoning is
the other part, and two of the seven criteria — autonomy and collaboration —
are things a judge reads off the *behaviour* and the *explanation*, not off a
scoreboard.

Practical consequence: a coordinator that visibly hands off between tower,
quad and fixed-wing, and can be *explained* in five minutes, scores on axes
that a marginally better search pattern does not.

`SPEC.md` §6 and the whole scorer design assume **four** axes — coverage,
collaboration, efficiency, tracking accuracy. That is wrong. Detection speed,
tracking duration and **autonomy** are all separately named, and autonomy in
particular is not something the current design measures at all.

**Submission is an HTTP API, not a file:**

```
POST http://<SIM-IP>:8010/api/tracks
  {"name":"Sierra One","lat":71.9965,"lon":-94.8448}

POST http://<SIM-IP>:8010/api/tracks          # update: name must match
  {"name":"Sierra One","lat":71.9975,"lon":-94.8450,"heading":315,"speed":6.5}

GET  http://10.99.0.1:8010/api/tracks         # list
```

A create returns `{"ok":true,"created":true,"uuid":"entity-…",…}`; an update
returns `created:false`. **Confirm with DD staff once a test detection has
been submitted.**

## 6. Getting it running

```
git clone https://github.com/Dominion-Dynamics/arctic-sim
cd arctic-sim
cp .env.example .env
docker compose up --build
```

**System requirements: 8 cores (12 recommended), 16 GB RAM (24 recommended),
50 GB disk.** A free `MAPBOX_TOKEN` gives hi-res terrain; without it the
terrain images are low-res.

**If the machine does not meet those requirements**, Dominion Dynamics will
supply a **WireGuard config to a dedicated cloud instance** — they need the
team roster (max 10) to issue it.

**This is the first human action to resolve.** The sim is a heavyweight Docker
stack, not a library, and nothing else can be validated until it runs
somewhere.

## 7. What this does to the plan

**Dead or badly reduced — they supply all of it:**

- the terrain generator, vehicle kinematics, sensor model, target movers and
  episode loop (`whiteout/sim/`, tickets #7, #8, #10, #11, #25, #24)
- the `sitl` transport and its Docker acquisition (#45, #46, #47, #48) — the
  sim *is* ArduPilot SITL, already containerised
- a four-axis scorer (#26) as specified

**The strategic bet is damaged and needs re-deciding.** `DECISION.md` bets on
reimplementing the scorer and tuning a policy over **thousands of fast
simulated episodes**. Against this arena that is not available: the run is
judged once on Sunday, there is no leaderboard to calibrate against, the
scoring axes are seven and unweighted, and their sim is a heavyweight Docker
stack that will not run at 100× real time. A fast local surrogate is still
useful for policy development, but "tune against a faithful model of their
scorer" has lost its ground truth.

**Vindicated, and now central:**

- **The transport seam.** `arena` is real and its interface is known: MAVLink
  per asset plus the tracks API. This is exactly what the seam was for.
- **Choke points.** A 25 × 2 km strait is a choke point. Two fixed towers with
  60° FOV covering the narrows is a strong, cheap search posture.
- **Negative information**, sharpened by exact camera FOVs.

**Missing from the backlog entirely, and now on the critical path:**

1. **Vision** — detect a small vessel in 640×480/640×360 frames among ice
   floes, and project a pixel to a lat/lon.
2. **The tracks API client** — the only thing the judges actually read.
3. **Inter-asset communication** — free, but ours to build.
4. **Tower control** — pan/tilt by raw servo PWM, and `mode scan`. Plus
   **tower placement**, which is the one sanctioned `.env` lever and a genuine
   strategic decision on a 25 × 2 km strait.
5. **Getting the sim running**, including the cloud-instance path.
6. **A track-maintenance loop** — once the boat is found, keep posting its
   lat/lon every few seconds. *Tracking duration* and *accuracy* are two of
   the seven criteria, and both are about what happens **after** detection.
7. **The five-minute presentation**, and the autonomy story it has to carry.

## 7a. The shape this suggests

Nothing here is decided, but the arena points somewhere specific:

- **Two towers, placed deliberately, watching the narrows** — free, permanent,
  never run out, and a strait is exactly where a tripwire works.
- **The fixed-wing sweeping the long axis** — it is the faster asset with the
  better sensor and a 69° FOV, and 25 km is a long way.
- **The quadcopter as the confirm-and-hold asset** — 114.6° FOV, hovers,
  slower; the classic cueing chain ends with it.
- **Then the track-maintenance loop** feeding the tracks API.

That chain is also the *collaboration* story the presentation needs, which is
one of the seven criteria and cannot be won by a search pattern alone.

## 8. Still unknown

- **The relative weights of the seven criteria** (Q3). Ask DD staff.
- How *accuracy* is measured — distance error, time-weighted, or a threshold.
- Whether `heading` and `speed` are scored or optional.
- What "autonomy" is judged on. It is named as a criterion and nothing in the
  current plan addresses it.
