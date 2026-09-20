# Dogwatch

**Four machines keep watch over a two-kilometre strait, and pass the watch
between them until the dark ship is found and held.**

Built at Hack the North 2026 for **Dominion Dynamics WHITEOUT**: a quadcopter,
a fixed-wing and two fixed towers search Bellot Strait in the Northwest Passage
for an unlit vessel — no AIS, random spawn, a course nobody tells us — and hold
a track on it once found.

A *dogwatch* is a deliberately shortened watch at sea, rotated so no crew
stands the same hours twice. It is the handover that gives it the name, and the
handover is what this project is about.

---

## The idea, in one paragraph

Most of a search problem is not searching. A 25 km × 2 km channel is a corridor,
not an area, and a vessel confined to water in a corridor has nowhere to be that
is not on a line. So Dogwatch does not fly a lawnmower pattern over a box. It
keeps a probability field over the **water only**, erodes it wherever a camera
has actually looked, and sends each asset to whichever stretch of channel is
currently worth the most — which is rarely the stretch nearest the asset.

The part worth arguing about is the erosion. **Looking somewhere and seeing
nothing is a measurement**, and the field is owed an update for it. That single
change is the difference between a fleet that sweeps and a fleet that reasons:
before it existed the belief field was mathematically frozen — 400 ticks, one
distinct entropy value — while the aircraft flew exactly the same paths.

---

## Run it

No keys, no accounts, no network. Python 3.11+.

```bash
pip install -e ".[dev]"
```

```bash
python -m whiteout.cli run --seed 7 --ticks 400 --out artifacts/episode.jsonl
```

```bash
python -m whiteout.cli serve
```

`serve` prints a URL; open it and load `fixtures/episodes/demo.jsonl`, or the
episode you just wrote. The viewer is static HTML and a `<canvas>` — no build
step, no `node_modules`.

Everything runs against the `kinematic` transport by default, which never opens
a socket. `WHITEOUT_TRANSPORT=arena` points the same code at the real ArcticSim
fleet over MAVLink.

---

## Architecture

The coordinator reaches the world through **one interface**. Observations in,
intents out, and a `Track` leaving by a second path to the scored API.

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
              │  no deps │ │ harness  │   │ the real one │
              └──────────┘ └──────────┘   └──────────────┘
```

| package | what it owns |
|---|---|
| `whiteout/transport/` | The seam and its implementations. **Adapters only** — no belief, no policy, no scoring ever lives here, and a test walks the package to enforce it. |
| `whiteout/belief/` | The channel-shaped probability field, its diffusion, the detection update and the **non-detection** update. |
| `whiteout/policy/` | Search over the channel as a curve: which stretch is worth the most, and which asset should go. |
| `whiteout/vision/` | Camera models, pixel-to-lat/lon projection, the vessel detector, motion gating, stand-off geometry. |
| `whiteout/tracks/` | The tracks API client and the hold that decides what is worth posting. |
| `viz/` | The run viewer. Static HTML over the episode log. |

**The seam earned its keep.** It was written before we knew what the sponsor's
interface was. The arena turned out **not** to be a single MAVLink socket — it
is MAVLink per asset *plus* an HTTP tracks API — and the cost of that surprise
was one adapter. Nothing above it changed.

---

## The four ideas worth reading the code for

### 1. A non-detection is evidence, and the falloff is derived

`whiteout/belief/negative.py`

A camera that swept water and saw nothing must reduce belief there **in
proportion to how well it could have seen**, and never to zero. A cell is
looked at when the ray to it lands inside the frame — off-frame, the likelihood
is *exactly* 1.0, because a sweep says nothing whatever about water it did not
cover.

How much it is worth is not a tuned curve. Pixels on a target of length *L* at
slant range *r* from height *h* go as

> n = fx · fy · L² · h / r³

— one power from the angular width, two from the depression angle
foreshortening the along-range axis — and the detector's own statistic
saturates in that pixel count. **A sweep at twice the range says an eighth as
much**, because of geometry and not because we picked eight.

### 2. Certainty is refused at construction

The same file. A detector with a false-negative rate that was allowed to drive
belief to zero would empty the field of the water the vessel is actually in, on
one empty frame — and the fleet would never look there again. `SweepParams`
**raises** on `p_max = 1.0` rather than clamping it. Twenty empty looks are what
empty a cell, and twenty multiplications is how.

### 3. Hovering over a contact does not see it

`whiteout/vision/standoff.py`

The quadcopter's gimbal is not commandable — `MAV_CMD_DO_MOUNT_CONTROL` returns
`MAV_RESULT_FAILED` on the arena — so its camera looks wherever the airframe
does, roughly at the horizon. The obvious way to hold a contact is to hover over
it, which puts the contact **straight down**, which is exactly where that camera
cannot see. Flown live, the quadcopter ended up on top of the vessel and saw
nothing. It now holds from a stand-off computed from the real field of view.

### 4. One module converts between frames, and a test proves it

`whiteout/geo.py`, enforced by `tests/test_geo.py`

Every position in the log is geodetic. A second converter written in a hurry at
hour 20 — `METRES_PER_DEGREE = 111320.0` — disagrees with the real ellipsoid by
30 m at the arena's corner, and posts a *plausible* wrong lat/lon to a scored
API. So an AST walk over the whole repository fails any literal in the range an
earth radius could occupy, any name containing a conversion word, and any call
to trigonometry outside the one module allowed it.

It caught two real mistakes during the build, both in code that looked fine.

---

## Checking the claims

Every number above is reproducible from this checkout.

| claim | check |
|---|---|
| It builds, lints, typechecks, tests and runs | `python scripts/gate.py` |
| The test suite | `python -m pytest -q -m "not slow"` |
| The same seed gives a byte-identical episode | run twice, `cmp` the outputs |
| The belief field is worked, not decorative | `fixtures/episodes/README.md` carries the measured entropy curve and the command that regenerates it |
| The transport package contains no belief or policy | `python -m pytest tests/test_transport.py -k no_leak` |
| One module converts frames | `python -m pytest tests/test_geo.py` |

---

## What is honest about the state of this

The parts below are written down because a reader will find them anyway, and
finding them unannounced is worse.

**The scorer is a stub.** `whiteout/cli.py`'s `score` prints four zeroes and
does not read the log. Our own scoring model was cut when the sponsor published
seven criteria of their own; the numbers that matter are theirs, computed from
what we post to their API.

**The detector has never been measured on compressed imagery.** Its noise
estimator is calibrated on uncompressed frames, and the arena publishes MJPEG.
`scripts/jpeg_quantisation.py` measures the gap and it is not small. The failure
direction is the bad one — toward confident false positives — and it is tracked
rather than fixed.

**Terrain occlusion is not modelled** in the non-detection update. There is no
terrain model of ours, so a ridge between a camera and the water reads as swept.
This is the one place the belief update errs *against* safety; the mitigation is
that a single sweep can never claim certainty, so an occluded cell recovers the
moment anything else looks at it.

**We cannot measure our own accuracy.** The arena publishes no ground truth.
Nothing ever tells us whether a posted lat/lon was right, which is why this
codebase treats a confident wrong detection as worse than silence, and why
several modules refuse rather than guess.

---

## Repository

| path | what |
|---|---|
| `hackathon/ARENA.md` | Ground truth for the sponsor's simulator: the fleet, the cameras, the tracks API |
| `hackathon/SPEC.md` | The build plan, and what changed when the arena turned out to be real |
| `hackathon/PRESENTATION.md` | The five-minute pitch |
| `hackathon/DEMO.md` | The run-through, for a live demo or a capture |
| `docs/` | The harness this was built with: an ideation loop and a reviewed-PR build loop |
| `fixtures/episodes/` | Committed real runs, regenerable by one documented command |
