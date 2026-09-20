# Arena runbook

How to bring the arena up, fly a run against it, and tell what went wrong when
it does not work. `hackathon/ARENA.md` is what the arena **is**; this is what
you **do**, in order, with the commands that exist in this repository today.

Two rules for editing this file. Every command here has to be one that runs —
a runbook that lists a flag we never added is worse than no runbook, because it
is read at the one moment nobody has time to check it. And where the system
cannot yet do something, this file says so in the place you would have looked
for it, rather than leaving a gap that reads as "should have worked".

---

## 1. Addresses

The arena answers on the WireGuard tunnel's first address, and **which /24 that
is depends on the config pack**. The SIM-5 pack routes `10.99.4.0/24`, so the
host is `10.99.4.1`. An earlier probe recorded nothing because it asked
`10.99.0.1`, which belongs to a different pack — if everything times out, check
this line first.

| what | where | used by |
|---|---|---|
| MAVLink, quadcopter | `udp 14550` (sys 1) | `whiteout.transport.arena` |
| MAVLink, fixed-wing | `udp 14560` (sys 2) | ” |
| MAVLink, tower-1 | `udp 14580` (sys 4) | ” |
| MAVLink, tower-2 | `udp 14590` (sys 5) | ” |
| MAVLink, boat/rover | `udp 14570` (sys 3) | **not driven** — role idle |
| Camera, quadcopter | `http://HOST:8600/stream` | `whiteout.vision.sightings` |
| Camera, fixed-wing | `:8610` | ” |
| Camera, tower-1 / tower-2 | `:8630` / `:8640` | ” |
| Tracks API | `http://HOST:8010/api/tracks` | `whiteout.tracks.client` |
| Site API | `http://HOST:8090/api/site` | `scripts/arena_site.py` |
| gzweb (viewer, and pose truth) | `http://HOST:8080` | `scripts/truth_probe.py` |

Environment, in `.env.local` (gitignored — nothing here is a secret, but the
addresses change per pack):

```sh
WHITEOUT_TRANSPORT=arena
WHITEOUT_ARENA_ENDPOINT=10.99.4.1
WHITEOUT_TRACKS_ENDPOINT=http://10.99.4.1:8010
# WHITEOUT_ARENA_ROSTER=path/to/roster.json   # only to change ports or add assets
```

**Two of those are not in `.env.example`**, so copying that file and filling it
in leaves you without them: `WHITEOUT_TRACKS_ENDPOINT` and
`WHITEOUT_ARENA_ROSTER` were added after `SPEC.md` §7's table was written, and
`tests/test_gate_config.py` pins the template to that table. Adding them means
amending §7 as well, which is its own change; until then, set them by hand.

There is no `--transport` flag: `WHITEOUT_TRANSPORT` selects it, and the
default is the kinematic fake. A run that was meant to be live and quietly was
not is the failure this line exists to prevent — the run prints
`(transport=…)` on its last line, so read it.

---

## 2. Bring-up, in order

Each step tells you what a good answer looks like. Do not go on until you get
it; every later failure is harder to read than the one you skipped.

**1. The sim is up.** Either theirs on your machine
(`git clone …/arctic-sim && cp .env.example .env && docker compose up --build`,
8 cores and 16 GB minimum) or the cloud instance behind the WireGuard pack. Open
`http://HOST:8080` — you should see the strait. This is also where `Reset` is.

**2. The tunnel reaches it.**

```sh
python scripts/arena_site.py --endpoint 10.99.4.1
```

Prints the arena's own answers about the site it is rendering. If it hangs or
refuses, nothing below will work: check the pack's /24 (§1), then that the
container is up.

Add `--write` to record the answer into `whiteout/data/site.json`. Do that once
per arena, not per run: `whiteout/site.py` reads the record and never opens a
socket, so a stale record is a silently wrong convergence angle.

**3. The fleet answers.** The cheapest honest check is a short dry run — it
opens every link, waits for a heartbeat from each asset, and sends nothing:

```sh
WHITEOUT_TRANSPORT=arena WHITEOUT_ARENA_ENDPOINT=10.99.4.1 \
  python -m whiteout.cli run --ticks 5 --dry-run --out artifacts/hello.jsonl
```

Good: `run: wrote 5 records to artifacts/hello.jsonl (transport=arena)`.
Bad: `no heartbeat from … within 8s` — see §5.

**4. The tracks API accepts a fix.** It is the only artifact the judges read,
so prove it end to end before the run rather than during it:

```sh
WHITEOUT_TRACKS_ENDPOINT=http://10.99.4.1:8010 python - <<'PY'
from whiteout.tracks.client import TrackFix, TracksClient, endpoint_from_env
client = TracksClient(endpoint_from_env())
fix = TrackFix(name="Sierra One", lat_deg=71.9965, lon_deg=-94.8448)
print(client.post(fix))          # TrackAck(ok=True, created=True, …)
print(client.post(fix))          # created=False — the update path
PY
```

The first post answers `created=True`, a second under the same name answers
`created=False` — that pair is the whole contract, and a run that posts
`created=True` every time is creating a new track per fix instead of keeping
one. `GET http://HOST:8010/api/tracks` lists them.

**ARENA.md §5 says to confirm a test detection with DD staff once submitted.**
Do that here, not on Sunday.

---

## 3. Flying a run

```sh
WHITEOUT_TRANSPORT=arena WHITEOUT_ARENA_ENDPOINT=10.99.4.1 \
  python -m whiteout.cli run --ticks 400 --out artifacts/judged.jsonl
```

That connects to all four assets, ticks the coordinator, commands waypoints and
writes the episode log.

### What that command does not do yet

Read this before the judged run, because each line is a human action that
nothing in `whiteout run` performs:

- **It does not arm or launch anything.** `ArenaTransport.arm_and_launch`
  exists and pipelines the arm and the takeoff (the arm holds about three
  seconds — `ARENA.md` §3), but no CLI path calls it. Until one does, the
  aircraft have to be launched by hand, by MAVProxy or from a REPL holding the
  connected transport:

  ```python
  # shapes from tests/test_transport_arena.py; not exercised from a machine
  # without the arena
  transport.arm_and_launch("quadcopter", altitude_m=60.0)
  transport.arm_and_launch("fixed-wing")
  transport.scan("tower-1")     # a tower is never armed: arm_and_launch
  transport.scan("tower-2")     # refuses one, and `scan` is its fallback mode
  ```

- **It posts nothing to the tracks API.** `cmd_run` builds its coordinator with
  no `TrackPoster`, so the hold that decides what to post is never created. The
  client, the hold and the stub server are all tested and merged; the wire from
  the run to them is missing. **A judged run made with this command scores
  nothing**, whatever the log says.

- **It runs no vision.** No `SightingSource` is passed either, so the fleet
  searches and the belief ages, but nothing ever sees the vessel. The camera
  path (`VisionSightings`, `MotionGate`) is built and tested; it is not wired.

Those three are one wiring ticket, and it is the only thing between this
repository and a scored run. If you are reading this runbook to prepare the
judged run, that wiring is the work — not the arena.

### While it runs

- `python -m whiteout.cli serve`, then open the viewer and load the log for a
  read of what the fleet believed.
- The rail's contacts panel shows each contact's fix synchronisation, and a line
  counting fixes the camera path refused and why. An empty contacts panel with
  refusals counted means the cameras are working and the poses are not.

---

## 4. After the run

```sh
python -m whiteout.cli score artifacts/judged.jsonl --weights fixtures/weights/equal.json
python scripts/truth_probe.py --host 10.99.4.1 --seconds 60 \
  --compare http://10.99.4.1:8010 --track "Sierra One"
```

`score` prints five axes: coverage, detection speed, tracking duration, search
efficiency and accuracy. They are five of the seven the sponsor judges on
(`ARENA.md` §5) — autonomy and collaboration are not axes at all, because they
are read off the fleet's behaviour and the explanation rather than off a log.
Treat the numbers as our own instrument, not as a prediction of the result.

**On a judged log, expect three numbers and two lines of words**, and that is
not a regression:

- **`search_efficiency` prints `no energy recorded`.**
  `ArenaTransport._pose` reports `energy_used=0.0` on every pose on every tick
  (`whiteout/transport/arena.py:395`), so the episode carries no spend to
  weigh coverage against. It waits on the transport reporting real energy,
  not on the scorer.
- **`accuracy` prints `not measured (no truth in log)`.** Nothing populates
  `Truth.targets` during a run, so no log carries truth. `truth_probe` below
  is the live comparison, and it is deliberately not an input to the log.

The total is then weighted over the axes the log answered for, and says so in
its parenthetical.

A **kinematic** log answers for fewer still — two, `coverage` and
`search_efficiency` — because nothing gives that transport a sighting source,
so it carries no contacts and `detection_speed` and `tracking_duration` are
words too. The committed `fixtures/episodes/demo.jsonl` scores exactly that.
An arena log is the one that can answer for the detection axes, because it is
the one with cameras behind it.

`truth_probe` reads the arena's ground truth from gzweb and diffs it against
what we posted. It is a **measurement instrument, not an input**: nothing under
`whiteout/` imports it, and no detection, belief, policy or tracking path may
read it. If a comparison comes out about 1500 m off, the grid-convergence
rotation was skipped — that is the arena's EPSG:3413 grid north against true
north, −49.80° at this site, and the probe's docstring has the measurements.

---

## 5. When it breaks

| symptom | cause | do this |
|---|---|---|
| `no heartbeat from … within 8s` | arena down, wrong /24, or a roster port that does not match | check `http://HOST:8080` in a browser, then §1's pack note, then `WHITEOUT_ARENA_ROSTER` |
| Connect refuses with "endpoint is not set" | `WHITEOUT_ARENA_ENDPOINT` unset | set it; the adapter refuses at connect rather than guessing a host |
| Link opens, no telemetry ever arrives | the arena binds and waits — it is `udpout`, not `udpin`, and says nothing until a GCS says hello | our adapter sends the heartbeat already; if you are using another client, make it do the same |
| Vehicle sits disarmed after an arm | the arm holds ~3 s and a round trip missed the window | use `arm_and_launch`, which pipelines the two |
| Everything was fine and now nothing responds | the sim crashed — it does | press `Reset` in the web UI, **wait five minutes**, then ask DD staff |
| Tracks API answers `created=True` on every post | the `name` is not matching, so each fix makes a new track | one name for the whole run (`Sierra One`), set once |
| Run completes but the fleet never moved | `--dry-run`, or the transport was the kinematic fake | read the run's last line: it prints `(transport=…)` |
| Truth comparison off by ~1500 m | grid convergence not applied | §4 |

Nothing in this table is a reason to edit the arena's own `.env` beyond tower
lat/lon. DD's deck says it twice, and so does `ARENA.md` §3: **tower placement
is the one sanctioned lever**; the rest breaks the sim easily.

---

## 6. Still to ask DD staff

Four questions are open (`ARENA.md` §8). None blocks a run; each one changes
what we should spend the remaining hours on, so ask early and write the answer
into `ARENA.md` where the next reader will find it.

**1. The relative weights of the seven criteria.** *(Q3, never answered.)*
Changes which axis is worth the last hours: a heavy *coverage* weight favours
the sweep, a heavy *tracking duration* favours the hold and the coast window.
Our weights are parameters (`fixtures/weights/`), so this costs minutes, not a
rebuild. Answer: ______________________________________________

**2. How is *accuracy* measured** — distance error, time-weighted, or a
threshold? Changes whether a coasting track should keep posting its last fix or
go quiet, which is the rule `whiteout/tracks/maintain.py` is built around, and
it is what `scripts/truth_probe.py` should be measuring against.
Answer: ______________________________________________

**3. Are `heading` and `speed` scored, or optional?** The tracks API accepts
both. The hold can compute a course over a baseline; if they are scored it is
worth posting, and if they are not, posting a noisy course is a liability.
Answer: ______________________________________________

**4. What is *autonomy* judged on?** It is a named criterion that nothing in
the current design measures. If it is read off the behaviour, the answer is the
cueing chain running unattended; if it is read off the presentation, it is a
thing to say rather than a thing to build.
Answer: ______________________________________________
