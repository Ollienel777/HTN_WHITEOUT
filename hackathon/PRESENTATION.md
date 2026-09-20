# The five-minute presentation

Issue #69. Dominion Dynamics judge this alongside the scored run, and they
named what they weigh: **how you thought about the problem**, **how you decided
to implement your solution**, **autonomy rated higher than manual control**,
and **collaboration meaning how the robots team up with each other**. Two of
the seven criteria — autonomy and collaboration — are read off the behaviour
and the explanation, not off a scoreboard.

`DECISION.md` says this track is "scored, not judged — won by a number, not a
performance". That was wrong, and this document is the correction.

**Rehearse against a clock.** A deck written and unread is worth nothing here;
the timings below are targets, and the only way to know they hold is to say it
out loud.

---

## The through-line

> We did not build a better search pattern. We worked out what the target
> actually does, and then most of the search stopped being necessary.

Everything below serves that sentence. If a beat does not, cut it.

---

## Beat 1 — the task, in our words (0:00–0:30)

One shadow vessel. No AIS, random spawn, a course we do not know, 3 m/s,
somewhere in 6.5 km of Bellot Strait. Four assets: a quadcopter, a fixed-wing,
and two towers that cannot move. Find it, classify it, hold it, and keep
reporting it.

Show the gzweb scene with the four assets and no vessel visible. **Do not
explain the architecture yet.**

---

## Beat 2 — how we thought about it (0:30–2:00) · **the beat that matters**

This is the one Dominion said they weigh most, and it is the only part of the
run nobody else will have.

**The naive read is that the vessel moves randomly.** Your own deck says
"random movement path", and a belief field that diffuses outward from the last
sighting is the obvious response. That is what we built first, and it was
wrong twice over.

**Your simulator is open source, so we read it.** `terrain/course.py` and
`sim/plugins/VesselPathPlugin.cc`:

1. The course is a **widest-path Dijkstra** between two far-apart navigable
   points, with step cost `1 + 3×(1 − clearance/max)`. One times mid-channel,
   four times hugging the bank. **It provably hugs the channel centreline.**
2. Every waypoint clears the shore by at least 120 m, repaired until it does.
   **The vessel is confined to water we can compute ourselves from the same
   DEM.**
3. The plugin moves it at **constant 3.0 m/s** along that polyline, and
   `<loop>false</loop>` makes it **reverse at each end and retrace the same
   line.**

So the target's support is not a two-dimensional field. **It is a
one-dimensional curve, travelled at a known speed, periodically.** That is a
far smaller estimation problem than the one we started on.

**What it changed in our code, concretely:** our belief field was diffusing at
8.0 m/s, a figure we had picked from loss asymmetry because we thought we had
no measurement. Variance goes as `v²`, so we were spreading belief over **7.1
times too much water per unit time** — σ of 339 m after a minute instead of
127 m. PR #91 is that one-line change and its arithmetic.

**Say the uncomfortable part out loud:** reading the simulator's source is
reading the answer sheet, and it is fair game only because you published it. If
the scored run uses `pond_inlet` or `resolute` instead — and `competition.lock`
pins all three — none of this is hardcoded, because all of it is derived from
the heightmap at runtime.

---

## Beat 3 — how we decided to implement it (2:00–3:00)

**One seam, decided before anything was built.** The coordinator talks to the
world through a single interface: poses in, waypoint intents out. Three
implementations behind it — a kinematic fake for development, and the arena.
We made that call when we did not yet know whether your interface was MAVLink
or something else, and it meant the answer cost us an adapter rather than a
rewrite.

**The tracks API is the only thing that scores, so it was built first**, with a
stub reproducing `created: true` then `created: false` so everything above it
could be tested without the arena.

Then show the adapter working: connect, four assets, live poses at their known
positions. **`udpout`, not `udpin`** — the arena sends nothing until a ground
station says hello, and that one detail is the difference between "the fleet is
down" and a working link.

---

## Beat 4 — collaboration, shown rather than described (3:00–3:50)

**This must be visible on screen. A described handoff scores nothing.**

The cueing chain, in order:

1. A **tower** trips first. They are free, permanent, never run out of
   endurance, and on a 2 km-wide channel they are the natural tripwire. Their
   pan and tilt are servo 1 and servo 2 over the same MAVLink link as
   everything else.
2. The **fixed-wing** diverts to confirm. It covers ground fastest and 25 km is
   a long way.
3. The **quadcopter** takes the hold. It is the only asset that can stop and
   stare.

**And the track never notices.** The tracks API keys on the name, so a handoff
is just the next fix under the same name — no gap, no re-created track, no
protocol to get wrong. Show `holder` changing from `tower-1` to `fixed-wing`
to `quadcopter` while one track's `fixes` count climbs without interruption.

### The tower numbers, and why they belong in beat 2

If beat 4 runs short, **move this into beat 2** — it is a "how we thought about
the problem" beat, not a collaboration one, and it is the only decision in the
run that was ours to make.

Tower lat/lon is the one `.env` edit Dominion sanctioned: the aircraft start
near-optimal and should be left alone. Stock placement puts the two towers
1,656 m and 1,716 m off a channel whose half-width is about 690 m, so they
cover almost nothing.

| | tower-1 | tower-2 | 
|---|---|---|
| **as shipped** | 0.0% | 1.2% |
| **re-sited** | **27.8%** | **19.4%** |

Two free, permanent sensors went from contributing nothing to covering a
quarter of the channel each, for two lines of configuration.

**The part worth saying out loud:** our own siting tool had no terrain model.
It scored every candidate at the height the towers happen to stand at now — so
its own recommendation put a tower at **0.0 m**, the waterline, where the
detector refuses the frame outright because the camera is not above the water
plane. We caught it by reading ground elevation out of the arena's heightmap
and re-scoring. The tool's warning about this was right there in its output and
both published recommendations walked straight into it.

That is a better story than the coverage number: a measured decision, and a
measurement that caught its own tool.

---

## Beat 5 — what is autonomous, and what is not (3:50–4:20)

**Overclaiming here to a panel that can watch the run is the worst available
move.** State it flatly, in this order:

- Autonomous: the link, the fix-to-position projection, the track hold, the
  decision of when a track is held, coasting or lost, and the posting.
- **Not autonomous, as of this run:** _(fill in honestly at rehearsal — the
  search policy and the initial tasking are the likely entries)_.
- Started by one command, and then not touched.

If a human flew any part of it, say which part, unprompted. They will see it
anyway, and being the one to point at it is worth more than the beat costs.

---

## Beat 6 — what did not work (4:20–5:00)

A named criterion is "how you thought about the problem", and a team with no
failures to report has not looked hard.

**The candidate, and it is a good one:** we ordered the entire build around a
strategy of out-measuring the field — reimplement the scorer, run thousands of
episodes overnight, tune the policy against the score. Two things killed it.
Your arena runs at `SPEEDUP=1`, so it cannot be swept. And the scoring is seven
criteria, not the four on the Devpost block, of which **autonomy and
collaboration are not numbers at all**. We had built a plan to optimise a
scoreboard that does not exist in the shape we assumed.

What we kept from it: the seam, and the habit of writing down why a number is
what it is. What we dropped: about a day of planned tuning infrastructure.

_(Second candidate if there is time: the belief speed. We had a careful,
written, wrong argument for 8.0 m/s and it survived review because the
reasoning was sound — it just rested on a premise we had not checked.)_

---

## Rehearsal checklist

- [ ] Run it against a clock. Five minutes, not six.
- [ ] Beat 2 lands in 90 seconds. It is the one to protect if time runs short.
- [ ] Beat 4 shows a handoff on screen, not a description of one.
- [ ] Beat 5's "not autonomous" list is filled in honestly and read aloud.
- [ ] Every number said out loud is one we can show the source of.
- [ ] Nobody says "shared intelligence", "swarm" or "synergy".

## What is still missing from this draft

- **Beat 5's list.** It cannot be written until the run is fixed, and writing
  it optimistically now is how a team ends up overclaiming on stage.
- **The screen recording for beat 4.** A live handoff is better; a recording is
  the fallback, and having none is not an option.
- **Who says what**, if it is delivered by two people.
