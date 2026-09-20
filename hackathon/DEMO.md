# The run-through

Issue #42. One script for two jobs: the **live demo** a judge watches, and the
**capture** that becomes the video. They are the same beats — the recording is
the fallback for the live one, and rehearsing it is how the live one stops
being improvised.

`PRESENTATION.md` is what you *say*. This is what you *do*, keystroke by
keystroke, and what to do when a step fails.

**Budget: three minutes of capture, five minutes live.** The extra two minutes
live are beats 5 and 6, which are spoken over a still screen and are not worth
recording.

---

## Before you start

Run this. All of it. It takes about ninety seconds and it is the difference
between a demo and an apology.

```bash
python scripts/gate.py
```

Then these four, in order. Each one is a thing that has failed before.

| # | command | what you are checking |
|---|---|---|
| 1 | `python -m whiteout.cli run --seed 7 --ticks 400 --out artifacts/demo.jsonl` | It writes 400 records and says so |
| 2 | `python -c "import json;rows=[json.loads(l) for l in open('artifacts/demo.jsonl')];print(len({r['belief_digest']['entropy'] for r in rows}),'distinct entropy values')"` | **Not 1.** One means the belief field is frozen and the demo is hollow |
| 3 | `python -m whiteout.cli serve` | It prints a URL. Open it |
| 4 | Load `fixtures/episodes/demo.jsonl` in the viewer | It renders, and the fleet rows read `quadcopter`, `fixed-wing`, `tower-1`, `tower-2` |

**Close every other window.** Terminal font at least 16pt. Browser at 1280×720
if you are capturing — not fullscreen on a 4K display, where nothing is legible
at video bitrate.

### What beat 2 needs before it can be performed

**One thing: #20.** Beat 2 *is* the demo, so this is worth being exact about.

The data is there. #124 landed as #125, and every record now carries a
`belief_field` — 850 quantised cells with the geometry on the first record.
Check it:

```bash
python -c "import json;print(sorted(json.loads(open('fixtures/episodes/demo.jsonl').readline())['belief_field']))"
```

What is missing is the drawing. `drawField()` in `viz/viewer.js` renders a grid
and a scale bar; `belief_field` appears in that file exactly once, in a list of
record keys, and nothing reads it. So preflight check 4 gives you a correct
instrument with an empty middle.

Until #20 lands, rehearse beats 1, 3, 4, 5 and 6 and leave a hole where beat 2
goes. Do not rehearse a beat 2 that narrates a static screen — it reads worse
than admitting the gap.

---

## The beats

### Beat 1 — the strait (0:00–0:20)

**Screen:** the viewer, episode loaded, paused at tick 0.

**Do:** nothing. Let it sit.

**Say:** one shadow vessel, no AIS, random spawn. Twenty-five kilometres of
channel, two across. Four machines, and two of them cannot move.

> The point of holding still here is that the next twenty seconds are the only
> ones where the screen is simple. Do not narrate the UI.

---

### Beat 2 — the field, eroding (0:20–1:20) · **the beat that matters**

**Screen:** press play. Let it run the full episode at replay speed.

**Do:** say nothing for the first five seconds. Let a judge notice the field
changing before anyone explains it.

**Say, while it runs:**

- Every asset that looks and sees nothing erodes the water it could have seen.
- In proportion to how well it could have seen it — a sweep at twice the range
  says an eighth as much, and that eighth is geometry, not a tuning constant.
- Never to zero. A detector that misses once must not be able to empty the
  water the vessel is in.

**Then point at the number.** Take it from `fixtures/episodes/README.md`,
which carries the current measured curve — do not read a figure off this page,
because this page will go stale and that table will not. At the time of
writing it is entropy 6.74 -> 5.93 over 200 seconds, across 400 distinct
values.

**Say:** before this existed that was one value, four hundred times. The
aircraft flew the same paths. The search was decoration.

> This is the beat to protect if you run short. Everything else is context for
> it.

---

### Beat 3 — the handover (1:20–2:10)

**Screen:** scrub to a stretch where the fixed-wing and a tower cover different
thirds of the channel.

**Say:** the towers are free and permanent and cannot move, so they are aimed
rather than sent. The fixed-wing is the only thing fast enough for the middle
third, which neither tower reaches. The quadcopter is the only one that can stop
and stare, so it is what holds a contact once there is one.

**The line worth saying out loud:** hovering over a contact does not see it. The
gimbal is not commandable, the camera looks at the horizon, and directly below
is outside the frame. We found that by flying it.

---

### Beat 4 — it runs against the real arena (2:10–2:40)

**Screen:** terminal.

```bash
WHITEOUT_TRANSPORT=arena python -m whiteout.cli run --ticks 60 --out artifacts/arena.jsonl
```

**Say:** same coordinator, same belief, same policy. One adapter changed. The
interface turned out not to be what we guessed — MAVLink per asset plus an HTTP
tracks API, not one socket — and that cost us an adapter and nothing else.

**If WireGuard is down or the arena is unreachable:** do not debug it on
camera. Say "this is recorded because the tunnel is not something I want to
demo live", and cut to the capture from the rehearsal. Have that capture.

---

### Beat 5 — what is autonomous (2:40–3:00 capture, longer live)

**Screen:** still. No scrubbing while you talk.

**Say plainly, and do not soften it.** What runs unattended: the search, the
tasking, the belief update, the track hold and what gets posted. What does not:
siting the towers, and the altitude datum, which we measured by hand because
guessing it scales every range we report.

> Judges weigh autonomy above manual control. The way to win that is to be
> exact about the boundary, not to claim the boundary is further out than it is.

---

### Beat 6 — what did not work (live only, 3:00–4:00)

Three, honestly, and each with what it cost:

1. **The belief field was frozen for a day and a half.** Every part above the
   seam was merged and tested; nothing called any of them in order. The tests
   all passed the whole time.
2. **The fake transport reported heading in radians** where the arena reports
   degrees. Every camera was pointed fifty-seven times too little off North.
3. **The detector has never been measured on compressed imagery**, and the
   arena publishes JPEG. We measured the gap rather than assuming it away, and
   it is not fixed.

> Do not end on this. End on the sentence in the next section.

---

### The close

> We did not build a better search pattern. We worked out what a
> non-detection is worth, and then most of the searching stopped being
> necessary.

---

## When it goes wrong

| symptom | do this |
|---|---|
| Viewer shows the error state | Load `fixtures/episodes/demo.jsonl` — the committed one always parses. Do not debug live |
| Belief field looks flat | You are on a stale episode. Regenerate: `fixtures/episodes/README.md` has the command |
| `serve` refuses a port | `PORT` is set and something is on it. `unset PORT` and rerun — it takes an ephemeral port |
| Arena unreachable | Beat 4 goes to the recording. Say why in one sentence and move on |
| You are over time | Cut beat 4. Never cut beat 2 |

---

## Capture settings

- 1280×720, 30 fps. Higher resolution makes the text smaller, not better.
- Audible narration. A silent screen recording of a probability field is not a
  demo of anything.
- **Watch beat 2 back at video bitrate before you ship it.** A field eroding is
  low-contrast motion and it is the first thing compression destroys. If it is
  not legible, raise the bitrate or slow the replay — do not re-record it
  louder.
- Under three minutes. Trim the pre-roll where you click into the terminal.

---

## Rehearse it

Once, against a clock, out loud, start to finish, before you record. Every
failure in the table above was found that way and not by reading.
