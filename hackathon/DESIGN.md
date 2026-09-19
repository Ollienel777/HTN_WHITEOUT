# DESIGN — WHITEOUT run viewer

The only UI in this project is the **run viewer**: a static page that replays an
episode log. It is specified in `SPEC.md` §4 as a **debugging tool first**. It
earns its polish because the team stares at it for twenty hours, and because it
is the one artifact a sponsor engineer or a judge can read in ten seconds.

This file is the direction every implementer builds against and every reviewer
checks against. It is decisive on purpose. Where it offers no option, do not
invent one.

---

## Tone and references

- **Foreflight and a modern EFB / mission console** — for the fact that dense
  operational data can be calm. Borrowed: dark ground, luminous data, thin
  strokes, no chrome competing with the map.
- **Linear** — borrowed: density without clutter, one accent used sparingly,
  keyboard-first, text that is small and confident rather than large and shouty.
- **Observable / D3 notebook plots** — borrowed: axis and legend restraint.
  Labels are small, grey, and right next to the thing they label.

Explicitly **not** borrowed: the military-console aesthetic. No scan lines, no
crosshair reticles, no all-caps green monospace everywhere, no fake radar
sweep. This is a scientific instrument, not a prop.

## Primary theme: dark. Only dark.

The centre of the screen is a **probability field over terrain**. A luminous
field on a dark ground has an order of magnitude more usable contrast than a
dark field on white, and the signature moment (§Signature) is *probability mass
being erased* — visible as luminance leaving, which needs a dark floor to
leave toward.

Ship dark only. A light theme costs a full re-tune of every field colour ramp
for a screen nobody will look at.

## Tokens

CSS custom properties on `:root`, in `viz/tokens.css`. **Nothing in the viewer
uses a raw hex value or a one-off pixel size.**

### Colour

```css
:root {
  /* neutral scale — cool, not pure grey; the ground is ice */
  --n-950: #07090c;   /* page */
  --n-900: #0c1015;   /* surface 0 — panels */
  --n-850: #12171e;   /* surface 1 — cards, raised */
  --n-800: #1a212a;   /* surface 2 — hover, inputs */
  --n-700: #2a3541;   /* border strong */
  --n-600: #3d4a59;   /* border */
  --n-400: #6f8095;   /* text tertiary */
  --n-300: #93a3b6;   /* text secondary */
  --n-100: #d7e0ea;   /* text primary */
  --n-000: #f4f8fc;   /* text emphatic — numbers */

  /* accent — one, ice cyan. primary action and the current selection only */
  --accent:        #38d8e0;
  --accent-hover:  #5ae6ec;
  --accent-active: #22b4bc;
  --accent-dim:    #38d8e01f;

  /* semantic — meaning only, never decoration */
  --ok:    #4ade80;   /* tracked, gate green, score improved */
  --warn:  #fbbf24;   /* confirming, degraded, score flat */
  --danger:#f87171;   /* lost, error, score worse */

  /* belief field ramp — sequential, perceptually ordered, dark floor */
  --belief-0: #07090c00;
  --belief-1: #10264a;
  --belief-2: #1d4e8f;
  --belief-3: #2f7fc4;
  --belief-4: #6fb6e8;
  --belief-5: #d6ecff;

  /* asset classes — categorical, distinguishable in greyscale and at 6px */
  --cls-wing:  #ffb454;   /* fixed-wing  — warm, fast, long strokes */
  --cls-quad:  #a78bfa;   /* quadcopter  — violet */
  --cls-rover: #4ade80;   /* rover       — green, ground */
  --cls-tower: #93a3b6;   /* tower       — neutral, immobile */
  --truth:     #f87171;   /* ground truth target — the only red drawn on the
                             field. Same hex as --danger deliberately: red
                             means "the thing you must not miss" everywhere.
                             Off the field (score deltas, errors) that is
                             --danger; on the field it is only ever this. */

  /* terrain */
  --terrain-lo: #0a0f16;
  --terrain-hi: #263442;
  --cut-line:   #38d8e0;  /* flow-network cuts */
}
```

Rules that follow from this and are not negotiable:

- **`--truth` is used for exactly one thing**: the ground-truth target. If red
  appears anywhere else on the map, the viewer is lying about what it knows.
- **The accent is for the primary action and the current selection.** Not for
  headings, not for borders, not for the logo.
- **The belief ramp is never used for anything but belief.** Scores, charts and
  assets use the neutral scale and the class colours.

### Type

One family plus one monospace.

```css
--font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
--font-mono: "JetBrains Mono", ui-monospace, "SF Mono", monospace;

--t-12: 12px/16px;   /* labels, axis, legend */
--t-14: 14px/20px;   /* body, controls */
--t-16: 16px/24px;   /* panel titles */
--t-20: 20px/28px;   /* section heads */
--t-24: 24px/32px;   /* score values */
--t-48: 48px/1.0;    /* the hero readout: the four axis numbers, and the clock */
```

Weights in use: **400, 500, 600.** Three, not five.

**Every number uses `--font-mono` with `font-variant-numeric: tabular-nums`.**
Scores, timestamps, coordinates, confidences, counts. A score that jitters
horizontally as it ticks is the single fastest way to look unfinished.

Fonts are self-hosted and preloaded. No layout shift on first paint.

### Space, radius, elevation, motion, breakpoints

```css
--s-1: 4px;  --s-2: 8px;  --s-3: 12px; --s-4: 16px;
--s-6: 24px; --s-8: 32px; --s-12: 48px;

--r-sm: 4px;   /* inputs, chips, badges */
--r-md: 8px;   /* panels, cards */
/* two radii. never a third. */

--e-1: 0 1px 2px #00000059;                      /* raised panel */
--e-2: 0 8px 24px #00000073;                     /* the one overlay */

--m-fast: 150ms; --m-base: 200ms; --m-slow: 300ms;
--m-ease: cubic-bezier(0.2, 0, 0, 1);            /* ease-out, no bounce */

--bp-demo: 1440px;   /* the target. everything is designed here. */
--bp-video: 1280px;
--bp-narrow: 960px;  /* panels stack below this. mobile is not supported. */
```

`@media (prefers-reduced-motion: reduce)` disables replay interpolation,
transitions and the erosion animation; the viewer snaps between ticks instead.

## Layout

A three-region shell. No navigation — there is one screen.

```
┌──────────────────────────────────────────────────────────────────────┐
│ TOPBAR 48px   seed · transport · policy · terrain    │ four score dials│
├───────────────────────────────────────┬──────────────────────────────┤
│                                       │  RAIL 320px                  │
│                                       │  ┌────────────────────────┐  │
│            THE FIELD                  │  │ CONTACTS               │  │
│            (canvas, fills)            │  │  ribbon per contact    │  │
│                                       │  ├────────────────────────┤  │
│   terrain hillshade                   │  │ FLEET                  │  │
│   + belief field                      │  │  6 rows, class colour, │  │
│   + cut chords                        │  │  task, energy bar      │  │
│   + asset sprites + intent lines      │  ├────────────────────────┤  │
│   + truth marker                      │  │ INSPECTOR              │  │
│                                       │  │  selected cell/asset   │  │
│                                       │  └────────────────────────┘  │
├───────────────────────────────────────┴──────────────────────────────┤
│ TRANSPORT BAR 64px  ◀ ▶ ⏸  ──────●────────────  t=0:41 / 4:00  20×   │
└──────────────────────────────────────────────────────────────────────┘
```

- **The field takes every pixel it can get.** It is the product. The rail is
  fixed at 320px and the topbar and transport bar are fixed height; the canvas
  absorbs the rest.
- **Content width is the viewport.** This is an instrument, not a document. No
  max-width, no centred column.
- **The primary action is the play/scrub control** in the transport bar,
  bottom-left, accent-filled, 40px. It is the only accent-filled control on
  screen.
- Grid: the rail's contents align on a single 16px gutter. Everything in the
  rail shares a left edge.

## Components

**No component library.** Vanilla JS, custom elements where useful, one
`tokens.css` plus one `viewer.css`. Adding a framework or a component library
is a stack change and therefore a human decision (`SPEC.md` §11).

Custom pieces, and they are the entire inventory:

| piece | notes |
|---|---|
| `ScoreDial` | Four in the topbar. Label at `--t-12` `--n-400`, value at `--t-24` mono tabular. A thin 2px arc at `--accent`, 0–1 normalised. Delta chevron in `--ok`/`--warn`/`--danger`. |
| `FieldCanvas` | One `<canvas>`, layered draws in one pass: terrain → belief → cuts → footprints → intents → assets → truth. No DOM overlay on the map except the tooltip. |
| `ContactRibbon` | The lifecycle: four segments, `unconfirmed · confirming · tracked · handed off`. Filled segments in `--warn` then `--ok`. Handoff shows the two class colours meeting. |
| `FleetRow` | Class dot, callsign in mono, current task in `--n-300`, a 2px energy bar. |
| `TransportBar` | Play/pause, step, a scrub track, speed chips (1× 5× 20× 100×), `t` and duration in mono. |
| `Inspector` | Key/value pairs, mono values, right-aligned on a shared edge. |
| `WeightsField` | Four numeric inputs and a re-score button. Demo beat 7. |

**Icons: none, beyond the three transport glyphs**, drawn as inline SVG paths
in `viz/icons.js`. No icon font, no icon library, and never an emoji.

## The signature element

**The field that erodes.**

A dark terrain hillshade under a luminous probability field. As sensor
footprints sweep and report *nothing*, the field is visibly carved away along
the swept cone — luminance draining out of the map in the shape of what was
looked at. Then thin bright `--cut-line` chords snap across the terrain's
saddles, and an asset sprite settles onto a chord rather than continuing its
raster.

This is the picture nobody else at the event will have, because nobody else's
belief update runs on non-detections. It is tied directly to demo beat 3.

Specification:

- Erosion animates over `--m-slow` per tick with `--m-ease`. It is the **only**
  animation on the field, and it is never decorative — one erosion event per
  actual negative report, no more.
- The swept cone draws for `--m-base` at 30% opacity, then leaves the erosion
  behind. It does not linger.
- Cut chords enter by drawing from the midpoint outward, `--m-base`, and hold.
  They do not pulse, glow or shimmer.
- A posted asset gets a 1px `--accent` ring when it is on a cut. That ring is
  the whole visual vocabulary for "this asset is guarding a gap".

## The hero screen, at each beat

The hero screen is the only screen. What changes is the camera and what is
switched on.

- **Beat 1** — full extent, terrain + fleet + score dials at zero. Belief flat
  and dim. Transport bar at `t=0`. Calm.
- **Beat 2** — the field splits into two canvases sharing the topbar and
  transport bar. Left labelled `frontier baseline`, right labelled `tuned`, at
  `--t-12` `--n-400`, top-left of each. Both run from the same scrub position.
  Eight score dials, four per side, the tuned side's deltas in `--ok`.
- **Beat 3 — the wow beat.** Camera zooms to a quarter extent over `--m-slow`.
  Belief opacity up. The erosion is the only thing moving. Cut chords enter.
  The `TRACKED` badge appears in the field's top-left, `--t-48` mono,
  `--ok`, with the clock — big enough to read from the back of the room, and
  it is the only time type that size appears on the map. (`--t-48` is also the
  topbar dials and the clock; the topbar is not the map. D34 builds this
  badge.)
- **Beat 4** — camera holds, rail's CONTACTS section expands, handoff arcs draw
  between assets on the field in the two class colours.
- **Beat 5** — the field is replaced by the **tuning report**: a scatter of
  score against parameter per panel, the chosen operating point marked with an
  `--accent` ring, and the ablation table below. Same tokens, same type, small
  multiples on a 4-column grid.
- **Beat 6** — back to the field; topbar `transport` reads `sitl`, and the
  fleet rows show MAVLink sysids. Nothing else changes, and that is the point.
  If SITL could not be obtained at all (`SPEC.md` §2 beat 6), this beat is the
  `pymavlink` loopback D46 records on overrun instead, shown as terminal
  output — no field view, and no topbar reading `sitl`.
- **Beat 7** — `WeightsField` in the rail, four numbers typed, the dials and
  the beat-5 operating point move.

**Every state is designed**, and each has a ticket: no log loaded (explains and
offers the bundled fixture episode), log loading (skeleton in the shape of the
final layout — field block, rail blocks, dials — never a spinner), log
malformed (names the failing line and the schema version), episode running,
episode finished, zero contacts (the rail says what would put one there), and
many contacts (the ribbon list scrolls; the field does not).

## Data and copy voice

**Fixtures are real episodes**, not invented rows. The bundled demo fixture is
a committed episode log from a real seeded run, which means every number on
screen is a number the system actually produced. There is no path by which a
fake value reaches the screen — if the viewer shows it, the sim computed it.

Callsigns are terse and class-legible: `WING-1`, `QUAD-1`, `QUAD-2`, `ROVR-1`,
`TOWR-1`, `TOWR-2`. Never `Vehicle 1`, never `Test Asset`.

Voice: **flat, specific, lowercase-label / sentence-case-sentence.** Labels are
nouns (`coverage`, `handed off`, `min turn radius`). Sentences state what
happened (`Swept sector 14, no detection. Belief reduced 0.31 → 0.04.`). No
exclamation marks. No first person. No "Welcome". No reassurance — the user is
an engineer who wants the number.

## Don'ts for this product

Beyond the generic list in the `ui-craft` skill:

- **No radar-sweep animation, no scan lines, no reticles, no all-caps green
  monospace field.** Military cosplay reads as a student project.
- **No red on the field except ground truth.** The truth marker is the only
  red drawn inside `FieldCanvas`. Off the field, red is `--danger` and means
  only "worse" — a falling score chevron is the one other place it appears.
- **No belief-ramp colours outside the belief field.**
- **No tooltip that follows the cursor continuously.** It appears on hover after
  120ms, anchored to the cell, and it does not animate.
- **Nothing glows.** No `box-shadow` used as a halo, no `filter: blur()` behind
  a card, no bloom on the field. Luminance comes from the ramp, not from
  effects.
- **No modal.** There is one screen and nothing on it needs interrupting.
- **No progress bar for the sweep.** The tuning report shows results, not a
  loading experience.
- **Do not make the map prettier at the cost of legibility.** Hillshade is
  background: it never exceeds `--terrain-hi` and it never competes with
  belief. If you cannot tell belief from terrain at a glance, the terrain is
  too bright.
