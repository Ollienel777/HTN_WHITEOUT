# viz

The episode-log viewer: static HTML plus a `<canvas>`, vanilla JS, no build
step, opens from `file://`. Nothing in `whiteout/` imports from here.

## Files

| file | what it holds |
|---|---|
| `index.html` | the three-region shell — topbar, field, 320px rail, transport bar — and the markup for every state |
| `tokens.css` | **the only file allowed a raw value.** `DESIGN.md`'s token block, plus the structural sizes it states in prose |
| `viewer.css` | the shell's layout and components, in tokens only |
| `viewer.js` | the state machine, the episode-log reader, the replay clock, and the field's chrome |
| `basemap.js` | **generated** by `scripts/make_basemap.py` — the channel's two shores, in lat/lon, from `whiteout.belief.geometry.DEFAULT_STRAIT`. The field fills everything outside them as land. Optional: a page without it draws the dark ground it always did |
| `icons.js` | the three transport glyphs, as inline SVG paths. There are no other icons |

## Running it

```bash
PORT=8123 python -m whiteout.cli serve     # or omit PORT for an ephemeral one
```

`serve` hands out the repository root, so the viewer at `/viz/` can read the
episode fixtures at `/fixtures/` beside it. It never reuses a running server:
a `PORT` something is already listening on is refused, not shared.

Opening `viz/index.html` straight off disk works too. The bundled episode
cannot be fetched from a `file://` origin — the page says so and offers the
file picker, which can read any log on the machine.

`?log=<path>` loads a log on open.

## Where the field's content is

`viewer.js` draws the field's **chrome** — the extent graticule and the scale
bar. The layered content draw (terrain hillshade, belief field, cut chords,
sensor footprints, intent lines, asset sprites, the truth marker) is
`FieldCanvas`, and lands with its own ticket.
