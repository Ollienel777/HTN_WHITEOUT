"""The basemap's registration, as assertions.

Issue #131. ``viz/basemap.js`` draws land under the belief field, and the
only thing that makes it worth drawing is that it lands *on* the water cells
rather than near them. A coastline a few hundred metres out is worse than no
coastline: it says the vessel is searching rock.

So the checks here are about registration and nothing else:

* the committed file is what ``scripts/make_basemap.py`` writes from
  ``DEFAULT_STRAIT`` today — it cannot drift when the geometry moves
* every shore point sits **on** the boundary ``is_water`` tests against:
  water a metre inboard, land a metre outboard
* every water cell of the default belief grid falls inside the committed
  ring, by a point-in-polygon test that does not go back through
  :meth:`StraitGeometry.is_water` and so is not the ribbon checking itself
* the page degrades to its old dark ground when the file is not there

Rendering is not checked here. That is `ui-craft`'s screenshot pass.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

from whiteout.belief.geometry import DEFAULT_STRAIT, ChannelPoint
from whiteout.belief.grid import ChannelBeliefGrid

REPO_ROOT = Path(__file__).resolve().parents[1]
BASEMAP = REPO_ROOT / "viz" / "basemap.js"
VIZ = REPO_ROOT / "viz"


def _generator() -> ModuleType:
    """``scripts/make_basemap.py``, loaded by path.

    ``scripts/`` is not a package — nothing imports from it in the product —
    so this is loaded the way the gate loads its own steps rather than by
    putting the directory on the path for every other test.
    """
    spec = importlib.util.spec_from_file_location(
        "make_basemap", REPO_ROOT / "scripts" / "make_basemap.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shores() -> dict[str, list[list[float]]]:
    """The committed shores, read out of the generated script."""
    text = BASEMAP.read_text(encoding="utf-8")
    payload = text.split("window.WHITEOUT_BASEMAP = ", 1)[1].rsplit(";", 1)[0]
    shores: dict[str, list[list[float]]] = json.loads(payload)["shores"]
    return shores


def test_the_committed_basemap_is_what_the_geometry_generates_today() -> None:
    """Regenerate and compare. The file is a record, not a hand edit."""
    generator = _generator()
    assert BASEMAP.read_text(encoding="utf-8") == generator.render(
        generator.sample_shores(DEFAULT_STRAIT)
    ), (
        "viz/basemap.js has drifted from whiteout.belief.geometry.DEFAULT_STRAIT. "
        "Run `python scripts/make_basemap.py` and commit the result."
    )


def test_every_shore_point_sits_on_the_water_boundary() -> None:
    """Water one metre inboard, land one metre outboard, at every sample.

    This is the registration claim stated as a predicate: the line the viewer
    fills land against is the line ``is_water`` switches on, so the land edge
    and the belief mask cannot disagree by more than the mask's own
    discretisation.
    """
    shores = _shores()
    for side, sign in (("left", 1.0), ("right", -1.0)):
        points = shores[side]
        assert len(points) > 100, f"{side} shore is too coarse to read as a curve"
        for lat, lon in points:
            point = DEFAULT_STRAIT.to_channel(lat, lon)
            half = DEFAULT_STRAIT.half_width_at(point.s_m)
            assert abs(abs(point.w_m) - half) < 1.0, (
                f"{side} shore at s={point.s_m:.0f} is {abs(point.w_m) - half:+.1f} m "
                "off the half-width"
            )
            assert point.w_m * sign > 0.0, f"{side} shore is on the wrong side"
            # The ends of the ribbon are mouths, not shore: `is_water` is
            # false at s <= 0 and s >= length whatever the offset, so the two
            # end samples are exempt from the inboard half of the check.
            if 1.0 < point.s_m < DEFAULT_STRAIT.length_m - 1.0:
                inboard = DEFAULT_STRAIT.to_position(
                    ChannelPoint(s_m=point.s_m, w_m=sign * (half - 1.0))
                )
                assert DEFAULT_STRAIT.is_water(*inboard), "a metre inboard of the shore is land"
            outboard = DEFAULT_STRAIT.to_position(
                ChannelPoint(s_m=point.s_m, w_m=sign * (half + 1.0))
            )
            assert not DEFAULT_STRAIT.is_water(*outboard), "a metre outboard of the shore is water"


def _ring(shores: dict[str, list[list[float]]]) -> list[tuple[float, float]]:
    """The closed water ring the viewer fills against: left out, right back."""
    return [(lon, lat) for lat, lon in shores["left"]] + [
        (lon, lat) for lat, lon in reversed(shores["right"])
    ]


def _inside(ring: list[tuple[float, float]], x: float, y: float) -> bool:
    """Ray-cast point-in-polygon, in (lon, lat) degrees.

    Degrees rather than metres on purpose: it is the committed numbers being
    tested, in the units they are committed in, with no projection in
    between.
    """
    inside = False
    count = len(ring)
    for index in range(count):
        x0, y0 = ring[index]
        x1, y1 = ring[(index + 1) % count]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def test_every_water_cell_of_the_default_grid_falls_inside_the_committed_ring() -> None:
    """The alignment the demo actually shows: cells on water, land around them.

    The belief field the viewer draws is this grid's water cells. If a cell
    centre fell outside the ring, the page would paint probability on top of
    land it had just drawn — the exact failure the ticket calls worse than no
    basemap.
    """
    grid = ChannelBeliefGrid(geometry=DEFAULT_STRAIT)
    ring = _ring(_shores())
    along = grid.along_centres_m()
    across = grid.across_centres_m()
    mask = grid.water_mask()
    water = 0
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            if not mask[row][column]:
                continue
            water += 1
            lat, lon = DEFAULT_STRAIT.to_position(
                ChannelPoint(s_m=float(along[row]), w_m=float(across[column]))
            )
            assert _inside(ring, lon, lat), (
                f"water cell ({row}, {column}) at {lat:.5f}, {lon:.5f} is outside the shores"
            )
    assert water > 100, "the default grid has no water in it to check"


def test_the_viewer_draws_the_land_under_the_graticule_and_over_nothing_else() -> None:
    """Order, as with every other field layer: background, land, grid, content."""
    source = (VIZ / "viewer.js").read_text(encoding="utf-8")
    body = source.split("function drawField(")[1].split("function drawContent(")[0]
    assert body.index("--n-950") < body.index("drawBasemap(") < body.index("--n-900"), (
        "the basemap is not between the background fill and the graticule"
    )


def test_the_viewer_renders_with_the_basemap_absent() -> None:
    """The dark ground is the fallback, not an error.

    ``basemap.js`` is a plain script that sets one global, so a page without
    it defines nothing and the guard below is the whole of the fallback.
    Nothing in the shell may require it.
    """
    source = (VIZ / "viewer.js").read_text(encoding="utf-8")
    guard = source.split("function projectBasemap(")[1].split("function projectShore(")[0]
    assert "if (!data || !data.shores || !state.origin) { return null; }" in guard
    drawn = source.split("function drawBasemap(")[1]
    assert 'if (!map || state.phase !== "ready") { return false; }' in drawn
    markup = (VIZ / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<script src="basemap\.js" defer></script>', markup), (
        "the page does not load the basemap"
    )


def test_the_basemap_credits_its_source_in_the_page() -> None:
    """Whatever is drawn under the field says what it is, in the page itself.

    These shores are a parameterisation of ARENA.md and not a survey, and not
    licensed imagery; the credit line says so where a judge reading the
    screen will meet it.
    """
    text = BASEMAP.read_text(encoding="utf-8")
    assert "ARENA.md" in text and "not a survey" in text
    markup = (VIZ / "index.html").read_text(encoding="utf-8")
    assert 'id="field-credit"' in markup
    css = (VIZ / "viewer.css").read_text(encoding="utf-8")
    assert ".field-credit" in css, "the credit has no position on the field"


def test_the_basemap_obeys_the_same_phase_gate_the_content_does() -> None:
    """A failed load must not leave the last episode's coastline on screen.

    ``loadText`` and ``loadUrl`` both clear ``state.records`` on failure and
    neither clears ``state.basemap``, so the gate in ``drawBasemap`` is what
    stops the "episode log is malformed" card sitting over the previous
    episode's land, at the previous episode's zoom, still credited.

    This is a source assertion and does not drive the page: it pins the
    guard against a silent removal, and is not evidence that the error state
    renders correctly. Driving the viewer is `test_viz_shell.py`'s job and
    neither file does it for this path yet.
    """
    source = (VIZ / "viewer.js").read_text(encoding="utf-8")
    drawn = source.split("function drawBasemap(")[1].split("function ")[0]
    assert 'state.phase !== "ready"' in drawn, (
        "drawBasemap no longer gates on the phase, so a stale coastline can "
        "outlive the episode that produced it"
    )
    content = source.split("function drawContent(")[1].split("function ")[0]
    assert 'state.phase !== "ready"' in content, (
        "the gate this one is matched to has moved; re-check both"
    )


def test_the_canvas_labels_over_land_clear_AA() -> None:
    """Both bottom corners sit over the land tone, so both take ``--n-300``.

    ``--n-400`` is the tertiary colour the canvas uses for labels over the
    bare frame, where it measures 4.93:1. Over the land blend it is 4.25:1,
    under AA. The credit line was raised for this reason when the basemap
    landed; the scale bar is the opposite corner of the same ground.

    A source assertion, not a contrast measurement: it pins the token, not
    the ratio.
    """
    source = (VIZ / "viewer.js").read_text(encoding="utf-8")
    bar = source.split("function drawScaleBar(")[1].split("function ")[0]
    assert 'token("--n-300")' in bar, "the scale-bar label is back under AA over land"
    assert 'token("--n-400")' not in bar
    css = (VIZ / "viewer.css").read_text(encoding="utf-8")
    credit = css.split(".field-credit {")[1].split("}")[0]
    assert "--n-300" in credit, "the credit line is back under AA over land"
