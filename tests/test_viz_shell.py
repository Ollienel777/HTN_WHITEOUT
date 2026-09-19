"""The viewer shell's acceptance criteria, as assertions.

Issue #18 asks for four things this file can check without a browser:

* no raw hex value and no one-off pixel size anywhere outside ``tokens.css``
* the seven states in ``DESIGN.md`` are all present, and loading is a skeleton
  in the final layout's shape rather than a spinner
* no build step: no ``package.json``, no ``node_modules``, and a page that
  still opens from ``file://``
* the bundled episode the empty state offers is a real, valid episode log

What a browser is needed for — that each state *renders*, at the demo
viewport — is verified by looking at the running app, and is issue #44's
standing job.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from whiteout.geo import ARENA_ORIGIN, WGS84_A, WGS84_F, LocalPoint, local_to_geodetic
from whiteout.log import SCHEMA_VERSION, validate_episode_log

REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ = REPO_ROOT / "viz"
TOKENS = VIZ / "tokens.css"
DESIGN = REPO_ROOT / "hackathon" / "DESIGN.md"
BUNDLED_EPISODE = REPO_ROOT / "fixtures" / "episodes" / "demo.jsonl"

#: Everything the viewer ships. tokens.css is the one file allowed a raw value.
SHELL_FILES = ("index.html", "viewer.css", "viewer.js", "icons.js")

HEX = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b")
PIXELS = re.compile(r"\b\d+(?:\.\d+)?px\b")
DECLARATION = re.compile(r"--([a-z0-9-]+)\s*:\s*([^;]+);")


def _read(name: str) -> str:
    return (VIZ / name).read_text(encoding="utf-8")


def _without_comments(text: str) -> str:
    """Strip ``/* ... */`` and ``<!-- ... -->``.

    Comments quote DESIGN.md, which is full of hex values and pixel sizes; the
    rule is about what the browser is told, not about what a reader is told.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)


def _declarations(text: str) -> dict[str, str]:
    return {
        name: " ".join(value.split())
        for name, value in DECLARATION.findall(_without_comments(text))
    }


@pytest.mark.parametrize("name", SHELL_FILES)
def test_no_raw_hex_value_outside_tokens_css(name: str) -> None:
    found = HEX.findall(_without_comments(_read(name)))
    assert found == [], f"{name} spells a colour instead of using a token: {found}"


@pytest.mark.parametrize("name", SHELL_FILES)
def test_no_one_off_pixel_size_outside_tokens_css(name: str) -> None:
    """Every length is a token.

    The single carve-out is the ``@media`` condition at the foot of
    ``viewer.css``: a media query cannot read a custom property, so the
    breakpoint's value has to appear there literally. The test below holds it
    equal to ``--bp-narrow``.
    """
    offending = [
        line.strip()
        for line in _without_comments(_read(name)).splitlines()
        if PIXELS.search(line) and "@media" not in line
    ]
    assert offending == [], f"{name} spells a size instead of using a token: {offending}"


def test_every_token_design_md_states_is_defined_with_its_stated_value() -> None:
    """``tokens.css`` is DESIGN.md's token block, not a reinterpretation of it."""
    stated: dict[str, str] = {}
    for block in re.findall(r"```css\n(.*?)```", DESIGN.read_text(encoding="utf-8"), re.DOTALL):
        stated.update(_declarations(block))
    defined = _declarations(TOKENS.read_text(encoding="utf-8"))
    assert stated, "DESIGN.md states no tokens; the test is looking in the wrong place"
    missing = sorted(name for name in stated if name not in defined)
    assert missing == [], f"tokens.css does not define: {missing}"
    wrong = {
        name: (stated[name], defined[name]) for name in stated if defined[name] != stated[name]
    }
    assert wrong == {}, f"tokens.css disagrees with DESIGN.md: {wrong}"


def test_the_narrow_media_query_still_equals_the_bp_narrow_token() -> None:
    narrow = _declarations(TOKENS.read_text(encoding="utf-8"))["bp-narrow"]
    conditions = re.findall(r"@media \(max-width: (\d+px)\)", _read("viewer.css"))
    assert conditions == [narrow], f"the one media query should read {narrow}, got {conditions}"


# ── the seven states ──────────────────────────────────────────────────


def test_the_three_overlay_states_are_present() -> None:
    """No log loaded, log loading, log malformed — states 1 to 3 of 7."""
    markup = _read("index.html")
    for view in ("empty", "loading", "error"):
        assert f'data-view="{view}"' in markup, f"no designed state for {view}"


def test_the_empty_state_explains_and_offers_the_bundled_episode() -> None:
    markup = _read("index.html")
    empty = markup.split('data-view="empty"')[1].split("</section>")[0]
    assert "no episode loaded" in empty
    assert 'data-act="load-bundled"' in empty, "the empty state offers no next action"
    assert 'data-act="open-file"' in empty, "a page opened from file:// needs the picker"
    assert "fixtures/episodes/demo.jsonl" in empty


def test_loading_is_a_skeleton_in_the_shape_of_the_layout_and_never_a_spinner() -> None:
    markup = _without_comments(_read("index.html"))
    css = _without_comments(_read("viewer.css"))
    loading = markup.split('data-view="loading"')[1].split("</section>")[0]
    assert "sk-field" in loading, "the field has no skeleton block"
    # The skeleton also stands in for the dials and the rail cards, so that it
    # is the final layout's shape and not one lonely rectangle.
    assert markup.count("data-skeleton") >= 4
    assert "sk-dial" in markup and "sk-block" in markup and "sk-line" in markup
    assert "spinner" not in markup.lower() and "spinner" not in css.lower()
    # A spinner is a rotation that never ends. The one animation here is a
    # slow change of opacity.
    assert "rotate" not in css.split("@keyframes")[-1]


def test_the_malformed_state_names_the_failing_line_and_the_schema_version() -> None:
    markup = _read("index.html")
    error = markup.split('data-view="error"')[1].split("</section>")[0]
    assert "data-error-line" in error
    assert "data-error-schema" in error
    assert "data-error-detail" in error
    assert "reader schema" in error, "the reader's own schema version is not named"
    js = _read("viewer.js")
    # The detail line is the reader's own message, both versions included.
    assert "is not supported, expected" in js
    assert "String(SCHEMA_VERSION)" in js


def test_the_running_and_finished_states_are_both_named() -> None:
    """States 4 and 5 of 7, which the transport bar reports."""
    js = _read("viewer.js")
    assert '"episode running"' in js
    assert '"episode finished"' in js
    assert "data-run-state" in _read("index.html")


def test_zero_contacts_says_what_would_post_one() -> None:
    """State 6 of 7."""
    markup = _read("index.html")
    empty = markup.split('data-empty="contacts"')[1].split("</p>")[0]
    assert "No contacts" in empty
    assert "posts one here" in empty


def test_many_contacts_scrolls_the_rail_and_not_the_field() -> None:
    """State 7 of 7."""
    markup = _read("index.html")
    css = _without_comments(_read("viewer.css"))
    contacts = markup.split('class="panel panel-contacts"')[1].split("</section>")[0]
    assert "panel-scroll" in contacts
    assert re.search(r"\.panel-scroll \{[^}]*overflow-y: auto", css, re.DOTALL)
    assert re.search(r"\.field \{[^}]*overflow: hidden", css, re.DOTALL)


# ── no build step ─────────────────────────────────────────────────────


def test_there_is_no_build_step() -> None:
    for artifact in ("package.json", "package-lock.json", "node_modules", "tsconfig.json"):
        assert not (VIZ / artifact).exists(), f"viz/{artifact} is a build step"
        assert not (REPO_ROOT / artifact).exists(), f"{artifact} is a build step"


def test_the_page_opens_from_a_file_url() -> None:
    """No module scripts, no absolute paths, no third-party origin.

    A ``file://`` origin is opaque: a module script fails CORS, and an
    absolute ``/viz/...`` path resolves to the filesystem root. Both make the
    page work under ``whiteout serve`` and fail on a stranger's laptop, which
    is the one place SPEC.md §5 says this choice has to pay off.
    """
    markup = _without_comments(_read("index.html"))
    assert 'type="module"' not in markup
    for reference in re.findall(r'(?:src|href)="([^"]+)"', markup):
        assert not reference.startswith("/"), f"{reference} is absolute"
        assert "//" not in reference, f"{reference} leaves the page's own directory"
    assert _read("viewer.js").lstrip().startswith("/*")


def test_the_only_icons_are_the_three_transport_glyphs() -> None:
    """DESIGN.md: no icon font, no icon library, and never an emoji."""
    paths = re.findall(r"^\s{4}(\w+): \"[Mm]", _read("icons.js"), re.MULTILINE)
    assert sorted(paths) == ["pause", "play", "step"]
    for name in SHELL_FILES:
        body = _without_comments(_read(name))
        assert not re.search(r"[\U0001f300-\U0001faff\u2600-\u27bf\u2b00-\u2bff]", body), (
            f"{name} draws with an emoji"
        )


# ── the bundled episode ───────────────────────────────────────────────


def test_the_bundled_episode_is_a_real_valid_episode_log() -> None:
    """The empty state's primary action must never land in the error state."""
    assert BUNDLED_EPISODE.is_file(), "the empty state offers an episode that is not there"
    assert validate_episode_log(BUNDLED_EPISODE) > 0


def test_the_viewer_reads_the_same_schema_version_as_the_python_reader() -> None:
    js = _read("viewer.js")
    declared = re.search(r"var SCHEMA_VERSION = (\d+);", js)
    assert declared is not None
    assert int(declared.group(1)) == SCHEMA_VERSION


def test_the_viewer_points_at_the_bundled_episode_that_exists() -> None:
    js = _read("viewer.js")
    declared = re.search(r'var BUNDLED_EPISODE = "([^"]+)";', js)
    assert declared is not None
    resolved = (VIZ / declared.group(1)).resolve()
    assert resolved == BUNDLED_EPISODE.resolve()


# ── the drawing frame (issue #73) ─────────────────────────────────────


def test_the_viewer_reads_positions_in_the_one_frame_of_record() -> None:
    """``SPEC.md`` §5: the log is lat/lon, so the viewer reads lat/lon.

    The page is the one consumer that *wants* metres, which makes it the one
    most likely to quietly become a second frame of record.
    """
    js = _without_comments(_read("viewer.js"))
    assert "pose.lat" in js and "pose.lon" in js
    assert "pose.x" not in js and "pose.y" not in js


def test_the_viewers_local_frame_is_named_as_a_drawing_frame() -> None:
    js = _read("viewer.js")
    assert "\u2500\u2500 the drawing frame" in js, "the section is not named"
    frame = js.split("\u2500\u2500 the drawing frame")[1].split("var WGS84_A")[0]
    assert "never written" in frame
    assert "never posted" in frame
    assert "frame of record" in frame


#: The viewer's drawing frame, cut out of ``viewer.js`` so ``node`` can run
#: it without the page around it. Both markers are asserted before the cut,
#: so moving the block fails the test rather than silently narrowing it.
_FRAME_OPENS = "var WGS84_A"
_FRAME_CLOSES = "/* The origin is the fleet's centroid"

#: Offsets in metres about ``ARENA_ORIGIN``, taken to the corners of the
#: 25 km x 2 km strait and to a couple of points inside it. The axes differ
#: by a factor of 3.2 up here, so a projection that crosses them is out by
#: kilometres at the far end and this catches it at every case.
_PINNED_OFFSETS = (
    (0.0, 0.0),
    (1000.0, 0.0),
    (0.0, 1000.0),
    (12500.0, 1000.0),
    (-12500.0, -1000.0),
    (-3000.0, 750.0),
)


def _viewer_frame_metres(offsets: tuple[tuple[float, float], ...]) -> list[dict[str, float]]:
    """Run ``viewer.js``'s ``toLocal`` under ``node`` and return its metres."""
    node = shutil.which("node")
    assert node is not None, (
        "node is needed to pin the viewer's arithmetic to whiteout.geo. It is "
        "already a dependency of this repository (scripts/toutc.mjs) and is "
        "present on the CI runner; this test is not skipped because skipping "
        "it is how the viewer's formulae stopped being checked at all."
    )
    js = _read("viewer.js")
    assert _FRAME_OPENS in js and _FRAME_CLOSES in js, "the drawing frame block has moved"
    frame = js[js.index(_FRAME_OPENS) : js.index(_FRAME_CLOSES)]
    assert "function toLocal" in frame, "the cut did not include toLocal"

    places = [local_to_geodetic(ARENA_ORIGIN, LocalPoint(east, north)) for east, north in offsets]
    driver = (
        frame
        + "var origin = { lat: "
        + repr(ARENA_ORIGIN.lat_deg)
        + ", lon: "
        + repr(ARENA_ORIGIN.lon_deg)
        + " };\n"
        + "var places = "
        + json.dumps([[place.lat_deg, place.lon_deg] for place in places])
        + ";\n"
        + "process.stdout.write(JSON.stringify(places.map(function (p) {\n"
        + "  return toLocal(origin, p[0], p[1]);\n"
        + "})));\n"
    )
    done = subprocess.run(  # noqa: S603
        [node, "-e", driver], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, f"node refused the drawing frame: {done.stderr}"
    return json.loads(done.stdout)


def test_the_viewers_projection_returns_the_same_metres_as_whiteout_geo() -> None:
    """The half of the pin that is a number rather than a constant.

    The constants test below compares ``WGS84_A``, ``WGS84_F`` and the
    origin, which leaves the formulae between them unpinned: swapping the two
    radii inside ``toLocal`` keeps every constant correct, makes the East
    axis 3.2x too large and the North axis 3.2x too small, and used to leave
    the whole suite green. This runs the block and compares metres, so the
    arithmetic is pinned too.
    """
    metres = _viewer_frame_metres(_PINNED_OFFSETS)
    assert len(metres) == len(_PINNED_OFFSETS)
    for (east, north), got in zip(_PINNED_OFFSETS, metres, strict=True):
        assert got["east"] == pytest.approx(east, abs=1e-6)
        assert got["north"] == pytest.approx(north, abs=1e-6)


def test_the_viewers_projection_is_pinned_to_whiteout_geo() -> None:
    """The page has no Python, so its copy of the arithmetic is pinned here.

    A viewer that drew with a different ellipsoid or a different origin would
    disagree with the posted lat/lon by metres at first and by kilometres
    once someone read a position off the canvas.
    """
    js = _read("viewer.js")
    semi_major = re.search(r"var WGS84_A = ([\d.]+);", js)
    flattening = re.search(r"var WGS84_F = 1 / ([\d.]+);", js)
    origin = re.search(r"var ARENA_ORIGIN = \{ lat: (-?[\d.]+), lon: (-?[\d.]+) \};", js)
    assert semi_major is not None and flattening is not None and origin is not None
    assert float(semi_major.group(1)) == WGS84_A
    assert 1.0 / float(flattening.group(1)) == WGS84_F
    assert float(origin.group(1)) == ARENA_ORIGIN.lat_deg
    assert float(origin.group(2)) == ARENA_ORIGIN.lon_deg
