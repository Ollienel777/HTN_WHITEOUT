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
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from whiteout.geo import (
    ARENA_ORIGIN,
    WGS84_A,
    WGS84_F,
    GeoPoint,
    LocalPoint,
    geodetic_to_local,
    local_to_geodetic,
)
from whiteout.log import SCHEMA_VERSION, read_episode_log, validate_episode_log
from whiteout.score import AXES as SCORE_AXES
from whiteout.types import SYNC_STATUSES
from whiteout.vision.camera import CAMERAS, CameraModel
from whiteout.vision.projection import (
    CameraPose,
    max_flat_plane_range_m,
    project_pixel_to_ground,
)

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


def test_a_contact_says_whether_its_fix_and_attitude_were_one_instant() -> None:
    """``Contact.sync``, on screen. A fix nobody synchronised must look unlike one
    that was: the failure being fixed is a 5.5 m error with every field
    populated and nothing to look at."""
    markup = _read("index.html")
    js = _read("viewer.js")
    css = _without_comments(_read("viewer.css"))
    contact_row = markup.split('id="tpl-contact-row"')[1].split("</template>")[0]
    assert "data-contact-sync" in contact_row
    # Every state the log can carry, plus the absence, reaches the row.
    for status in SYNC_STATUSES:
        assert status in js, f"the viewer has no word for {status}"
    assert "unrecorded" in js, "a contact with no sync recorded reads as a state"
    assert "skew " in js, "a measured skew is the finding and is not shown"
    # Semantic colour only, and from tokens: amber for unknown, red for known bad.
    sync_rules = "".join(css.split(".contact-sync")[1:])
    assert "--warn" in sync_rules
    assert "--danger" in sync_rules


def test_a_refused_fix_is_named_in_the_rail_and_not_only_dropped() -> None:
    """`EpisodeRecord.refusals`, on screen.

    A refused sighting leaves no contact, so without this line the rail cannot
    tell "nobody can see the vessel" from "two cameras saw something we would
    not stand behind" — and in the artifact the second must not read as an
    empty sea.
    """
    markup = _without_comments(_read("index.html"))
    js = _read("viewer.js")
    css = _without_comments(_read("viewer.css"))
    contacts = markup.split('class="panel panel-contacts"')[1].split("</section>")[0]
    assert "data-refused" in contacts, "the contacts panel never says what was refused"
    assert '"refusals"' in js, "the viewer does not read the record's refusals"
    assert "refused" in js
    assert "--warn" in "".join(css.split(".panel-note")[1:])


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


def test_the_bundled_episode_shows_a_fleet_doing_something() -> None:
    """Valid is not the same as worth looking at, and this is the difference.

    The committed fixture was a valid episode log for a day and a half while
    being 120 ticks of four parked assets, an empty intent every tick and a
    1x1 placeholder belief digest with zero mass. Every test above passed the
    whole time. A judge opening the viewer would have seen a correctly
    rendered instrument reporting that nothing was happening.

    So this asserts the three things that make it a demo rather than a
    well-formed file: the fleet is tasked, at least one asset actually goes
    somewhere, and the belief field is a real grid rather than the
    placeholder. It deliberately says nothing about *how much* the field
    moves -- the negative-information update (#13) is not merged, so the
    field is still uniform, and pinning a number here would have to be
    rewritten the day it lands.
    """
    records = read_episode_log(BUNDLED_EPISODE)
    last = records[-1]
    assert last.belief_digest.grid_shape != (1, 1), "the bundled episode carries a placeholder"
    assert last.belief_digest.mass > 0.0
    assert any(record.intent.intents for record in records), "nothing was ever tasked"
    assert any(pose.energy_used > 0.0 for record in records for pose in record.observation.poses), (
        "no asset moved for the whole episode"
    )


def test_the_viewer_reads_the_same_schema_version_as_the_python_reader() -> None:
    js = _read("viewer.js")
    declared = re.search(r"var SCHEMA_VERSION = (\d+);", js)
    assert declared is not None
    assert int(declared.group(1)) == SCHEMA_VERSION


def test_the_viewer_shows_the_axes_the_scorer_can_answer_for() -> None:
    """The dial row and ``whiteout score`` must name the same axes.

    They drifted: the viewer kept the superseded four — ``coverage``,
    ``collaboration``, ``efficiency``, ``tracking_accuracy`` — while the
    scorer moved to the five an episode log can answer for. Two of them named
    nothing the scorer computes, and one of those was **collaboration**,
    which :func:`whiteout.score.unscorable_criteria` exists to say a log
    cannot answer for at all. A dial for a number that can never arrive reads
    as a broken instrument for the whole run, in front of the judges it is
    wrong in front of.
    """
    js = _read("viewer.js")
    block = re.search(r"var AXES = \[(.*?)\];", js, re.DOTALL)
    assert block is not None, "the viewer's axis list has moved"
    declared = tuple(re.findall(r'key:\s*"(\w+)"', block.group(1)))
    assert declared == SCORE_AXES


def test_the_viewer_names_no_axis_the_scorer_refuses_to_invent() -> None:
    """Autonomy and collaboration are judged, not computed. Not dials."""
    js = _without_comments(_read("viewer.js"))
    block = re.search(r"var AXES = \[(.*?)\];", js, re.DOTALL)
    assert block is not None
    for refused in ("collaboration", "autonomy"):
        assert refused not in block.group(1), (
            f"{refused!r} is read off the fleet's behaviour, not off a log"
        )


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


# ── the field canvas (issue #20) ──────────────────────────────────────


#: The footprint block, cut out of ``viewer.js`` so ``node`` can run it. It
#: reads the ellipsoid out of the drawing frame above it, so both blocks are
#: handed to node together.
#: Opens at `radians`, not at the earth radius below it: the unit conversion
#: is the first thing in this section and `groundFootprint` does not run
#: without it.
_FOOTPRINT_OPENS = "function radians"
_FOOTPRINT_CLOSES = "function cameraFor"


def _run_viewer_block(blocks: str, tail: str) -> object:
    node = shutil.which("node")
    assert node is not None, (
        "node is needed to pin the viewer's field canvas to whiteout.vision. "
        "It is already a dependency of this repository (scripts/toutc.mjs) "
        "and is present on the CI runner; this test is not skipped because "
        "skipping it is how the viewer's formulae stopped being checked."
    )
    done = subprocess.run(  # noqa: S603
        [node, "-e", blocks + tail], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, f"node refused the block: {done.stderr}"
    return json.loads(done.stdout)


def _footprint_source() -> str:
    js = _read("viewer.js")
    for marker in (_FRAME_OPENS, _FRAME_CLOSES, _FOOTPRINT_OPENS, _FOOTPRINT_CLOSES):
        assert marker in js, f"the field canvas has moved: {marker!r} is gone"
    frame = js[js.index(_FRAME_OPENS) : js.index(_FRAME_CLOSES)]
    footprint = js[js.index(_FOOTPRINT_OPENS) : js.index(_FOOTPRINT_CLOSES)]
    assert "function groundFootprint" in footprint, "the cut did not include groundFootprint"
    return frame + footprint


#: Poses whose whole frame lands on the water: the bottom row short of the
#: nadir and the top row short of the horizon, so ``project_pixel_to_ground``
#: answers for both and the comparison is against a number rather than
#: against a clamp. The quadcopter is absent on purpose — its 99.4 degree
#: lens has no pitch at which both rows land, which is a fact about the
#: arena's optics and is asserted separately below.
_PINNED_FOOTPRINTS = (
    ("tower", 60.0, -45.0),
    ("fixed-wing", 500.0, -50.0),
    ("tower", 60.0, -70.0),
)

#: A boresight that is nobody's default and reads differently in each unit:
#: taken as radians, 137 wraps to 289.5 degrees, so a viewer that confused
#: the two cannot pass by accident. Zero was the old value and is the one
#: heading at which degrees and radians agree.
_PINNED_HEADING_DEG = 137.0


def _ground_range_m(camera: CameraModel, pose: CameraPose, py: float) -> float:
    """Ground range of the frame's centre column at row ``py``, in metres."""
    fix = project_pixel_to_ground(camera, pose, camera.width / 2.0, py)
    here = GeoPoint(pose.lat_deg, pose.lon_deg)
    offset = geodetic_to_local(here, GeoPoint(fix.lat_deg, fix.lon_deg))
    return math.hypot(offset.east_m, offset.north_m)


def test_the_viewers_footprint_is_the_frame_whiteout_vision_projects() -> None:
    """The near and far edges are the frame's bottom and top rows, in metres.

    The viewer draws a wedge rather than tracing four corner rays, so the
    thing that can silently go wrong is the range: a footprint computed off
    the boresight instead of the frame edges looks entirely plausible and is
    wrong by hundreds of metres. This runs the viewer's own function under
    node and compares it to ``whiteout.vision.projection`` at the same two
    pixels.
    """
    # Degrees, because that is what a `Pose` carries. Feeding radians here
    # would let the viewer read the field in either unit and still pass, and
    # a revision of it did: see `test_the_footprint_reads_a_pose_in_degrees`.
    cases = [
        (
            {"z": alt, "heading": _PINNED_HEADING_DEG, "pitch": pitch, "cls": name},
            {"hfov": CAMERAS[name].hfov_deg, "vfov": CAMERAS[name].vfov_deg},
        )
        for name, alt, pitch in _PINNED_FOOTPRINTS
    ]
    tail = (
        "var cases = " + json.dumps(cases) + ";\n"
        "process.stdout.write(JSON.stringify(cases.map(function (c) {\n"
        "  return groundFootprint(c[0], c[1]);\n"
        "})));\n"
    )
    got = _run_viewer_block(_footprint_source(), tail)
    assert isinstance(got, list)

    for (name, alt, pitch), drawn in zip(_PINNED_FOOTPRINTS, got, strict=True):
        camera = CAMERAS[name]
        pose = CameraPose(
            lat_deg=ARENA_ORIGIN.lat_deg,
            lon_deg=ARENA_ORIGIN.lon_deg,
            alt_m=alt,
            yaw_deg=_PINNED_HEADING_DEG,
            pitch_deg=pitch,
        )
        assert drawn is not None, f"{name} at {pitch} deg drew no footprint"
        assert drawn["bearing"] == pytest.approx(math.radians(_PINNED_HEADING_DEG))
        assert drawn["near"] == pytest.approx(
            _ground_range_m(camera, pose, float(camera.height)), rel=1e-6
        )
        assert drawn["far"] == pytest.approx(_ground_range_m(camera, pose, 0.0), rel=1e-6)
        assert drawn["halfAngle"] == pytest.approx(math.radians(camera.hfov_deg) / 2.0)


def _class_cameras() -> dict[str, str]:
    """The viewer's own vehicle-class to camera-name map, read off the page.

    Parsed rather than restated because the log spells a class (``quad``) and
    ``whiteout.vision.camera`` spells a camera (``quadcopter``), and a third
    copy of that correspondence is a third thing to keep in step.
    """
    block = re.search(r"var CAMERA_OF_CLASS = \{(.*?)\};", _read("viewer.js"), re.DOTALL)
    assert block is not None, "the viewer's class-to-camera map has moved"
    found = dict(re.findall(r'([\w-]+):\s*"([\w-]+)"', block.group(1)))
    assert found, "no classes were parsed out of the viewer"
    return found


def test_the_footprint_reads_a_pose_in_degrees() -> None:
    """A `Pose`'s heading is degrees, and the wedge has to be drawn there.

    Driven with a pose lifted out of the committed episode rather than one
    built here, because the unit is a property of the log and a hand-built
    case can be written in whichever unit the viewer happens to want. Read as
    radians the arena's headings wrap — 279.2468 becomes 2.787 rad, 240
    degrees off — so the wedge keeps its shape and size and points at the
    wrong water, which is the failure least likely to be caught by eye.

    `whiteout.vision.sightings` settles the unit: it builds the camera pose
    as ``CameraPose(yaw_deg=pose.heading, pitch_deg=pose.pitch, ...)``.
    """
    cameras = _class_cameras()
    record = read_episode_log(BUNDLED_EPISODE)[0]
    poses = [pose for pose in record.observation.poses if pose.cls in cameras]
    assert poses, "the committed episode carries no pose with a camera"

    cases = [
        (
            {"z": pose.z, "heading": pose.heading, "pitch": pose.pitch, "cls": pose.cls},
            {
                "hfov": CAMERAS[cameras[pose.cls]].hfov_deg,
                "vfov": CAMERAS[cameras[pose.cls]].vfov_deg,
            },
        )
        for pose in poses
    ]
    tail = (
        "var cases = " + json.dumps(cases) + ";\n"
        "process.stdout.write(JSON.stringify(cases.map(function (c) {\n"
        "  return groundFootprint(c[0], c[1]);\n"
        "})));\n"
    )
    got = _run_viewer_block(_footprint_source(), tail)
    assert isinstance(got, list)

    # Some of these headings exceed 2*pi, so a viewer reading them as radians
    # does not merely scale the bearing -- it wraps it, and no assertion on a
    # single pose would be enough to say which unit was read.
    assert any(pose.heading > 2 * math.pi for pose in poses), (
        "the committed episode no longer carries a heading that wraps; pick "
        "a record that does, or this test cannot tell the two units apart"
    )
    for pose, drawn in zip(poses, got, strict=True):
        assert drawn is not None, f"{pose.asset_id} drew no footprint"
        assert drawn["bearing"] == pytest.approx(math.radians(pose.heading)), (
            f"{pose.asset_id}'s wedge is drawn at "
            f"{math.degrees(drawn['bearing']) % 360:.1f} deg, not {pose.heading:.1f}"
        )


def test_a_level_camera_stops_where_the_flat_water_plane_does() -> None:
    """The quadcopter looks at the horizon (#109), and the wedge must not.

    Its top row is above the horizontal, so there is no projection to compare
    the far edge to — that is the point. ``project_pixel_to_ground`` refuses
    a fix past the range where the flat plane's own error reaches a tenth of
    it, so the wedge stops there rather than drawing ground the projection
    would not stand behind. The near edge still projects, and is pinned.
    """
    camera = CAMERAS["quadcopter"]
    pose = CameraPose(
        lat_deg=ARENA_ORIGIN.lat_deg,
        lon_deg=ARENA_ORIGIN.lon_deg,
        alt_m=120.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
    )
    tail = (
        "process.stdout.write(JSON.stringify(groundFootprint({ z: 120, heading: 0, pitch: 0 },"
        " { hfov: " + repr(camera.hfov_deg) + ", vfov: " + repr(camera.vfov_deg) + " })));\n"
    )
    drawn = _run_viewer_block(_footprint_source(), tail)
    assert isinstance(drawn, dict)
    assert drawn["far"] == pytest.approx(max_flat_plane_range_m(120.0), rel=1e-9)
    assert drawn["near"] == pytest.approx(
        _ground_range_m(camera, pose, float(camera.height)), rel=1e-6
    )


def test_a_camera_pitched_past_its_own_nadir_starts_underneath_itself() -> None:
    """A 99.4 degree lens at 60 degrees down has the ground below it in frame.

    The near edge is then zero rather than a range behind the asset, and
    without that branch ``h / tan(theta)`` goes negative and the wedge is
    drawn inside out.
    """
    camera = CAMERAS["quadcopter"]
    tail = (
        "process.stdout.write(JSON.stringify(groundFootprint("
        "{ z: 120, heading: 0, pitch: "
        + repr(-60.0)
        + " }, { hfov: "
        + repr(camera.hfov_deg)
        + ", vfov: "
        + repr(camera.vfov_deg)
        + " })));\n"
    )
    drawn = _run_viewer_block(_footprint_source(), tail)
    assert isinstance(drawn, dict)
    assert drawn["near"] == 0.0
    assert drawn["far"] > 0.0


def test_the_viewer_declares_the_cameras_whiteout_vision_publishes() -> None:
    """One table of fields of view, and the page's copy is held to it."""
    js = _read("viewer.js")
    block = re.search(r"var CAMERAS = \{(.*?)\};", js, re.DOTALL)
    assert block is not None, "the viewer's camera table has moved"
    declared = {
        name: (float(hfov), float(vfov))
        for name, hfov, vfov in re.findall(
            r'"([\w-]+)":\s*\{ hfov: ([\d.]+), vfov: ([\d.]+) \}', block.group(1)
        )
    }
    assert declared, "no cameras were parsed out of the viewer"
    assert set(declared) == set(CAMERAS)
    for name, (hfov, vfov) in declared.items():
        assert hfov == CAMERAS[name].hfov_deg
        assert vfov == CAMERAS[name].vfov_deg


def test_the_field_draws_the_ticketed_layers_in_the_ticketed_order() -> None:
    """Issue #20: outline, belief over water, footprints, assets, track.

    Order is the whole of it — belief drawn over the assets hides the fleet,
    and footprints drawn under belief cannot be seen at all.
    """
    js = _without_comments(_read("viewer.js"))
    body = js.split("function drawContent")[1].split("function waterCells")[0]
    calls = [name for name in re.findall(r"draw(Belief|Outline|Footprints|Assets|Track)\(", body)]
    assert calls == ["Belief", "Outline", "Footprints", "Assets", "Track"]


def test_belief_is_drawn_only_over_water() -> None:
    """Nothing may imply the vessel could be on land.

    The cells the canvas fills come from the mask's set bits and from nothing
    else, and the count is checked against the frame before a single quad is
    drawn — a field and a layout that disagree draw nothing rather than draw
    the wrong water.
    """
    js = _without_comments(_read("viewer.js"))
    cells = js.split("function waterCells")[1].split("function outlineEdges")[0]
    assert "if (water[cell])" in cells, "the cell list is not filtered by the water mask"
    belief = js.split("function drawBelief")[1].split("function screenAngle")[0]
    assert "field.cells[i]" in belief, "the quads are not indexed through the water cells"
    assert "codes.length !== field.cells.length" in belief, "a mismatched frame is not refused"


def test_the_belief_ramp_is_used_for_belief_and_nothing_else() -> None:
    """DESIGN.md: no belief-ramp colours outside the belief field."""
    js = _without_comments(_read("viewer.js"))
    uses = [line for line in js.splitlines() if "--belief-" in line]
    assert len(uses) == 1, f"the belief ramp is read in {len(uses)} places: {uses}"
    assert "beliefRamp" in js.split("--belief-")[0].rsplit("function", 1)[-1]
    css = _without_comments(_read("viewer.css"))
    assert "--belief-" not in css, "the belief ramp reached the stylesheet"


def test_nothing_on_the_field_glows() -> None:
    """DESIGN.md's flat rule, and the easiest one to break by accident.

    Named against the canvas properties rather than the bare words: the
    viewer is full of ``Array.filter``, and a guard that fails on that is a
    guard somebody deletes.
    """
    js = _without_comments(_read("viewer.js"))
    for banned in (
        "shadowBlur",
        "shadowColor",
        "shadowOffset",
        "ctx.filter",
        "globalCompositeOperation",
    ):
        assert banned not in js, f"the canvas uses {banned}"
    css = _without_comments(_read("viewer.css"))
    assert "blur(" not in css
    assert "filter:" not in css
