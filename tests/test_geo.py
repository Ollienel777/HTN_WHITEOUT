"""Issue #73's acceptance criteria, one test each.

The decision under test is ``SPEC.md`` §5's: **the frame of record is geodetic
lat/lon, a local East–North frame exists only as a projection for drawing, and
``whiteout/geo.py`` is the one module that converts.**

Two of these are static guards rather than behaviour tests, and deliberately
so. The failure this ticket exists to prevent is not a crash — it is a second
converter appearing in a second module at hour 20, disagreeing with the first
by a few kilometres, and posting a *plausible* lat/lon to the scored API. A
reviewer will not catch that; :mod:`ast` will.

Everything numeric here is evaluated at Bellot Strait's latitude, ~71.99 N.
At the equator the two radii of curvature are within half a percent of each
other, so a projection that confuses East with North, or that drops
:math:`\\cos\\varphi` entirely, round-trips almost perfectly and every test
passes. At 71.99 N it does not.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest

from whiteout import types as record_types
from whiteout.geo import (
    ARENA_ORIGIN,
    WGS84_A,
    WGS84_E2,
    WGS84_F,
    GeoError,
    GeoPoint,
    LocalPoint,
    enu_to_geodetic,
    geodetic_to_local,
    local_to_geodetic,
    radii_of_curvature,
)
from whiteout.types import Contact, Detection, Pose, RecordError

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "whiteout"
THE_ONE_CONVERTER = PACKAGE / "geo.py"
THE_GUARD = Path(__file__).resolve()

#: The strait, ``ARENA.md`` §2: 25 km along the channel and 2 km across it.
#: Offsets are taken to the corners, which is the worst case the arena has.
STRAIT_HALF_LENGTH_M = 12_500.0
STRAIT_HALF_WIDTH_M = 1_000.0


# --- A1: the frame of record is lat/lon, and the log schema matches ---------


def test_no_record_type_carries_a_local_coordinate() -> None:
    """``SPEC.md`` §5: nothing in the episode log is x/y.

    The names are the check. A field called ``x`` in a log that the scorer,
    the viewer and the tracks poster all read is an invitation for one of them
    to treat metres as degrees.
    """
    banned = {"x", "y", "xy", "target_xy", "peak_xy", "east", "north"}
    for name in record_types.__all__:
        member = getattr(record_types, name)
        if not hasattr(member, "__dataclass_fields__"):
            continue
        found = {field.name for field in fields(member)} & banned
        assert not found, f"{name} carries {', '.join(sorted(found))}"


def test_every_position_in_the_log_is_a_geodetic_pair() -> None:
    """A ``lat`` without a ``lon`` is half a position, and vice versa."""
    for name in record_types.__all__:
        member = getattr(record_types, name)
        if not hasattr(member, "__dataclass_fields__"):
            continue
        present = {field.name for field in fields(member)}
        for latitude in sorted(field for field in present if field.endswith("lat")):
            assert f"{latitude[:-3]}lon" in present, f"{name}.{latitude} has no longitude"


def test_a_swapped_lat_lon_at_the_strait_is_refused_at_construction() -> None:
    """The mix-up this ticket exists to prevent, caught where it is made.

    Bellot Strait is 71.99 N, -94.84 E. Swapped, the latitude is -94.84, which
    is not a latitude — so the one frame error that would otherwise post a
    confident wrong answer to the scored API cannot be represented at all.
    """
    for build in (
        lambda lat, lon: Pose("a", "quad", 0.0, lat, lon, 0.0, 0.0, 0.0, 0.0),
        lambda lat, lon: Contact("c", 0.0, "tracked", lat, lon, 0.5, "vessel", None),
        lambda lat, lon: Detection("d", lat, lon, 0.5, "vessel"),
    ):
        build(ARENA_ORIGIN.lat_deg, ARENA_ORIGIN.lon_deg)
        with pytest.raises(RecordError) as caught:
            build(ARENA_ORIGIN.lon_deg, ARENA_ORIGIN.lat_deg)
        assert "-94.84" in str(caught.value)


# --- A2: exactly one module converts ---------------------------------------


#: Directories with no source of ours in them. ``.claude`` is the live one: in
#: the main checkout it holds every agent's gitignored worktree, so a walk that
#: descends into it reads other branches' source as this tree's and reports
#: their offenders here -- 124 files walked instead of 25, most of the failures
#: someone else's. ``SPEC.md`` §6 already excludes ``.claude/**`` from every
#: lint, typecheck and test glob; this is the same rule for the guard's own
#: walk, and nothing of ours is tracked under it.
_NOT_OURS = {
    ".git",
    ".claude",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

#: The two files the guards below skip, and the only two. ``whiteout/geo.py``
#: is the rule; this file is where the rule is written, so it has to be free
#: to spell an ellipsoid constant, name a conversion and run the
#: trigonometry its reference implementation needs. Everything else in the
#: repository is walked -- ``whiteout/``, ``scripts/`` and the rest of
#: ``tests/`` -- because a second converter is no less dangerous for living
#: in a script that seeds the arena or in a test that sets one up.
_EXEMPT = (THE_ONE_CONVERTER, THE_GUARD)


def _is_ours(rel_path: str) -> bool:
    """Whether a repository-relative path is source of ours, and so walked.

    A pure predicate over the path's parts, lifted out of :func:`_sources`
    so that it pins without a filesystem. It is the one change in #82 that
    *removes* files from the guards' reach, so it is the one that most needs
    a test -- and being inline was the only thing that made it look untestable.
    """
    parts = Path(rel_path).parts
    return not (set(parts) & _NOT_OURS) and not any(part.endswith(".egg-info") for part in parts)


def _sources() -> list[Path]:
    """Every Python file in the repository but the converter and this guard."""
    sources = [
        path
        for path in sorted(REPO_ROOT.rglob("*.py"))
        if _is_ours(path.relative_to(REPO_ROOT).as_posix())
    ]
    for required in _EXEMPT:
        assert required in sources, f"{required} is not in the tree"
    # The walk is half of what this guard is worth, so assert that it left
    # the package: a `_sources()` that quietly stopped at `whiteout/` is how
    # the first version of this rule came to be narrower than the sentence it
    # was enforcing.
    reached = {path.relative_to(REPO_ROOT).parts[0] for path in sources}
    assert {"whiteout", "scripts", "tests"} <= reached, f"the walk only reached {reached}"
    return [source for source in sources if source not in _EXEMPT]


#: A metre-scale constant is how a second frame starts, and an allowlist of
#: exact literals is the wrong shape for catching one: the naive
#: ``METRES_PER_DEGREE = 111320.0`` that someone writes in a hurry would
#: never have been on such a list, and as a two-axis converter it disagrees
#: with ``whiteout.geo`` by 30.0 m at the arena's worst corner -- measured,
#: against ``geodetic_to_local``. So the guard is a *range*. An earth radius, a
#: polar semi-axis, a metres-per-degree factor and either circumference all
#: fall in it, and nothing else in this repository does. A computed spelling
#: is usually caught by its first term -- ``6_378_000.0 + 137.0`` is -- but
#: not when the first term is an exempted wire factor: see ``_WIRE_FACTORS``,
#: which is the hole this range knowingly has.
_ELLIPSOID_SCALE_M = (1.0e5, 1.0e8)

#: The values inside that range this guard lets through, #82. Both are
#: dimensionless MAVLink wire factors rather than metres, and the one concern
#: here is one sentence long: **the metre-scale range must not adjudicate
#: MAVLink wire factors.**
#:
#: - ``1e7``: ``GLOBAL_POSITION_INT`` reports ``lat`` and ``lon`` as integers
#:   of 1e-7 degrees, so every adapter above the transport seam divides by it
#:   to get the degrees :mod:`whiteout.geo` and the episode log take. The
#:   range as first written rejected the `arena` and `sitl` transports on
#:   sight -- citing earth radii.
#: - ``1e6``: ``time_usec`` is microseconds, the same class of fact about the
#:   same wire format. It adds no derived-converter risk beyond ``1e7``, for
#:   an exact reason: ``1e6 / 9`` and ``1e7 / 90`` are the same float
#:   (111 111.111), so every metres-per-degree a ``1e6`` exemption can reach,
#:   the ``1e7`` exemption already reached. And nothing geodetic lives near
#:   1e6 -- ``1e6 / 90`` is 11 111 and ``1e6 / 360`` is 2 778; no radius, no
#:   circumference, no quadrant -- unlike 1e7, which sits 2 km from the
#:   meridian quadrant and is exempted anyway. It is here rather than in a
#:   follow-up so that #64 and #68 do not have to open and defend a change to
#:   this file mid-event.
#:
#: Exempted **by value** rather than by narrowing the top of the range, which
#: was the other way to spell this. Narrowing below 1e7 would free the whole
#: decade above it, and that decade is not empty: the equatorial circumference
#: is 40 075 017 m and the polar meridian 40 007 863 m, and either divided by
#: 360 is a metres-per-degree factor -- the second converter this guard exists
#: to stop, reached without ever writing 111320. Exempting two literals costs
#: two literals; narrowing the range costs a decade of them.
#:
#: **What the exemption costs, measured at the strait rather than estimated.**
#: It is not small, and the honest statement is that the guards do not catch
#: what it opens:
#:
#: - ``north / (1e7 / 90)`` is a latitude scale the range can no longer see.
#:   1e7 m is within 2 km of the quadrant of the meridian, so it is only
#:   0.02% off the *global mean* degree -- but this guard exists to stop a
#:   converter disagreeing with :mod:`whiteout.geo`, and at 71.99 N
#:   ``radii_of_curvature`` gives M = 6 393 414.1 m, i.e. 111 586.1 m/deg.
#:   ``1e7 / 90`` = 111 111.1 is **0.426% low: about 4.3 m per km North, and
#:   53.44 m at the arena's 12.5 km half-length.**
#: - **The longitude half does not need a cosine call, so no guard stops it.**
#:   A precomputed ``COS_ORIGIN = 0.3093051`` is neither an ``ast.Attribute``
#:   nor a ``from math import``, and it is far below this range, so the whole
#:   two-axis converter --
#:   ``lat + north / (1e7 / 90)``, ``lon + east / ((1e7 / 90) * COS_ORIGIN)``
#:   -- passes all three guards, with a worst disagreement of **53.63 m** over
#:   the arena. That is within 3 m of the 51 m converter option 1 was rejected
#:   to prevent. Option 2 is still the right call, because it opens one door
#:   and option 1 opened the decade; but the door it opens is ~54 m wide, not
#:   sub-metre. This is the same shape as PR #78's R2-N1 precomputed-cosine
#:   bypass, which is still open, and closing that closes this too.
#:
#: Put plainly, so that the next person to widen this cannot cite a
#: sub-metre cost: **this exemption opens a two-axis converter that is ~54 m
#: out at the arena corner, and none of the three guards catches it.** The
#: naive ``111320.0`` that ``_ELLIPSOID_SCALE_M`` above exists to catch is
#: 30.0 m out -- the guard refuses the less wrong spelling and admits the more
#: wrong one. That is accepted, not overlooked: the alternative was option 1,
#: which frees a whole decade of them.
#:
#: What the exemption does contain: it is these values and not a band. Both
#: neighbours of each are planted as breaches below, and the quadrant's own
#: value, 10 001 966 m, is still in range.
#:
#: Each carries what it is for, so the failure message can name the whole
#: list rather than the first of it, and so a third entry cannot be added
#: without saying which wire field it decodes.
_WIRE_FACTORS: dict[float, str] = {
    1.0e7: "degE7 lat/lon",
    1.0e6: "time_usec microseconds",
}

#: The exemptions as the failure message spells them, derived so it cannot
#: drift from the dict above.
_WIRE_FACTORS_NAMED = ", ".join(f"{value:.0e} ({why})" for value, why in _WIRE_FACTORS.items())

#: The dimensionless half of the ellipsoid -- too small for the range above,
#: and the other way every copy of this arithmetic starts.
_ELLIPSOID_SHAPES = (1.0 / WGS84_F, WGS84_F, WGS84_E2)

#: Substrings that name a frame conversion. A helper called
#: ``_to_latlon`` in the detector is exactly the duplicate this forbids.
_CONVERSION_WORDS = ("geodetic", "latlon", "lat_lon", "to_enu", "enu_to", "wgs84")

#: A converter that dodges both lists above still has to turn a latitude into
#: a scale, and that is trigonometry.
_TRIGONOMETRY = frozenset(
    {"sin", "cos", "tan", "asin", "acos", "atan", "atan2", "radians", "degrees"}
)

#: Repository-relative paths allowed to call trigonometry anyway. Add a path
#: here with a reason rather than shrinking the set above: the entry is the
#: reviewable record that somebody looked at the module and agreed its
#: trigonometry is not a second frame.
#:
#: #72 is the whole list. ``whiteout/vision/camera.py`` turns a field of view
#: into a focal length and ``whiteout/vision/projection.py`` turns an
#: attitude into a ray -- trigonometry on angles that are not latitudes, on a
#: camera rather than an ellipsoid -- and ``tests/test_vision_projection.py``
#: works an arctangent by hand to pin the frame edges. None of the three
#: converts a frame any more: ``projection.py`` imports ``enu_to_geodetic``
#: and the ellipsoid from ``whiteout.geo``, so their geodesy is still covered
#: by the two guards above, which have no allowlist and walk these files.
#:
#: #66 adds ``whiteout/vision/scene.py``, the synthetic-imagery generator.
#: Its trigonometry is entirely inside the image: ``_ellipse`` rotates a
#: painted shape about a pixel centre, and ``_scatter_floes`` turns one sun
#: azimuth per frame into the pixel offset every floe's shadow is cast along.
#: The module imports no position type, names no distance in metres and takes
#: no latitude -- its only import from the package is ``CameraModel``, for a
#: width and a height -- so there is no frame here for a second converter to
#: be. It is not exempted from the two guards above, which walk it and pass:
#: it spells no ellipsoid constant and defines no conversion.
#:
#: #84 adds ``scripts/jpeg_quantisation.py``, whose cosine is the DCT basis --
#: a fixed 8x8 matrix of ``cos((2k+1) j pi / 16)``, with no angle in it that
#: came from anywhere and no coordinate of any kind. It is measurement
#: apparatus and lives in ``scripts/`` for that reason: nothing under
#: ``whiteout/`` imports it, so unlike ``scene.py`` above it cannot become
#: something the product depends on. That distinction is the one worth keeping
#: as this list grows -- an entry for a tool is cheap, an entry for a module
#: the product ships is a standing invitation.
_TRIGONOMETRY_ALLOWED: frozenset[str] = frozenset(
    {
        "whiteout/vision/camera.py",
        "whiteout/vision/projection.py",
        "whiteout/vision/scene.py",
        # One `atan2`, and it is not geodesy. #71 flies the fake fleet toward
        # its commanded waypoint, and a moving asset has to report which way
        # it is pointing. Both endpoints reach `whiteout.geo` first, so the
        # call takes an East and a North *in metres* off the tangent plane
        # that module returned and gives a bearing in radians: no latitude
        # enters it, and no scale leaves it. The check that this stays true
        # is not this list -- it is that the ellipsoid guard and the
        # conversion-name guard both walk this file with no allowlist at all,
        # so a second converter here still has to spell a constant or name
        # itself, and both of those are still refused.
        "whiteout/transport/kinematic.py",
        # Camera-field trigonometry, not latitude trigonometry: it turns a
        # field of view and a height into a depression angle and a ground
        # range, and does its one frame conversion through whiteout.geo like
        # everything else. Same category as camera.py and projection.py
        # above, which is why it sits beside them rather than widening the
        # rule.
        "whiteout/vision/standoff.py",
        "tests/test_vision_projection.py",
        "tests/test_vision_standoff.py",
        # #20 pins the viewer's camera footprint against
        # whiteout.vision.projection, and the pitches it pins it at are
        # written in degrees because that is how ARENA.md publishes them.
        # The `radians` calls are on a camera's attitude and on a field of
        # view; the one position in the file reaches whiteout.geo like every
        # other caller. Same category as the two entries above it, and the
        # file's own subject is a page rather than a frame.
        "tests/test_viz_shell.py",
        "scripts/jpeg_quantisation.py",
        # A rotation between two map grids, not a latitude-to-scale step.
        # Gazebo's world axes are EPSG:3413 grid and grid north is not true
        # north; the probe turns x/y by minus the site's published grid
        # convergence to get East/North, then hands those to
        # whiteout.geo.local_to_geodetic like every other caller. The sin and
        # cos are of a convergence angle the arena reports, never of a
        # latitude, and this file owns no ellipsoid and no second converter.
        # It is also a measurement instrument that nothing under whiteout/
        # imports, so the one-converter rule is not weakened by it.
        "scripts/truth_probe.py",
    }
)


def _defined_names(tree: ast.AST) -> set[str]:
    """Every name a module binds, at any level.

    ``AnnAssign`` and ``AugAssign`` are collected here because they are not
    ``ast.Assign``. An earlier version of this guard walked only the latter,
    so an annotated ``WGS84_E2: float = 0.00669...`` was invisible to it.
    """
    defined: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    return defined


#: Each guard below takes ``(where, text)`` rather than reading the file
#: itself, so that the planted breaches at the end of this section run the
#: same code the walk runs. Every failure message names its guard and what
#: that guard refuses, because the reader who hits one is holding a diff and
#: not this file -- #82 exists because "an earth radius or a metres-per-degree
#: factor" sent the first person to hit it looking for geodesy they had not
#: written.
def _check_the_ellipsoid(where: str, text: str) -> None:
    """A second copy of the constants is a second answer to the same question."""
    low, high = _ELLIPSOID_SCALE_M
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, int | float):
            continue
        if isinstance(node.value, bool):
            continue
        value = abs(float(node.value))
        if value in _WIRE_FACTORS:
            continue
        assert not low <= value <= high, (
            f"{where} line {node.lineno} spells out {node.value!r}. The ellipsoid "
            f"guard (tests/test_geo.py::test_only_whiteout_geo_defines_the_ellipsoid) "
            f"fails any literal whose magnitude falls in [{low:.0e}, {high:.0e}], because "
            f"a constant that size is an earth radius, a circumference or a "
            f"metres-per-degree factor -- and the ellipsoid lives in whiteout/geo.py "
            f"and nowhere else. Its only exemptions are {_WIRE_FACTORS_NAMED}. If "
            f"yours is another dimensionless wire factor rather than a second frame, "
            f"name it beside those in test_geo.py's _WIRE_FACTORS; do not widen "
            f"the range"
        )
        for shape in _ELLIPSOID_SHAPES:
            assert not math.isclose(value, shape, rel_tol=1e-6), (
                f"{where} line {node.lineno} spells out {node.value!r}. The ellipsoid "
                f"guard (tests/test_geo.py::test_only_whiteout_geo_defines_the_ellipsoid) "
                f"also fails the dimensionless half -- f, 1/f and e^2, to a relative "
                f"1e-6 -- and the ellipsoid lives in whiteout/geo.py and nowhere else"
            )


def _check_a_frame_conversion_name(where: str, text: str) -> None:
    """Importing the one converter is fine; defining a second one is not."""
    for name in sorted(_defined_names(ast.parse(text))):
        lowered = name.lower()
        for word in _CONVERSION_WORDS:
            assert word not in lowered, (
                f"{where} defines {name!r}. The frame-conversion-name guard "
                f"(tests/test_geo.py::test_only_whiteout_geo_defines_a_frame_conversion) "
                f"fails any bound name containing {', '.join(_CONVERSION_WORDS)}, "
                f"matched without regard to case -- whiteout/geo.py is the one "
                f"converter, and every other caller imports it"
            )


def _check_the_trigonometry(where: str, text: str) -> None:
    """The check the two lists above cannot make.

    A duplicate converter can be spelt with constants that are on no list and
    a name that reads as innocent -- ``offset_position`` over a
    ``METRES_PER_DEGREE`` it computed -- but it cannot avoid a sine or a
    cosine, because that is what turns a latitude into a scale. Nothing else
    in this repository has any reason to call one.
    """
    if where in _TRIGONOMETRY_ALLOWED:
        return
    named = ", ".join(sorted(_TRIGONOMETRY))
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute) and node.attr in _TRIGONOMETRY:
            raise AssertionError(
                f"{where} line {node.lineno} calls {node.attr}(). The trigonometry "
                f"guard (tests/test_geo.py::"
                f"test_nothing_but_whiteout_geo_does_trigonometry_on_a_latitude) "
                f"fails any of {named} outside _TRIGONOMETRY_ALLOWED -- only "
                f"whiteout/geo.py turns a latitude into a scale"
            )
        if isinstance(node, ast.ImportFrom) and node.module in {"math", "numpy"}:
            for alias in node.names:
                assert alias.name not in _TRIGONOMETRY, (
                    f"{where} line {node.lineno} imports {alias.name}. The "
                    f"trigonometry guard (tests/test_geo.py::"
                    f"test_nothing_but_whiteout_geo_does_trigonometry_on_a_latitude) "
                    f"fails any of {named} outside _TRIGONOMETRY_ALLOWED -- only "
                    f"whiteout/geo.py turns a latitude into a scale"
                )


#: Every static guard this section runs over the tree. The planted breaches
#: below are checked against this tuple, and :func:`_walk` refuses a guard
#: that is not in it while :func:`test_every_walking_guard_is_registered`
#: refuses a ``_check_*`` that never joined it -- so a guard added later
#: cannot land without a planted breach, whether by being dropped from this
#: tuple or by never being added to it.
_GUARDS: tuple[Callable[[str, str], None], ...] = (
    _check_the_ellipsoid,
    _check_a_frame_conversion_name,
    _check_the_trigonometry,
)


def _walk(guard: Callable[[str, str], None]) -> None:
    # The walking tests below name their guard directly rather than looping
    # over `_GUARDS`, because each failure message embeds its own test name
    # and pytest has to be able to run one of them. That left `_GUARDS` a
    # registry nothing consulted: dropping an entry went red, but *never
    # adding* one did not. Driving the walk through this assertion, and
    # `test_every_walking_guard_is_registered` below, closes the other half.
    assert guard in _GUARDS, (
        f"{guard.__name__} is walked over the tree but is not in _GUARDS, so "
        f"test_every_guard_owns_a_planted_breach never asked it for a planted "
        f"breach. Add it to _GUARDS and plant one"
    )
    for source in _sources():
        guard(source.relative_to(REPO_ROOT).as_posix(), source.read_text(encoding="utf-8"))


def test_only_whiteout_geo_defines_the_ellipsoid() -> None:
    """A second copy of the constants is a second answer to the same question."""
    _walk(_check_the_ellipsoid)


def test_only_whiteout_geo_defines_a_frame_conversion() -> None:
    """Importing the one converter is fine; defining a second one is not."""
    _walk(_check_a_frame_conversion_name)


def test_nothing_but_whiteout_geo_does_trigonometry_on_a_latitude() -> None:
    """A converter that dodges both lists still has to turn a latitude into a scale."""
    _walk(_check_the_trigonometry)


def test_every_walking_guard_is_registered() -> None:
    """A fourth ``_check_*`` cannot land outside ``_GUARDS``.

    ``_walk`` asserts the guard it is handed is registered, but only for
    guards somebody remembered to walk. This is the other direction: every
    ``_check_*`` this module defines must be in the tuple, so that
    :func:`test_every_guard_owns_a_planted_breach` sees it and demands a
    planted breach for it.
    """
    defined = {
        value for name, value in globals().items() if name.startswith("_check_") and callable(value)
    }
    missing = sorted(guard.__name__ for guard in defined - set(_GUARDS))
    assert not missing, f"defined but not in _GUARDS, so never asked for a breach: {missing}"


def test_the_exclusion_of_other_agents_worktrees_holds() -> None:
    """``.claude`` is the one change here that *removes* files from the walk.

    It needs no filesystem to pin: :func:`_is_ours` is a pure predicate over
    a path's parts. Getting this wrong in the permissive direction is how the
    walk came to read another branch's source as this tree's (PR #78's R2-M2);
    getting it wrong in the strict direction would silently unguard a
    directory. ``.claude/**`` is outside every ruff, mypy and pytest glob in
    ``SPEC.md`` §6, so a converter landing there would be unguarded by
    everything at once.
    """
    assert not _is_ours(".claude/worktrees/agent-x/whiteout/geo.py")
    assert not _is_ours(".claude/hooks/session_start.py")
    assert not _is_ours("whiteout.egg-info/thing.py")
    assert not _is_ours(".venv/lib/python3.12/site-packages/pymavlink/mavutil.py")
    assert _is_ours("whiteout/geo.py")
    assert _is_ours("scripts/gate.py")
    assert _is_ours("tests/test_geo.py")
    # Not a prefix match: a directory merely *starting* with a skipped name
    # is ours, and so is a file whose name happens to contain one.
    assert _is_ours("whiteout/build_plan.py")
    assert _is_ours("scripts/claude_report.py")


#: What a MAVLink adapter above the transport seam actually writes: degE7 in,
#: degrees out, and not one line of geodesy. #64 and #68 are next to write it,
#: and ``SPEC.md`` §3 and §9 make it demo beat 6.
_A_MAVLINK_ADAPTER = '''\
"""The `arena` transport's position decoding, in the shape #64 will write it."""

from whiteout.geo import GeoPoint

#: `GLOBAL_POSITION_INT` reports lat/lon as integers of 1e-7 degrees.
DEGE7 = 1e7

#: `time_usec` is microseconds since boot.
USEC_PER_S = 1e6


def position_of(message: object) -> tuple[float, GeoPoint]:
    return (
        message.time_usec / USEC_PER_S,
        GeoPoint(message.lat / DEGE7, message.lon / 1e7),
    )
'''


def test_a_mavlink_adapter_may_spell_the_wire_factors() -> None:
    """#82's acceptance: degE7 in ``whiteout/transport/`` passes all three guards.

    The range contains both 1e7 and 1e6, so before the exemption this file
    failed the gate with a message about earth radii, and the `arena` and
    `sitl` adapters were blocked on sight. This runs the guards the walk
    runs, at the path the adapter lands on, which is not in any allowlist.
    """
    for guard in _GUARDS:
        guard("whiteout/transport/mavlink.py", _A_MAVLINK_ADAPTER)


#: Source each guard must reject, and the guard that owns it. The idiom is
#: ``tests/test_transport_conformance.py``'s ``_PLANTED_BREACHES``: a guard
#: nothing is planted against is a guard nobody has shown to be load-bearing,
#: and the gap is invisible for as long as the tree happens to be clean --
#: which is exactly how long it takes for the exemption above to be widened
#: into a hole nobody measures.
#:
#: Every spelling PR #78's reviews defeated a guard with is here, so that
#: closing the degE7 hole cannot quietly reopen one: the naive
#: metres-per-degree, a computed constant, an ``AnnAssign``, and a path
#: outside ``whiteout/``. Both neighbours of both wire factors are here too,
#: because "allow 1e7 and 1e6" must mean two values and not two
#: neighbourhoods.
_PLANTED_BREACHES: tuple[tuple[str, str, Callable[[str, str], None]], ...] = (
    ("a second semi-major axis", "WGS84_A = 6378137.0\n", _check_the_ellipsoid),
    ("the naive metres-per-degree", "METRES_PER_DEGREE = 111320.0\n", _check_the_ellipsoid),
    (
        "the equatorial circumference, which narrowing the range would have freed",
        "EARTH_CIRCUMFERENCE_M = 40075016.686\n",
        _check_the_ellipsoid,
    ),
    (
        "the polar meridian, the other way to a metres-per-degree",
        "MERIDIAN_M = 40007862.917\n",
        _check_the_ellipsoid,
    ),
    (
        "an annotated mean radius, which an ast.Assign-only collection would miss",
        "EARTH_R: float = 6371008.7714\n",
        _check_the_ellipsoid,
    ),
    (
        "a computed axis, caught by its first term",
        "AXIS = 6_378_000.0 + 137.0\n",
        _check_the_ellipsoid,
    ),
    ("the dimensionless half", "E2: float = 0.0066943799901413165\n", _check_the_ellipsoid),
    (
        "the upper neighbour of the degE7 exemption, which is one value and not a gap",
        "SCALE = 10000001.0\n",
        _check_the_ellipsoid,
    ),
    (
        "the lower neighbour of the degE7 exemption",
        "SCALE = 9999999.0\n",
        _check_the_ellipsoid,
    ),
    (
        "the upper neighbour of the time_usec exemption",
        "SCALE = 1000001.0\n",
        _check_the_ellipsoid,
    ),
    (
        "the lower neighbour of the time_usec exemption",
        "SCALE = 999999.0\n",
        _check_the_ellipsoid,
    ),
    (
        "the quadrant of the meridian, which 1e7 only approximates",
        "QUADRANT_M = 10001965.729\n",
        _check_the_ellipsoid,
    ),
    (
        "a second converter by name",
        "def to_latlon(east, north):\n    return east, north\n",
        _check_a_frame_conversion_name,
    ),
    (
        "a conversion named by an annotated assignment",
        "enu_to_geodetic_scale: float = 2.0\n",
        _check_a_frame_conversion_name,
    ),
    (
        "a converter with innocent names over constants it computed",
        "import math\n\n\ndef offset(lat, east):\n    return lat + east / math.cos(lat)\n",
        _check_the_trigonometry,
    ),
    (
        "trigonometry imported rather than reached through the module",
        "from math import radians\n",
        _check_the_trigonometry,
    ),
)


@pytest.mark.parametrize(
    ("source", "guard"),
    [(source, guard) for _, source, guard in _PLANTED_BREACHES],
    ids=[name for name, _, _ in _PLANTED_BREACHES],
)
def test_a_guard_fails_the_source_it_exists_to_catch(
    source: str, guard: Callable[[str, str], None]
) -> None:
    """Planted in ``whiteout/transport/``, the directory #82 unblocks."""
    with pytest.raises(AssertionError):
        guard("whiteout/transport/probe.py", source)


def test_a_guard_failure_names_the_guard_and_what_it_refuses() -> None:
    """#82's acceptance: a hit is diagnosable without opening this file.

    The message that sent #82's author hunting for geodesy they had not
    written said only "an earth radius or a metres-per-degree factor". A
    reader holding a diff needs the guard's name, the rule it applies, and
    the exemption that exists, in the failure itself.
    """
    low, high = _ELLIPSOID_SCALE_M
    with pytest.raises(AssertionError) as caught:
        _check_the_ellipsoid("whiteout/transport/probe.py", "METRES_PER_DEGREE = 111320.0\n")
    message = str(caught.value)
    assert "test_only_whiteout_geo_defines_the_ellipsoid" in message, message
    assert f"[{low:.0e}, {high:.0e}]" in message, message
    # The whole exemption list, not just its first entry: a reader who hits
    # this while decoding `time_usec` has to be told 1e6 is already allowed.
    for value, why in _WIRE_FACTORS.items():
        assert f"{value:.0e}" in message, message
        assert why in message, message
    assert "_WIRE_FACTORS" in message, message

    # The other two guards are pinned on their test name too, not merely on
    # "tests/test_geo.py::" -- otherwise a regression that dropped the name
    # and kept the path would pass here (R1-N1).
    for guard, source, named in (
        (
            _check_a_frame_conversion_name,
            "def to_latlon(east, north):\n    return east\n",
            "test_only_whiteout_geo_defines_a_frame_conversion",
        ),
        (
            _check_the_trigonometry,
            "from math import radians\n",
            "test_nothing_but_whiteout_geo_does_trigonometry_on_a_latitude",
        ),
    ):
        with pytest.raises(AssertionError) as caught:
            guard("whiteout/transport/probe.py", source)
        message = str(caught.value)
        assert f"tests/test_geo.py::{named}" in message, message


def test_the_name_guard_says_it_matches_regardless_of_case() -> None:
    """``Geodetic`` is rejected by a guard that lists only ``geodetic`` (R1-N2).

    The match is on ``name.lower()``, so a reader who sees a capitalised name
    refused has to open this file to learn why unless the message says so.
    """
    with pytest.raises(AssertionError) as caught:
        _check_a_frame_conversion_name("whiteout/transport/probe.py", "class GeodeticFix: ...\n")
    message = str(caught.value)
    assert "GeodeticFix" in message, message
    assert "case" in message.lower(), message


def test_every_guard_owns_a_planted_breach() -> None:
    """A guard nothing is planted against has not been shown able to fail."""
    covered = {guard for _, _, guard in _PLANTED_BREACHES}
    unguarded = sorted(guard.__name__ for guard in _GUARDS if guard not in covered)
    assert not unguarded, f"no planted breach for: {unguarded}"
    unknown = sorted(guard.__name__ for guard in covered if guard not in _GUARDS)
    assert not unknown, f"a breach is planted against a guard the walk does not run: {unknown}"


def test_the_transport_takes_its_positions_from_the_one_converter() -> None:
    """The stub fleet is placed in metres and reported in lat/lon.

    Not a style point: the adapter is the first thing above the seam that has
    to name a position, and it is where a private ``x + 0.0001`` frame would
    otherwise be born.
    """
    source = (PACKAGE / "transport" / "kinematic.py").read_text(encoding="utf-8")
    assert "from whiteout.geo import" in source


# --- A3: the round trip, at the strait's latitude ---------------------------


def test_geodetic_to_local_and_back_round_trips_across_the_strait() -> None:
    """Every corner of the arena, out and back, to floating-point noise.

    The tolerance is 1e-6 m, and it is **not** a claim about the tangent-plane
    approximation. That model error is metres at this range, not microns —
    2.2 m at 3 km East, 38 m at the corner — and it cancels exactly here,
    because both directions read their radii from the same origin latitude.
    What this measures is that the inverse is the inverse;
    ``test_the_tangent_plane_error_is_the_one_the_module_documents`` measures
    the model.
    """
    for east in (-STRAIT_HALF_LENGTH_M, -1.0, 0.0, 1.0, STRAIT_HALF_LENGTH_M):
        for north in (-STRAIT_HALF_WIDTH_M, 0.0, STRAIT_HALF_WIDTH_M):
            offset = LocalPoint(east, north)
            there = local_to_geodetic(ARENA_ORIGIN, offset)
            back = geodetic_to_local(ARENA_ORIGIN, there)
            assert back.east_m == pytest.approx(east, abs=1e-6)
            assert back.north_m == pytest.approx(north, abs=1e-6)


def test_a_geodetic_position_round_trips_through_the_local_frame() -> None:
    """The other direction: lat/lon in, lat/lon out, at 71.99 N."""
    for lat, lon in ((71.9965, -94.8448), (71.9975, -94.8450), (71.9820, -94.9000)):
        point = GeoPoint(lat, lon)
        back = local_to_geodetic(ARENA_ORIGIN, geodetic_to_local(ARENA_ORIGIN, point))
        assert back.lat_deg == pytest.approx(lat, abs=1e-12)
        assert back.lon_deg == pytest.approx(lon, abs=1e-12)


#: WGS-84 radii of curvature at 71.99 N, in metres, computed from the
#: defining constants (a = 6378137, 1/f = 298.257223563) at 40 significant
#: figures, outside this codebase. ``M = a(1 - e^2) / w^{3/2}`` and
#: ``N = a / w^{1/2}`` with ``w = 1 - e^2 sin^2 phi``.
WGS84_MERIDIONAL_AT_THE_STRAIT_M = 6393414.135
WGS84_PRIME_VERTICAL_AT_THE_STRAIT_M = 6397533.132


def test_the_longitude_convergence_at_the_strait_is_real() -> None:
    """~0.31, the number ``ARENA.md`` and #73 both quote.

    This is the test that would pass at the equator and must not. A degree of
    longitude at Bellot Strait is a third of a degree of latitude, so a
    projection that drops the cosine, or that hands East to the latitude
    slot, lands about 2 km out at the far end of the channel — a plausible
    position, in the water, wrong.
    """
    meridional, east = radii_of_curvature(ARENA_ORIGIN.lat_deg)
    assert east / meridional == pytest.approx(0.31, abs=0.005)


def test_the_radii_at_the_strait_are_the_wgs84_ones_to_the_metre() -> None:
    """The ratio above is not enough, because a sphere has the same one.

    A sphere of radius ``a`` gives ``east / meridional = 0.309`` at 71.99 N —
    within half a percent of the ellipsoid's 0.309 — so the ratio, and the
    round trip, and the equator check this replaced, all stay green while
    ``local_to_geodetic(ARENA_ORIGIN, LocalPoint(12500, 1000))`` moves 38 m.
    Absolute radii to the metre are what separates the two: a sphere is 15 km
    out on ``M`` and 19 km out on ``N``.
    """
    meridional, east = radii_of_curvature(ARENA_ORIGIN.lat_deg)
    prime_vertical = east / math.cos(math.radians(ARENA_ORIGIN.lat_deg))
    assert meridional == pytest.approx(WGS84_MERIDIONAL_AT_THE_STRAIT_M, abs=1.0)
    assert prime_vertical == pytest.approx(WGS84_PRIME_VERTICAL_AT_THE_STRAIT_M, abs=1.0)


def _exact_offset(lat_deg: float, lon_deg: float, east_m: float, north_m: float) -> GeoPoint:
    """The reference ``whiteout.geo`` approximates: ENU → ECEF → geodetic.

    No tangent plane and no dropped cross term — the ENU displacement is
    rotated into earth-centred cartesian, added, and converted back by
    iterating Bowring's relation to convergence. It agrees with a Vincenty
    direct solution along azimuth 90 to under a centimetre at 10 km, which is
    two independent references for the table in ``whiteout/geo.py``.

    This is a second piece of geodesy in the repository, which is what the A2
    guards forbid — and it is why they exempt this file and no other. A
    claim about an approximation's error cannot be tested without something
    exact to test it against.
    """
    phi, lam = math.radians(lat_deg), math.radians(lon_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_phi * sin_phi)
    x = prime_vertical * cos_phi * cos_lam - sin_lam * east_m - sin_phi * cos_lam * north_m
    y = prime_vertical * cos_phi * sin_lam + cos_lam * east_m - sin_phi * sin_lam * north_m
    z = prime_vertical * (1.0 - WGS84_E2) * sin_phi + cos_phi * north_m

    radius = math.hypot(x, y)
    latitude = math.atan2(z, radius * (1.0 - WGS84_E2))
    for _ in range(100):
        curvature = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(latitude) ** 2)
        height = radius / math.cos(latitude) - curvature
        turned = math.atan2(z, radius * (1.0 - WGS84_E2 * curvature / (curvature + height)))
        if abs(turned - latitude) < 1e-16:
            latitude = turned
            break
        latitude = turned
    return GeoPoint(math.degrees(latitude), math.degrees(math.atan2(y, x)))


def _metres_apart(one: GeoPoint, other: GeoPoint) -> float:
    """Straight-line metres between two nearby positions, through the earth.

    Chord rather than arc: over the tens of metres this is used for, the two
    differ by less than a picometre.
    """
    places = []
    for point in (one, other):
        phi, lam = math.radians(point.lat_deg), math.radians(point.lon_deg)
        prime_vertical = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(phi) ** 2)
        places.append(
            (
                prime_vertical * math.cos(phi) * math.cos(lam),
                prime_vertical * math.cos(phi) * math.sin(lam),
                prime_vertical * (1.0 - WGS84_E2) * math.sin(phi),
            )
        )
    return math.dist(*places)


#: ``whiteout/geo.py``'s "How wrong the model is" table, to the millimetre.
#: The docstring is the contract #66 and #67 are being written against, so it
#: is a tested number rather than an adjective. An earlier draft of that
#: paragraph claimed "under a centimetre at 3 km" — 500x out on the East
#: axis, which is the one *accuracy* is scored on.
_DOCUMENTED_ERROR_M = (
    (0.0, 3_000.0, 0.004),
    (0.0, 10_000.0, 0.055),
    (500.0, 0.0, 0.060),
    (1_000.0, 0.0, 0.240),
    (1_000.0, 1_000.0, 0.538),
    (2_000.0, 0.0, 0.962),
    (3_000.0, 0.0, 2.164),
    (10_000.0, 0.0, 24.039),
    (STRAIT_HALF_LENGTH_M, STRAIT_HALF_WIDTH_M, 38.033),
)


def test_the_tangent_plane_error_is_the_one_the_module_documents() -> None:
    """The approximation's error, measured, not asserted in prose.

    The round-trip tests above cannot see this: both directions share the
    model, so its error cancels exactly. Only an independent reference shows
    it, and it is badly asymmetric — a few millimetres along the meridian,
    38 m at the far corner of the strait along the parallel.
    """
    for east, north, documented in _DOCUMENTED_ERROR_M:
        approximate = local_to_geodetic(ARENA_ORIGIN, LocalPoint(east, north))
        exact = _exact_offset(ARENA_ORIGIN.lat_deg, ARENA_ORIGIN.lon_deg, east, north)
        assert _metres_apart(approximate, exact) == pytest.approx(documented, abs=0.001), (
            f"the error at ({east} E, {north} N) is not the {documented} m "
            f"whiteout/geo.py documents"
        )


def test_the_projection_error_is_negligible_only_near_its_origin() -> None:
    """The rule ``whiteout/geo.py`` asks callers to follow, as a test.

    A camera converting its own detection is projecting about an origin a
    kilometre or so away, where the error is under a metre. Offsetting
    :data:`ARENA_ORIGIN` ten kilometres down the channel is a different
    thing, and this is the assertion that says so out loud rather than
    leaving it in a docstring.
    """
    origin = ARENA_ORIGIN
    near = local_to_geodetic(origin, LocalPoint(1_000.0, 1_000.0))
    assert _metres_apart(near, _exact_offset(origin.lat_deg, origin.lon_deg, 1e3, 1e3)) < 1.0

    far = local_to_geodetic(origin, LocalPoint(10_000.0, 0.0))
    assert _metres_apart(far, _exact_offset(origin.lat_deg, origin.lon_deg, 1e4, 0.0)) > 20.0


def test_a_thousand_metres_east_is_not_a_thousand_metres_north() -> None:
    """The same offset in the two axes moves a different number of degrees."""
    east_only = local_to_geodetic(ARENA_ORIGIN, LocalPoint(1000.0, 0.0))
    north_only = local_to_geodetic(ARENA_ORIGIN, LocalPoint(0.0, 1000.0))
    moved_lon = abs(east_only.lon_deg - ARENA_ORIGIN.lon_deg)
    moved_lat = abs(north_only.lat_deg - ARENA_ORIGIN.lat_deg)
    assert moved_lon / moved_lat == pytest.approx(1.0 / 0.31, rel=0.02)


def test_enu_to_geodetic_is_the_positional_form_of_local_to_geodetic() -> None:
    """The camera projection calls the positional one; they must not diverge."""
    positional = enu_to_geodetic(ARENA_ORIGIN.lat_deg, ARENA_ORIGIN.lon_deg, 750.0, -250.0)
    typed = local_to_geodetic(ARENA_ORIGIN, LocalPoint(750.0, -250.0))
    assert positional == typed


# --- A4: nothing leaves in any frame but lat/lon ----------------------------


def test_the_type_that_leaves_for_the_tracks_api_carries_only_lat_lon() -> None:
    """``ARENA.md`` §5's body is ``{"name", "lat", "lon"}`` and nothing else.

    :class:`~whiteout.geo.GeoPoint` is what the camera projection returns and
    what the tracks poster (#67) takes, so a metre, a pixel or an altitude
    cannot reach the scored API without someone widening this type first.
    """
    assert [field.name for field in fields(GeoPoint)] == ["lat_deg", "lon_deg"]


def test_a_local_offset_is_not_a_position() -> None:
    """:class:`~whiteout.geo.LocalPoint` has no lat/lon to post by mistake."""
    assert [field.name for field in fields(LocalPoint)] == ["east_m", "north_m"]


def test_an_impossible_position_is_refused_rather_than_clamped() -> None:
    for lat, lon in ((91.0, 0.0), (-90.5, 0.0), (0.0, 181.0), (float("nan"), 0.0)):
        with pytest.raises(GeoError):
            GeoPoint(lat, lon)


def test_a_pole_has_no_local_frame() -> None:
    """The East radius vanishes, so metres East stop meaning degrees."""
    with pytest.raises(GeoError):
        geodetic_to_local(GeoPoint(90.0, 0.0), GeoPoint(89.0, 10.0))
