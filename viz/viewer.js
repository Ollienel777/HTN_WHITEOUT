/* viewer.js — the shell's state machine, log reader and replay clock.
 *
 * SPEC.md section 5: static HTML plus a <canvas>, vanilla JS, no build step,
 * opens from file://. So this is a classic script publishing one global, not
 * a module: a module fetch fails CORS on a file:// origin, and the viewer has
 * to open on someone else's laptop at a sponsor booth with nothing installed.
 *
 * The seven states DESIGN.md requires, and where each one lives:
 *
 *   1 no log loaded   field overlay, [data-view="empty"]  — explains, and
 *                     offers the bundled episode
 *   2 log loading     field overlay, [data-view="loading"] plus the
 *                     [data-skeleton] blocks in the topbar and the rail — a
 *                     skeleton in the shape of the final layout, no spinner
 *   3 log malformed   field overlay, [data-view="error"] — names the failing
 *                     line and the schema version
 *   4 episode running the shell, with the transport playing
 *   5 episode finished the shell, transport parked at the last tick
 *   6 zero contacts   the rail's contacts panel, which says what would post one
 *   7 many contacts   the rail's contacts panel scrolls; the field does not
 *
 * The field draws in one pass, back to front: the chrome (the extent
 * graticule and the scale bar), the belief field over the water the vessel
 * could be in, and the fleet over that. #20, scoped to those two layers —
 * camera footprints, the contact track and its age are the rest of that
 * ticket and are not here.
 *
 * Three things the original #20 asked for are deliberately absent, because
 * the arena turned out not to have them: there is no terrain hillshade (the
 * sponsor renders Bellot Strait themselves), no cut chords (#14 is closed, a
 * min-cut over one channel is degenerate) and no truth marker (the arena
 * gives no ground truth, so drawing one would be drawing a number we do not
 * have).
 */
(function (global) {
  "use strict";

  var doc = global.document;

  /* Kept in step with whiteout.log.SCHEMA_VERSION by a test: the viewer must
   * reject a log this build cannot read, with the same version in the message
   * the Python reader would have printed. */
  var SCHEMA_VERSION = 6;

  /* Relative, so it resolves the same from `whiteout serve` and from file://
   * (where the fetch is blocked, and the file picker takes over). */
  var BUNDLED_EPISODE = "../fixtures/episodes/demo.jsonl";

  var AXES = [
    { key: "coverage", label: "coverage" },
    { key: "collaboration", label: "collaboration" },
    { key: "efficiency", label: "efficiency" },
    { key: "tracking_accuracy", label: "accuracy" }
  ];

  var RECORD_FIELDS = [
    "schema_version", "t", "observation", "intent", "belief_digest",
    "contacts", "truth", "refusals", "belief_field"
  ];

  /* The em dash the shell shows wherever the episode log does not carry a
   * value. A zero would be a number the system never produced. */
  var UNKNOWN = "—";

  /* How many colours the belief ramp is resolved to. Finer than the eye
   * separates in a 100 m cell, and a quarter of the 255 the log quantises to,
   * so the ramp is never what loses a gradient. */
  var PALETTE_STEPS = 64;

  var state = {
    phase: "empty",          /* empty | loading | error | ready */
    records: [],
    index: 0,
    playing: false,
    speed: 1,
    source: "",
    selected: null,
    extent: 1000,            /* metres of half-width the field frames */
    extentNorth: 1000,       /* the same across the channel, which is far less */
    origin: null,            /* the drawing frame's origin, as {lat, lon} */
    lattice: null,           /* the belief cells' corners, in metres */
    failure: null            /* { line, reason } */
  };

  var el = {};
  var clock = { last: 0, carry: 0, frame: 0 };

  /* ── tokens ──────────────────────────────────────────────────────── */

  function token(name) {
    return global.getComputedStyle(doc.documentElement).getPropertyValue(name).trim();
  }

  /* `--t-12` is `12px/16px`; canvas wants `12px <family>`. Reading the step
   * rather than spelling it keeps the type scale in tokens.css, including for
   * the text drawn on the canvas. */
  function canvasFont(step, family) {
    return token(step).split("/")[0] + " " + token(family);
  }

  /* A length token as a number, for the canvas, which takes no units. */
  function size(name) {
    return parseFloat(token(name)) || 0;
  }

  function reducedMotion() {
    return global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  /* ── log reading ─────────────────────────────────────────────────── */

  function LogError(line, reason) {
    this.name = "LogError";
    this.line = line;
    this.reason = reason;
    this.message = "line " + line + ": " + reason;
  }
  LogError.prototype = Object.create(Error.prototype);

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  /* Mirrors whiteout.log.validate_line, message shape included: a reader who
   * has seen the Python error must recognise this one. */
  function validateLine(raw, number) {
    var record;
    try {
      record = JSON.parse(raw);
    } catch (error) {
      throw new LogError(number, "not valid JSON: " + error.message);
    }
    if (!isObject(record)) {
      throw new LogError(number, "record: expected an object");
    }
    if (record.schema_version !== SCHEMA_VERSION) {
      throw new LogError(
        number,
        "schema_version " + JSON.stringify(record.schema_version) +
          " is not supported, expected " + SCHEMA_VERSION
      );
    }
    var missing = RECORD_FIELDS.filter(function (name) {
      return !Object.prototype.hasOwnProperty.call(record, name);
    });
    if (missing.length) {
      throw new LogError(number, "record: missing field(s) " + missing.join(", "));
    }
    var unknown = Object.keys(record).filter(function (name) {
      return RECORD_FIELDS.indexOf(name) === -1;
    }).sort();
    if (unknown.length) {
      throw new LogError(number, "record: unknown field(s) " + unknown.join(", "));
    }
    if (typeof record.t !== "number" || !isFinite(record.t)) {
      throw new LogError(number, "record.t: expected a finite number");
    }
    if (!isObject(record.observation) || !Array.isArray(record.observation.poses)) {
      throw new LogError(number, "record.observation.poses: expected an array");
    }
    if (!Array.isArray(record.contacts)) {
      throw new LogError(number, "record.contacts: expected an array");
    }
    return record;
  }

  function parseEpisodeLog(text) {
    var lines = text.split("\n");
    var records = [];
    for (var i = 0; i < lines.length; i += 1) {
      var raw = lines[i].replace(/\r$/, "");
      if (raw === "") { continue; }
      records.push(validateLine(raw, i + 1));
    }
    if (!records.length) {
      throw new LogError(1, "the episode log is empty");
    }
    return records;
  }

  /* ── state transitions ───────────────────────────────────────────── */

  function setPhase(phase) {
    state.phase = phase;
    doc.body.setAttribute("data-state", phase);
    var loading = phase === "loading";
    each("[data-skeleton]", function (node) { node.hidden = !loading; });
    each("[data-view]", function (node) {
      node.hidden = node.getAttribute("data-view") !== phase;
    });
    /* The topbar and the rail carry the same skeleton, so their real content
     * steps aside for it rather than showing half-filled. */
    el.dialRow.hidden = loading;
    each(".panel-body:not([data-skeleton])", function (node) { node.hidden = loading; });
    render();
  }

  function loadText(text, source) {
    state.source = source;
    try {
      state.records = parseEpisodeLog(text);
    } catch (error) {
      state.records = [];
      state.failure = { line: error.line, reason: error.reason };
      stop();
      setPhase("error");
      return;
    }
    state.failure = null;
    measureExtent();
    state.index = 0;
    state.selected = null;
    state.playing = false;
    setPhase("ready");
  }

  function loadUrl(url) {
    state.source = url;
    setPhase("loading");
    global.fetch(url, { cache: "no-store" }).then(function (response) {
      if (!response.ok) {
        throw new Error("the server answered " + response.status + " " + response.statusText);
      }
      return response.text();
    }).then(function (text) {
      loadText(text, url);
    }).catch(function (error) {
      state.records = [];
      state.failure = {
        line: null,
        reason: global.location.protocol === "file:"
          ? "a page opened from file:// may not read " + url +
            ", because the file:// origin is opaque. Open a log with the file " +
            "picker, or serve the viewer with `whiteout serve`."
          : "could not read " + url + ": " + error.message
      };
      stop();
      setPhase("error");
    });
  }

  /* ── replay clock ────────────────────────────────────────────────── */

  function tickSeconds() {
    if (state.records.length < 2) { return 0.5; }
    return Math.max(1e-6, state.records[1].t - state.records[0].t);
  }

  function play() {
    if (state.phase !== "ready" || !state.records.length) { return; }
    if (state.index >= state.records.length - 1) { state.index = 0; }
    state.playing = true;
    clock.last = global.performance.now();
    clock.carry = 0;
    clock.frame = global.requestAnimationFrame(step);
    render();
  }

  function stop() {
    state.playing = false;
    if (clock.frame) { global.cancelAnimationFrame(clock.frame); }
    clock.frame = 0;
  }

  function pause() {
    stop();
    render();
  }

  /* Advances by whole ticks. There is no interpolation between ticks in the
   * shell, which is also what prefers-reduced-motion asks for, so the reduced
   * path is the same path at the same rate. */
  function step(now) {
    if (!state.playing) { return; }
    var elapsed = (now - clock.last) / 1000;
    clock.last = now;
    clock.carry += (elapsed * state.speed) / tickSeconds();
    var whole = Math.floor(clock.carry);
    if (whole > 0) {
      clock.carry -= whole;
      state.index = Math.min(state.records.length - 1, state.index + whole);
      render();
    }
    if (state.index >= state.records.length - 1) {
      stop();
      render();
      return;
    }
    clock.frame = global.requestAnimationFrame(step);
  }

  function seek(index) {
    state.index = Math.max(0, Math.min(state.records.length - 1, index));
    render();
  }

  /* ── rendering ───────────────────────────────────────────────────── */

  function each(selector, fn) {
    Array.prototype.forEach.call(doc.querySelectorAll(selector), fn);
  }

  function text(node, value) {
    node.textContent = value;
  }

  function clear(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  function clockLabel(seconds) {
    if (!isFinite(seconds) || seconds < 0) { return "0:00"; }
    var whole = Math.floor(seconds);
    var minutes = Math.floor(whole / 60);
    var rest = whole % 60;
    return minutes + ":" + (rest < 10 ? "0" : "") + rest;
  }

  function fixed(value, places) {
    return typeof value === "number" && isFinite(value) ? value.toFixed(places) : UNKNOWN;
  }

  function current() {
    return state.records.length ? state.records[state.index] : null;
  }

  function render() {
    renderMeta();
    renderDials();
    renderTransport();
    renderRail();
    renderError();
    drawField();
  }

  function renderMeta() {
    /* seed, transport, policy and terrain are episode provenance, and the
     * record shape does not carry them at any version this viewer reads — so
     * the slots read as unknown rather than as a value the viewer invented. */
    each("[data-meta]", function (node) {
      text(node, UNKNOWN);
      node.classList.add("is-unknown");
    });
  }

  function renderDials() {
    if (el.dialRow.childNodes.length) { return; }
    AXES.forEach(function (axis) {
      var node = el.tplDial.content.firstElementChild.cloneNode(true);
      node.setAttribute("data-axis", axis.key);
      text(node.querySelector("[data-label]"), axis.label);
      node.querySelector("[data-value]").classList.add("is-unknown");
      var arc = node.querySelector("[data-arc]");
      var circumference = 2 * Math.PI * Number(arc.getAttribute("r"));
      arc.style.strokeDasharray = String(circumference);
      arc.style.strokeDashoffset = String(circumference);
      el.dialRow.appendChild(node);
    });
  }

  function renderTransport() {
    var ready = state.phase === "ready" && state.records.length > 0;
    var last = state.records.length ? state.records.length - 1 : 0;
    var finished = ready && !state.playing && state.index >= last && last > 0;

    el.play.disabled = !ready;
    el.stepBack.disabled = !ready;
    el.stepForward.disabled = !ready;
    el.scrub.disabled = !ready;
    el.play.classList.toggle("btn-primary", ready);
    each(".chip", function (chip) {
      chip.disabled = !ready;
      chip.setAttribute("aria-pressed", String(Number(chip.getAttribute("data-speed")) === state.speed));
    });

    clear(el.play);
    el.play.appendChild(global.WHITEOUT_ICONS.glyph(state.playing ? "pause" : "play"));
    el.play.setAttribute("aria-label", state.playing ? "pause" : finished ? "replay from the first tick" : "play");

    el.scrub.max = String(last);
    el.scrub.value = String(state.index);

    var record = current();
    var duration = state.records.length ? state.records[last].t : 0;
    text(el.clockT, clockLabel(record ? record.t : 0));
    text(el.clockDuration, clockLabel(duration));

    el.runState.hidden = !ready;
    if (ready) {
      var phase = state.playing ? "running" : finished ? "finished" : "paused";
      el.runState.setAttribute("data-run-state", phase);
      text(el.runState, phase === "running" ? "episode running"
        : phase === "finished" ? "episode finished" : "paused");
    }
  }

  function renderRail() {
    var record = current();
    renderContacts(record);
    renderFleet(record);
    renderInspector(record);
  }

  /* How a contact's fix and the attitude it was projected with stood in time
   * (`whiteout.types.PoseSync`). A row says it in three or four words, because
   * the alternative it replaces is saying nothing: a position assembled out of
   * two instants a quarter-second apart is 5.5 m wrong on the fixed-wing, with
   * every field populated and nothing to look at.
   *
   * `synchronised` is spelt "in sync" and not left blank. A blank cell reads as
   * "no data", which is what a *null* sync is, and those two must not look
   * alike: one says the pair was checked and stood together, the other says
   * nobody checked. The em dash is the shell's word for the second. */
  var SYNC_LABELS = {
    synchronised: "in sync",
    telemetry_missing: "no fix time",
    attitude_missing: "no attitude"
  };

  function syncLabel(sync) {
    if (!sync || typeof sync !== "object") { return UNKNOWN; }
    /* The stale row is the only one that carries a number, because it is the
     * only state where the skew is both known and out of bounds — the size of
     * the error is the finding. */
    if (sync.status === "telemetry_stale") {
      return "skew " + fixed(sync.skew_s, 2) + " s";
    }
    return SYNC_LABELS[sync.status] || UNKNOWN;
  }

  function syncKind(sync) {
    if (!sync || typeof sync !== "object") { return "unrecorded"; }
    return String(sync.status);
  }

  /* What the tick's cameras declined to turn into a fix. A refused sighting
   * leaves no contact, so without this line the rail cannot tell "nobody can
   * see the vessel" from "two cameras saw something and we would not stand
   * behind where it was" — and the second is the one the operator can act on. */
  function refusedLine(record) {
    var refusals = record && Array.isArray(record.refusals) ? record.refusals : [];
    if (!refusals.length) { return ""; }
    var reasons = [];
    refusals.forEach(function (refusal) {
      var label = syncLabel(refusal ? refusal.sync : null);
      if (reasons.indexOf(label) === -1) { reasons.push(label); }
    });
    var fixes = refusals.length === 1 ? "1 fix" : refusals.length + " fixes";
    return fixes + " refused — " + reasons.join(", ");
  }

  function renderContacts(record) {
    var contacts = record ? record.contacts : [];
    text(el.contactCount, record ? String(contacts.length) : UNKNOWN);
    el.contactEmpty.hidden = contacts.length > 0;
    clear(el.contactList);
    contacts.forEach(function (contact) {
      var row = el.tplContact.content.firstElementChild.cloneNode(true);
      row.setAttribute("data-state", String(contact.state));
      row.setAttribute("data-sync", syncKind(contact.sync));
      text(row.querySelector("[data-contact-id]"), String(contact.contact_id));
      text(row.querySelector("[data-contact-sync]"), syncLabel(contact.sync));
      text(row.querySelector("[data-contact-state]"), String(contact.state).replace("_", " "));
      text(row.querySelector("[data-contact-confidence]"), fixed(contact.confidence, 2));
      el.contactList.appendChild(row);
    });
    var refused = refusedLine(record);
    el.contactRefused.hidden = refused === "";
    text(el.contactRefused, refused);
  }

  function taskFor(record, assetId) {
    var intents = record && record.intent && Array.isArray(record.intent.intents)
      ? record.intent.intents : [];
    for (var i = 0; i < intents.length; i += 1) {
      if (intents[i].asset_id === assetId) { return String(intents[i].reason); }
    }
    return "no intent this tick";
  }

  function renderFleet(record) {
    var poses = record ? record.observation.poses : [];
    text(el.fleetCount, record ? String(poses.length) : UNKNOWN);
    el.fleetEmpty.hidden = poses.length > 0;
    clear(el.fleetList);
    var peak = poses.reduce(function (most, pose) {
      return Math.max(most, Number(pose.energy_used) || 0);
    }, 0);
    poses.forEach(function (pose) {
      var row = el.tplFleet.content.firstElementChild.cloneNode(true);
      row.setAttribute("aria-selected", String(state.selected === pose.asset_id));
      row.querySelector("[data-dot]").setAttribute("data-cls", String(pose.cls));
      text(row.querySelector("[data-callsign]"), String(pose.asset_id));
      text(row.querySelector("[data-task]"), taskFor(record, pose.asset_id));
      /* An empty track under every callsign reads as a rule between the rows,
       * so the meter appears only once an asset has spent something. */
      row.querySelector(".energy").hidden = peak <= 0;
      row.querySelector("[data-energy]").style.width =
        (peak > 0 ? ((Number(pose.energy_used) || 0) / peak) * 100 : 0).toFixed(1) + "%";
      row.querySelector("[data-select]").addEventListener("click", function () {
        state.selected = state.selected === pose.asset_id ? null : pose.asset_id;
        render();
      });
      el.fleetList.appendChild(row);
    });
  }

  function inspectorRows(record) {
    if (!record) { return []; }
    if (state.selected) {
      var pose = record.observation.poses.filter(function (item) {
        return item.asset_id === state.selected;
      })[0];
      if (pose) {
        return [
          ["callsign", String(pose.asset_id)],
          ["class", String(pose.cls)],
          ["latitude", fixed(pose.lat, 5) + "\u00b0"],
          ["longitude", fixed(pose.lon, 5) + "\u00b0"],
          ["altitude", fixed(pose.z, 1) + " m"],
          ["heading", fixed((Number(pose.heading) * 180) / Math.PI, 1) + "°"],
          ["speed", fixed(pose.speed, 2) + " m/s"],
          ["energy used", fixed(pose.energy_used, 2)],
          ["task", taskFor(record, pose.asset_id)]
        ];
      }
    }
    var digest = record.belief_digest || {};
    var grid = Array.isArray(digest.grid_shape) ? digest.grid_shape : null;
    return [
      ["source", state.source.split("/").pop() || UNKNOWN],
      ["schema", String(SCHEMA_VERSION)],
      ["ticks", String(state.records.length)],
      ["tick", String(state.index + 1)],
      ["t", fixed(record.t, 1) + " s"],
      ["belief grid", grid ? grid[0] + " × " + grid[1] : UNKNOWN],
      ["belief mass", fixed(digest.mass, 3)],
      ["covered", fixed(digest.covered_fraction, 3)]
    ];
  }

  function renderInspector(record) {
    var rows = inspectorRows(record);
    text(el.inspectorCount, state.selected ? "asset" : record ? "episode" : UNKNOWN);
    el.inspectorEmpty.hidden = rows.length > 0;
    clear(el.inspectorKv);
    rows.forEach(function (pair) {
      var row = el.tplKv.content.firstElementChild.cloneNode(true);
      text(row.querySelector("[data-key]"), pair[0]);
      text(row.querySelector("[data-val]"), pair[1]);
      el.inspectorKv.appendChild(row);
    });
  }

  function renderError() {
    if (!state.failure) { return; }
    /* A log that failed to arrive is not a log that failed to validate, and
     * the two want different first words. */
    var validated = state.failure.line !== null;
    text(el.errorTitle, validated
      ? "episode log is malformed"
      : "episode log could not be read");
    text(el.errorSource, state.source || UNKNOWN);
    text(el.errorLine, validated ? String(state.failure.line) : "not reached");
    text(el.errorSchema, String(SCHEMA_VERSION));
    text(el.errorDetail, validated
      ? "line " + state.failure.line + ": " + state.failure.reason
      : state.failure.reason);
    text(el.errorSummary, validated
      ? "The log stopped validating at line " + state.failure.line +
        ", and nothing after it was read."
      : "The log could not be read, so no record was parsed.");
  }

  /* ── the drawing frame ────────────────────────────────────── */

  /* The log is lat/lon and nothing else: SPEC.md section 5 names geodetic
   * WGS-84 degrees as the frame of record, and whiteout/geo.py is the one
   * module that converts. A canvas needs metres, so this block is the
   * viewer's local projection and is *only* that: an East-North offset in
   * metres about an origin this file picks, used to place pixels. It is
   * never written anywhere, never posted anywhere, and no number it produces
   * leaves the browser.
   *
   * It is a second implementation of whiteout.geo's arithmetic because the
   * page has no Python; tests/test_viz_shell.py pins the constants and the
   * origin below against whiteout.geo, so the two cannot drift apart. The
   * formulae are whiteout/geo.py's, with the radii of curvature taken at the
   * origin's latitude. */
  var WGS84_A = 6378137.0;
  var WGS84_F = 1 / 298.257223563;
  var WGS84_E2 = WGS84_F * (2 - WGS84_F);

  /* Bellot Strait, ARENA.md section 2. The fallback origin, for an episode
   * whose first record carries no pose to centre on. */
  var ARENA_ORIGIN = { lat: 71.99, lon: -94.84 };

  /* Metres per radian of latitude, and metres per radian of longitude, at
   * `latDeg`. The second collapses toward the pole: at 71.99 N it is about
   * 0.31 of the first, which is why a frame mix-up up here looks plausible
   * instead of absurd. */
  function radii(latDeg) {
    var phi = (latDeg * Math.PI) / 180;
    var sinPhi = Math.sin(phi);
    var w = 1 - WGS84_E2 * sinPhi * sinPhi;
    return {
      meridional: (WGS84_A * (1 - WGS84_E2)) / Math.pow(w, 1.5),
      east: (WGS84_A / Math.sqrt(w)) * Math.cos(phi)
    };
  }

  /* lat/lon -> {east, north} metres about `origin`. The drawing frame, and
   * the only conversion this file performs. */
  function toLocal(origin, latDeg, lonDeg) {
    var scale = radii(origin.lat);
    return {
      east: ((lonDeg - origin.lon) * Math.PI * scale.east) / 180,
      north: ((latDeg - origin.lat) * Math.PI * scale.meridional) / 180
    };
  }

  /* validateLine checks that `observation.poses` is an array, not what is in
   * it, so a pose with a missing or non-numeric lat/lon can reach the two
   * measures below. They have to agree about what to do with it: `|| 0` in
   * one and a raw `Number()` in the other would drag the origin toward
   * (0, 0) *and* make the extent NaN, which leaves the grid undrawn and the
   * scale bar reading "NaN m" with no error state to explain it. Both skip
   * such a pose instead. */
  function isPlaced(pose) {
    return isFinite(Number(pose.lat)) && isFinite(Number(pose.lon));
  }

  /* The origin is the fleet's centroid in the first record, so the axes cross
   * where the episode starts rather than at an arbitrary meridian. Measured
   * once per load, with the extent. */
  function measureOrigin() {
    var first = state.records[0];
    var poses = (first ? first.observation.poses : []).filter(isPlaced);
    if (!poses.length) { return ARENA_ORIGIN; }
    var lat = 0;
    var lon = 0;
    poses.forEach(function (pose) {
      lat += Number(pose.lat);
      lon += Number(pose.lon);
    });
    return { lat: lat / poses.length, lon: lon / poses.length };
  }

  /* ── the field's chrome ──────────────────────────────────────────── */

  var DEFAULT_EXTENT = 1000;   /* metres of half-width before a log says otherwise */

  /* The extent is a property of the episode, so it is measured once per load
   * rather than per frame. */
  /* The two half-extents, measured apart rather than as one square.
   *
   * A square frame is the wrong shape for this world. Bellot Strait is about
   * 6.25 km along and 1.5 km across, so squaring the extent sizes everything
   * to the long axis and then shows the channel at a quarter of the width the
   * canvas has for it, inside a margin of empty water on all four sides. Held
   * apart, whichever axis actually binds sets the scale, and on a 1440-wide
   * field that is the along-channel one — the strait fills the frame it is
   * given. */
  function measureExtent() {
    var span = DEFAULT_EXTENT;
    var across = DEFAULT_EXTENT;
    state.origin = measureOrigin();
    state.lattice = measureLattice();
    state.records.forEach(function (record) {
      record.observation.poses.filter(isPlaced).forEach(function (pose) {
        var local = toLocal(state.origin, Number(pose.lat), Number(pose.lon));
        span = Math.max(span, Math.abs(local.east) * 1.2);
        across = Math.max(across, Math.abs(local.north) * 1.2);
      });
    });
    /* The channel has to fit too, not just the fleet. The assets start inside
     * the water and stay there, so sizing on poses alone framed the episode
     * tightly enough to clip the ends of the belief ribbon — the thing the
     * field is mostly showing. No 1.2 margin here: the lattice is the world's
     * own edge, and padding it would leave the channel floating in dead
     * space. */
    var lattice = state.lattice;
    if (lattice) {
      for (var i = 0; i < lattice.east.length; i += 1) {
        span = Math.max(span, Math.abs(lattice.east[i]));
        across = Math.max(across, Math.abs(lattice.north[i]));
      }
    }
    state.extent = span;
    state.extentNorth = across;
  }

  /* ── the belief field ────────────────────────────────────────────── */

  /* base64 to bytes. `atob` is in every browser this opens in, and the
   * viewer has no build step to bring in anything else. */
  function bytesOf(text) {
    var binary = global.atob(text);
    var out = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i += 1) { out[i] = binary.charCodeAt(i); }
    return out;
  }

  /* The episode-constant cell lattice, parsed once per load.
   *
   * `whiteout.types.BeliefGeometry` is the contract: `water` is a row-major
   * bitfield over `shape`, most significant bit first, and `corners` is the
   * (along + 1) x (across + 1) corner lattice as interleaved big-endian
   * float32 lat/lon. Cell (row, column) is the quadrilateral on corners
   * (row, column), (row + 1, column), (row + 1, column + 1), (row, column + 1).
   *
   * **Projected here rather than per frame.** The origin is fixed for the
   * episode, so a corner's metres never change; `toLocal` runs a sine, a
   * cosine and a power per call, and doing that for 3400 corners on all 400
   * ticks is 1.4M transcendental calls to redraw a constant. Held as metres,
   * a redraw is one multiply-add per corner.
   *
   * Returns null for a log carrying no field, which is every episode written
   * before schema 6 and any run of a transport that reports none. The field
   * simply is not drawn; that is not an error state. */
  function measureLattice() {
    var carrier = null;
    for (var r = 0; r < state.records.length; r += 1) {
      if (state.records[r].belief_field && state.records[r].belief_field.geometry) {
        carrier = state.records[r].belief_field.geometry;
        break;
      }
    }
    if (!carrier) { return null; }
    var along = Number(carrier.shape[0]);
    var across = Number(carrier.shape[1]);
    var mask = bytesOf(carrier.water);
    var raw = bytesOf(carrier.corners);
    var view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    var corners = (along + 1) * (across + 1);
    var east = new Float64Array(corners);
    var north = new Float64Array(corners);
    for (var i = 0; i < corners; i += 1) {
      var local = toLocal(state.origin, view.getFloat32(i * 8, false), view.getFloat32(i * 8 + 4, false));
      east[i] = local.east;
      north[i] = local.north;
    }
    /* The set bits, in mask order: byte k of a frame's `cells` is cell
     * `water[k]`, which is what makes a byte's position mean a place. */
    var water = [];
    for (var c = 0; c < along * across; c += 1) {
      if ((mask[c >> 3] >> (7 - (c & 7))) & 1) { water.push(c); }
    }
    return { along: along, across: across, east: east, north: north, water: water };
  }

  /* `#rrggbb` or `#rrggbbaa` as [r, g, b, a]. The ramp's floor carries an
   * alpha of 00 so that an emptied cell reveals the graticule under it
   * rather than painting a near-black patch over it. */
  function rgbaOf(hex) {
    var text = hex.replace("#", "");
    return [
      parseInt(text.substring(0, 2), 16),
      parseInt(text.substring(2, 4), 16),
      parseInt(text.substring(4, 6), 16),
      text.length >= 8 ? parseInt(text.substring(6, 8), 16) / 255 : 1
    ];
  }

  /* The belief ramp as a fixed number of ready-made `rgba()` strings.
   *
   * Built per draw rather than per cell: `token()` is a getComputedStyle
   * call, and six of those plus 850 string builds every frame is work the
   * ramp does not need — it is the same ramp for every cell. 64 steps is
   * finer than the eye resolves in a 100 m cell and a quarter of the
   * quantiser's own 255. */
  function beliefPalette() {
    var stops = [];
    for (var s = 0; s <= 5; s += 1) { stops.push(rgbaOf(token("--belief-" + s))); }
    var palette = [];
    for (var i = 0; i < PALETTE_STEPS; i += 1) {
      var x = (i / (PALETTE_STEPS - 1)) * (stops.length - 1);
      var low = Math.min(Math.floor(x), stops.length - 2);
      var f = x - low;
      var a = stops[low];
      var b = stops[low + 1];
      palette.push(
        "rgba(" + Math.round(a[0] + (b[0] - a[0]) * f) +
        "," + Math.round(a[1] + (b[1] - a[1]) * f) +
        "," + Math.round(a[2] + (b[2] - a[2]) * f) +
        "," + (a[3] + (b[3] - a[3]) * f).toFixed(3) + ")"
      );
    }
    return palette;
  }

  /* The belief field, one filled quadrilateral per water cell.
   *
   * **Byte value, not absolute probability.** A frame's bytes are already
   * relative to that frame's own peak (`scale`), and drawing them as they
   * come is what makes the erosion legible: the most probable cell holds
   * full luminance throughout while swept water drains toward the floor, so
   * what moves on screen is the shape of what the fleet has ruled out. An
   * absolute ramp would instead dim the whole field together as the peak
   * climbed, which is the same information rendered as a fade.
   *
   * Each quad is stroked in its own fill colour as well as filled. Adjacent
   * antialiased fills leave a hairline of background between them, and 850
   * of those read as a grid laid over the field rather than as water. */
  function drawBelief(ctx, cx, cy, scale) {
    var record = state.records[state.index];
    var frame = record && record.belief_field;
    var lattice = state.lattice;
    if (!frame || !lattice) { return; }
    var bytes = bytesOf(frame.cells);
    var palette = beliefPalette();
    var last = palette.length - 1;
    var stride = lattice.across + 1;
    var east = lattice.east;
    var north = lattice.north;
    var water = lattice.water;
    var count = Math.min(water.length, bytes.length);
    ctx.lineWidth = 1;
    ctx.lineJoin = "round";
    for (var k = 0; k < count; k += 1) {
      var cell = water[k];
      var row = (cell / lattice.across) | 0;
      var column = cell % lattice.across;
      var a = row * stride + column;
      var b = a + stride;
      var colour = palette[Math.round((bytes[k] / 255) * last)];
      ctx.fillStyle = colour;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.moveTo(cx + east[a] * scale, cy - north[a] * scale);
      ctx.lineTo(cx + east[b] * scale, cy - north[b] * scale);
      ctx.lineTo(cx + east[b + 1] * scale, cy - north[b + 1] * scale);
      ctx.lineTo(cx + east[a + 1] * scale, cy - north[a + 1] * scale);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }

  /* The four assets, over the field.
   *
   * A tower is a square because it cannot move; an aircraft is a triangle
   * pointing along its heading, which is degrees clockwise from North
   * (`whiteout.types.Pose`). Both carry a dark outline so the mark holds its
   * shape over the ramp's bright end as well as over the empty floor, and a
   * label, because a judge watching a handoff needs to know which airframe
   * moved. Nothing glows. */
  function drawAssets(ctx, cx, cy, scale) {
    var record = state.records[state.index];
    if (!record) { return; }
    var poses = record.observation.poses.filter(isPlaced);
    ctx.font = canvasFont("--t-12", "--font-mono");
    ctx.textBaseline = "middle";
    ctx.lineJoin = "miter";
    /* Where a label has already been put, so the next one can step over it.
     * Two assets converging on the same waypoint is the normal end of a
     * handoff, not an edge case — the quadcopter and the fixed-wing finish
     * this episode a few metres apart — and two labels on the same baseline
     * overprint into a word that reads as a rendering fault. */
    var placed = [];
    poses.forEach(function (pose) {
      var local = toLocal(state.origin, Number(pose.lat), Number(pose.lon));
      var x = cx + local.east * scale;
      var y = cy - local.north * scale;
      var reach = size("--s-2");
      ctx.lineWidth = size("--w-hairline") * 2;
      ctx.strokeStyle = token("--n-950");
      ctx.fillStyle = pose.cls === "tower" ? token("--n-300") : token("--n-000");
      ctx.beginPath();
      if (pose.cls === "tower") {
        ctx.rect(x - reach, y - reach, reach * 2, reach * 2);
      } else {
        /* Heading is degrees clockwise from North, and the canvas y axis
         * points down, so North is -y: forward is (sin, -cos) and the
         * airframe's right is that turned 90 degrees clockwise, (cos, sin).
         * The triangle is nose-forward with its base behind the position, so
         * the mark's centre of area sits on the fix rather than ahead of it. */
        var heading = ((Number(pose.heading) || 0) * Math.PI) / 180;
        var forwardX = Math.sin(heading);
        var forwardY = -Math.cos(heading);
        var rightX = Math.cos(heading);
        var rightY = Math.sin(heading);
        var nose = reach * 1.9;
        var back = reach * 0.9;
        var beam = reach * 1.0;
        ctx.moveTo(x + nose * forwardX, y + nose * forwardY);
        ctx.lineTo(x - back * forwardX + beam * rightX, y - back * forwardY + beam * rightY);
        ctx.lineTo(x - back * forwardX - beam * rightX, y - back * forwardY - beam * rightY);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.fill();
      var labelX = x + reach * 2.2;
      var labelY = y;
      var step = size("--s-3") || 12;
      var width = ctx.measureText(pose.asset_id).width;
      for (var tries = 0; tries < placed.length + 1; tries += 1) {
        var clash = placed.some(function (box) {
          return Math.abs(box.y - labelY) < step &&
            labelX < box.x + box.width && box.x < labelX + width;
        });
        if (!clash) { break; }
        labelY += step;
      }
      placed.push({ x: labelX, y: labelY, width: width });
      ctx.fillStyle = token("--n-300");
      ctx.strokeStyle = token("--n-950");
      ctx.lineWidth = size("--w-hairline") * 3;
      ctx.strokeText(pose.asset_id, labelX, labelY);
      ctx.fillText(pose.asset_id, labelX, labelY);
    });
  }

  /* A round grid step that puts between four and ten lines across the view. */
  function gridStep(span) {
    var raw = (span * 2) / 8;
    var power = Math.pow(10, Math.floor(Math.log10(raw)));
    var candidates = [1, 2, 5, 10];
    for (var i = 0; i < candidates.length; i += 1) {
      if (candidates[i] * power >= raw) { return candidates[i] * power; }
    }
    return 10 * power;
  }

  function metres(value) {
    return value >= 1000 ? (value / 1000) + " km" : value + " m";
  }

  function drawField() {
    var canvas = el.canvas;
    var ratio = global.devicePixelRatio || 1;
    var box = canvas.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) { return; }
    var wanted = [Math.round(box.width * ratio), Math.round(box.height * ratio)];
    /* Assigning width or height clears and reallocates the backing store, so
     * it happens only when the element actually changed size. */
    if (canvas.width !== wanted[0]) { canvas.width = wanted[0]; }
    if (canvas.height !== wanted[1]) { canvas.height = wanted[1]; }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, box.width, box.height);
    ctx.fillStyle = token("--n-950");
    ctx.fillRect(0, 0, box.width, box.height);

    var span = state.extent;
    /* Whichever axis runs out of room first sets the scale, so the channel is
     * framed by the canvas it has rather than by the square that contains it.
     * One scale for both axes still, because the field is a map and a map
     * with two scales is a lie. */
    var scale = Math.min(box.width / (span * 2), box.height / (state.extentNorth * 2));
    var cx = box.width / 2;
    var cy = box.height / 2;
    var step = gridStep(span);

    /* The grid covers what the canvas can show, not the extent square, or the
     * field is left with an unruled band down each side. */
    var half = size("--w-hairline") / 2;
    var reach = Math.ceil(Math.max(box.width, box.height) / 2 / scale / step) * step;
    ctx.lineWidth = size("--w-hairline");
    ctx.strokeStyle = token("--n-900");
    ctx.beginPath();
    for (var v = -reach; v <= reach + 1e-6; v += step) {
      var x = Math.round(cx + v * scale) + half;
      var y = Math.round(cy + v * scale) + half;
      ctx.moveTo(x, 0); ctx.lineTo(x, box.height);
      ctx.moveTo(0, y); ctx.lineTo(box.width, y);
    }
    ctx.stroke();

    /* The two axes through the origin sit one step above the grid, so the
     * frame reads without competing with anything drawn into it. */
    ctx.strokeStyle = token("--n-800");
    ctx.beginPath();
    ctx.moveTo(Math.round(cx) + half, 0); ctx.lineTo(Math.round(cx) + half, box.height);
    ctx.moveTo(0, Math.round(cy) + half); ctx.lineTo(box.width, Math.round(cy) + half);
    ctx.stroke();

    /* Layered in one pass, back to front: the chrome above, then the water
     * the vessel could be in, then the fleet over it. The scale bar goes
     * last so it stays readable where the field runs bright under it. */
    drawBelief(ctx, cx, cy, scale);
    drawAssets(ctx, cx, cy, scale);
    drawScaleBar(ctx, box, step, scale);
  }

  /* One grid step, drawn as a bar with end ticks and its length in metres:
   * the Observable habit of putting the label right next to the thing it
   * labels, rather than round the edge of the plot. */
  function drawScaleBar(ctx, box, step, scale) {
    var pad = size("--s-4");
    var tick = size("--s-1");
    var half = size("--w-hairline") / 2;
    var width = step * scale;
    var y = box.height - pad;
    var x0 = pad;
    ctx.strokeStyle = token("--n-600");
    ctx.lineWidth = size("--w-hairline");
    ctx.beginPath();
    ctx.moveTo(x0 + half, y - tick); ctx.lineTo(x0 + half, y + half);
    ctx.lineTo(x0 + width + half, y + half); ctx.lineTo(x0 + width + half, y - tick);
    ctx.stroke();
    ctx.fillStyle = token("--n-400");
    ctx.font = canvasFont("--t-12", "--font-mono");
    ctx.textBaseline = "alphabetic";
    ctx.fillText(metres(step), x0 + width + size("--s-2"), y + half);
  }

  /* ── wiring ──────────────────────────────────────────────────────── */

  function act(name) {
    if (name === "load-bundled") { loadUrl(BUNDLED_EPISODE); return; }
    if (name === "open-file") { el.fileInput.click(); return; }
    if (name === "play") {
      if (state.playing) { pause(); } else { play(); }
      return;
    }
    if (name === "step-back") { stop(); seek(state.index - 1); return; }
    if (name === "step-forward") { stop(); seek(state.index + 1); }
  }

  function wire() {
    el.dialRow = doc.getElementById("dial-row");
    el.canvas = doc.getElementById("field-canvas");
    el.fileInput = doc.getElementById("file-input");
    el.play = doc.getElementById("play");
    el.stepBack = doc.getElementById("step-back");
    el.stepForward = doc.getElementById("step-forward");
    el.scrub = doc.getElementById("scrub");
    el.clockT = doc.querySelector("[data-clock='t']");
    el.clockDuration = doc.querySelector("[data-clock='duration']");
    el.runState = doc.querySelector("[data-run-state]");
    el.contactList = doc.getElementById("contact-list");
    el.contactEmpty = doc.querySelector("[data-empty='contacts']");
    el.contactCount = doc.querySelector("[data-count='contacts']");
    el.contactRefused = doc.querySelector("[data-refused]");
    el.fleetList = doc.getElementById("fleet-list");
    el.fleetEmpty = doc.querySelector("[data-empty='fleet']");
    el.fleetCount = doc.querySelector("[data-count='fleet']");
    el.inspectorKv = doc.getElementById("inspector-kv");
    el.inspectorEmpty = doc.querySelector("[data-empty='inspector']");
    el.inspectorCount = doc.querySelector("[data-count='inspector']");
    el.errorSource = doc.querySelector("[data-error-source]");
    el.errorLine = doc.querySelector("[data-error-line]");
    el.errorSchema = doc.querySelector("[data-error-schema]");
    el.errorDetail = doc.querySelector("[data-error-detail]");
    el.errorSummary = doc.querySelector("[data-error-summary]");
    el.errorTitle = doc.querySelector("[data-error-title]");
    el.tplDial = doc.getElementById("tpl-dial");
    el.tplFleet = doc.getElementById("tpl-fleet-row");
    el.tplContact = doc.getElementById("tpl-contact-row");
    el.tplKv = doc.getElementById("tpl-kv-row");

    el.stepBack.appendChild(global.WHITEOUT_ICONS.glyph("step", true));
    el.stepForward.appendChild(global.WHITEOUT_ICONS.glyph("step"));

    each("[data-act]", function (node) {
      node.addEventListener("click", function () { act(node.getAttribute("data-act")); });
    });

    each(".chip", function (chip) {
      chip.addEventListener("click", function () {
        state.speed = Number(chip.getAttribute("data-speed"));
        render();
      });
    });

    el.scrub.addEventListener("input", function () {
      stop();
      seek(Number(el.scrub.value));
    });

    el.fileInput.addEventListener("change", function () {
      var file = el.fileInput.files && el.fileInput.files[0];
      if (!file) { return; }
      state.source = file.name;
      setPhase("loading");
      var reader = new global.FileReader();
      reader.onload = function () { loadText(String(reader.result), file.name); };
      reader.onerror = function () {
        state.failure = { line: null, reason: "could not read " + file.name };
        setPhase("error");
      };
      reader.readAsText(file);
    });

    doc.addEventListener("keydown", function (event) {
      if (event.target !== doc.body && event.target !== doc.documentElement) { return; }
      if (event.key === " ") { event.preventDefault(); act("play"); }
      if (event.key === "ArrowLeft") { event.preventDefault(); act("step-back"); }
      if (event.key === "ArrowRight") { event.preventDefault(); act("step-forward"); }
    });

    if (global.ResizeObserver) {
      new global.ResizeObserver(function () { drawField(); }).observe(el.canvas);
    } else {
      global.addEventListener("resize", drawField);
    }

    setPhase("empty");

    /* `?log=<path>` loads a log on open, so that a rehearsal or a screenshot
     * run lands straight in the state it wants. */
    var wanted = new global.URLSearchParams(global.location.search).get("log");
    if (wanted) { loadUrl(wanted); }
  }

  global.WHITEOUT_VIEWER = {
    SCHEMA_VERSION: SCHEMA_VERSION,
    BUNDLED_EPISODE: BUNDLED_EPISODE,
    parseEpisodeLog: parseEpisodeLog,
    toLocal: toLocal,
    state: state,
    reducedMotion: reducedMotion
  };

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})(window);
