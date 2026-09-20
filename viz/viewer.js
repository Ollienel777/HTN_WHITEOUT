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
 * The field's layered content draw — terrain, belief, cuts, footprints,
 * intents, sprites, truth — is FieldCanvas's, and lands with its own ticket.
 * What is here is the field's chrome: the extent graticule and the scale bar,
 * so that the canvas reads as an instrument at rest rather than a blank area.
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

  /* The published camera fields of view, ARENA.md section 4, and the only
   * thing the arena tells us about the optics. Keyed by the camera names in
   * whiteout.vision.camera.CAMERAS, which tests/test_viz_shell.py reads this
   * table against, so the two cannot drift. `cls` is the vehicle class the
   * episode log spells, which is not the same word. */
  var CAMERAS = {
    "quadcopter": { hfov: 114.6, vfov: 99.4 },
    "fixed-wing": { hfov: 69.0, vfov: 42.6 },
    "tower": { hfov: 60.0, vfov: 36.1 }
  };

  var CAMERA_OF_CLASS = {
    quad: "quadcopter",
    fixedwing: "fixed-wing",
    tower: "tower"
  };

  /* The em dash the shell shows wherever the episode log does not carry a
   * value. A zero would be a number the system never produced. */
  var UNKNOWN = "—";

  var state = {
    phase: "empty",          /* empty | loading | error | ready */
    records: [],
    index: 0,
    playing: false,
    speed: 1,
    source: "",
    selected: null,
    extent: 1000,            /* metres of half-width the field frames */
    origin: null,            /* the drawing frame's origin, as {lat, lon} */
    field: null,             /* the belief grid's layout, decoded once */
    basemap: null,           /* the shores, projected once — null when absent */
    fixes: {},               /* contact id -> the t its position last moved */
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
    state.basemap = projectBasemap();
    state.field = decodeField(state.records);
    state.fixes = measureFixAges(state.records);
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
          ["heading", fixed(Number(pose.heading), 1) + "°"],
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
  function measureExtent() {
    var span = DEFAULT_EXTENT;
    state.origin = measureOrigin();
    state.records.forEach(function (record) {
      record.observation.poses.filter(isPlaced).forEach(function (pose) {
        var local = toLocal(state.origin, Number(pose.lat), Number(pose.lon));
        span = Math.max(span, Math.abs(local.east) * 1.2, Math.abs(local.north) * 1.2);
      });
    });
    state.extent = span;
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

  /* ── the basemap ─────────────────────────────────────────────────── */

  /* How much of `--terrain-hi` the land carries over the land tone. Land has
   * to be a different *tone* from water, because that difference is the whole
   * of what a basemap says; `--terrain-lo` alone is a shade off the page and
   * says nothing. A third of the way to `--terrain-hi` is the darkest value
   * that still reads at the demo viewport, and it stays far below the belief
   * ramp's floor, which is where DESIGN.md wants terrain.
   *
   * **It is flat on purpose.** A first revision graded the tone up toward the
   * shoreline to suggest rock rising out of the water. On screen that is a
   * halo around the belief field — indistinguishable from the bloom DESIGN.md
   * forbids, and unearned besides: there is no heightmap here, so any relief
   * would be invented. Flat land, one tone, no shading. */
  var LAND_TONE = 0.34;

  /* `viz/basemap.js`'s two shores, in the drawing frame's metres.
   *
   * Projected once per load, like the belief grid's corners, and through the
   * same `toLocal` the field's every other position goes through — which is
   * what makes the coastline land on the water cells rather than near them.
   * The shores themselves are `whiteout.belief.geometry.DEFAULT_STRAIT`'s
   * ribbon boundary, sampled by `scripts/make_basemap.py`, and that ribbon is
   * what the belief grid masks its water with.
   *
   * `null` whenever there is nothing to draw: the file absent (a `file://`
   * page missing its sibling, a deploy that dropped it), the global missing
   * its shores, or no origin to project about. The field then draws exactly
   * as it did before this layer existed — the dark ground is the fallback,
   * not an error. */
  function projectBasemap() {
    var data = global.WHITEOUT_BASEMAP;
    if (!data || !data.shores || !state.origin) { return null; }
    var left = projectShore(data.shores.left);
    var right = projectShore(data.shores.right);
    if (left.length < 2 || right.length < 2) { return null; }
    return { credit: String(data.credit || ""), left: left, right: right };
  }

  function projectShore(points) {
    var out = [];
    if (!Array.isArray(points)) { return out; }
    points.forEach(function (point) {
      var lat = Number(point[0]);
      var lon = Number(point[1]);
      if (!isFinite(lat) || !isFinite(lon)) { return; }
      var local = toLocal(state.origin, lat, lon);
      out.push([local.east, local.north]);
    });
    return out;
  }

  /* Land, as everything the channel is not.
   *
   * The water is one closed ring — the left shore out, the right shore back —
   * and the land is the canvas with that ring punched out of it, in one
   * even-odd fill. The ring closes across each mouth because the model does
   * too: `StraitGeometry.is_water` is false past either end of the
   * centreline, so belief there is identically zero and the viewer must not
   * suggest otherwise.
   *
   * It is the bottom layer and it stays the dimmest thing on the field: the
   * land tone never reaches `--terrain-hi` (DESIGN.md, "Do not make the map
   * prettier at the cost of legibility"). The shoreline itself is not
   * stroked here — `drawOutline` already draws it from the belief mask, and
   * two shorelines in two places is how they come to disagree.
   *
   * Returns whether it drew, so the credit line can be shown only when there
   * is something to credit. */
  function drawBasemap(ctx, box, cx, cy, scale) {
    var map = state.basemap;
    if (!map) { return false; }

    var water = new global.Path2D();
    var index;
    for (index = 0; index < map.left.length; index += 1) {
      var out = map.left[index];
      var x = cx + out[0] * scale;
      var y = cy - out[1] * scale;
      if (index === 0) { water.moveTo(x, y); } else { water.lineTo(x, y); }
    }
    for (index = map.right.length - 1; index >= 0; index -= 1) {
      var back = map.right[index];
      water.lineTo(cx + back[0] * scale, cy - back[1] * scale);
    }
    water.closePath();

    var land = new global.Path2D();
    land.rect(0, 0, box.width, box.height);
    land.addPath(water);
    ctx.fillStyle = token("--terrain-lo");
    ctx.fill(land, "evenodd");
    ctx.globalAlpha = LAND_TONE;
    ctx.fillStyle = token("--terrain-hi");
    ctx.fill(land, "evenodd");
    ctx.globalAlpha = 1;
    return true;
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
    var scale = Math.min(box.width, box.height) / (span * 2);
    var cx = box.width / 2;
    var cy = box.height / 2;
    var step = gridStep(span);

    /* The land goes under the graticule: the grid is chrome over the world,
     * not a fence in front of it. */
    var grounded = drawBasemap(ctx, box, cx, cy, scale);
    if (el.credit) {
      text(el.credit, grounded ? (state.basemap.credit || "") : "");
      el.credit.hidden = !grounded;
    }

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

    drawContent(ctx, cx, cy, scale);
    drawScaleBar(ctx, box, step, scale);
  }

  /* The field's content, in one pass, over the chrome. Everything below is
   * skipped when there is no record to draw — the empty, loading and error
   * states get the frame alone, which is the instrument at rest.
   *
   * **Measured, not assumed** (#20). Driving `drawField` over all 400 ticks
   * of the committed episode — 850 water cells, four assets, 876x308 CSS
   * pixels at devicePixelRatio 1.25 — costs **2.9 ms a frame**, three runs
   * agreeing to 0.06 ms. The episode's own tick is 500 ms, so real time has
   * 170x the budget it needs and even the 100x speed chip has room to spare.
   * Re-measure with `WHITEOUT_VIEWER.drawField` rather than trusting this. */
  function drawContent(ctx, cx, cy, scale) {
    var record = current();
    if (!record || state.phase !== "ready") { return; }
    function project(east, north) {
      return [cx + east * scale, cy - north * scale];
    }
    if (state.field) {
      drawBelief(ctx, project, state.field, record.belief_field);
      drawOutline(ctx, project, state.field);
    }
    drawFootprints(ctx, project, record, scale);
    drawAssets(ctx, project, record);
    drawTrack(ctx, project, record, state.fixes[state.index]);
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

  /* ── the field's content ─────────────────────────────────────────── */

  /* The layers, in the order they are drawn, and there is exactly one pass:
   * the channel outline, the belief field over its water, the camera
   * footprints, the four assets, the current track. Nothing here glows:
   * there is no shadowBlur, no filter and no second pass at a larger radius
   * anywhere below. Luminance comes from the ramp (DESIGN.md "Don'ts").
   *
   * Everything that does not change during an episode — the cell corners in
   * metres, the water mask, the outline's edges — is computed once in
   * `decodeField` and not per frame. A tick redraws 850 quadrilaterals and
   * six sprites, and nothing else. */

  function decodeBase64(text) {
    var binary = global.atob(text);
    var bytes = new global.Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i += 1) { bytes[i] = binary.charCodeAt(i); }
    return bytes;
  }

  /* The episode's belief layout: the water mask, and every cell corner as an
   * offset in metres from the drawing frame's origin.
   *
   * `whiteout.types.BeliefGeometry` ships this once, on the log's first
   * frame, so this looks for the first record that carries one and stops.
   * A log with none — a run with no belief behind it — leaves the field
   * layers off, and the chrome is drawn alone. */
  function decodeField(records) {
    var carrier = null;
    for (var i = 0; i < records.length && !carrier; i += 1) {
      var frame = records[i].belief_field;
      if (frame && frame.geometry) { carrier = frame.geometry; }
    }
    if (!carrier || !Array.isArray(carrier.shape)) { return null; }
    var along = Number(carrier.shape[0]);
    var across = Number(carrier.shape[1]);
    if (!(along > 0 && across > 0)) { return null; }

    var packed = decodeBase64(carrier.water);
    var water = new global.Uint8Array(along * across);
    for (var cell = 0; cell < water.length; cell += 1) {
      water[cell] = (packed[cell >> 3] >> (7 - (cell & 7))) & 1;
    }

    /* Big-endian float32 lat/lon pairs over the (along + 1) x (across + 1)
     * corner lattice. Big-endian is the log's wire order and `false` is what
     * asks DataView for it; the platform's own order would read as garbage
     * on half the machines in the world and as correct here. */
    var raw = decodeBase64(carrier.corners);
    var view = new global.DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    var corners = (along + 1) * (across + 1);
    var east = new global.Float64Array(corners);
    var north = new global.Float64Array(corners);
    for (var corner = 0; corner < corners; corner += 1) {
      var local = toLocal(
        state.origin,
        view.getFloat32(corner * 8, false),
        view.getFloat32(corner * 8 + 4, false)
      );
      east[corner] = local.east;
      north[corner] = local.north;
    }

    var field = {
      along: along,
      across: across,
      water: water,
      east: east,
      north: north,
      cells: waterCells(water, along, across),
      outline: outlineEdges(water, along, across)
    };
    growExtentToField(field);
    return field;
  }

  /* The (row, column) of each water cell, in the order the mask's set bits
   * run — which is the order `BeliefFrame.cells` carries its bytes in, and
   * so the only order that puts a probability on the right piece of water. */
  function waterCells(water, along, across) {
    var cells = [];
    for (var cell = 0; cell < water.length; cell += 1) {
      if (water[cell]) { cells.push([Math.floor(cell / across), cell % across]); }
    }
    return cells;
  }

  /* Every cell side with water on one side of it and not the other: the
   * shoreline, as the only thing this viewer knows about it. There is no
   * terrain here to shade — the arena renders Bellot Strait itself — so the
   * outline is what says "the vessel is confined to this". */
  function outlineEdges(water, along, across) {
    var edges = [];
    function wet(row, column) {
      if (row < 0 || column < 0 || row >= along || column >= across) { return 0; }
      return water[row * across + column];
    }
    for (var row = 0; row < along; row += 1) {
      for (var column = 0; column < across; column += 1) {
        if (!wet(row, column)) { continue; }
        if (!wet(row - 1, column)) { edges.push([row, column, row, column + 1]); }
        if (!wet(row + 1, column)) { edges.push([row + 1, column, row + 1, column + 1]); }
        if (!wet(row, column - 1)) { edges.push([row, column, row + 1, column]); }
        if (!wet(row, column + 1)) { edges.push([row, column + 1, row + 1, column + 1]); }
      }
    }
    return edges;
  }

  /* The extent is measured from the fleet, which on this arena sits inside
   * the channel, so the channel's own ends fall outside the view unless they
   * are counted too. */
  function growExtentToField(field) {
    var span = state.extent;
    for (var corner = 0; corner < field.east.length; corner += 1) {
      span = Math.max(span, Math.abs(field.east[corner]) * 1.1, Math.abs(field.north[corner]) * 1.1);
    }
    state.extent = span;
  }

  /* ── the belief ramp ─────────────────────────────────────────────── */

  var ramp = null;

  /* `--belief-0` through `--belief-5` as 256 steps, built once. The stops are
   * read from tokens.css rather than spelled here, so the ramp stays one
   * definition; DESIGN.md forbids these colours anywhere but this layer and
   * tests/test_viz_shell.py holds the file to it. */
  function beliefRamp() {
    if (ramp) { return ramp; }
    var stops = [];
    for (var i = 0; i <= 5; i += 1) { stops.push(parseColour(token("--belief-" + i))); }
    ramp = [];
    for (var code = 0; code < 256; code += 1) {
      var position = (code / 255) * (stops.length - 1);
      var low = Math.min(stops.length - 1, Math.floor(position));
      var high = Math.min(stops.length - 1, low + 1);
      ramp.push(mixColour(stops[low], stops[high], position - low));
    }
    return ramp;
  }

  function parseColour(value) {
    var hex = value.replace("#", "");
    if (hex.length === 6) { hex += "ff"; }
    return [
      parseInt(hex.slice(0, 2), 16),
      parseInt(hex.slice(2, 4), 16),
      parseInt(hex.slice(4, 6), 16),
      parseInt(hex.slice(6, 8), 16) / 255
    ];
  }

  function mixColour(low, high, weight) {
    var out = [];
    for (var channel = 0; channel < 3; channel += 1) {
      out.push(Math.round(low[channel] + (high[channel] - low[channel]) * weight));
    }
    var alpha = low[3] + (high[3] - low[3]) * weight;
    return "rgba(" + out[0] + "," + out[1] + "," + out[2] + "," + alpha.toFixed(3) + ")";
  }

  /* ── the camera footprint ────────────────────────────────────────── */

  /* **The episode log mixes angle units by record type, and this is the
   * trap.** A `Pose`'s `heading`, `pitch` and `roll` are **degrees** — they
   * reach `whiteout.vision.projection` as `CameraPose(yaw_deg=pose.heading,
   * pitch_deg=pose.pitch, ...)`, and the committed episode's first record
   * reads `heading: 279.2468`, which is no radian. A `SightingFootprint`'s
   * `heading` and `half_angle` are **radians**, and its own docstring says
   * so. So every pose angle crosses into this file's trigonometry through
   * `radians()`, and nothing else does.
   *
   * Getting this backwards is not a subtle failure. Read as radians, that
   * 279.2468 wraps to 2.787 rad and the wedge is drawn 240 degrees off — the
   * right shape, the right size, pointing at the wrong water. An earlier
   * revision of this file did exactly that, in `groundFootprint`,
   * `drawAssets` and the pose readout alike. `tests/test_viz_shell.py` now
   * feeds these functions the degrees a log actually carries. */
  function radians(degrees) {
    return (Number(degrees) * Math.PI) / 180;
  }

  /* WGS-84 mean radius, from the ellipsoid this file already carries.
   * whiteout.vision.projection.EARTH_MEAN_RADIUS_M, same expression. */
  var EARTH_MEAN_RADIUS_M = (2 * WGS84_A + WGS84_A * (1 - WGS84_F)) / 3;

  /* The flat water plane's error reaches a tenth of the range at
   * sqrt(0.1) x the horizon, and past that the model means nothing. It is
   * whiteout.vision.projection's own bound, and it is why a level camera's
   * footprint stops somewhere rather than running to the horizon. */
  var MAX_FLAT_PLANE_RANGE_ERROR = 0.10;

  function horizonRangeM(heightM) {
    return Math.sqrt(2 * EARTH_MEAN_RADIUS_M * heightM);
  }

  /* The ground the camera's frame covers, as an annular sector about the
   * asset: `near` and `far` in metres along the ground, `bearing` the
   * boresight clockwise from North, `halfAngle` the horizontal half field of
   * view.
   *
   * **Degrees in, radians out.** `pose.heading` and `pose.pitch` arrive in
   * the degrees the log spells them in, and the camera's fields of view in
   * the degrees `ARENA.md` publishes; everything returned is radians, which
   * is what `screenAngle` and the canvas take. See `radians()` above on why
   * that seam is where it is.
   *
   * The near and far edges are the bottom and top rows of the frame:
   * depression theta +/- VFOV/2, and a range of h / tan(depression). A ray
   * at or above the horizontal never meets the water, so the far edge is
   * clamped to the flat-plane bound rather than run to infinity. This is the
   * shape of whiteout.vision.projection's frame corners and not a second
   * answer to it; tests/test_viz_shell.py runs both and compares metres.
   *
   * `null` when there is nothing honest to draw: no camera for the class, no
   * height above the water, or a frame pointing entirely at the sky.
   *
   * `limitM` is how far the caller will stand behind a ground range, and it
   * defaults to the flat plane's own bound. The canvas passes the arena's
   * extent instead, because `project_pixel_to_ground`'s own docstring asks
   * it to: the model bound is 12.4 km from the quadcopter and Bellot Strait
   * is 6.5 km end to end, so past the arena there is no water to have seen
   * and a wedge drawn out there is a claim about somebody else's ocean. */
  function groundFootprint(pose, camera, limitM) {
    var height = Number(pose.z);
    if (!camera || !isFinite(height) || height <= 0) { return null; }
    if (!isFinite(Number(pose.heading))) { return null; }
    var bearing = radians(pose.heading);
    /* Pitch is optional on a Pose and nose-up positive, so the depression
     * below the horizontal is its negation, and an absent attitude reads as
     * level rather than as a guess at a tilt. */
    var pitch = isFinite(Number(pose.pitch)) ? radians(pose.pitch) : 0;
    var halfVertical = radians(camera.vfov) / 2;
    var model = Math.sqrt(MAX_FLAT_PLANE_RANGE_ERROR) * horizonRangeM(height);
    var limit = isFinite(Number(limitM)) ? Math.min(model, Number(limitM)) : model;
    var lower = -pitch + halfVertical;
    var upper = -pitch - halfVertical;
    if (lower <= 0) { return null; }
    /* Past the nadir the frame's bottom row looks behind the asset, which on
     * the quadcopter's 99.4 degree lens is the normal case rather than an
     * edge one. The near edge is then the ground under it. */
    var near = lower >= Math.PI / 2 ? 0 : Math.min(limit, height / Math.tan(lower));
    var far = upper <= 0 ? limit : Math.min(limit, height / Math.tan(upper));
    if (!(far > near)) { return null; }
    return { near: near, far: far, bearing: bearing, halfAngle: radians(camera.hfov) / 2 };
  }

  function cameraFor(pose) {
    return CAMERAS[CAMERA_OF_CLASS[String(pose.cls)]] || null;
  }

  /* ── track age ───────────────────────────────────────────────────── */

  /* When each contact's position last moved, per record. A track that has
   * been coasting on a two-minute-old fix and one posted four seconds ago
   * are the same record shape, and the acceptance for this canvas is that
   * they must not look the same — so the age is measured here, from the log,
   * rather than asked of a field the log does not carry. */
  function measureFixAges(records) {
    var ages = [];
    var last = {};
    records.forEach(function (record) {
      var here = {};
      (record.contacts || []).forEach(function (contact) {
        var id = String(contact.contact_id);
        var held = last[id];
        if (!held || held.lat !== contact.lat || held.lon !== contact.lon) {
          held = { lat: contact.lat, lon: contact.lon, since: record.t };
          last[id] = held;
        }
        here[id] = record.t - held.since;
      });
      ages.push(here);
    });
    return ages;
  }

  /* ── the layers ──────────────────────────────────────────────────── */

  function drawOutline(ctx, project, field) {
    var across = field.across;
    /* `--n-600`, not the fainter border above it, because once the field
     * erodes this line is the only thing left saying that cleared water is
     * still water. Swept cells fall below the ramp's floor and draw nothing,
     * so without the shoreline the channel would look shorter every tick. */
    ctx.strokeStyle = token("--n-600");
    ctx.lineWidth = size("--w-hairline");
    ctx.beginPath();
    field.outline.forEach(function (edge) {
      var a = project(field.east[edge[0] * (across + 1) + edge[1]], field.north[edge[0] * (across + 1) + edge[1]]);
      var b = project(field.east[edge[2] * (across + 1) + edge[3]], field.north[edge[2] * (across + 1) + edge[3]]);
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    });
    ctx.stroke();
  }

  /* One quadrilateral per water cell, filled from the ramp.
   *
   * **The ramp is indexed by how much likelier a cell is than chance, not by
   * its raw probability**, and that is the whole of what makes this layer
   * readable. A field over 850 cells that has learned nothing still sums to
   * 1, so every cell holds about 0.0012 and the brightest cell is brightest
   * only by rounding — a ramp over the raw values paints a uniform field at
   * full luminance and says "the vessel is certainly everywhere".
   *
   * Against chance it says the true thing instead. A cell nobody has looked
   * at sits at 1x and draws in the ramp's low blue: DESIGN.md's "flat and
   * dim". Water a camera has swept and found empty falls below 1x and drains
   * toward the transparent floor. Water a detection has landed in climbs. So
   * the picture is *information*, which is what the fleet is actually
   * accumulating, and the erosion the negative-information update produces
   * is visible rather than a rounding difference in the top eighth of a ramp.
   *
   * The bounds are an eighth of chance and thirty-two times it, on a log
   * scale, so the two directions are symmetric and neither saturates here. */
  var BELIEF_FLOOR = 1 / 8;
  var BELIEF_CEILING = 32;
  var BELIEF_ALPHA = 0.85;

  /* A byte of `BeliefFrame.cells` to a ramp colour, for one frame. Built per
   * tick rather than per cell: the decode is linear in the code, so 256
   * entries cover all 850 cells. */
  function frameRamp(frame, waterCells) {
    var colours = beliefRamp();
    var uniform = 1 / waterCells;
    var low = Math.log(BELIEF_FLOOR);
    var span = Math.log(BELIEF_CEILING) - low;
    var table = [];
    for (var code = 0; code < 256; code += 1) {
      var probability = (code / 255) * Number(frame.scale);
      if (!(probability > 0)) { table.push(null); continue; }
      var place = (Math.log(probability / uniform) - low) / span;
      table.push(colours[Math.max(0, Math.min(255, Math.round(place * 255)))]);
    }
    return table;
  }

  function drawBelief(ctx, project, field, frame) {
    if (!frame || !frame.cells) { return; }
    var codes = decodeBase64(frame.cells);
    if (codes.length !== field.cells.length) { return; }
    var table = frameRamp(frame, field.cells.length);
    var across = field.across;

    /* Batched by colour, one path per distinct code, and this is not only a
     * speed choice. Adjacent quads share an edge, and filling them one at a
     * time leaves an antialiased hairline of background along every seam —
     * 850 cells of it reads as woven fabric rather than as water. A single
     * `fill()` over a path that already contains both neighbours covers the
     * shared edge exactly once, so the seam is gone and no pixel is painted
     * twice. A uniform field is then one path and one fill. */
    var paths = {};
    for (var i = 0; i < codes.length; i += 1) {
      var colour = table[codes[i]];
      if (!colour) { continue; }
      var path = paths[colour];
      if (!path) { path = paths[colour] = new global.Path2D(); }
      var row = field.cells[i][0];
      var column = field.cells[i][1];
      var topLeft = row * (across + 1) + column;
      var bottomLeft = (row + 1) * (across + 1) + column;
      var a = project(field.east[topLeft], field.north[topLeft]);
      var b = project(field.east[topLeft + 1], field.north[topLeft + 1]);
      var c = project(field.east[bottomLeft + 1], field.north[bottomLeft + 1]);
      var d = project(field.east[bottomLeft], field.north[bottomLeft]);
      path.moveTo(a[0], a[1]);
      path.lineTo(b[0], b[1]);
      path.lineTo(c[0], c[1]);
      path.lineTo(d[0], d[1]);
      path.closePath();
    }

    ctx.save();
    ctx.globalAlpha = BELIEF_ALPHA;
    Object.keys(paths).forEach(function (colour) {
      ctx.fillStyle = colour;
      ctx.fill(paths[colour]);
    });
    ctx.restore();
  }

  /* A bearing clockwise from North, as the canvas angle for a frame whose x
   * runs East and whose y runs *down* the screen while north runs up it. */
  function screenAngle(bearing) {
    return bearing - Math.PI / 2;
  }

  function drawFootprints(ctx, project, record, scale) {
    ctx.save();
    ctx.lineWidth = size("--w-hairline");
    record.observation.poses.filter(isPlaced).forEach(function (pose) {
      var footprint = groundFootprint(pose, cameraFor(pose), state.extent);
      if (!footprint) { return; }
      var local = toLocal(state.origin, Number(pose.lat), Number(pose.lon));
      var apex = project(local.east, local.north);
      var from = screenAngle(footprint.bearing - footprint.halfAngle);
      var to = screenAngle(footprint.bearing + footprint.halfAngle);
      ctx.beginPath();
      ctx.arc(apex[0], apex[1], footprint.near * scale, from, to, false);
      ctx.arc(apex[0], apex[1], footprint.far * scale, to, from, true);
      ctx.closePath();
      /* The class colour at low alpha, so "who is looking where" is readable
       * without the wedge competing with the belief underneath it. */
      ctx.globalAlpha = 0.10;
      ctx.fillStyle = classColour(pose.cls);
      ctx.fill();
      ctx.globalAlpha = 0.45;
      ctx.strokeStyle = classColour(pose.cls);
      ctx.stroke();
    });
    ctx.restore();
  }

  function classColour(cls) {
    var name = String(cls);
    var known = { fixedwing: "--cls-wing", quad: "--cls-quad", rover: "--cls-rover", tower: "--cls-tower" };
    return token(known[name] || "--n-400");
  }

  /* An asset is a dot with a heading tick, in its class colour. A tower's
   * tick is its pan: the thing about a tower that moves is where it looks. */
  function drawAssets(ctx, project, record) {
    var radius = size("--s-1") + 1;
    record.observation.poses.filter(isPlaced).forEach(function (pose) {
      var local = toLocal(state.origin, Number(pose.lat), Number(pose.lon));
      var at = project(local.east, local.north);
      var colour = classColour(pose.cls);
      if (isFinite(Number(pose.heading))) {
        var angle = screenAngle(radians(pose.heading));
        ctx.strokeStyle = colour;
        ctx.lineWidth = size("--w-hairline");
        ctx.beginPath();
        ctx.moveTo(at[0], at[1]);
        ctx.lineTo(at[0] + Math.cos(angle) * radius * 3, at[1] + Math.sin(angle) * radius * 3);
        ctx.stroke();
      }
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(at[0], at[1], radius, 0, Math.PI * 2);
      ctx.fill();
      if (state.selected === pose.asset_id) {
        ctx.strokeStyle = token("--accent");
        ctx.lineWidth = size("--w-hairline");
        ctx.beginPath();
        ctx.arc(at[0], at[1], radius * 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
  }

  /* The track, drawn so that it cannot be mistaken for belief: a hard accent
   * cross where belief is a soft blue wash, and the age of the fix in
   * seconds beside it. A fix that has stopped moving loses its ring first
   * and then goes dashed, so a two-minute-old hold does not read as a live
   * one at a glance. */
  var FIX_STALE_S = 10;

  function drawTrack(ctx, project, record, ages) {
    var arm = size("--s-2");
    (record.contacts || []).forEach(function (contact) {
      if (!isPlaced({ lat: contact.lat, lon: contact.lon })) { return; }
      var local = toLocal(state.origin, Number(contact.lat), Number(contact.lon));
      var at = project(local.east, local.north);
      var age = ages ? ages[String(contact.contact_id)] : undefined;
      var stale = typeof age === "number" && age >= FIX_STALE_S;
      ctx.save();
      ctx.strokeStyle = contact.state === "lost" ? token("--warn") : token("--accent");
      ctx.lineWidth = size("--w-bar");
      if (stale) { ctx.setLineDash([size("--s-1"), size("--s-1")]); }
      ctx.beginPath();
      ctx.moveTo(at[0] - arm, at[1]);
      ctx.lineTo(at[0] + arm, at[1]);
      ctx.moveTo(at[0], at[1] - arm);
      ctx.lineTo(at[0], at[1] + arm);
      ctx.stroke();
      if (!stale) {
        ctx.setLineDash([]);
        ctx.lineWidth = size("--w-hairline");
        ctx.beginPath();
        ctx.arc(at[0], at[1], arm * 1.6, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
      if (typeof age === "number") {
        ctx.fillStyle = token(stale ? "--n-400" : "--n-100");
        ctx.font = canvasFont("--t-12", "--font-mono");
        ctx.textBaseline = "middle";
        ctx.fillText("fix " + age.toFixed(0) + " s", at[0] + arm * 2, at[1]);
      }
    });
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
    el.credit = doc.getElementById("field-credit");
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
    CAMERAS: CAMERAS,
    parseEpisodeLog: parseEpisodeLog,
    toLocal: toLocal,
    groundFootprint: groundFootprint,
    decodeField: decodeField,
    measureFixAges: measureFixAges,
    /* Exported so the canvas can be driven and timed from outside the replay
     * clock. #20 asks for the frame rate measured rather than assumed, and a
     * canvas nobody can ask to redraw cannot be measured. */
    drawField: drawField,
    state: state,
    reducedMotion: reducedMotion
  };

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})(window);
