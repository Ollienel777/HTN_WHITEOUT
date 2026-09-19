/* icons.js — the transport glyphs, and nothing else.
 *
 * DESIGN.md "Components": "Icons: none, beyond the three transport glyphs,
 * drawn as inline SVG paths in viz/icons.js. No icon font, no icon library,
 * and never an emoji."
 *
 * Three paths: play, pause, step. Step-back is step mirrored, so the pair
 * stays one shape and cannot drift apart. Every glyph is drawn in a 24-unit
 * square, fills with `currentColor`, and carries no stroke, so a button's own
 * colour token is the only thing that decides how it looks.
 *
 * A classic script, not a module: index.html opens from file://, where a
 * module fetch fails CORS. It publishes one global.
 */
(function (global) {
  "use strict";

  var VIEWBOX = "0 0 24 24";

  var PATHS = {
    play: "M8 5.2 L19 12 L8 18.8 Z",
    pause: "M8.5 5.5 h2.6 v13 h-2.6 Z M12.9 5.5 h2.6 v13 h-2.6 Z",
    step: "M7 6.6 L14.4 12 L7 17.4 Z M15.8 6.2 h1.9 v11.6 h-1.9 Z"
  };

  /* Returns a fresh <svg> for `name`. `mirror` flips it horizontally, which
   * is how step-back is drawn. The node is decorative: the button around it
   * carries the accessible name. */
  function glyph(name, mirror) {
    var path = PATHS[name];
    if (!path) {
      throw new Error("icons: no glyph named " + name);
    }
    var ns = "http://www.w3.org/2000/svg";
    var svg = global.document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", VIEWBOX);
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    svg.setAttribute("class", "glyph" + (mirror ? " glyph-mirrored" : ""));
    var node = global.document.createElementNS(ns, "path");
    node.setAttribute("d", path);
    node.setAttribute("fill", "currentColor");
    svg.appendChild(node);
    return svg;
  }

  global.WHITEOUT_ICONS = { glyph: glyph, names: Object.keys(PATHS) };
})(window);
