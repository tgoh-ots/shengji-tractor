/*
 * Standalone strategy-page script.
 *
 * Loaded as an EXTERNAL, same-origin <script src="strategy.js"> AFTER
 * guide-common.js, so it runs under the production Content-Security-Policy
 * (script-src 'self' 'wasm-unsafe-eval'). There is intentionally NO inline JS
 * and NO inline event-handler attributes on the page.
 *
 * The Basics/Advanced switch itself is CSS-ONLY (hidden radio inputs +
 * :checked sibling selectors in guide.css), so the page is fully usable with
 * JavaScript disabled. This file only adds the niceties:
 *   1. Language: ?lang=en|zh, plus the shared top nav.
 *   2. Tab deep-link: ?tab=advanced opens the Advanced panel, and selecting a
 *      tab rewrites the URL so the page can be linked/shared/reloaded in place.
 *   3. Card examples: render the static glyph spans as REAL SVG card faces.
 */
(function () {
  "use strict";

  var guide = window.ShengjiGuide;

  // --- 1. Language + shared navigation. ---
  var lang = guide.readLang();
  guide.applyLang(lang);
  guide.markNav(lang);
  var body = document.getElementById("body");

  // --- 2. Tab deep-link (?tab=basics|advanced). ---
  (function () {
    var params = new URLSearchParams(window.location.search);
    var basics = document.getElementById("tab-basics");
    var advanced = document.getElementById("tab-advanced");
    if (!basics || !advanced) return;

    if ((params.get("tab") || "").toLowerCase() === "advanced") {
      advanced.checked = true;
    }

    // Keep the address bar in sync so a reader can copy the URL of the tab
    // they are actually looking at. replaceState (not a navigation) leaves the
    // CSS-only switch untouched.
    function syncUrl(tab) {
      if (!window.history || !window.history.replaceState) return;
      var next = new URLSearchParams(window.location.search);
      next.set("tab", tab);
      window.history.replaceState(
        null,
        "",
        window.location.pathname + "?" + next.toString(),
      );
    }
    basics.addEventListener("change", function () {
      if (basics.checked) syncUrl("basics");
    });
    advanced.addEventListener("change", function () {
      if (advanced.checked) syncUrl("advanced");
    });
  })();

  // --- 3. Card examples: render real SVG card faces. ---
  var fourColor = body.classList.contains("four-color");
  guide
    .loadCardArt()
    .then(function (svgMap) {
      guide.renderStaticCards(svgMap, fourColor);
    })
    .catch(function (err) {
      // If the art can't be fetched the glyphs remain as their text fallback
      // and every example still reads correctly from the prose.
      // eslint-disable-next-line no-console
      console.error("strategy.js: failed to load card art", err);
    });
})();
