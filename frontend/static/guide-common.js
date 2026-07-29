/*
 * Shared helpers for the standalone guide pages (rules.html, strategy.html).
 *
 * Loaded as an EXTERNAL, same-origin script BEFORE each page's own script, so
 * everything here runs under the production Content-Security-Policy
 * (script-src 'self' 'wasm-unsafe-eval'). There is intentionally NO inline JS
 * and NO inline event-handler attributes on either page.
 *
 * Exposes a single global, `window.ShengjiGuide`:
 *   readLang()                     -> "en" | "zh" (from ?lang=, default en)
 *   applyLang(lang)                -> toggle the body lang-en/lang-zh classes
 *   markNav(lang)                  -> highlight the current page in the top nav
 *                                     and carry ?lang= across its links
 *   loadCardArt()                  -> Promise<svgMap> from rules-cards.json
 *   renderStaticCards(svgMap, 4c)  -> replace `.card` glyph spans with real SVG
 */
(function () {
  "use strict";

  // --- Language: read ?lang=en|zh (default en). ---
  function readLang() {
    var params = new URLSearchParams(window.location.search);
    var lang = (params.get("lang") || "en").toLowerCase();
    return lang === "zh" ? "zh" : "en";
  }

  function applyLang(lang) {
    var body = document.getElementById("body");
    if (!body) return;
    body.classList.remove("lang-en", "lang-zh");
    body.classList.add(lang === "zh" ? "lang-zh" : "lang-en");
    document.documentElement.setAttribute("lang", lang === "zh" ? "zh" : "en");
  }

  // --- Top navigation: mark the current page, keep ?lang= on every hop. ---
  function markNav(lang) {
    var nav = document.getElementById("guide-nav");
    if (!nav) return;
    var current = nav.getAttribute("data-guide");
    var links = nav.querySelectorAll("a[data-nav]");
    links.forEach(function (link) {
      var target = link.getAttribute("data-nav");
      if (target === current) {
        link.setAttribute("aria-current", "page");
      }
      // Preserve the reader's language across pages (and back into the app,
      // which reads its own localStorage but accepts the same query shape).
      var href = link.getAttribute("href");
      if (href && href.indexOf("?") === -1) {
        link.setAttribute("href", href + "?lang=" + lang);
      }
    });
  }

  // --- Card art: the SVG map shipped alongside these pages. ---
  function loadCardArt() {
    return fetch("rules-cards.json").then(function (r) {
      return r.json();
    });
  }

  /*
   * Replace the STATIC card examples embedded in the prose:
   *   <span class="card ♤">🂢</span>  ->  same span, innerHTML = real SVG.
   * The glyph in the span's text is the lookup key. Spans that already hold an
   * <svg> (dynamically built ones) are skipped, and an unknown glyph is left as
   * its text fallback so something always renders.
   */
  function renderStaticCards(svgMap, fourColor) {
    var spans = document.querySelectorAll(".card");
    spans.forEach(function (span) {
      if (span.querySelector("svg")) return;
      var glyph = (span.textContent || "").trim();
      if (!glyph) return;
      var entry = svgMap[glyph];
      if (entry) {
        span.innerHTML = fourColor ? entry.fourColor : entry.normal;
        span.setAttribute("role", "img");
      }
    });
  }

  window.ShengjiGuide = {
    readLang: readLang,
    applyLang: applyLang,
    markNav: markNav,
    loadCardArt: loadCardArt,
    renderStaticCards: renderStaticCards,
  };
})();
