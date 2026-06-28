/*
 * Standalone rules-page script.
 *
 * This file is loaded as an EXTERNAL, same-origin <script src="rules.js">, so
 * it runs under the production Content-Security-Policy (script-src 'self'
 * 'wasm-unsafe-eval'), which blocks inline scripts. There is intentionally NO
 * inline JS and NO inline event-handler attributes anywhere on the page.
 *
 * Responsibilities:
 *   1. Language: read ?lang=en|zh and toggle body classes.
 *   2. Mode deep-link: ?mode=friends checks the Finding-Friends radio. (The
 *      Tractor/Finding-Friends tab itself is CSS-only and works without JS.)
 *   3. Deck selector: recompute the round-result point-threshold table.
 *   4. Card examples: render REAL card faces as inline SVG (fetched from
 *      rules-cards.json, which mirrors the app's /cards.json SVG art) — both
 *      the dynamically-generated example plays and the static glyph examples
 *      embedded in the prose. No unicode playing-card glyphs are relied upon.
 */
(function () {
  "use strict";

  // --- 1. Language: read ?lang=en|zh (default en) and toggle body classes. ---
  var params = new URLSearchParams(window.location.search);
  var lang = (params.get("lang") || "en").toLowerCase();
  if (lang !== "zh") lang = "en";
  var body = document.getElementById("body");
  body.classList.remove("lang-en", "lang-zh");
  body.classList.add(lang === "zh" ? "lang-zh" : "lang-en");
  document.documentElement.setAttribute("lang", lang === "zh" ? "zh" : "en");

  // --- 2. Round-result deck selector: recompute the point-threshold table. ---
  (function () {
    var sel = document.getElementById("n-select");
    if (!sel) return;
    function setPointThresholds(evt) {
      var v = evt.target.value;
      function set(id, text) {
        var el = document.getElementById(id);
        if (el) el.innerText = text;
      }
      if (v === "n") {
        set("n-1", "5 to n-5");
        set("n-2", "n to 2n-5");
        set("n-3", "2n to 3n-5");
        set("n-4", "3n to 4n-5");
        set("n-5", "4n to 5n-5");
        set("n-6", "5n and above");
      } else {
        var n = parseInt(v, 10) * 20;
        set("n-1", "5 to " + (n - 5));
        set("n-2", n + " to " + (2 * n - 5));
        set("n-3", 2 * n + " to " + (3 * n - 5));
        set("n-4", 3 * n + " to " + (4 * n - 5));
        set("n-5", 4 * n + " to " + (5 * n - 5));
        set("n-6", 5 * n + " and above");
      }
    }
    sel.addEventListener("change", setPointThresholds);
  })();

  // --- 3. Card examples: render real SVG card faces. ---
  var fourColor = body.classList.contains("four-color");

  // Build an <svg> card face element for a given card glyph (display_value).
  // Falls back to a labelled placeholder if the glyph is unknown.
  function makeCardEl(glyph, typClass, svgMap) {
    var span = document.createElement("span");
    span.className = "card" + (typClass ? " " + typClass : "");
    var entry = svgMap[glyph];
    if (entry) {
      span.innerHTML = fourColor ? entry.fourColor : entry.normal;
      span.setAttribute("role", "img");
    } else {
      // Last-resort fallback: show the glyph as text so something renders.
      span.textContent = glyph;
    }
    return span;
  }

  // Append a list of card-info objects (from cards.json) into the element #id.
  function addCards(id, cards, svgMap) {
    var el = document.getElementById(id);
    if (!el) return;
    cards.forEach(function (c) {
      el.appendChild(makeCardEl(c.display_value, c.typ, svgMap));
    });
  }

  // Replace the STATIC card examples embedded in the prose:
  //   <span class="card ♤">🂢</span>  ->  same span, innerHTML = real SVG.
  // The glyph in the span's text is the lookup key.
  function renderStaticCards(svgMap) {
    var spans = document.querySelectorAll(".card");
    spans.forEach(function (span) {
      // Skip spans we already rendered (dynamic ones contain an <svg>).
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

  // Build the dynamic example plays (suits, point cards, trump orderings,
  // jokers) that live in #id containers — mirrors the original logic.
  function renderDynamicCards(cardsJson, svgMap) {
    var CARDS = cardsJson.cards;
    var REVERSED = CARDS.slice().reverse();
    var isJoker = function (c) {
      return c.value === "🃏" || c.typ === "🃟";
    };

    addCards(
      "spades",
      REVERSED.filter(function (c) {
        return c.typ === "♤";
      }),
      svgMap,
    );
    addCards(
      "hearts",
      REVERSED.filter(function (c) {
        return c.typ === "♡";
      }),
      svgMap,
    );
    addCards(
      "diamonds",
      REVERSED.filter(function (c) {
        return c.typ === "♢";
      }),
      svgMap,
    );
    addCards(
      "clubs",
      REVERSED.filter(function (c) {
        return c.typ === "♧";
      }),
      svgMap,
    );
    addCards("jokers", CARDS.filter(isJoker), svgMap);

    addCards(
      "points-d",
      REVERSED.filter(function (c) {
        return c.points > 0 && c.typ === "♢";
      }),
      svgMap,
    );
    addCards(
      "points-c",
      REVERSED.filter(function (c) {
        return c.points > 0 && c.typ === "♧";
      }),
      svgMap,
    );
    addCards(
      "points-h",
      REVERSED.filter(function (c) {
        return c.points > 0 && c.typ === "♡";
      }),
      svgMap,
    );
    addCards(
      "points-s",
      REVERSED.filter(function (c) {
        return c.points > 0 && c.typ === "♤";
      }),
      svgMap,
    );

    // Trump suit when specification is 4♤ (spades): non-4 spades, then off-suit
    // 4s, then 4♤, then jokers (highest to lowest as laid out left->right).
    addCards(
      "trump-4s",
      REVERSED.filter(function (c) {
        return c.typ === "♤" && c.number !== "4";
      }),
      svgMap,
    );
    addCards(
      "trump-4s",
      CARDS.filter(function (c) {
        return c.typ !== "♤" && c.number === "4";
      }),
      svgMap,
    );
    addCards(
      "trump-4s",
      CARDS.filter(function (c) {
        return c.typ === "♤" && c.number === "4";
      }),
      svgMap,
    );
    addCards("trump-4s", CARDS.filter(isJoker), svgMap);
    addCards(
      "trump-4s-hearts",
      REVERSED.filter(function (c) {
        return c.typ === "♡" && c.number !== "4";
      }),
      svgMap,
    );

    // Trump suit when specification is 4NT (no suit): all 4s, then jokers.
    addCards(
      "trump-4nt",
      CARDS.filter(function (c) {
        return c.number === "4";
      }),
      svgMap,
    );
    addCards("trump-4nt", CARDS.filter(isJoker), svgMap);
    addCards(
      "trump-4nt-hearts",
      REVERSED.filter(function (c) {
        return c.typ === "♡" && c.number !== "4";
      }),
      svgMap,
    );

    // "Not identical / not a pair": one big + one small joker.
    addCards("jokers2", CARDS.filter(isJoker), svgMap);
  }

  // Fetch both the card metadata (/cards.json, served by the backend in prod)
  // and the SVG art map (rules-cards.json, shipped alongside this page).
  Promise.all([
    fetch("cards.json").then(function (r) {
      return r.json();
    }),
    fetch("rules-cards.json").then(function (r) {
      return r.json();
    }),
  ])
    .then(function (results) {
      var cardsJson = results[0];
      var svgMap = results[1];
      renderDynamicCards(cardsJson, svgMap);
      renderStaticCards(svgMap);
    })
    .catch(function (err) {
      // If the data can't be fetched, the dynamic examples stay empty and the
      // static glyphs remain as their (text) fallback; the prose still
      // explains everything.
      // eslint-disable-next-line no-console
      console.error("rules.js: failed to load card art", err);
    });
})();
