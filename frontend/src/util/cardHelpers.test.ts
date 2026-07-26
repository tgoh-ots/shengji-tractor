import { cardAriaLabel, tryUnicodeToCard, unicodeToCard } from "./cardHelpers";

describe("Card helpers", () => {
  // `unicodeToCard` throws on an unrecognised glyph, and card strings arrive
  // from the server. The throwing form used to be called unguarded inside
  // `InlineCard`, which renders all over the app, so one unexpected glyph
  // escaped to the error boundary and blanked the whole game UI for that
  // client. `tryUnicodeToCard` is the form render paths must use.
  describe("tryUnicodeToCard", () => {
    it("agrees with unicodeToCard on everything it can parse", () => {
      for (const glyph of ["🂤", "🂾", "🃞", "🃂", "🃟", "🃏", "🂠"]) {
        expect(tryUnicodeToCard(glyph)).toEqual(unicodeToCard(glyph));
      }
    });

    it("returns null instead of throwing on anything it cannot", () => {
      // Same inputs the strict form is asserted to throw on above, including
      // the knight cards this deck does not use.
      for (const bogus of ["", "a", "🂷 ", "🂬", "🂼", "🃌", "🃜", "🙂"]) {
        expect(() => tryUnicodeToCard(bogus)).not.toThrow();
        expect(tryUnicodeToCard(bogus)).toBeNull();
      }
    });
  });

  describe("cardAriaLabel", () => {
    it("never throws, whatever it is handed", () => {
      for (const glyph of ["🂠", "🃟", "🃏", "🂤", "", "a", "🂬", "🙂"]) {
        expect(() => cardAriaLabel(glyph)).not.toThrow();
        expect(typeof cardAriaLabel(glyph)).toBe("string");
      }
    });

    it("describes a known card and degrades to a neutral label otherwise", () => {
      expect(cardAriaLabel("🂠")).toBe("face-down card");
      expect(cardAriaLabel("a")).toBe("card");
      expect(cardAriaLabel("🂤")).toContain("spades");
    });
  });

  describe("unicodeToCard", () => {
    it("throws with invalid strings", () => {
      expect(() => unicodeToCard("")).toThrow();
      expect(() => unicodeToCard("a")).toThrow();
      expect(() => unicodeToCard("🂷 ")).toThrow();
    });

    it("works with various cards", () => {
      expect(unicodeToCard("🂤")).toEqual({
        type: "suit_card",
        rank: "4",
        suit: "spades",
      });
      expect(unicodeToCard("🂾")).toEqual({
        type: "suit_card",
        rank: "K",
        suit: "hearts",
      });
      expect(unicodeToCard("🃞")).toEqual({
        type: "suit_card",
        rank: "K",
        suit: "clubs",
      });
      expect(unicodeToCard("🃂")).toEqual({
        type: "suit_card",
        rank: "2",
        suit: "diamonds",
      });
    });

    it("ignores knight cards", () => {
      expect(() => unicodeToCard("🂬")).toThrow();
      expect(() => unicodeToCard("🂼")).toThrow();
      expect(() => unicodeToCard("🃌")).toThrow();
      expect(() => unicodeToCard("🃜")).toThrow();
    });

    it("works with jokers", () => {
      expect(unicodeToCard("🃟")).toEqual({ type: "little_joker" });
      expect(unicodeToCard("🃏")).toEqual({ type: "big_joker" });
    });
  });
});
