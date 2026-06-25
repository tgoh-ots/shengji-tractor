import * as React from "react";
import { useTranslation } from "./i18n";
import About from "./About";

import type { JSX } from "react";

/*
 * Top-right control cluster: About and the language toggle. Rendered as a
 * floating rail so it's reachable on every screen.
 *
 * (The app is dark-only, so there is no theme toggle; the four-color-suit
 * toggle lives in the Settings pane.)
 */
const Toolbar = (): JSX.Element => {
  const { t, lang, toggleLang } = useTranslation();
  const [aboutOpen, setAboutOpen] = React.useState<boolean>(false);

  const aboutLabel = lang === "zh" ? "关于" : "About";

  return (
    <div className="sj-rail fixed right-3 top-3 z-50 flex items-center gap-1 p-1">
      <button
        type="button"
        className="sj-btn sj-btn-ghost !min-h-[40px] !px-3 !text-[var(--text-primary)]"
        onClick={() => setAboutOpen(true)}
        aria-label={aboutLabel}
        title={aboutLabel}
      >
        <span aria-hidden="true">ℹ️</span>
      </button>
      <About isOpen={aboutOpen} onRequestClose={() => setAboutOpen(false)} />
      <button
        type="button"
        className="sj-btn sj-btn-ghost !min-h-[40px] !px-3 !text-[var(--text-primary)]"
        onClick={toggleLang}
        aria-label={t("toolbar.language")}
        title={t("toolbar.language")}
      >
        {t("toolbar.language")}
      </button>
    </div>
  );
};

export default Toolbar;
