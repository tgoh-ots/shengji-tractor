import * as React from "react";
import { useTheme } from "./ThemeProvider";
import { useTranslation } from "./i18n";
import { AppStateContext } from "./AppStateProvider";

import type { JSX } from "react";

/*
 * Top-right control cluster: theme toggle, language toggle, and four-color
 * suit toggle. Rendered as a floating rail so it's reachable on every screen.
 */
const Toolbar = (): JSX.Element => {
  const { theme, toggleTheme } = useTheme();
  const { t, toggleLang } = useTranslation();
  const { state, updateState } = React.useContext(AppStateContext);

  const toggleFourColor = (): void => {
    updateState({
      settings: { ...state.settings, fourColor: !state.settings.fourColor },
    });
  };

  return (
    <div className="sj-rail fixed right-3 top-3 z-50 flex items-center gap-1 p-1">
      <button
        type="button"
        className="sj-btn sj-btn-ghost !min-h-[40px] !px-3 !text-[var(--text-primary)]"
        onClick={toggleTheme}
        aria-label={
          theme === "dark"
            ? t("toolbar.theme.toLight")
            : t("toolbar.theme.toDark")
        }
        title={
          theme === "dark"
            ? t("toolbar.theme.toLight")
            : t("toolbar.theme.toDark")
        }
      >
        <span aria-hidden="true">{theme === "dark" ? "☀️" : "🌙"}</span>
      </button>
      <button
        type="button"
        className="sj-btn sj-btn-ghost !min-h-[40px] !px-3 !text-[var(--text-primary)]"
        onClick={toggleLang}
        aria-label={t("toolbar.language")}
        title={t("toolbar.language")}
      >
        {t("toolbar.language")}
      </button>
      <button
        type="button"
        className={
          "sj-btn !min-h-[40px] !px-3 " +
          (state.settings.fourColor ? "sj-btn-primary" : "sj-btn-ghost")
        }
        onClick={toggleFourColor}
        aria-pressed={state.settings.fourColor}
        title={t("toolbar.fourColor")}
      >
        <span aria-hidden="true">🎨</span>
      </button>
    </div>
  );
};

export default Toolbar;
