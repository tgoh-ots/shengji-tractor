import * as React from "react";

import type { JSX } from "react";

/*
 * Theme management for the M3 redesign.
 *
 * We drive the palette from a [data-theme="light"|"dark"] attribute on <html>.
 * For backward compatibility with the existing card-glyph color logic (which
 * historically keyed off a `.dark-mode` class on <body>, see style.css /
 * Card.tsx), we ALSO toggle that class.
 *
 * The choice persists under the legacy "dark_mode" localStorage key ("on"/"off")
 * so it stays in sync with the existing Settings toggle.
 */

export type Theme = "light" | "dark";

const STORAGE_KEY = "dark_mode";

const loadTheme = (): Theme => {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "on") return "dark";
    if (stored === "off") return "light";
    if (
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    ) {
      return "dark";
    }
  } catch {
    // ignore
  }
  return "light";
};

const applyTheme = (theme: Theme): void => {
  const root = document.documentElement;
  root.setAttribute("data-theme", theme);
  if (theme === "dark") {
    document.body.classList.add("dark-mode");
  } else {
    document.body.classList.remove("dark-mode");
  }
};

interface ThemeContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

export const ThemeContext = React.createContext<ThemeContextValue>({
  theme: "light",
  setTheme: () => {},
  toggleTheme: () => {},
});

interface IProps {
  children: React.ReactNode;
}

export const ThemeProvider = (props: IProps): JSX.Element => {
  const [theme, setThemeState] = React.useState<Theme>(() => loadTheme());

  React.useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const setTheme = React.useCallback((next: Theme) => {
    setThemeState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next === "dark" ? "on" : "off");
    } catch {
      // ignore
    }
  }, []);

  const value = React.useMemo<ThemeContextValue>(
    () => ({
      theme,
      setTheme,
      toggleTheme: () => setTheme(theme === "dark" ? "light" : "dark"),
    }),
    [theme, setTheme],
  );

  return (
    <ThemeContext.Provider value={value}>
      {props.children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextValue => React.useContext(ThemeContext);
