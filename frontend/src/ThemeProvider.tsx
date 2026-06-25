import * as React from "react";

import type { JSX } from "react";

/*
 * Theme management for the M3 redesign.
 *
 * The app is DARK-ONLY. We drive the palette from a [data-theme="dark"]
 * attribute on <html>, and (for backward compatibility with the card-glyph
 * color logic that historically keyed off a `.dark-mode` class on <body>, see
 * style.css / Card.tsx) we ALSO toggle that class.
 *
 * The light theme was removed; this provider keeps the same `useTheme()` shape
 * so existing consumers compile, but `theme` is always "dark" and the
 * setters are no-ops.
 */

export type Theme = "dark";

const applyTheme = (): void => {
  const root = document.documentElement;
  root.setAttribute("data-theme", "dark");
  document.body.classList.add("dark-mode");
};

interface ThemeContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

export const ThemeContext = React.createContext<ThemeContextValue>({
  theme: "dark",
  setTheme: () => {},
  toggleTheme: () => {},
});

interface IProps {
  children: React.ReactNode;
}

export const ThemeProvider = (props: IProps): JSX.Element => {
  React.useEffect(() => {
    applyTheme();
  }, []);

  const value = React.useMemo<ThemeContextValue>(
    () => ({
      theme: "dark",
      setTheme: () => {},
      toggleTheme: () => {},
    }),
    [],
  );

  return (
    <ThemeContext.Provider value={value}>
      {props.children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextValue => React.useContext(ThemeContext);
