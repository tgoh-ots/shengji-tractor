import * as React from "react";

import type { JSX } from "react";

/*
 * Presentational primitives for the lobby game-settings UI.
 *
 * These are pure layout/styling helpers — they carry NO settings logic and
 * dispatch NO actions. They exist so that Initialize.tsx and the individual
 * selector components (NumDecksSelector, KittySizeSelector, RankSelector,
 * LandlordSelector, etc.) can present every <select>/<input>/<button> with a
 * consistent, themed look that matches the rest of the redesigned app
 * (.sj-panel / .sj-input / .sj-btn from theme.css).
 */

interface SettingsSectionProps {
  title: string;
  subtitle?: string;
  /** Optional control rendered on the right of the section header. */
  action?: React.ReactNode;
  children: React.ReactNode;
  /** Render a slightly inset/soft surface instead of the full panel. */
  inset?: boolean;
}

/** A titled card surface that groups a set of related settings. */
export const SettingsSection = (props: SettingsSectionProps): JSX.Element => (
  <section
    className={
      (props.inset
        ? "border border-[var(--border-subtle)] bg-[var(--surface-panel-soft)]"
        : "sj-panel") + " mb-4 p-4 sm:p-5"
    }
    style={props.inset ? { borderRadius: "var(--radius-2xl)" } : undefined}
  >
    <header className="mb-3 flex flex-wrap items-baseline justify-between gap-2 border-b border-[var(--border-subtle)] pb-2">
      <div>
        <h3 className="m-0 text-base font-bold tracking-tight text-[var(--text-primary)]">
          {props.title}
        </h3>
        {props.subtitle !== undefined ? (
          <p className="m-0 mt-0.5 text-xs text-[var(--text-secondary)]">
            {props.subtitle}
          </p>
        ) : null}
      </div>
      {props.action !== undefined ? <div>{props.action}</div> : null}
    </header>
    <div className="flex flex-col gap-3">{props.children}</div>
  </section>
);

interface SettingRowProps {
  label: React.ReactNode;
  hint?: React.ReactNode;
  htmlFor?: string;
  children: React.ReactNode;
}

/*
 * A single labeled setting: a description on the left, the control on the
 * right. Stacks vertically on narrow screens. Use this to wrap a <select>,
 * <input>, toggle, or button cluster.
 */
export const SettingRow = (props: SettingRowProps): JSX.Element => (
  <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
    <label
      htmlFor={props.htmlFor}
      className="flex flex-col text-sm font-medium text-[var(--text-primary)] sm:max-w-[60%]"
    >
      <span>{props.label}</span>
      {props.hint !== undefined ? (
        <span className="mt-0.5 text-xs font-normal text-[var(--text-secondary)]">
          {props.hint}
        </span>
      ) : null}
    </label>
    <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
      {props.children}
    </div>
  </div>
);

/*
 * A themed <select>. Forwards every prop straight through, so callers keep
 * using `value`/`onChange` exactly as before — only the styling changes.
 */
export const SettingSelect = (
  props: React.SelectHTMLAttributes<HTMLSelectElement>,
): JSX.Element => {
  const { className, children, ...rest } = props;
  return (
    <select
      className={
        "sj-input !min-h-[40px] w-full !py-1 sm:w-auto sm:min-w-[14rem]" +
        (className !== undefined ? ` ${className}` : "")
      }
      {...rest}
    >
      {children}
    </select>
  );
};

/*
 * A themed <input> (text/number/etc). Same passthrough contract as
 * SettingSelect.
 */
export const SettingInput = (
  props: React.InputHTMLAttributes<HTMLInputElement>,
): JSX.Element => {
  const { className, ...rest } = props;
  return (
    <input
      className={
        "sj-input !min-h-[40px] !py-1" +
        (className !== undefined ? ` ${className}` : "")
      }
      {...rest}
    />
  );
};

/*
 * A themed secondary button — replacement for the legacy `button.normal`
 * styling inside the settings UI.
 */
export const SettingButton = (
  props: React.ButtonHTMLAttributes<HTMLButtonElement>,
): JSX.Element => {
  const { className, children, ...rest } = props;
  return (
    <button
      className={
        "sj-btn !min-h-[40px] !px-3 !py-1 !text-sm" +
        (className !== undefined ? ` ${className}` : "")
      }
      {...rest}
    >
      {children}
    </button>
  );
};
