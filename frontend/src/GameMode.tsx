import * as React from "react";
import { GameModeSettings, GameMode } from "./gen-types";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  gameMode: GameModeSettings | GameMode;
}
const GameModeE = (props: IProps): JSX.Element => {
  const { lang } = useTranslation();
  const rules = (
    <a
      href={`rules.html?lang=${lang}`}
      target="_blank"
      rel="noreferrer"
      className="ml-2 align-middle text-sm font-semibold text-[var(--accent)] underline decoration-[var(--accent)]/40 underline-offset-2"
    >
      rules
    </a>
  );
  const isZh = lang === "zh";
  if (props.gameMode === "Tractor") {
    return (
      <span>
        {isZh && "升级 "}
        <span className="text-[var(--accent)]">Tractor</span>
        {rules}
      </span>
    );
  } else {
    return (
      <span>
        {isZh && "找朋友 "}
        <span className="text-[var(--accent)]">Finding Friends</span>
        {rules}
      </span>
    );
  }
};

export default GameModeE;
