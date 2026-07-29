import * as React from "react";
import { GameModeSettings, GameMode } from "./gen-types";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  gameMode: GameModeSettings | GameMode;
}
const GameModeE = (props: IProps): JSX.Element => {
  const { lang, t } = useTranslation();
  const linkClass =
    "ml-2 align-middle text-sm font-semibold text-[var(--accent)] underline decoration-[var(--accent)]/40 underline-offset-2";
  const guides = (
    <>
      <a
        href={`rules.html?lang=${lang}`}
        target="_blank"
        rel="noreferrer"
        className={linkClass}
      >
        {t("gameMode.rules")}
      </a>
      <a
        href={`strategy.html?lang=${lang}`}
        target="_blank"
        rel="noreferrer"
        className={linkClass}
      >
        {t("gameMode.strategy")}
      </a>
    </>
  );
  const isZh = lang === "zh";
  if (props.gameMode === "Tractor") {
    return (
      <span>
        {isZh && "升级 "}
        <span className="text-[var(--accent)]">Tractor</span>
        {guides}
      </span>
    );
  } else {
    return (
      <span>
        {isZh && "找朋友 "}
        <span className="text-[var(--accent)]">Finding Friends</span>
        {guides}
      </span>
    );
  }
};

export default GameModeE;
