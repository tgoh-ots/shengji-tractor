import * as React from "react";
import { GameModeSettings, GameMode } from "./gen-types";

import type { JSX } from "react";

interface IProps {
  gameMode: GameModeSettings | GameMode;
}
const GameModeE = (props: IProps): JSX.Element => {
  const rules = (
    <a
      href="rules.html"
      target="_blank"
      rel="noreferrer"
      className="ml-2 align-middle text-sm font-semibold text-[var(--accent)] underline decoration-[var(--accent)]/40 underline-offset-2"
    >
      rules
    </a>
  );
  if (props.gameMode === "Tractor") {
    return (
      <span>
        升级 <span className="text-[var(--accent)]">Tractor</span>
        {rules}
      </span>
    );
  } else {
    return (
      <span>
        找朋友 <span className="text-[var(--accent)]">Finding Friends</span>
        {rules}
      </span>
    );
  }
};

export default GameModeE;
