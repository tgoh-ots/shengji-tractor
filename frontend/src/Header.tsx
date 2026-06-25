import * as React from "react";
import GameMode from "./GameMode";
import GameStatisticsButton from "./GameStatisticsButton";
import SettingsButton from "./SettingsButton";
import { GameModeSettings } from "./gen-types";

import type { JSX } from "react";

interface IProps {
  gameMode: GameModeSettings;
  chatLink?: string | null;
}

const Header = (props: IProps): JSX.Element => (
  <div className="mb-4">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h1 className="m-0 text-lg font-bold tracking-tight text-[var(--text-on-felt)] sm:text-xl">
        <GameMode gameMode={props.gameMode} />
      </h1>
      <div className="flex items-center gap-1 text-[var(--text-on-felt)]">
        <SettingsButton />
        <GameStatisticsButton />
      </div>
    </div>
    {props.chatLink !== undefined && props.chatLink !== null ? (
      <p className="mt-1 text-sm text-[var(--text-on-felt-soft)]">
        Join the chat at{" "}
        <a
          href={props.chatLink}
          target="_blank"
          rel="noreferrer"
          className="text-[var(--accent)] underline"
        >
          {props.chatLink}
        </a>
      </p>
    ) : null}
  </div>
);

export default Header;
