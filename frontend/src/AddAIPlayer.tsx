import * as React from "react";
import { WebsocketContext } from "./WebsocketProvider";
import { useTranslation } from "./i18n";
import { BotDifficulty } from "./gen-types";

import type { JSX } from "react";

/*
 * Lobby control to add / remove AI players (Milestone 1 protocol).
 *
 * Dispatches over the SAME websocket `send` mechanism every other lobby action
 * uses (see Players.tsx / Initialize.tsx), wrapping the action in { Action: ... }:
 *   - Add:    { Action: { AddAIPlayer: { difficulty } } }
 *   - Remove: { Action: { RemoveAIPlayer: playerId } }
 */

const DIFFICULTIES: BotDifficulty[] = ["Easy", "Hard", "Expert", "Omniscient"];

const AddAIPlayer = (): JSX.Element => {
  const { send } = React.useContext(WebsocketContext);
  const { t } = useTranslation();
  const [difficulty, setDifficulty] = React.useState<BotDifficulty>("Hard");

  const addBot = (): void => {
    send({ Action: { AddAIPlayer: { difficulty } } });
  };

  return (
    <div className="sj-rail mb-4 flex flex-wrap items-center gap-3 p-3">
      <span className="text-sm font-semibold" aria-hidden="true">
        🤖 {t("ai.heading")}
      </span>
      <label className="flex items-center gap-2 text-sm">
        <span className="text-[var(--text-secondary)]">
          {t("ai.difficulty")}
        </span>
        <select
          className="sj-input !min-h-[40px] !py-1"
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value as BotDifficulty)}
          aria-label={t("ai.difficulty")}
        >
          {DIFFICULTIES.map((d) => (
            <option key={d} value={d}>
              {t(`ai.difficulty.${d}`)}
            </option>
          ))}
        </select>
      </label>
      <button type="button" className="sj-btn sj-btn-primary" onClick={addBot}>
        + {t("ai.add")}
      </button>
    </div>
  );
};

export default AddAIPlayer;
