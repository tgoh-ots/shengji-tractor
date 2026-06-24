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
 *
 * "Omniscient" is a deliberate CHEATER tier: those bots see every player's cards
 * and play with perfect information. It is surfaced here with a distinct warning
 * badge so nobody confuses it with the three fair (honest) tiers.
 */

const HONEST_DIFFICULTIES: BotDifficulty[] = ["Easy", "Medium", "Hard"];
const CHEATER_DIFFICULTY: BotDifficulty = "Omniscient";

const isCheater = (d: BotDifficulty): boolean => d === CHEATER_DIFFICULTY;

const AddAIPlayer = (): JSX.Element => {
  const { send } = React.useContext(WebsocketContext);
  const { t } = useTranslation();
  const [difficulty, setDifficulty] = React.useState<BotDifficulty>("Medium");

  const addBot = (): void => {
    send({ Action: { AddAIPlayer: { difficulty } } });
  };

  const cheater = isCheater(difficulty);

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
          {HONEST_DIFFICULTIES.map((d) => (
            <option key={d} value={d}>
              {t(`ai.difficulty.${d}`)}
            </option>
          ))}
          {/* Visually set the cheater tier apart with an eye prefix. */}
          <option key={CHEATER_DIFFICULTY} value={CHEATER_DIFFICULTY}>
            👁 {t(`ai.difficulty.${CHEATER_DIFFICULTY}`)}
          </option>
        </select>
      </label>
      {cheater && (
        <span
          className="inline-flex items-center gap-1 rounded border border-red-500 bg-red-100 px-2 py-1 text-xs font-bold uppercase tracking-wide text-red-700"
          role="alert"
          title={t("ai.cheaterWarning")}
        >
          👁 {t("ai.cheaterBadge")}
        </span>
      )}
      <button type="button" className="sj-btn sj-btn-primary" onClick={addBot}>
        + {t("ai.add")}
      </button>
      {cheater && (
        <p className="w-full text-xs text-red-700" role="note">
          {t("ai.cheaterWarning")}
        </p>
      )}
    </div>
  );
};

export default AddAIPlayer;
