import * as React from "react";

import classNames from "classnames";
import { MovePlayerLeft, MovePlayerRight } from "./MovePlayerButton";
import { Player, BotRegistration } from "./gen-types";
import { WebsocketContext } from "./WebsocketProvider";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  players: Player[];
  observers: Player[];
  bots?: BotRegistration[];
  landlord?: number | null;
  landlords_team?: number[];
  movable?: boolean;
  next?: number | null;
  name: string;
}

const Players = (props: IProps): JSX.Element => {
  const {
    players,
    observers,
    bots,
    landlord,
    landlords_team,
    movable,
    next,
    name,
  } = props;
  const { send } = React.useContext(WebsocketContext);
  const { t } = useTranslation();

  const botById = React.useMemo(() => {
    const m: Record<number, BotRegistration> = {};
    (bots ?? []).forEach((b) => {
      m[b.player_id] = b;
    });
    return m;
  }, [bots]);

  const makeDescriptor = (p: Player): Array<JSX.Element | string> => {
    if (p.metalevel <= 1) {
      return [`${p.name} (${t("common.rank")} ${p.level})`];
    } else {
      return [
        `${p.name} (${t("common.rank")} ${p.level}`,
        <sup key={`meta-${p.id}`}>{p.metalevel}</sup>,
        ")",
      ];
    }
  };

  return (
    <div className="players flex flex-wrap gap-2">
      {players.map((player) => {
        const isLandlord =
          player.id === landlord || landlords_team?.includes(player.id);
        const isNext = player.id === next;
        const bot = botById[player.id];

        const className = classNames(
          "player relative flex min-w-[8.5rem] flex-col rounded-xl border px-3 py-2 text-sm transition",
          {
            "border-[var(--accent)] bg-[color-mix(in_srgb,var(--accent)_14%,transparent)]":
              isLandlord,
            "border-[var(--border-subtle)] bg-[var(--surface-panel-soft)]":
              !isLandlord,
            "ring-2 ring-[var(--accent)]": isNext,
            movable,
          },
        );

        const descriptor = makeDescriptor(player);
        if (player.id === landlord) {
          descriptor.push(` ${t("term.banker")}`);
        }
        if (player.name === name) {
          descriptor.push(` (${t("common.you")})`);
        }

        return (
          <div key={player.id} className={className}>
            <span className="flex items-center gap-1 font-semibold">
              {bot !== undefined && (
                <span
                  aria-hidden="true"
                  title={`${t("ai.botLabel")} · ${t(
                    `ai.difficulty.${bot.difficulty}`,
                  )}`}
                >
                  🤖
                </span>
              )}
              {descriptor}
            </span>
            {bot !== undefined && (
              <span className="sj-chip sj-chip-accent mt-1 w-fit">
                {t(`ai.difficulty.${bot.difficulty}`)}
              </span>
            )}
            {movable && (
              <span className="mt-2 flex items-center justify-center gap-1">
                <MovePlayerLeft players={players} player={player} />
                {bot !== undefined ? (
                  <button
                    type="button"
                    className="sj-btn sj-btn-ghost !min-h-[28px] !px-2 !text-xs !text-[var(--text-primary)]"
                    onClick={() =>
                      send({ Action: { RemoveAIPlayer: player.id } })
                    }
                    title={t("ai.remove")}
                  >
                    {t("ai.remove")}
                  </button>
                ) : (
                  <span
                    role="button"
                    tabIndex={0}
                    style={{ cursor: "pointer" }}
                    onClick={() =>
                      send({ Action: { MakeObserver: player.id } })
                    }
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        send({ Action: { MakeObserver: player.id } });
                      }
                    }}
                    title="Make observer"
                  >
                    ✔️
                  </span>
                )}
                <MovePlayerRight players={players} player={player} />
              </span>
            )}
          </div>
        );
      })}
      {observers.map((player) => {
        const descriptor = makeDescriptor(player);
        if (player.name === name) {
          descriptor.push(` (${t("common.you")})`);
        }

        return (
          <div
            key={player.id}
            className="player observer flex min-w-[8.5rem] flex-col rounded-xl border border-dashed border-[var(--border-strong)] px-3 py-2 text-sm text-[var(--text-secondary)]"
          >
            <span style={{ textDecoration: "line-through" }}>{descriptor}</span>
            {movable && (
              <span className="mt-2 flex items-center justify-center">
                <span
                  role="button"
                  tabIndex={0}
                  style={{ cursor: "pointer" }}
                  onClick={() => send({ Action: { MakePlayer: player.id } })}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      send({ Action: { MakePlayer: player.id } });
                    }
                  }}
                  title="Make player"
                >
                  💤
                </span>
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default Players;
