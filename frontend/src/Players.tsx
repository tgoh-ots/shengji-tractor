import * as React from "react";

import classNames from "classnames";
import { MovePlayerLeft, MovePlayerRight } from "./MovePlayerButton";
import { Player, BotRegistration, GameModeSettings } from "./gen-types";
import { WebsocketContext } from "./WebsocketProvider";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  players: Player[];
  observers: Player[];
  bots?: BotRegistration[];
  landlord?: number | null;
  landlords_team?: number[];
  /**
   * The configured game mode. In fixed-partnership "Tractor" the lobby can
   * already show who is on the local player's team (partners sit across — i.e.
   * the engine teams players by seat-index parity; see
   * core/src/game_state/exchange_phase.rs `idx % 2 == landlord_position % 2`).
   * In FindingFriends there are no fixed pre-game teams, so we don't fake them.
   */
  gameMode?: GameModeSettings;
  movable?: boolean;
  next?: number | null;
  name: string;
}

interface IBotRenameProps {
  player: Player;
}

/*
 * Inline rename control shown for a seated bot in the lobby. A pencil affordance
 * swaps the bot's name for a small text input that dispatches a RenameBot action
 * over the same websocket `send` mechanism as the other lobby controls:
 *   { Action: { RenameBot: { player: playerId, name } } }
 * Only bots are renamable; humans never get this control.
 */
const BotRename = ({ player }: IBotRenameProps): JSX.Element => {
  const { send } = React.useContext(WebsocketContext);
  const { t } = useTranslation();
  const [editing, setEditing] = React.useState<boolean>(false);
  const [draft, setDraft] = React.useState<string>(player.name);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const open = (): void => {
    setDraft(player.name);
    setEditing(true);
  };

  const cancel = (): void => {
    setEditing(false);
  };

  const submit = (): void => {
    const trimmed = draft.trim();
    // No-op on empty or unchanged names; the server also validates.
    if (trimmed.length > 0 && trimmed !== player.name) {
      send({ Action: { RenameBot: { player: player.id, name: trimmed } } });
    }
    setEditing(false);
  };

  React.useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  if (!editing) {
    return (
      <button
        type="button"
        className="sj-btn sj-btn-ghost !min-h-[28px] !px-2 !text-xs !text-[var(--text-primary)]"
        onClick={open}
        title={t("ai.rename")}
        aria-label={t("ai.rename")}
      >
        ✏️
      </button>
    );
  }

  return (
    <span className="flex items-center gap-1">
      <input
        ref={inputRef}
        type="text"
        className="sj-input !min-h-[28px] !w-28 !py-0.5 !text-xs"
        value={draft}
        maxLength={32}
        aria-label={t("ai.renameLabel")}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            submit();
          } else if (e.key === "Escape") {
            cancel();
          }
        }}
      />
      <button
        type="button"
        className="sj-btn sj-btn-primary !min-h-[28px] !px-2 !text-xs"
        onClick={submit}
        title={t("ai.renameSave")}
        aria-label={t("ai.renameSave")}
      >
        ✓
      </button>
      <button
        type="button"
        className="sj-btn sj-btn-ghost !min-h-[28px] !px-2 !text-xs !text-[var(--text-primary)]"
        onClick={cancel}
        title={t("ai.renameCancel")}
        aria-label={t("ai.renameCancel")}
      >
        ✕
      </button>
    </span>
  );
};

const Players = (props: IProps): JSX.Element => {
  const {
    players,
    observers,
    bots,
    landlord,
    landlords_team,
    gameMode,
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

  // In fixed-partnership Tractor, partnerships are decided purely by seat
  // (the engine teams players whose seat index shares the landlord's parity —
  // i.e. partners sit across, +2). With no landlord chosen yet we can still
  // show the local player which seats are on their team vs the opponents:
  // everyone at the same seat-parity as "you" is a teammate. Only meaningful
  // once all 4 seats are filled. FindingFriends has no fixed pre-game teams.
  const isTractor = gameMode === "Tractor";
  const selfSeatIndex = players.findIndex((p) => p.name === name);
  const showLobbyTeams =
    isTractor && players.length === 4 && selfSeatIndex >= 0;
  const teamRoleOfSeat = (seatIndex: number): "self-team" | "opponent" =>
    seatIndex % 2 === selfSeatIndex % 2 ? "self-team" : "opponent";

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

  // A small explanatory line under the lobby roster so players understand how
  // teams form before the game starts (and we never fake teams in FF).
  const teamNote: string | null =
    gameMode === undefined
      ? null
      : isTractor
        ? showLobbyTeams
          ? t("lobby.team.fixedHint")
          : null
        : t("lobby.team.decidedInPlay");

  return (
    <>
      <div className="players flex flex-wrap gap-2">
        {players.map((player, seatIndex) => {
          const isLandlord =
            player.id === landlord || landlords_team?.includes(player.id);
          const isNext = player.id === next;
          const bot = botById[player.id];

          // Only color by fixed team when a landlord hasn't taken over the
          // highlight (continuation games can pre-set a landlord).
          const teamRole =
            showLobbyTeams && !isLandlord ? teamRoleOfSeat(seatIndex) : null;

          const className = classNames(
            "player relative flex min-w-[8.5rem] flex-col rounded-xl border px-3 py-2 text-sm transition",
            {
              "border-[var(--accent)] bg-[color-mix(in_srgb,var(--accent)_14%,transparent)]":
                isLandlord,
              "sj-lobby-team-self": teamRole === "self-team",
              "sj-lobby-team-opponent": teamRole === "opponent",
              "border-[var(--border-subtle)] bg-[var(--surface-panel-soft)]":
                !isLandlord && teamRole === null,
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
              {teamRole !== null && (
                <span
                  className={classNames("sj-lobby-team-tag", {
                    "sj-lobby-team-tag-self": teamRole === "self-team",
                    "sj-lobby-team-tag-opponent": teamRole === "opponent",
                  })}
                >
                  {teamRole === "self-team"
                    ? t("lobby.team.yourTeam")
                    : t("lobby.team.opponents")}
                </span>
              )}
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
                    <>
                      <BotRename player={player} />
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
                    </>
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
              <span style={{ textDecoration: "line-through" }}>
                {descriptor}
              </span>
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
      {teamNote !== null && (
        <p className="mt-2 text-xs text-[var(--text-secondary)]">{teamNote}</p>
      )}
    </>
  );
};

export default Players;
