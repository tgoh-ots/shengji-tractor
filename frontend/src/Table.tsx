import * as React from "react";
import classNames from "classnames";
import { Player, BotRegistration, GameMode } from "./gen-types";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

/*
 * In-game 4-seat table layout.
 *
 * Players are arranged around a central trick area in seating (turn) order,
 * rotated so the local player ("you") is always at the bottom. The opponents
 * fill the left / top (across) / right seats. On portrait/mobile the grid
 * reflows so the opponents become a compact top strip and the hand stays at
 * the bottom (see .sj-table in style.css).
 *
 * Each seat is color-coded by team allegiance so players can tell at a glance
 * who is on whose side (see TeamRole / seatTeamRole below):
 *   - "declarer"  — the landlord who set trump (amber team color + 👑)
 *   - "teammate"  — a revealed member of the landlord's team (amber team color)
 *   - "opponent"  — a known member of the opposing team (cool cyan color)
 *   - "unknown"   — allegiance not yet revealed (FindingFriends only; neutral)
 */

export type SeatKey = "bottom" | "left" | "top" | "right";

/** A player's allegiance, as far as it can be known right now. */
type TeamRole = "declarer" | "teammate" | "opponent" | "unknown";

interface IProps {
  players: Player[];
  bots?: BotRegistration[];
  landlord?: number | null;
  landlordsTeam?: number[];
  /**
   * The active game mode. In fixed-partnership "Tractor" every player not on
   * the landlord's team is a *known* opponent. In FindingFriends, friends are
   * revealed over the hand, so an unrevealed player has unknown allegiance.
   */
  gameMode?: GameMode;
  next?: number | null;
  /** The local player's id (-1 if spectating). */
  selfId: number;
  /** Status rail rendered across the top of the table. */
  status?: React.ReactNode;
  /** The center trick area. */
  center: React.ReactNode;
}

const SEAT_ORDER: SeatKey[] = ["bottom", "left", "top", "right"];
const SEAT_CLASS: Record<SeatKey, string> = {
  bottom: "sj-seat-bottom",
  left: "sj-seat-left",
  top: "sj-seat-top",
  right: "sj-seat-right",
};

const ROLE_CLASS: Record<TeamRole, string> = {
  declarer: "is-team-declarer",
  teammate: "is-team-declarer",
  opponent: "is-team-opponent",
  unknown: "is-team-unknown",
};

/**
 * Determine a player's team role given the current game state.
 *
 * Pure presentation — does not change any game logic.
 */
const seatTeamRole = (
  id: number,
  landlord: number | null | undefined,
  landlordsTeam: number[] | undefined,
  gameMode: GameMode | undefined,
): TeamRole => {
  if (landlord !== undefined && landlord !== null && id === landlord) {
    return "declarer";
  }
  if (landlordsTeam?.includes(id) ?? false) {
    return "teammate";
  }
  // Not on the landlord's team. In fixed-partnership Tractor this player is a
  // definitively-known opponent. In FindingFriends a player who isn't (yet) on
  // the landlord's team could be an unrevealed friend OR an opponent, so we
  // mark them as unknown until they're revealed.
  const isFindingFriends = gameMode !== undefined && gameMode !== "Tractor";
  return isFindingFriends ? "unknown" : "opponent";
};

const Seat = (props: {
  player: Player | null;
  seat: SeatKey;
  role: TeamRole;
  isSelf: boolean;
  isTurn: boolean;
  bot?: BotRegistration;
}): JSX.Element | null => {
  const { t } = useTranslation();
  if (props.player === null) {
    return null;
  }
  const seatLabelKey =
    props.seat === "bottom"
      ? "play.seat.you"
      : props.seat === "left"
        ? "play.seat.left"
        : props.seat === "top"
          ? "play.seat.across"
          : "play.seat.right";

  const role = props.role;
  const roleTitleKey =
    role === "declarer"
      ? "team.declarerTitle"
      : role === "teammate"
        ? "team.teammateTitle"
        : role === "opponent"
          ? "team.opponentTitle"
          : "team.unknownTitle";

  // Team membership is conveyed purely by the seat pill's COLOR (amber =
  // declarer's side, cyan = opponents, dashed slate = unrevealed) plus the 👑
  // on the declarer and a single "You" tag on the local player's own seat.
  // No redundant per-seat role text — that kept the felt cluttered.
  return (
    <div className={classNames("sj-seat", SEAT_CLASS[props.seat])}>
      <span
        className={classNames("sj-seat-name", ROLE_CLASS[role], {
          "is-self": props.isSelf,
          "is-turn": props.isTurn,
        })}
        title={`${props.player.name} · ${t(seatLabelKey)} · ${t(roleTitleKey)}`}
      >
        {props.bot !== undefined && <span aria-hidden="true">🤖</span>}
        <span className="sj-seat-name-text">{props.player.name}</span>
        {role === "declarer" && (
          <span aria-hidden="true" title={t("term.banker")}>
            👑
          </span>
        )}
        {props.isSelf && (
          <span className="sj-seat-you-tag">{t("team.you")}</span>
        )}
      </span>
      {props.bot !== undefined && (
        <span className="sj-seat-badge">
          {t(`ai.difficulty.${props.bot.difficulty}`)}
        </span>
      )}
      <span className="sr-only">{t(roleTitleKey)}</span>
      {props.isTurn && <span className="sr-only">{t("rail.yourTurn")}</span>}
    </div>
  );
};

const Table = (props: IProps): JSX.Element => {
  const { t } = useTranslation();
  const botById = React.useMemo(() => {
    const m: Record<number, BotRegistration> = {};
    (props.bots ?? []).forEach((b) => {
      m[b.player_id] = b;
    });
    return m;
  }, [props.bots]);

  // Rotate the player list so the local player sits at the bottom, then assign
  // the remaining players to left / top / right in seating order.
  const seatAssignments = React.useMemo<
    Array<{ player: Player; seat: SeatKey }>
  >(() => {
    const players = props.players;
    if (players.length === 0) {
      return [];
    }
    let startIdx = players.findIndex((p) => p.id === props.selfId);
    if (startIdx < 0) {
      startIdx = 0;
    }
    const rotated: Player[] = [];
    for (let i = 0; i < players.length; i++) {
      rotated.push(players[(startIdx + i) % players.length]);
    }
    // Distribute up to 4 visible seats. For >4 players we only show the four
    // canonical seats; everyone is still listed in the player list above.
    return rotated.slice(0, 4).map((player, i) => ({
      player,
      seat: SEAT_ORDER[i],
    }));
  }, [props.players, props.selfId]);

  const roleOf = (id: number): TeamRole =>
    seatTeamRole(id, props.landlord, props.landlordsTeam, props.gameMode);

  // A neutral allegiance only exists in FindingFriends.
  const hasUnknownSeats = seatAssignments.some(
    ({ player }) => roleOf(player.id) === "unknown",
  );

  return (
    <div className="sj-table-wrap">
      <div className="sj-table" role="group" aria-label="game table">
        {props.status !== undefined && (
          <div className="sj-table-status">{props.status}</div>
        )}
        {seatAssignments.map(({ player, seat }) => (
          <Seat
            key={player.id}
            player={player}
            seat={seat}
            role={roleOf(player.id)}
            isSelf={player.id === props.selfId}
            isTurn={player.id === props.next}
            bot={botById[player.id]}
          />
        ))}
        <div className="sj-center">{props.center}</div>
      </div>
      {/* One small, fixed legend in its own row directly under the felt. It only
       * decodes the team colors — no per-seat duplication, no "you're on the X
       * side" banner — so nothing overlaps the seats or the trick. */}
      <div className="sj-team-legend" aria-hidden="true">
        <span className="sj-team-legend-key">
          <span className="sj-team-swatch sj-team-swatch-declarer" />
          {t("team.legend.declarerSide")}
        </span>
        <span className="sj-team-legend-key">
          <span className="sj-team-swatch sj-team-swatch-opponent" />
          {t("team.legend.opponents")}
        </span>
        {hasUnknownSeats && (
          <span className="sj-team-legend-key">
            <span className="sj-team-swatch sj-team-swatch-unknown" />
            {t("team.legend.unknown")}
          </span>
        )}
      </div>
    </div>
  );
};

export default Table;
