import * as React from "react";
import classNames from "classnames";
import { Player, BotRegistration } from "./gen-types";
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
 */

export type SeatKey = "bottom" | "left" | "top" | "right";

interface IProps {
  players: Player[];
  bots?: BotRegistration[];
  landlord?: number | null;
  landlordsTeam?: number[];
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

const Seat = (props: {
  player: Player | null;
  seat: SeatKey;
  isLandlord: boolean;
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

  return (
    <div className={classNames("sj-seat", SEAT_CLASS[props.seat])}>
      <span
        className={classNames("sj-seat-name", {
          "is-landlord": props.isLandlord,
          "is-turn": props.isTurn,
        })}
        title={`${props.player.name} · ${t(seatLabelKey)}`}
      >
        {props.bot !== undefined && <span aria-hidden="true">🤖</span>}
        <span className="overflow-hidden text-ellipsis">
          {props.player.name}
        </span>
        {props.isLandlord && (
          <span aria-hidden="true" title={t("term.banker")}>
            👑
          </span>
        )}
      </span>
      {props.bot !== undefined && (
        <span className="sj-seat-badge">
          {t(`ai.difficulty.${props.bot.difficulty}`)}
        </span>
      )}
      {props.isTurn && <span className="sr-only">{t("rail.yourTurn")}</span>}
    </div>
  );
};

const Table = (props: IProps): JSX.Element => {
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

  const isLandlord = (id: number): boolean =>
    id === props.landlord || (props.landlordsTeam?.includes(id) ?? false);

  return (
    <div className="sj-table" role="group" aria-label="game table">
      {props.status !== undefined && (
        <div className="sj-table-status">{props.status}</div>
      )}
      {seatAssignments.map(({ player, seat }) => (
        <Seat
          key={player.id}
          player={player}
          seat={seat}
          isLandlord={isLandlord(player.id)}
          isTurn={player.id === props.next}
          bot={botById[player.id]}
        />
      ))}
      <div className="sj-center">{props.center}</div>
    </div>
  );
};

export default Table;
