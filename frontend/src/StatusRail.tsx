import * as React from "react";
import { Trump } from "./gen-types";
import InlineCard from "./InlineCard";
import preloadedCards from "./preloadedCards";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  trump: Trump;
  declarerName: string | null;
  points: number;
  turnName: string | null;
  isYourTurn: boolean;
}

const TrumpDisplay = ({ trump }: { trump: Trump }): JSX.Element => {
  const { t } = useTranslation();
  if ("Standard" in trump) {
    const { suit, number: rank } = trump.Standard;
    const match = preloadedCards.find(
      (v) => v.typ === suit && v.number === rank,
    );
    return (
      <span className="flex items-center gap-1">
        {match ? <InlineCard card={match.value} /> : suit}
        <span className="text-[var(--text-secondary)]">· {rank}</span>
      </span>
    );
  } else if (
    trump.NoTrump.number !== undefined &&
    trump.NoTrump.number !== null
  ) {
    return (
      <span>
        {t("term.noTrump")} · {trump.NoTrump.number}
      </span>
    );
  } else {
    return <span>{t("term.noTrump")}</span>;
  }
};

const Cell = (props: {
  label: string;
  children: React.ReactNode;
  highlight?: boolean;
}): JSX.Element => (
  <div className="flex min-w-[6rem] flex-col">
    <span className="text-[0.7rem] font-semibold uppercase tracking-wide text-[var(--text-secondary)]">
      {props.label}
    </span>
    <span
      className={
        "text-base font-semibold " +
        (props.highlight
          ? "text-[var(--accent)]"
          : "text-[var(--text-primary)]")
      }
    >
      {props.children}
    </span>
  </div>
);

/*
 * Compact status rail summarizing the live game state: trump suit + level,
 * declarer (庄家), current points (分), and whose turn it is. Rendered above
 * the trick area on the in-game table.
 */
const StatusRail = (props: IProps): JSX.Element => {
  const { t } = useTranslation();
  return (
    <div className="sj-rail mb-4 flex flex-wrap items-center gap-x-6 gap-y-3 p-3">
      <Cell label={`${t("rail.trump")} / 主`}>
        <TrumpDisplay trump={props.trump} />
      </Cell>
      <Cell label={`${t("rail.declarer")} / 庄`}>
        {props.declarerName ?? "—"}
      </Cell>
      <Cell label={`${t("rail.points")} / 分`}>{props.points}</Cell>
      <Cell label={`${t("rail.turn")} / 出牌`} highlight={props.isYourTurn}>
        {props.isYourTurn ? t("rail.yourTurn") : (props.turnName ?? "—")}
      </Cell>
    </div>
  );
};

export default StatusRail;
