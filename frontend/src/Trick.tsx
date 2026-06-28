import * as React from "react";

import { Tooltip } from "react-tooltip";
import classNames from "classnames";

import LabeledPlay from "./LabeledPlay";
import { PlayedCards, Player, Trick } from "./gen-types";
import ArrayUtils from "./util/array";

import type { JSX } from "react";

interface IProps {
  players: Player[];
  landlord?: number | null;
  landlord_suffix: string;
  landlords_team?: number[];
  trick: Trick;
  next?: number | null;
  name: string;
  showTrickInPlayerOrder: boolean;
}

/**
 * The footprint of a play, expressed in card "columns" (cards visible across)
 * and "rows" (stacked rows — e.g. a throw that overflows into a `more` strip).
 * Used so every seat's play slot in the compass can be sized to the LARGEST
 * play in the trick, keeping the cross balanced even when one player throws a
 * pair / tractor against three singles.
 */
export interface TrickFootprint {
  cols: number;
  rows: number;
}

/** One player's rendered compass play plus the metadata to lay it out. */
export interface SeatPlay {
  /** The rendered <LabeledPlay> for this player (cards only — no name label). */
  node: JSX.Element;
  /** True when this player is the current winner of the trick. */
  winning: boolean;
  /** True when this player is the "better player" of an attempted throw. */
  better: boolean;
}

/**
 * Shared trick decomposition: turns the raw `Trick` into, per player, the
 * grouped cards. Both the flat previous-trick list and the in-table compass
 * layout build on this so the card-coalescing logic lives in exactly one place.
 */
const decomposeTrick = (
  trick: Trick,
): {
  playedByID: { [id: number]: PlayedCards };
  cardsFromMappingByID: { [id: number]: string[][] };
  betterPlayer: number | null;
  blankCards: string[];
} => {
  const blankCards =
    trick.played_cards.length > 0
      ? Array(trick.played_cards[0].cards.length).fill("🂠")
      : ["🂠"];
  const betterPlayer =
    trick.played_cards.length > 0
      ? (trick.played_cards[0].better_player ?? null)
      : null;

  const playedByID: { [id: number]: PlayedCards } = {};
  const cardsFromMappingByID: { [id: number]: string[][] } = {};

  trick.played_cards.forEach((played, idx) => {
    playedByID[played.id] = played;
    const m = trick.played_card_mappings
      ? trick.played_card_mappings[idx]
      : undefined;
    if (m !== undefined && m !== null && m.length > 0) {
      // We should coalesce blocks of `Repeated` of count 1 together, since
      // that displays more nicely.
      const mapping: string[][] = [];
      const singles: string[] = [];

      m.forEach((mm) => {
        if ("Repeated" in mm && mm.Repeated.count === 1) {
          singles.push(mm.Repeated.card.card);
        } else if ("Repeated" in mm) {
          mapping.push(
            ArrayUtils.range(mm.Repeated.count, (_) => mm.Repeated.card.card),
          );
        } else if ("Tractor" in mm) {
          mapping.push(
            mm.Tractor.members.flatMap((mmm) =>
              ArrayUtils.range(mm.Tractor.count, (_) => mmm.card),
            ),
          );
        }
      });
      mapping.push(singles);

      cardsFromMappingByID[played.id] = mapping;
    }
  });

  return { playedByID, cardsFromMappingByID, betterPlayer, blankCards };
};

/**
 * Measure a play's footprint. `cols` = the most cards shown side-by-side in any
 * row; `rows` = stacked rows (grouped runs render left-to-right as one row; a
 * `more`/overflow strip adds a row). It only needs to be big enough that the
 * uniform slot never clips a play.
 */
const footprintOf = (
  cards: string[],
  groupedCards: string[][] | undefined,
  moreCards: string[] | undefined,
): TrickFootprint => {
  let cols: number;
  let rows = 1;
  if (groupedCards !== undefined && groupedCards.length > 0) {
    // Grouped runs render left-to-right on one row, so total width is the sum.
    cols = groupedCards.reduce((acc, g) => acc + g.length, 0);
  } else {
    cols = cards.length;
  }
  cols = Math.max(cols, 1);
  if (moreCards !== undefined && moreCards.length > 0) {
    rows += 1;
    cols = Math.max(cols, moreCards.length);
  }
  return { cols, rows };
};

/** The little "(!)" winner / "(-)" better-player marker shown beside a name. */
export const winnerSuffix = (
  winning: boolean,
  better: boolean,
): JSX.Element => {
  if (winning) {
    return (
      <>
        {" "}
        <Tooltip id="winningTip" place="bottom" />
        <span
          data-tooltip-id="winningTip"
          data-tooltip-content="Current winner of trick"
        >
          (<code>!</code>)
        </span>
      </>
    );
  }
  if (better) {
    return (
      <>
        {" "}
        <Tooltip id="betterTip" place="bottom" />
        <span
          data-tooltip-id="betterTip"
          data-tooltip-content="First player who can prevent the attempted throw"
        >
          (<code>-</code>)
        </span>
      </>
    );
  }
  return <></>;
};

/**
 * Build, for every player that has played this trick, the rendered play to drop
 * into their compass seat, plus the uniform footprint that all four slots should
 * adopt (= the largest play in the trick). The name label is intentionally left
 * OUT of the returned node — the table renders the team-colored name pill beside
 * each seat, so the play itself is just the cards.
 */
export const buildSeatPlays = (
  props: Pick<IProps, "trick" | "landlord" | "landlords_team" | "next">,
): { plays: { [id: number]: SeatPlay }; uniform: TrickFootprint } => {
  const { cardsFromMappingByID, betterPlayer } = decomposeTrick(props.trick);

  const plays: { [id: number]: SeatPlay } = {};
  let uniform: TrickFootprint = { cols: 1, rows: 1 };

  props.trick.played_cards.forEach((played) => {
    const id = played.id;
    const winning = props.trick.current_winner === id;
    const better = betterPlayer === id;
    const cards = played.cards;
    const groupedCards = cardsFromMappingByID[id];
    const moreCards = played.bad_throw_cards;
    const footprint = footprintOf(cards, groupedCards, moreCards);
    uniform = {
      cols: Math.max(uniform.cols, footprint.cols),
      rows: Math.max(uniform.rows, footprint.rows),
    };

    const className = classNames(
      winning ? "winning" : props.trick.player_queue[0] === id ? "notify" : "",
      {
        landlord: id === props.landlord || props.landlords_team?.includes(id),
      },
    );

    plays[id] = {
      winning,
      better,
      node: (
        <LabeledPlay
          id={id}
          label={null}
          className={className}
          groupedCards={groupedCards}
          cards={cards}
          trump={props.trick.trump}
          next={props.next}
          moreCards={moreCards}
        />
      ),
    };
  });

  return { plays, uniform };
};

const TrickE = (props: IProps): JSX.Element => {
  const namesById = ArrayUtils.mapObject(props.players, (p: Player) => [
    String(p.id),
    p.name,
  ]);

  const { playedByID, cardsFromMappingByID, betterPlayer, blankCards } =
    decomposeTrick(props.trick);

  let playOrder: number[] = [];
  if (props.showTrickInPlayerOrder) {
    playOrder = props.players.map((p) => p.id);
  } else {
    props.trick.played_cards.forEach((played) => playOrder.push(played.id));
    props.trick.player_queue.forEach((id) => playOrder.push(id));
  }

  const isRainbow = props.trick.trick_format?.is_rainbow === true;

  return (
    <div className="trick">
      {isRainbow && (
        <div style={{ fontWeight: "bold", marginBottom: "4px" }}>
          🌈 Rainbow trick — play same rank across ≥4 suits to counter
        </div>
      )}
      {playOrder.map((id) => {
        const winning = props.trick.current_winner === id;
        const better = betterPlayer === id;
        const cards = id in playedByID ? playedByID[id].cards : blankCards;
        const suffix = winnerSuffix(winning, better);

        const className = classNames(
          winning
            ? "winning"
            : props.trick.player_queue[0] === id
              ? "notify"
              : "",
          {
            landlord:
              id === props.landlord || props.landlords_team?.includes(id),
          },
        );

        return (
          <LabeledPlay
            key={id}
            id={id}
            label={
              <>
                {namesById[id] +
                  (id === props.landlord ? " " + props.landlord_suffix : "")}
                {suffix}
              </>
            }
            className={className}
            groupedCards={cardsFromMappingByID[id]}
            cards={cards}
            trump={props.trick.trump}
            next={props.next}
            moreCards={playedByID[id]?.bad_throw_cards}
          />
        );
      })}
    </div>
  );
};

export default TrickE;
