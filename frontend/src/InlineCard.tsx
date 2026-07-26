import * as React from "react";
import styled from "styled-components";
import {
  cardToUnicodeSuit,
  ISuitCard,
  tryUnicodeToCard,
} from "./util/cardHelpers";
import { SettingsContext } from "./AppStateProvider";
import { ISuitOverrides } from "./state/Settings";

import type { JSX } from "react";

const InlineCardBase = styled.span`
  padding-left: 0.1em;
  padding-right: 0.1em;
`;

function Suit(
  className: string,
): React.FunctionComponent<React.PropsWithChildren<object>> {
  const component = (props: object): JSX.Element => {
    const settings = React.useContext(SettingsContext);
    return (
      <InlineCardBase
        className={className}
        {...props}
        style={{
          color: settings.suitColorOverrides[className as keyof ISuitOverrides],
        }}
      />
    );
  };
  component.displayName = "Suit";
  return component;
}
const Diamonds = Suit("♢");
const Hearts = Suit("♡");
const Spades = Suit("♤");
const Clubs = Suit("♧");
const LittleJoker = Suit("🃟");
const BigJoker = Suit("🃏");
const Unknown = Suit("🂠");

const suitComponent = (
  suitCard: ISuitCard,
): React.FunctionComponent<React.PropsWithChildren<object>> => {
  switch (suitCard.suit) {
    case "diamonds":
      return Diamonds;
    case "hearts":
      return Hearts;
    case "clubs":
      return Clubs;
    case "spades":
      return Spades;
  }
};

interface IProps {
  card: string;
}

const InlineCard = (props: IProps): JSX.Element => {
  // Card glyphs come off the wire, so an unrecognised one is possible. Render it
  // as face-down rather than throwing: this component is used all over the app,
  // and a throw here would escape to the error boundary and blank the whole game.
  const card = tryUnicodeToCard(props.card);
  if (card === null) {
    return <Unknown>🂠</Unknown>;
  }
  switch (card.type) {
    case "unknown":
      return <Unknown>🂠</Unknown>;
    case "big_joker":
      return <BigJoker>HJ</BigJoker>;
    case "little_joker":
      return <LittleJoker>LJ</LittleJoker>;
    case "suit_card":
      // eslint-disable-next-line no-case-declarations
      const Component = suitComponent(card);
      return (
        <Component>
          {card.rank}
          {cardToUnicodeSuit(card)}
        </Component>
      );
  }
};

export default InlineCard;
