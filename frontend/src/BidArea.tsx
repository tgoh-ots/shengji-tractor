import * as React from "react";
import Cards from "./Cards";
import {
  Bid,
  Player,
  Hands,
  Trump,
  BidPolicy,
  BidReinforcementPolicy,
  JokerBidPolicy,
} from "./gen-types";
import { WebsocketContext } from "./WebsocketProvider";
import LabeledPlay from "./LabeledPlay";
import { useEngine } from "./useEngine";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IBidAreaProps {
  bids: Bid[];
  autobid: Bid | null;
  trump?: Trump;
  epoch: number;
  name: string;
  landlord: number | null;
  players: Player[];
  header?: JSX.Element | JSX.Element[];
  prefixButtons?: JSX.Element | JSX.Element[];
  suffixButtons?: JSX.Element | JSX.Element[];
  bidTakeBacksEnabled: boolean;
  bidPolicy: BidPolicy;
  bidReinforcementPolicy: BidReinforcementPolicy;
  jokerBidPolicy: JokerBidPolicy;
  hands: Hands;
  numDecks: number;
  // Whether the temporary "Done bidding" control should be offered to this
  // player (the deck is fully drawn so bidding is the only remaining choice and
  // this player is not the standing winner who finalizes via "Pick up kitty").
  showDoneBidding: boolean;
  // Whether this player has already clicked "Done bidding" (so we show the
  // waiting state instead).
  isDoneBidding: boolean;
}

const BidArea = (props: IBidAreaProps): JSX.Element => {
  const { send } = React.useContext(WebsocketContext);
  const engine = useEngine();
  const { t } = useTranslation();
  const [validBids, setValidBids] = React.useState<Bid[]>([]);
  const [isLoadingBids, setIsLoadingBids] = React.useState<boolean>(false);
  const trump = props.trump == null ? { NoTrump: {} } : props.trump;

  const takeBackBid = (evt: React.SyntheticEvent): void => {
    evt.preventDefault();
    send({ Action: "TakeBackBid" });
  };

  const markDoneBidding = (evt: React.SyntheticEvent, ready: boolean): void => {
    evt.preventDefault();
    send({ Action: { MarkBiddingDone: { ready } } });
  };

  const players: { [playerId: number]: Player } = {};
  let playerId = -1;
  props.players.forEach((p: Player): void => {
    players[p.id] = p;
    if (p.name === props.name) {
      playerId = p.id;
    }
  });

  // Load valid bids when player is not a spectator
  React.useEffect(() => {
    if (playerId >= 0) {
      setIsLoadingBids(true);
      engine
        .findValidBids({
          id: playerId,
          bids: props.bids,
          hands: props.hands,
          players: props.players,
          landlord: props.landlord,
          epoch: props.epoch,
          bid_policy: props.bidPolicy,
          bid_reinforcement_policy: props.bidReinforcementPolicy,
          joker_bid_policy: props.jokerBidPolicy,
          num_decks: props.numDecks,
        })
        .then((bids) => {
          // Sort the bids
          bids.sort((a, b) => {
            if (a.card < b.card) {
              return -1;
            } else if (a.card > b.card) {
              return 1;
            } else if (a.count < b.count) {
              return -1;
            } else if (a.count > b.count) {
              return 1;
            } else {
              return 0;
            }
          });
          setValidBids(bids);
          setIsLoadingBids(false);
        })
        .catch((error) => {
          console.error("Error finding valid bids:", error);
          setValidBids([]);
          setIsLoadingBids(false);
        });
    }
  }, [
    playerId,
    props.bids,
    props.hands,
    props.players,
    props.landlord,
    props.epoch,
    props.bidPolicy,
    props.bidReinforcementPolicy,
    props.jokerBidPolicy,
    props.numDecks,
    engine,
  ]);

  if (playerId === null || playerId < 0) {
    // Spectator mode
    return (
      <div>
        {props.header}
        {props.autobid !== null ? (
          <LabeledPlay
            label={`${players[props.autobid.id].name} ${t("bid.fromBottom")}`}
            trump={trump}
            cards={[props.autobid.card]}
          />
        ) : null}
        {props.bids.map((bid, idx) => {
          const name = players[bid.id].name;
          return (
            <LabeledPlay
              label={name}
              key={idx}
              trump={trump}
              cards={Array(bid.count).fill(bid.card)}
            />
          );
        })}
        {props.bids.length === 0 && props.autobid === null ? (
          <LabeledPlay trump={trump} label={t("bid.noBidsYet")} cards={["🂠"]} />
        ) : null}
      </div>
    );
  } else {
    const levelId =
      props.landlord !== null && props.landlord !== undefined
        ? props.landlord
        : playerId;

    const trump: any =
      props.trump !== null && props.trump !== undefined
        ? props.trump
        : {
            NoTrump: {
              number:
                players[levelId].level !== "NT" ? players[levelId].level : null,
            },
          };

    return (
      <div>
        <div>
          {props.header}
          {props.autobid !== null ? (
            <LabeledPlay
              label={`${players[props.autobid.id].name} ${t("bid.fromBottom")}`}
              cards={[props.autobid.card]}
              trump={trump}
            />
          ) : null}
          {props.bids.map((bid, idx) => {
            const name = players[bid.id].name;
            return (
              <LabeledPlay
                label={name}
                key={idx}
                trump={trump}
                cards={Array(bid.count).fill(bid.card)}
              />
            );
          })}
          {props.trump !== undefined &&
          "NoTrump" in props.trump &&
          props.trump?.NoTrump?.number === null ? (
            <>{t("bid.noBidsInNoTrump")}</>
          ) : props.bids.length === 0 && props.autobid === null ? (
            <LabeledPlay
              trump={trump}
              label={t("bid.noBidsYet")}
              cards={["🂠"]}
            />
          ) : null}
        </div>
        <div className="my-3 flex flex-wrap items-center gap-2">
          {props.prefixButtons}
          {props.bidTakeBacksEnabled ? (
            <button
              onClick={takeBackBid}
              disabled={
                props.bids.length === 0 ||
                props.bids[props.bids.length - 1].id !== playerId ||
                props.bids[props.bids.length - 1].epoch !== props.epoch
              }
              className="sj-btn"
            >
              {t("bid.takeBack")}
            </button>
          ) : null}
          {props.suffixButtons}
        </div>
        {props.showDoneBidding ? (
          props.isDoneBidding ? (
            <div className="my-3 flex flex-wrap items-center gap-2">
              <button
                onClick={(evt) => markDoneBidding(evt, false)}
                className="sj-btn"
              >
                {t("bid.undoDoneBidding")}
              </button>
              <span className="text-sm font-medium text-[var(--text-on-felt-soft)]">
                {t("bid.waitingForOthers")}
              </span>
            </div>
          ) : (
            <div className="my-3 flex flex-wrap items-center gap-2">
              <button
                onClick={(evt) => markDoneBidding(evt, true)}
                className="sj-btn sj-btn-primary"
              >
                {t("bid.doneBidding")}
              </button>
            </div>
          )
        ) : null}
        <p className="mb-1 text-sm font-semibold text-[var(--text-on-felt)]">
          {isLoadingBids
            ? t("bid.loading")
            : validBids.length > 0
              ? t("bid.clickToBid")
              : t("bid.noAvailable")}
        </p>
        {!isLoadingBids &&
          validBids.map((bid, idx) => {
            return (
              <LabeledPlay
                trump={trump}
                cards={Array(bid.count).fill(bid.card)}
                key={idx}
                label={t("bid.option", { n: idx + 1 })}
                onClick={() => {
                  send({ Action: { Bid: [bid.card, bid.count] } });
                }}
              />
            );
          })}
        <Cards hands={props.hands} playerId={playerId} trump={trump} />
      </div>
    );
  }
};

export default BidArea;
