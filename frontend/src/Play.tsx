import * as React from "react";
import { Tooltip } from "react-tooltip";
import ReactModal from "react-modal";
import {
  PlayPhase,
  TrickFormat,
  Hands,
  TrickDrawPolicy,
  FoundViablePlay,
  SuitGroup,
} from "./gen-types";
import Header from "./Header";
import Beeper from "./Beeper";
import Friends from "./Friends";
import Trick, { buildSeatPlays } from "./Trick";
import Cards from "./Cards";
import Points, { calculatePoints, ProgressBarDisplay } from "./Points";
import LabeledPlay from "./LabeledPlay";
import Table from "./Table";
import ArrayUtils from "./util/array";
import AutoPlayButton from "./AutoPlayButton";
import BeepButton from "./BeepButton";
import StatusRail from "./StatusRail";
import { useTranslation } from "./i18n";
import { WebsocketContext } from "./WebsocketProvider";
import { SettingsContext } from "./AppStateProvider";
import { useEngine } from "./useEngine";
import InlineCard from "./InlineCard";
import {
  prefillCardInfoCache,
  prefillExplainScoringCache,
} from "./util/cachePrefill";

import type { JSX } from "react";

const contentStyle: React.CSSProperties = {
  position: "absolute",
  top: "50%",
  left: "50%",
  width: "min(560px, 92vw)",
  padding: "1.25rem 1.5rem",
  transform: "translate(-50%, -50%)",
};

interface IProps {
  playPhase: PlayPhase;
  name: string;
  beepOnTurn: boolean;
  showLastTrick: boolean;
  unsetAutoPlayWhenWinnerChanges: boolean;
  showTrickInPlayerOrder: boolean;
}

const Play = (props: IProps): JSX.Element => {
  const { send } = React.useContext(WebsocketContext);
  const settings = React.useContext(SettingsContext);
  const { t } = useTranslation();
  const [selected, setSelected] = React.useState<string[]>([]);
  const [grouping, setGrouping] = React.useState<FoundViablePlay[]>([]);
  const engine = useEngine();
  const [lastPrefillTrump, setLastPrefillTrump] = React.useState<string | null>(
    null,
  );

  // Helper function to update selection and grouping
  const updateSelectionAndGrouping = async (
    newSelected: string[],
    trump: any,
    tractorRequirements: any,
  ) => {
    setSelected(newSelected);
    try {
      const plays = await engine.findViablePlays(
        trump,
        tractorRequirements,
        newSelected,
      );
      setGrouping(plays);
    } catch (error) {
      console.error("Error finding viable plays:", error);
      setGrouping([]);
    }
  };

  const playCards = (): void => {
    send({ Action: { PlayCardsWithHint: [selected, grouping[0].grouping] } });
    setSelected([]);
    setGrouping([]);
  };

  const sendEvent = (event: object) => () => send(event);
  const takeBackCards = sendEvent({ Action: "TakeBackCards" });
  const endTrick = sendEvent({ Action: "EndTrick" });
  const endGameEarly = sendEvent({ Action: "EndGameEarly" });
  const startNewGame = sendEvent({ Action: "StartNewGame" });

  const { playPhase } = props;

  // TODO: instead of telling who the player is by checking the name, pass in
  // the Player object
  let isSpectator = true;
  let currentPlayer = playPhase.propagated.players.find(
    (p) => p.name === props.name,
  );
  if (currentPlayer === undefined) {
    currentPlayer = playPhase.propagated.observers.find(
      (p) => p.name === props.name,
    );
  } else {
    isSpectator = false;
  }
  if (currentPlayer === undefined) {
    currentPlayer = {
      id: -1,
      name: props.name,
      level: "",
      metalevel: 0,
    };
  }

  // Prefill caches when trump or game parameters change
  React.useEffect(() => {
    const trumpKey = JSON.stringify(playPhase.trump);

    // Only prefill if trump has changed
    if (trumpKey !== lastPrefillTrump) {
      // Trump changed, prefill caches
      setLastPrefillTrump(trumpKey);

      // Prefill card info cache for all cards with the new trump
      prefillCardInfoCache(engine, playPhase.trump).catch((error) => {
        console.error("Failed to prefill card info cache:", error);
      });

      // Prefill explainScoring cache
      if (playPhase.propagated.game_scoring_parameters && playPhase.decks) {
        prefillExplainScoringCache(
          engine,
          playPhase.propagated.game_scoring_parameters,
          playPhase.decks,
        ).catch((error) => {
          console.error("Failed to prefill explainScoring cache:", error);
        });
      }
    }
  }, [
    playPhase.trump,
    playPhase.propagated.game_scoring_parameters,
    playPhase.decks,
    engine,
    lastPrefillTrump,
  ]);

  React.useEffect(() => {
    // When the hands change, our `selected` cards may become invalid, since we
    // could have raced and selected cards that we just played.
    //
    // In that case, let's fix the selected cards.
    const hand =
      currentPlayer.id in playPhase.hands.hands
        ? { ...playPhase.hands.hands[currentPlayer.id] }
        : {};
    selected.forEach((card) => {
      if (card in hand) {
        hand[card] = hand[card] - 1;
      } else {
        hand[card] = -1;
      }
    });

    const toRemove = Object.entries(hand)
      .filter((x) => x[1] < 0)
      .map((x) => x[0]);

    const newSelected = ArrayUtils.minus(selected, toRemove);

    if (toRemove.length > 0) {
      updateSelectionAndGrouping(
        newSelected,
        playPhase.trump,
        playPhase.propagated.tractor_requirements!,
      );
    }
  }, [playPhase.hands.hands, currentPlayer.id, selected]);

  const nextPlayer = playPhase.trick.player_queue[0];
  const lastPlay =
    playPhase.trick.played_cards[playPhase.trick.played_cards.length - 1];

  const [canPlay, setCanPlay] = React.useState(false);

  React.useEffect(() => {
    if (!isSpectator && selected.length > 0) {
      engine
        .canPlayCards({
          trick: playPhase.trick,
          id: currentPlayer!.id,
          hands: playPhase.hands,
          cards: selected,
          trick_draw_policy: playPhase.propagated.trick_draw_policy!,
        })
        .then((playable) => {
          // In order to play the first trick, the grouping must be disambiguated!
          if (lastPlay === undefined) {
            playable = playable && grouping.length === 1;
          }
          playable = playable && !playPhase.game_ended_early;
          setCanPlay(playable);
        })
        .catch((error) => {
          console.error("Error checking if cards can be played:", error);
          setCanPlay(false);
        });
    } else {
      setCanPlay(false);
    }
  }, [
    playPhase.trick,
    currentPlayer.id,
    playPhase.hands,
    selected,
    playPhase.propagated.trick_draw_policy,
    isSpectator,
    lastPlay,
    playPhase.game_ended_early,
    grouping,
    engine,
  ]);

  const isCurrentPlayerTurn = currentPlayer.id === nextPlayer;
  const canTakeBack =
    lastPlay !== undefined &&
    currentPlayer.id === lastPlay.id &&
    !playPhase.game_ended_early;

  // "Finish trick" advances the game once every seat has played. Only the
  // trick's winner (who leads the next trick) may finish it. This prevents a
  // human from finishing a bot-won trick: bots auto-finish their own tricks,
  // and a non-winner clicking "Finish trick" would otherwise be a no-op / poke.
  const trickComplete =
    playPhase.trick.player_queue.length === 0 && !playPhase.game_ended_early;
  const isTrickWinner =
    !isSpectator &&
    playPhase.trick.current_winner !== undefined &&
    playPhase.trick.current_winner !== null &&
    currentPlayer.id === playPhase.trick.current_winner;
  const canFinishTrick = trickComplete && isTrickWinner;

  const shouldBeBeeping =
    props.beepOnTurn && isCurrentPlayerTurn && !playPhase.game_ended_early;

  const remainingCardsInHands = ArrayUtils.sum(
    Object.values(playPhase.hands.hands).map((playerHand) =>
      ArrayUtils.sum(Object.values(playerHand)),
    ),
  );

  const { totalPointsPlayed, nonLandlordPointsWithPenalties } = calculatePoints(
    playPhase.propagated.players,
    playPhase.landlords_team,
    playPhase.points,
    playPhase.penalties,
  );

  const noCardsLeft =
    remainingCardsInHands === 0 && playPhase.trick.played_cards.length === 0;

  const canFinish = noCardsLeft || playPhase.game_ended_early;

  const [canEndGameEarly, setCanEndGameEarly] = React.useState(false);

  React.useEffect(() => {
    if (!canFinish && playPhase.decks) {
      engine
        .nextThresholdReachable({
          decks: playPhase.decks,
          params: playPhase.propagated.game_scoring_parameters!,
          non_landlord_points: nonLandlordPointsWithPenalties,
          observed_points: totalPointsPlayed,
        })
        .then((reachable) => {
          setCanEndGameEarly(!reachable);
        })
        .catch((error) => {
          console.error(
            "Error checking if next threshold is reachable:",
            error,
          );
          setCanEndGameEarly(false);
        });
    } else {
      setCanEndGameEarly(false);
    }
  }, [
    canFinish,
    playPhase.decks,
    playPhase.propagated.game_scoring_parameters,
    nonLandlordPointsWithPenalties,
    totalPointsPlayed,
    engine,
  ]);

  const landlordSuffix =
    playPhase.propagated.landlord_emoji !== undefined &&
    playPhase.propagated.landlord_emoji !== null &&
    playPhase.propagated.landlord_emoji !== ""
      ? playPhase.propagated.landlord_emoji
      : t("play.declarerMark");

  const landlordTeamSize = playPhase.landlords_team.length;
  let configFriendTeamSize = 0;
  let smallerTeamSize = false;
  if (playPhase.game_mode !== "Tractor") {
    configFriendTeamSize =
      playPhase.game_mode.FindingFriends.num_friends != null
        ? playPhase.game_mode.FindingFriends.num_friends + 1
        : playPhase.propagated.players.length / 2;
    smallerTeamSize = landlordTeamSize < configFriendTeamSize;
  }

  // For now, return unsorted cards since sortAndGroupCards needs to be async
  // This function is used in rendering and needs refactoring to handle async
  const getCardsFromHand = (pid: number): SuitGroup[] => {
    const cardsInHand =
      pid in playPhase.hands.hands
        ? Object.entries(playPhase.hands.hands[pid]).flatMap(([c, ct]) =>
            Array(ct).fill(c),
          )
        : [];
    // TODO: Make this async or cache the sorted results
    // For now, return all cards in a single group
    return cardsInHand.length > 0
      ? [
          {
            suit: null as any, // Will be replaced when async is properly handled
            cards: cardsInHand,
          },
        ]
      : [];
  };

  return (
    <div>
      {shouldBeBeeping ? <Beeper /> : null}
      <Header
        gameMode={playPhase.propagated.game_mode}
        chatLink={playPhase.propagated.chat_link}
      />
      <Table
        players={playPhase.propagated.players}
        bots={playPhase.propagated.bots}
        landlord={playPhase.landlord}
        landlordsTeam={playPhase.landlords_team}
        gameMode={playPhase.game_mode}
        next={nextPlayer}
        selfId={currentPlayer.id}
        status={
          <StatusRail
            trump={playPhase.trump}
            declarerName={
              playPhase.landlord !== undefined && playPhase.landlord !== null
                ? (playPhase.propagated.players.find(
                    (p) => p.id === playPhase.landlord,
                  )?.name ?? null)
                : null
            }
            points={nonLandlordPointsWithPenalties}
            turnName={
              playPhase.propagated.players.find((p) => p.id === nextPlayer)
                ?.name ?? null
            }
            isYourTurn={isCurrentPlayerTurn}
          />
        }
        compass={buildSeatPlays({
          trick: playPhase.trick,
          landlord: playPhase.landlord,
          landlords_team: playPhase.landlords_team,
          next: nextPlayer,
        })}
      />
      <Friends gameMode={playPhase.game_mode} showPlayed={true} />
      {playPhase.removed_cards!.length > 0 ? (
        <p className="text-sm text-[var(--text-on-felt-soft)]">
          {t("play.removedNote")}{" "}
          {playPhase.removed_cards!.map((c) => (
            <InlineCard key={c} card={c} />
          ))}
        </p>
      ) : null}
      {settings.showPointsAboveGame && (
        <ProgressBarDisplay
          points={playPhase.points}
          penalties={playPhase.penalties}
          decks={playPhase.decks!}
          trump={playPhase.trump}
          players={playPhase.propagated.players}
          landlordTeam={playPhase.landlords_team}
          landlord={playPhase.landlord}
          hideLandlordPoints={playPhase.propagated.hide_landlord_points!}
          gameScoringParameters={playPhase.propagated.game_scoring_parameters!}
          smallerTeamSize={smallerTeamSize}
        />
      )}
      <div className="my-3 flex flex-wrap items-center gap-2">
        <AutoPlayButton
          onSubmit={playCards}
          playDescription={
            grouping.length === 1 && lastPlay === undefined
              ? grouping[0].description
              : null
          }
          canSubmit={canPlay!}
          currentWinner={playPhase.trick.current_winner!}
          unsetAutoPlayWhenWinnerChanges={props.unsetAutoPlayWhenWinnerChanges}
          isCurrentPlayerTurn={isCurrentPlayerTurn}
        />
        {playPhase.propagated.play_takeback_policy === "AllowPlayTakeback" && (
          <button
            className="sj-btn"
            onClick={takeBackCards}
            disabled={!canTakeBack}
          >
            {t("play.takeBack")}
          </button>
        )}
        <button
          className="sj-btn"
          onClick={endTrick}
          disabled={!canFinishTrick}
        >
          {t("play.finishTrick")}
        </button>
        {canEndGameEarly && (
          <button
            className="sj-btn"
            onClick={() => {
              if (confirm(t("play.confirmEndEarly"))) {
                endGameEarly();
              }
            }}
          >
            {t("play.endEarly")}
          </button>
        )}
        {canFinish && (
          <button className="sj-btn sj-btn-primary" onClick={startNewGame}>
            {t("play.finishGame")}
          </button>
        )}
        <BeepButton />
      </div>
      {canFinish && !noCardsLeft && (
        <div>
          <p>{t("play.cardsRemaining")}</p>
          {playPhase.propagated.players.map((p) => (
            <LabeledPlay
              key={p.id}
              trump={playPhase.trump}
              label={p.name}
              cards={getCardsFromHand(p.id).flatMap((g) => g.cards)}
            />
          ))}
        </div>
      )}
      {!canFinish && (
        <>
          {playPhase.trick.trick_format !== null &&
          !isSpectator &&
          playPhase.trick.player_queue.includes(currentPlayer.id) ? (
            <TrickFormatHelper
              format={playPhase.trick.trick_format!}
              hands={playPhase.hands}
              playerId={currentPlayer.id}
              trickDrawPolicy={playPhase.propagated.trick_draw_policy!}
              setSelected={(newSelected) => {
                updateSelectionAndGrouping(
                  newSelected,
                  playPhase.trump,
                  playPhase.propagated.tractor_requirements!,
                );
              }}
            />
          ) : null}
          {lastPlay === undefined &&
            isCurrentPlayerTurn &&
            grouping.length > 1 && (
              <div className="sj-rail my-2 p-3">
                <p className="m-0 font-semibold">
                  {t("play.multiInterpretation")}
                </p>
                <p className="mb-2 mt-1 text-sm text-[var(--text-secondary)]">
                  {t("play.whichDidYouMean")}
                </p>
                <div className="flex flex-wrap gap-2">
                  {grouping.map((g, gidx) => (
                    <button
                      key={gidx}
                      onClick={(evt) => {
                        evt.preventDefault();
                        setGrouping([g]);
                      }}
                      className="sj-btn"
                    >
                      {g.description}
                    </button>
                  ))}
                </div>
              </div>
            )}
          <div className="sj-hand-fan mt-2">
            <div className="mb-1 flex items-center justify-between px-2">
              <span className="text-sm font-semibold text-[var(--text-on-felt)]">
                {t("play.yourHand")}
              </span>
              {isCurrentPlayerTurn && (
                <span className="sj-chip sj-chip-accent">
                  {t("rail.yourTurn")}
                </span>
              )}
            </div>
            <Cards
              hands={playPhase.hands}
              playerId={currentPlayer.id}
              trump={playPhase.trump}
              selectedCards={selected}
              onSelect={(newSelected) => {
                updateSelectionAndGrouping(
                  newSelected,
                  playPhase.trump,
                  playPhase.propagated.tractor_requirements!,
                );
              }}
              notifyEmpty={isCurrentPlayerTurn}
            />
          </div>
        </>
      )}
      {playPhase.last_trick !== undefined &&
      playPhase.last_trick !== null &&
      props.showLastTrick ? (
        <div>
          <p>{t("play.previousTrick")}</p>
          <Trick
            trick={playPhase.last_trick}
            players={playPhase.propagated.players}
            landlord={playPhase.landlord}
            landlord_suffix={landlordSuffix}
            landlords_team={playPhase.landlords_team}
            name={props.name}
            showTrickInPlayerOrder={props.showTrickInPlayerOrder}
          />
        </div>
      ) : null}
      {playPhase.propagated.game_scoring_parameters ? (
        <Points
          points={playPhase.points}
          penalties={playPhase.penalties}
          decks={playPhase.decks || []}
          players={playPhase.propagated.players}
          landlordTeam={playPhase.landlords_team}
          landlord={playPhase.landlord}
          trump={playPhase.trump}
          hideLandlordPoints={
            playPhase.propagated.hide_landlord_points || false
          }
          gameScoringParameters={playPhase.propagated.game_scoring_parameters}
          smallerTeamSize={smallerTeamSize}
        />
      ) : null}
      <LabeledPlay
        trump={playPhase.trump}
        className="kitty"
        cards={playPhase.kitty}
        label={t("term.kitty")}
      />
    </div>
  );
};

const HelperContents = (props: {
  format: TrickFormat;
  hands: Hands;
  playerId: number;
  trickDrawPolicy: TrickDrawPolicy;
  setSelected: (selected: string[]) => void;
}): JSX.Element => {
  const engine = useEngine();
  const [decomp, setDecomp] = React.useState<any[]>([]);
  const [loading, setLoading] = React.useState<boolean>(true);

  React.useEffect(() => {
    let cancelled = false;
    setLoading(true);

    engine
      .decomposeTrickFormat({
        trick_format: props.format,
        hands: props.hands,
        player_id: props.playerId,
        trick_draw_policy: props.trickDrawPolicy,
      })
      .then((result) => {
        if (!cancelled) {
          setDecomp(result);
          setLoading(false);
        }
      })
      .catch((error) => {
        console.error("Error decomposing trick format:", error);
        if (!cancelled) {
          setDecomp([]);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    props.format,
    props.hands,
    props.playerId,
    props.trickDrawPolicy,
    engine,
  ]);

  if (loading) {
    return <div>Loading...</div>;
  }

  if (props.format.is_rainbow) {
    const rainbows = decomp.filter((d) => d.playable.length > 0);
    return (
      <>
        <p>
          <strong>🌈 Rainbow trick!</strong> The leader played cards all of the
          same rank spanning at least 4 suits.
        </p>
        <p>
          If you have a <em>rainbow</em> — the same number of cards as the
          trick, all of the same rank — you <strong>must</strong> play one.
        </p>
        <p>
          The highest-rank rainbow wins. If no follower plays a rainbow, the
          leader wins.
        </p>
        {rainbows.length > 0 ? (
          <>
            <p>You can play:</p>
            <ul>
              {rainbows.map((r, idx) => (
                <li key={idx}>
                  <span
                    style={{ cursor: "pointer" }}
                    onClick={() => props.setSelected(r.playable)}
                  >
                    {r.playable.map((c: string, cidx: number) => (
                      <InlineCard key={cidx} card={c} />
                    ))}
                  </span>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <p>You have no rainbow — you may play any cards.</p>
        )}
      </>
    );
  }

  if (decomp.length === 0) {
    return <div>Unable to analyze format</div>;
  }

  const trickSuit = props.format.suit;
  const bestMatch = decomp.findIndex((d) => d.playable.length > 0);
  const modalContents = (
    <>
      <p>
        In order to win, you have to play {decomp[0].description} in {trickSuit}
      </p>
      {decomp[0].playable.length > 0 && (
        <p>
          It looks like you are able to match this format, e.g. with{" "}
          <span
            style={{ cursor: "pointer" }}
            onClick={() => props.setSelected(decomp[0].playable)}
          >
            {decomp[0].playable.map((c: string, cidx: number) => (
              <InlineCard key={cidx} card={c} />
            ))}
          </span>
        </p>
      )}

      {decomp.length > 1 && props.trickDrawPolicy !== "NoFormatBasedDraw" && (
        <>
          <p>
            If you can&apos;t play that, but you <em>can</em> play one of the
            following, you have to play it
          </p>
          <ol>
            {decomp.slice(1).map((d, idx) => (
              <li
                key={idx}
                style={{
                  fontWeight: idx === bestMatch - 1 ? "bold" : "normal",
                }}
              >
                {d.description} in {trickSuit}
                {idx === bestMatch - 1 && (
                  <>
                    {" "}
                    <span
                      style={{ cursor: "pointer" }}
                      onClick={() => props.setSelected(d.playable)}
                    >
                      (for example:{" "}
                      {d.playable.map((c: string, cidx: number) => (
                        <InlineCard key={cidx} card={c} />
                      ))}
                      )
                    </span>
                  </>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
      <p
        style={{
          fontWeight: bestMatch < 0 ? "bold" : "normal",
        }}
      >
        Otherwise, you have to play as many {trickSuit} as you can. The
        remaining cards can be anything.
      </p>
      {trickSuit !== "Trump" && (
        <p>
          If you have no cards in {trickSuit}, you can play{" "}
          {decomp[0].description} in Trump to potentially win the trick.
        </p>
      )}
    </>
  );

  return modalContents;
};

const TrickFormatHelper = (props: {
  format: TrickFormat;
  hands: Hands;
  playerId: number;
  trickDrawPolicy: TrickDrawPolicy;
  setSelected: (selected: string[]) => void;
}): JSX.Element => {
  const engine = useEngine();
  const [modalOpen, setModalOpen] = React.useState<boolean>(false);
  const [message, setMessage] = React.useState<string>("");
  const [isLoading, setIsLoading] = React.useState<boolean>(false);

  React.useEffect(() => {
    setMessage("");
  }, [props.hands]);

  return (
    <div className="my-2 flex flex-wrap items-center gap-2">
      <span className="text-sm font-semibold text-[var(--text-on-felt)]">
        Need help?
      </span>
      <Tooltip id="helpTip" place="top" />
      <button
        data-tooltip-id="helpTip"
        data-tooltip-content="Get help on what you can play"
        aria-label="What can I play?"
        className="sj-btn !min-h-[40px] !px-3"
        onClick={(evt) => {
          evt.preventDefault();
          setModalOpen(true);
        }}
      >
        ?
      </button>
      <Tooltip id="suggestTip" place="top" />
      <button
        data-tooltip-id="suggestTip"
        data-tooltip-content="Suggest a play (not guaranteed to succeed)"
        aria-label="Suggest a play"
        className="sj-btn !min-h-[40px] !px-3"
        disabled={isLoading}
        onClick={async (evt) => {
          evt.preventDefault();
          setIsLoading(true);
          try {
            const decomp = await engine.decomposeTrickFormat({
              trick_format: props.format,
              hands: props.hands,
              player_id: props.playerId,
              trick_draw_policy: props.trickDrawPolicy,
            });
            const bestMatch = decomp.findIndex((d) => d.playable.length > 0);
            if (bestMatch >= 0) {
              props.setSelected(decomp[bestMatch].playable);
              setMessage("success");
              setTimeout(() => setMessage(""), 500);
            } else {
              setMessage("cannot suggest a play");
              setTimeout(() => setMessage(""), 2000);
            }
          } catch (error) {
            console.error("Error getting play suggestion:", error);
            setMessage("error suggesting play");
            setTimeout(() => setMessage(""), 2000);
          } finally {
            setIsLoading(false);
          }
        }}
      >
        {isLoading ? "..." : "✨"}
      </button>
      {message !== "" && (
        <span
          className="cursor-pointer text-sm font-semibold text-[#ff6b6b]"
          onClick={() => setMessage("")}
        >
          {message}
        </span>
      )}
      <ReactModal
        isOpen={modalOpen}
        onRequestClose={() => setModalOpen(false)}
        shouldCloseOnOverlayClick
        shouldCloseOnEsc
        style={{ content: contentStyle }}
      >
        {modalOpen && (
          <HelperContents
            format={props.format}
            hands={props.hands}
            playerId={props.playerId}
            trickDrawPolicy={props.trickDrawPolicy}
            setSelected={(sel) => {
              props.setSelected(sel);
              setModalOpen(false);
            }}
          />
        )}
      </ReactModal>
    </div>
  );
};

export default Play;
