import * as React from "react";
import { Tooltip } from "react-tooltip";
import ReactModal from "react-modal";
import { EmojiStyle } from "emoji-picker-react";
import ReadyCheck from "./ReadyCheck";
import LandlordSelector from "./LandlordSelector";
import NumDecksSelector from "./NumDecksSelector";
import KittySizeSelector from "./KittySizeSelector";
import RankSelector from "./RankSelector";
import Kicker from "./Kicker";
import ArrayUtils from "./util/array";
import { RandomizePlayersButton } from "./RandomizePlayersButton";
import {
  CompoundFormats,
  InitializePhase,
  Player,
  PropagatedState,
  Deck,
  TractorRequirements,
} from "./gen-types";
import { WebsocketContext } from "./WebsocketProvider";

import Header from "./Header";
import Players from "./Players";
import AddAIPlayer from "./AddAIPlayer";
import { useTranslation } from "./i18n";
import { GameScoringSettings } from "./ScoringSettings";
import {
  SettingsSection,
  SettingRow,
  SettingSelect,
  SettingInput,
  SettingButton,
} from "./SettingsWidgets";

import type { JSX } from "react";

const Picker = React.lazy(async () => await import("emoji-picker-react"));

interface IDifficultyProps {
  state: InitializePhase;
  setFriendSelectionPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setMultipleJoinPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setAdvancementPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setMaxRank: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setHideLandlordsPoints: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setHidePlayedCards: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setKittyPenalty: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setThrowPenalty: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setPlayTakebackPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setBidTakebackPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
}

const contentStyle: React.CSSProperties = {
  position: "absolute",
  top: "50%",
  left: "50%",
  width: "min(680px, 92vw)",
  transform: "translate(-50%, -50%)",
  padding: "0",
};

interface ISettingsModalProps {
  title: string;
  subtitle?: string;
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

/*
 * Shared themed wrapper around ReactModal for the "advanced settings" groups.
 * Gives every settings modal a consistent header, padding, and Done button.
 */
const SettingsModal = (props: ISettingsModalProps): JSX.Element => (
  <ReactModal
    isOpen={props.isOpen}
    onRequestClose={props.onClose}
    shouldCloseOnOverlayClick
    shouldCloseOnEsc
    style={{ content: contentStyle }}
  >
    <div className="flex max-h-[85dvh] flex-col">
      <div className="flex items-start justify-between gap-3 border-b border-[var(--border-subtle)] p-4 sm:p-5">
        <div>
          <h2 className="m-0 text-lg font-bold tracking-tight text-[var(--text-primary)]">
            {props.title}
          </h2>
          {props.subtitle !== undefined ? (
            <p className="m-0 mt-1 text-xs text-[var(--text-secondary)]">
              {props.subtitle}
            </p>
          ) : null}
        </div>
        <button
          type="button"
          aria-label="Close"
          className="sj-btn sj-btn-ghost !min-h-[36px] !px-3 !text-[var(--text-primary)]"
          onClick={props.onClose}
        >
          ✕
        </button>
      </div>
      <div className="flex flex-col gap-3 overflow-y-auto p-4 sm:p-5">
        {props.children}
      </div>
      <div className="border-t border-[var(--border-subtle)] p-3 text-right sm:p-4">
        <button
          type="button"
          className="sj-btn sj-btn-primary !min-h-[40px]"
          onClick={props.onClose}
        >
          Done
        </button>
      </div>
    </div>
  </ReactModal>
);

const DifficultySettings = (props: IDifficultyProps): JSX.Element => {
  const [modalOpen, setModalOpen] = React.useState<boolean>(false);
  const s = (
    <>
      <SettingRow label="Friend selection restriction">
        <SettingSelect
          value={props.state.propagated.friend_selection_policy}
          onChange={props.setFriendSelectionPolicy}
        >
          <option value="Unrestricted">Non-trump cards</option>
          <option value="TrumpsIncluded">All cards, including trumps</option>
          <option value="HighestCardNotAllowed">
            Non-trump cards, except the highest
          </option>
          <option value="PointCardNotAllowed">
            Non-trump, non-point cards (except K when playing A)
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Multiple joining policy">
        <SettingSelect
          value={props.state.propagated.multiple_join_policy}
          onChange={props.setMultipleJoinPolicy}
        >
          <option value="Unrestricted">
            Players can join the defending team multiple times.
          </option>
          <option value="NoDoubleJoin">
            Each player can only join the defending team once.
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Rank advancement policy">
        <SettingSelect
          value={props.state.propagated.advancement_policy}
          onChange={props.setAdvancementPolicy}
        >
          <option value="Unrestricted">A must be defended</option>
          <option value="FullyUnrestricted">Unrestricted</option>
          <option value="DefendPoints">
            Points (5, 10, K) and A must be defended
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Max rank">
        <SettingSelect
          value={props.state.propagated.max_rank}
          onChange={props.setMaxRank}
        >
          <option value="NT">No trump</option>
          <option value="A">A</option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Point visibility">
        <SettingSelect
          value={props.state.propagated.hide_landlord_points ? "hide" : "show"}
          onChange={props.setHideLandlordsPoints}
        >
          <option value="show">Show all players&apos; points</option>
          <option value="hide">Hide defending team&apos;s points</option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Played card visibility (in chat)">
        <SettingSelect
          value={props.state.propagated.hide_played_cards ? "hide" : "show"}
          onChange={props.setHidePlayedCards}
        >
          <option value="show">Show played cards in chat</option>
          <option value="hide">Hide played cards in chat</option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Penalty for points left in the bottom">
        <SettingSelect
          value={props.state.propagated.kitty_penalty}
          onChange={props.setKittyPenalty}
        >
          <option value="Times">Twice the size of the last trick</option>
          <option value="Power">
            Two to the power of the size of the last trick
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Penalty for incorrect throws">
        <SettingSelect
          value={props.state.propagated.throw_penalty}
          onChange={props.setThrowPenalty}
        >
          <option value="None">No penalty</option>
          <option value="TenPointsPerAttempt">Ten points per bad throw</option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Play takeback">
        <SettingSelect
          value={props.state.propagated.play_takeback_policy}
          onChange={props.setPlayTakebackPolicy}
        >
          <option value="AllowPlayTakeback">Allow taking back plays</option>
          <option value="NoPlayTakeback">Disallow taking back plays</option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Bid takeback">
        <SettingSelect
          value={props.state.propagated.bid_takeback_policy}
          onChange={props.setBidTakebackPolicy}
        >
          <option value="AllowBidTakeback">Allow bid takeback</option>
          <option value="NoBidTakeback">No bid takeback</option>
        </SettingSelect>
      </SettingRow>
    </>
  );

  return (
    <SettingRow
      label="Difficulty settings"
      hint="Friend selection, rank advancement, penalties and takebacks."
    >
      <SettingButton
        onClick={(evt) => {
          evt.preventDefault();
          setModalOpen(true);
        }}
      >
        Open
      </SettingButton>
      <SettingsModal
        title="Difficulty settings"
        subtitle="Friend selection, rank advancement, point visibility, penalties and takebacks."
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
      >
        {s}
      </SettingsModal>
    </SettingRow>
  );
};

interface IDeckSettings {
  decks: Deck[];
  setSpecialDecks: (specialDecks: Deck[]) => void;
}

const DeckSettings = (props: IDeckSettings): JSX.Element => {
  const [modalOpen, setModalOpen] = React.useState<boolean>(false);
  const isNotDefault = (d: Deck): boolean =>
    !(d.min === "2" && !d.exclude_big_joker && !d.exclude_small_joker);
  const onChange = (decks: Deck[]): void => {
    // exclude the decks that are the same as default
    const filtered = decks.filter((d) => isNotDefault(d));
    props.setSpecialDecks(filtered);
  };

  const setDeckAtIndex = (deck: Deck, index: number): void => {
    const newDecks = [...props.decks];
    newDecks[index] = deck;
    onChange(newDecks);
  };
  const numbers = [
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "A",
  ];

  const s = (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {props.decks.map((d, i) => (
        <div
          key={i}
          className="rounded-[var(--radius-xl)] border border-[var(--border-subtle)] bg-[var(--surface-panel-soft)] p-3"
        >
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm font-semibold text-[var(--text-primary)]">
              Deck {i + 1}
            </span>
            <span
              className={
                "sj-chip !py-0.5 !text-xs" +
                (isNotDefault(d) ? " sj-chip-accent" : "")
              }
            >
              {isNotDefault(d) ? "modified" : "standard"}
            </span>
          </div>
          <div className="flex flex-col gap-2">
            <label className="flex items-center justify-between gap-2 text-sm text-[var(--text-primary)]">
              <span>Include HJ (大王)</span>
              <input
                type="checkbox"
                className="h-4 w-4 accent-[var(--accent)]"
                checked={!d.exclude_big_joker}
                onChange={(evt) =>
                  setDeckAtIndex(
                    { ...d, exclude_big_joker: !evt.target.checked },
                    i,
                  )
                }
              />
            </label>
            <label className="flex items-center justify-between gap-2 text-sm text-[var(--text-primary)]">
              <span>Include LJ (小王)</span>
              <input
                type="checkbox"
                className="h-4 w-4 accent-[var(--accent)]"
                checked={!d.exclude_small_joker}
                onChange={(evt) =>
                  setDeckAtIndex(
                    { ...d, exclude_small_joker: !evt.target.checked },
                    i,
                  )
                }
              />
            </label>
            <label className="flex items-center justify-between gap-2 text-sm text-[var(--text-primary)]">
              <span>Minimum card</span>
              <SettingSelect
                className="!min-w-[5rem] sm:!min-w-[5rem]"
                value={d.min}
                onChange={(evt) =>
                  setDeckAtIndex({ ...d, min: evt.target.value }, i)
                }
              >
                {numbers.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </SettingSelect>
            </label>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <SettingRow
      label="More deck customization"
      hint="Per-deck jokers and minimum card (short decks)."
    >
      <SettingButton
        onClick={(evt) => {
          evt.preventDefault();
          setModalOpen(true);
        }}
      >
        Open
      </SettingButton>
      <SettingsModal
        title="Deck customization"
        subtitle="Configure jokers and the minimum card for each deck."
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
      >
        {s}
      </SettingsModal>
    </SettingRow>
  );
};

interface ITractorRequirementsProps {
  tractorRequirements: TractorRequirements;
  numDecks: number;
  onChange: (requirements: TractorRequirements) => void;
}

const TractorRequirementsE = (
  props: ITractorRequirementsProps,
): JSX.Element => {
  return (
    <SettingRow label="Tractor requirements">
      <div className="flex flex-wrap items-center gap-2 text-sm text-[var(--text-primary)]">
        <SettingInput
          type="number"
          className="!w-16"
          onChange={(v) =>
            props.onChange({
              ...props.tractorRequirements,
              min_count: v.target.valueAsNumber,
            })
          }
          value={props.tractorRequirements.min_count}
          min="2"
          max={props.numDecks}
        />
        <span>cards wide by</span>
        <SettingInput
          type="number"
          className="!w-16"
          onChange={(v) =>
            props.onChange({
              ...props.tractorRequirements,
              min_length: v.target.valueAsNumber,
            })
          }
          value={props.tractorRequirements.min_length}
          min="2"
          max="12"
        />
        <span>tuples long</span>
      </div>
    </SettingRow>
  );
};

interface IScoringSettings {
  state: InitializePhase;
  decks: Deck[];
}
const ScoringSettings = (props: IScoringSettings): JSX.Element => {
  const [modalOpen, setModalOpen] = React.useState<boolean>(false);
  return (
    <SettingRow
      label="Scoring settings"
      hint="Step size, deadzone and level thresholds."
    >
      <SettingButton
        onClick={(evt) => {
          evt.preventDefault();
          setModalOpen(true);
        }}
      >
        Open
      </SettingButton>
      <SettingsModal
        title="Scoring settings"
        subtitle="Tune the point thresholds and how many levels each team gains."
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
      >
        <GameScoringSettings
          params={props.state.propagated.game_scoring_parameters!}
          decks={props.decks}
        />
      </SettingsModal>
    </SettingRow>
  );
};

interface IUncommonSettings {
  state: InitializePhase;
  numDecksEffective: number;
  setBidPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setBidReinforcementPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setJokerBidPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setShouldRevealKittyAtEndOfGame: (
    v: React.ChangeEvent<HTMLSelectElement>,
  ) => void;
  setFirstLandlordSelectionPolicy: (
    v: React.ChangeEvent<HTMLSelectElement>,
  ) => void;
  setGameStartPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setGameShadowingPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setKittyBidPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setJackVariation: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setHideThrowHaltingPlayer: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setTractorRequirements: (v: TractorRequirements) => void;
  setBombPolicy: (v: React.ChangeEvent<HTMLSelectElement>) => void;
  setCompoundFormats: (v: CompoundFormats) => void;
}

const UncommonSettings = (props: IUncommonSettings): JSX.Element => {
  const [modalOpen, setModalOpen] = React.useState<boolean>(false);
  const s = (
    <>
      <SettingRow label="Game shadowing policy">
        <SettingSelect
          value={props.state.propagated.game_shadowing_policy}
          onChange={props.setGameShadowingPolicy}
        >
          <option value="AllowMultipleSessions">
            Allow players to be shadowed by joining with the same name
          </option>
          <option value="SingleSessionOnly">
            Do not allow players to be shadowed
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Game start policy">
        <SettingSelect
          value={props.state.propagated.game_start_policy}
          onChange={props.setGameStartPolicy}
        >
          <option value="AllowAnyPlayer">
            Allow any player to start a game
          </option>
          <option value="AllowLandlordOnly">
            Allow only landlord to start a game
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Landlord selection from bid">
        <SettingSelect
          value={props.state.propagated.first_landlord_selection_policy}
          onChange={props.setFirstLandlordSelectionPolicy}
        >
          <option value="ByWinningBid">
            Winning bid decides both landlord and trump
          </option>
          <option value="ByFirstBid">
            First bid decides landlord, winning bid decides trump
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Trump policy for cards revealed from the bottom">
        <SettingSelect
          value={props.state.propagated.kitty_bid_policy}
          onChange={props.setKittyBidPolicy}
        >
          <option value="FirstCard">First card revealed</option>
          <option value="FirstCardOfLevelOrHighest">
            First card revealed of the appropriate rank
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Bid policy">
        <SettingSelect
          value={props.state.propagated.bid_policy}
          onChange={props.setBidPolicy}
        >
          <option value="JokerOrHigherSuit">
            Joker or higher suit bids to outbid non-joker bids with the same
            number of cards
          </option>
          <option value="JokerOrGreaterLength">
            Joker bids to outbid non-joker bids with the same number of cards
          </option>
          <option value="GreaterLength">
            All bids must have more cards than the previous bids
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Bid reinforcement policy">
        <SettingSelect
          value={props.state.propagated.bid_reinforcement_policy}
          onChange={props.setBidReinforcementPolicy}
        >
          <option value="ReinforceWhileWinning">
            The current winning bid can be reinforced
          </option>
          <option value="ReinforceWhileEquivalent">
            A bid can be reinforced after it is overturned
          </option>
          <option value="OverturnOrReinforceWhileWinning">
            The current winning bid can be overturned by the same bidder
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Joker bid policy">
        <SettingSelect
          value={props.state.propagated.joker_bid_policy}
          onChange={props.setJokerBidPolicy}
        >
          <option value="BothTwoOrMore">
            At least two jokers (or number of decks) to bid no trump
          </option>
          <option value="BothNumDecks">
            All the low or high jokers to bid no trump
          </option>
          <option value="LJNumDecksHJNumDecksLessOne">
            All the low jokers or all but one high joker to bid no trump
          </option>
          <option value="Disabled">No trump / joker bids disabled</option>
        </SettingSelect>
      </SettingRow>
      <TractorRequirementsE
        tractorRequirements={props.state.propagated.tractor_requirements!}
        numDecks={props.numDecksEffective}
        onChange={(req) => props.setTractorRequirements(req)}
      />
      {props.numDecksEffective >= 4 && (
        <SettingRow
          label="Bomb cards"
          hint="4+ identical cards beat any play of the same size."
        >
          <SettingSelect
            value={props.state.propagated.bomb_policy ?? "NoBombs"}
            onChange={props.setBombPolicy}
          >
            <option value="NoBombs">Disabled</option>
            <option value="AllowBombs">
              Enabled (any suit, no suit-following required)
            </option>
            <option value="AllowBombsSuitFollowing">
              Enabled (must follow suit)
            </option>
          </SettingSelect>
        </SettingRow>
      )}
      <SettingRow label="Rainbow tricks" hint="Same rank across ≥4 suits.">
        <SettingSelect
          value={
            props.state.propagated.compound_formats?.rainbows != null
              ? "enabled"
              : "disabled"
          }
          onChange={(evt) => {
            if (evt.target.value === "enabled") {
              props.setCompoundFormats({
                rainbows:
                  props.state.propagated.compound_formats?.rainbows ??
                  (props.state.propagated.num_decks ?? 2) * 2 + 1,
              });
            } else {
              props.setCompoundFormats({ rainbows: null });
            }
          }}
        >
          <option value="disabled">Disabled</option>
          <option value="enabled">Enabled</option>
        </SettingSelect>
        {props.state.propagated.compound_formats?.rainbows != null && (
          <label className="flex items-center gap-1.5 text-sm text-[var(--text-secondary)]">
            Min cards
            <SettingInput
              type="number"
              className="!w-16"
              min={4}
              value={props.state.propagated.compound_formats.rainbows}
              onChange={(evt) => {
                const n = parseInt(evt.target.value, 10);
                if (!isNaN(n) && n >= 4) {
                  props.setCompoundFormats({ rainbows: n });
                }
              }}
            />
          </label>
        )}
      </SettingRow>
      <SettingRow label="Should reveal kitty at end of game">
        <SettingSelect
          value={
            props.state.propagated.should_reveal_kitty_at_end_of_game
              ? "show"
              : "hide"
          }
          onChange={props.setShouldRevealKittyAtEndOfGame}
        >
          <option value="hide">
            Do not reveal contents of the kitty at the end of the game in chat
          </option>
          <option value="show">
            Reveal contents of the kitty at the end of the game in chat
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Show player which defeats throw">
        <SettingSelect
          value={
            props.state.propagated.hide_throw_halting_player ? "hide" : "show"
          }
          onChange={props.setHideThrowHaltingPlayer}
        >
          <option value="hide">
            Hide the player who defeats a potential throw
          </option>
          <option value="show">
            Show the player who defeats a potential throw
          </option>
        </SettingSelect>
      </SettingRow>
      <SettingRow label="Jacks variation">
        <SettingSelect
          value={props.state.propagated.jack_variation}
          onChange={props.setJackVariation}
        >
          <option value="SingleJack">
            Winning the last trick with a single J will set the leader&apos;s
            team to rank 2
          </option>
          <option value="Disabled">Disable the J variation</option>
        </SettingSelect>
      </SettingRow>
    </>
  );
  return (
    <SettingRow
      label="More game settings"
      hint="Bidding, kitty, throws, bombs and other rule tweaks."
    >
      <SettingButton
        onClick={(evt) => {
          evt.preventDefault();
          setModalOpen(true);
        }}
      >
        Open
      </SettingButton>
      <SettingsModal
        title="More game settings"
        subtitle="Less common rule tweaks for bidding, the kitty, throws and bombs."
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
      >
        {s}
      </SettingsModal>
    </SettingRow>
  );
};

interface IProps {
  state: InitializePhase;
  name: string;
}

const Initialize = (props: IProps): JSX.Element => {
  const { send } = React.useContext(WebsocketContext);
  const { t } = useTranslation();
  const [showPicker, setShowPicker] = React.useState<boolean>(false);
  const setGameMode = (evt: React.ChangeEvent<HTMLSelectElement>): void => {
    evt.preventDefault();
    if (evt.target.value === "Tractor") {
      send({ Action: { SetGameMode: "Tractor" } });
    } else {
      send({
        Action: {
          SetGameMode: {
            FindingFriends: {
              num_friends: null,
            },
          },
        },
      });
    }
  };

  const setNumFriends = (evt: React.ChangeEvent<HTMLSelectElement>): void => {
    evt.preventDefault();
    if (evt.target.value === "") {
      send({
        Action: {
          SetGameMode: {
            FindingFriends: {
              num_friends: null,
            },
          },
        },
      });
    } else {
      const num = parseInt(evt.target.value, 10);
      send({
        Action: {
          SetGameMode: {
            FindingFriends: {
              num_friends: num,
            },
          },
        },
      });
    }
  };

  const onSelectString =
    (action: string): ((evt: React.ChangeEvent<HTMLSelectElement>) => void) =>
    (evt: React.ChangeEvent<HTMLSelectElement>): void => {
      evt.preventDefault();
      if (evt.target.value !== "") {
        send({ Action: { [action]: evt.target.value } });
      }
    };

  const onSelectStringDefault =
    (
      action: string,
      defaultValue: null | string,
    ): ((evt: React.ChangeEvent<HTMLSelectElement>) => void) =>
    (evt: React.ChangeEvent<HTMLSelectElement>): void => {
      evt.preventDefault();
      if (evt.target.value !== "") {
        send({ Action: { [action]: evt.target.value } });
      } else {
        send({ Action: { [action]: defaultValue } });
      }
    };

  const setFriendSelectionPolicy = onSelectString("SetFriendSelectionPolicy");
  const setMultipleJoinPolicy = onSelectString("SetMultipleJoinPolicy");
  const setFirstLandlordSelectionPolicy = onSelectString(
    "SetFirstLandlordSelectionPolicy",
  );
  const setBidPolicy = onSelectString("SetBidPolicy");
  const setBidReinforcementPolicy = onSelectString("SetBidReinforcementPolicy");
  const setJokerBidPolicy = onSelectString("SetJokerBidPolicy");
  const setKittyTheftPolicy = onSelectString("SetKittyTheftPolicy");
  const setKittyBidPolicy = onSelectString("SetKittyBidPolicy");
  const setTrickDrawPolicy = onSelectString("SetTrickDrawPolicy");
  const setThrowEvaluationPolicy = onSelectString("SetThrowEvaluationPolicy");
  const setPlayTakebackPolicy = onSelectString("SetPlayTakebackPolicy");
  const setGameShadowingPolicy = onSelectString("SetGameShadowingPolicy");
  const setGameStartPolicy = onSelectString("SetGameStartPolicy");
  const setBidTakebackPolicy = onSelectString("SetBidTakebackPolicy");
  const setGameVisibility = onSelectString("SetGameVisibility");

  const setShouldRevealKittyAtEndOfGame = (
    evt: React.ChangeEvent<HTMLSelectElement>,
  ): void => {
    evt.preventDefault();
    if (evt.target.value !== "") {
      send({
        Action: {
          SetShouldRevealKittyAtEndOfGame: evt.target.value === "show",
        },
      });
    }
  };
  const setHideThrowHaltingPlayer = (
    evt: React.ChangeEvent<HTMLSelectElement>,
  ): void => {
    evt.preventDefault();
    if (evt.target.value !== "") {
      send({
        Action: {
          SetHideThrowHaltingPlayer: evt.target.value === "hide",
        },
      });
    }
  };
  const setJackVariation = (
    evt: React.ChangeEvent<HTMLSelectElement>,
  ): void => {
    evt.preventDefault();
    if (evt.target.value !== "") {
      send({
        Action: {
          SetJackVariation: evt.target.value,
        },
      });
    }
  };
  const setBombPolicy = onSelectString("SetBombPolicy");

  const setKittyPenalty = onSelectStringDefault("SetKittyPenalty", null);
  const setAdvancementPolicy = onSelectStringDefault(
    "SetAdvancementPolicy",
    "Unrestricted",
  );
  const setMaxRank = onSelectStringDefault("SetMaxRank", "NT");
  const setThrowPenalty = onSelectStringDefault("SetThrowPenalty", null);

  const setHideLandlordsPoints = (
    evt: React.ChangeEvent<HTMLSelectElement>,
  ): void => {
    evt.preventDefault();
    send({ Action: { SetHideLandlordsPoints: evt.target.value === "hide" } });
  };

  const setHidePlayedCards = (
    evt: React.ChangeEvent<HTMLSelectElement>,
  ): void => {
    evt.preventDefault();
    send({ Action: { SetHidePlayedCards: evt.target.value === "hide" } });
  };

  const startGame = (evt: React.SyntheticEvent): void => {
    evt.preventDefault();
    send({ Action: "StartGame" });
  };

  const setEmoji = (emoji: string): void => {
    send({
      Action: {
        SetLandlordEmoji: emoji,
      },
    });
  };

  const modeAsString =
    props.state.propagated.game_mode === "Tractor"
      ? "Tractor"
      : "FindingFriends";
  const numFriends =
    props.state.propagated.game_mode === "Tractor" ||
    props.state.propagated.game_mode.FindingFriends.num_friends === null
      ? ""
      : props.state.propagated.game_mode.FindingFriends.num_friends;
  const decksEffective =
    props.state.propagated.num_decks !== undefined &&
    props.state.propagated.num_decks !== null &&
    props.state.propagated.num_decks > 0
      ? props.state.propagated.num_decks
      : Math.max(Math.floor(props.state.propagated.players.length / 2), 1);
  const decks = [...(props.state.propagated.special_decks || [])];
  while (decks.length < decksEffective) {
    decks.push({
      exclude_big_joker: false,
      exclude_small_joker: false,
      min: "2",
    });
  }
  decks.length = decksEffective;

  let currentPlayer = props.state.propagated.players.find(
    (p: Player) => p.name === props.name,
  );
  if (currentPlayer === undefined) {
    currentPlayer = props.state.propagated.observers.find(
      (p) => p.name === props.name,
    );
  }
  if (currentPlayer === undefined) {
    currentPlayer = {
      id: -1,
      name: props.name,
      level: "",
      metalevel: 0,
    };
  }

  const landlordIndex = props.state.propagated.players.findIndex(
    (p: Player) => p.id === props.state.propagated.landlord,
  );
  const saveGameSettings = (evt: React.SyntheticEvent): void => {
    evt.preventDefault();
    localStorage.setItem(
      "gameSettingsInLocalStorage",
      JSON.stringify(props.state.propagated),
    );
  };

  const setGameSettings = (gameSettings: PropagatedState): void => {
    if (gameSettings !== null) {
      let kittySizeSet = false;
      let kittySize = null;
      for (const [key, value] of Object.entries(gameSettings)) {
        switch (key) {
          case "game_mode":
            send({
              Action: {
                SetGameMode: value,
              },
            });
            break;
          case "num_decks":
            send({
              Action: {
                SetNumDecks: value,
              },
            });
            if (kittySizeSet) {
              // reset the size again, as setting deck num resets kitty_size to default
              send({
                Action: {
                  SetKittySize: kittySize,
                },
              });
            }
            break;
          case "special_decks":
            send({
              Action: {
                SetSpecialDecks: value,
              },
            });
            break;
          case "kitty_size":
            send({
              Action: {
                SetKittySize: value,
              },
            });
            kittySizeSet = true;
            kittySize = value;
            break;
          case "friend_selection_policy":
            send({
              Action: {
                SetFriendSelectionPolicy: value,
              },
            });
            break;
          case "multiple_join_policy":
            send({
              Action: {
                SetMultipleJoinPolicy: value,
              },
            });
            break;
          case "first_landlord_selection_policy":
            send({
              Action: {
                SetFirstLandlordSelectionPolicy: value,
              },
            });
            break;
          case "hide_landlord_points":
            send({
              Action: {
                SetHideLandlordsPoints: value,
              },
            });
            break;
          case "hide_played_cards":
            send({ Action: { SetHidePlayedCards: value } });
            break;
          case "advancement_policy":
            send({
              Action: {
                SetAdvancementPolicy: value,
              },
            });
            break;
          case "max_rank":
            send({
              Action: {
                SetMaxRank: value,
              },
            });
            break;
          case "kitty_bid_policy":
            send({
              Action: {
                SetKittyBidPolicy: value,
              },
            });
            break;
          case "kitty_penalty":
            send({
              Action: {
                SetKittyPenalty: value,
              },
            });
            break;
          case "kitty_theft_policy":
            send({
              Action: {
                SetKittyTheftPolicy: value,
              },
            });
            break;
          case "throw_penalty":
            send({
              Action: {
                SetThrowPenalty: value,
              },
            });
            break;
          case "trick_draw_policy":
            send({
              Action: {
                SetTrickDrawPolicy: value,
              },
            });
            break;
          case "throw_evaluation_policy":
            send({
              Action: {
                SetThrowEvaluationPolicy: value,
              },
            });
            break;
          case "landlord_emoji":
            send({
              Action: {
                SetLandlordEmoji: value,
              },
            });
            break;
          case "bid_policy":
            send({
              Action: {
                SetBidPolicy: value,
              },
            });
            break;
          case "bid_reinforcement_policy":
            send({
              Action: {
                SetBidReinforcementPolicy: value,
              },
            });
            break;
          case "joker_bid_policy":
            send({
              Action: {
                SetJokerBidPolicy: value,
              },
            });
            break;
          case "should_reveal_kitty_at_end_of_game":
            send({
              Action: {
                SetShouldRevealKittyAtEndOfGame: value,
              },
            });
            break;
          case "hide_throw_halting_player":
            send({ Action: { SetHideThrowHaltingPlayer: value } });
            break;
          case "set_jack_variation":
            send({ Action: { SetJackVariation: value } });
            break;
          case "game_scoring_parameters":
            send({
              Action: {
                SetGameScoringParameters: value,
              },
            });
            break;
          case "play_takeback_policy":
            send({
              Action: {
                SetPlayTakebackPolicy: value,
              },
            });
            break;
          case "bid_takeback_policy":
            send({
              Action: {
                SetBidTakebackPolicy: value,
              },
            });
            break;
          case "game_shadowing_policy":
            send({
              Action: {
                SetGameShadowingPolicy: value,
              },
            });
            break;
          case "game_start_policy":
            send({
              Action: {
                SetGameStartPolicy: value,
              },
            });
            break;
          case "tractor_requirements":
            send({
              Action: {
                SetTractorRequirements: value,
              },
            });
            break;
          case "game_visibility":
            send({
              Action: {
                SetGameVisibility: value,
              },
            });
            break;
          case "compound_formats":
            send({
              Action: {
                SetCompoundFormats: value,
              },
            });
            break;
        }
      }
    }
  };

  const loadGameSettings = (evt: React.SyntheticEvent): void => {
    evt.preventDefault();
    const settings = localStorage.getItem("gameSettingsInLocalStorage");
    if (settings !== null) {
      let gameSettings: PropagatedState;
      try {
        gameSettings = JSON.parse(settings);

        const fetchAsync = async (): Promise<void> => {
          const fetchResult = await fetch("default_settings.json");
          const fetchJSON = await fetchResult.json();
          const combined = { ...fetchJSON, ...gameSettings };
          if (
            combined.bonus_level_policy !== undefined &&
            combined.game_scoring_parameters !== undefined &&
            combined.bonus_level_policy !==
              combined.game_scoring_parameters.bonus_level_policy
          ) {
            combined.game_scoring_parameters.bonus_level_policy =
              combined.bonus_level_policy;
          }
          setGameSettings(combined);
        };

        fetchAsync().catch((e) => {
          console.error(e);
          localStorage.setItem(
            "gameSettingsInLocalStorage",
            JSON.stringify(props.state.propagated),
          );
        });
      } catch {
        localStorage.setItem(
          "gameSettingsInLocalStorage",
          JSON.stringify(props.state.propagated),
        );
      }
    }
  };

  const resetGameSettings = (evt: React.SyntheticEvent): void => {
    evt.preventDefault();

    const fetchAsync = async (): Promise<void> => {
      const fetchResult = await fetch("default_settings.json");
      const fetchJSON = await fetchResult.json();
      setGameSettings(fetchJSON);
    };

    fetchAsync().catch((e) => console.error(e));
  };

  return (
    <div>
      <Header
        gameMode={props.state.propagated.game_mode}
        chatLink={props.state.propagated.chat_link}
      />
      <div className="sj-panel mb-4 p-4">
        <Players
          players={props.state.propagated.players}
          observers={props.state.propagated.observers}
          bots={props.state.propagated.bots}
          landlord={props.state.propagated.landlord}
          gameMode={props.state.propagated.game_mode}
          next={null}
          movable={true}
          name={props.name}
        />
        <p className="mt-3 text-sm text-[var(--text-secondary)]">
          {t("lobby.shareLink")}{" "}
          <a
            href={window.location.href}
            target="_blank"
            rel="noreferrer"
            className="text-[var(--accent)] underline"
          >
            <code>{window.location.href}</code>
          </a>
        </p>
      </div>
      <AddAIPlayer />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {props.state.propagated.players.length >= 4 ? (
          <>
            <button
              className="sj-btn sj-btn-primary"
              disabled={
                props.state.propagated.game_start_policy ===
                  "AllowLandlordOnly" &&
                landlordIndex !== -1 &&
                props.state.propagated.players[landlordIndex].name !==
                  props.name
              }
              onClick={startGame}
            >
              {t("lobby.start")}
            </button>
            <ReadyCheck />
          </>
        ) : (
          <h2 className="m-0 text-lg font-semibold text-[var(--text-on-felt-soft)]">
            {t("lobby.waiting")}
          </h2>
        )}
        <RandomizePlayersButton players={props.state.propagated.players}>
          {t("lobby.randomize")}
        </RandomizePlayersButton>
        <Kicker
          players={props.state.propagated.players}
          onKick={(playerId: number) => send({ Kick: playerId })}
        />
      </div>
      <div className="game-settings">
        <h2 className="mb-3 text-xl font-bold tracking-tight text-[var(--text-on-felt)]">
          Game settings
        </h2>

        <SettingsSection
          title="Players & teams"
          subtitle="Pick the game mode and, for Finding Friends, the team size."
        >
          <SettingRow label="Game mode" htmlFor="game-mode-selector">
            <SettingSelect
              id="game-mode-selector"
              value={modeAsString}
              onChange={setGameMode}
            >
              <option value="Tractor">升级 / Tractor</option>
              <option value="FindingFriends">找朋友 / Finding Friends</option>
            </SettingSelect>
          </SettingRow>
          {props.state.propagated.game_mode !== "Tractor" ? (
            <SettingRow
              label="Number of friends"
              htmlFor="num-friends-selector"
            >
              <SettingSelect
                id="num-friends-selector"
                value={numFriends}
                onChange={setNumFriends}
              >
                <option value="">default</option>
                {ArrayUtils.range(
                  Math.max(
                    Math.floor(props.state.propagated.players.length / 2) - 1,
                    0,
                  ),
                  (idx) => (
                    <option value={idx + 1} key={idx}>
                      {idx + 1}
                    </option>
                  ),
                )}
              </SettingSelect>
            </SettingRow>
          ) : null}
        </SettingsSection>

        <SettingsSection
          title="Decks & kitty"
          subtitle="How many decks are in play and how the bottom cards work."
        >
          <NumDecksSelector
            numPlayers={props.state.propagated.players.length}
            numDecks={props.state.propagated.num_decks}
            onChange={(newNumDecks: number | null) =>
              send({ Action: { SetNumDecks: newNumDecks } })
            }
          />
          <DeckSettings
            decks={decks}
            setSpecialDecks={(d) => send({ Action: { SetSpecialDecks: d } })}
          />
          <KittySizeSelector
            numPlayers={props.state.propagated.players.length}
            decks={decks}
            kittySize={props.state.propagated.kitty_size}
            onChange={(newKittySize: number | null) =>
              send({ Action: { SetKittySize: newKittySize } })
            }
          />
          <SettingRow
            label="Bids after cards are exchanged from the bottom"
            htmlFor="kitty-theft-selector"
          >
            <SettingSelect
              id="kitty-theft-selector"
              value={props.state.propagated.kitty_theft_policy}
              onChange={setKittyTheftPolicy}
            >
              <option value="AllowKittyTheft">Allowed (炒地皮)</option>
              <option value="NoKittyTheft">Not allowed</option>
            </SettingSelect>
          </SettingRow>
        </SettingsSection>

        <SettingsSection
          title="Gameplay rules"
          subtitle="How tricks, protections and throws are evaluated."
        >
          <SettingRow
            label="Card protection policy"
            htmlFor="trick-draw-selector"
          >
            <SettingSelect
              id="trick-draw-selector"
              value={props.state.propagated.trick_draw_policy}
              onChange={setTrickDrawPolicy}
            >
              <option value="NoProtections">No protections</option>
              <option value="LongerTuplesProtected">
                Longer tuple (triple) is protected from shorter (pair)
              </option>
              <option value="OnlyDrawTractorOnTractor">
                Only tractors can draw tractors
              </option>
              <option value="LongerTuplesProtectedAndOnlyDrawTractorOnTractor">
                Longer tuples are protected from shorter, and only tractors can
                draw tractors
              </option>
              <option value="NoFormatBasedDraw">
                No format-based requirements (pairs do not draw pairs)
              </option>
            </SettingSelect>
          </SettingRow>
          <SettingRow
            label="Multi-throw evaluation policy"
            htmlFor="throw-eval-selector"
          >
            <SettingSelect
              id="throw-eval-selector"
              value={props.state.propagated.throw_evaluation_policy}
              onChange={setThrowEvaluationPolicy}
            >
              <option value="All">
                Subsequent throw must beat all cards to win
              </option>
              <option value="Highest">
                Subsequent throw must beat highest card to win
              </option>
              <option value="TrickUnitLength">
                Subsequent throw must beat largest component to win
              </option>
            </SettingSelect>
          </SettingRow>
        </SettingsSection>

        <SettingsSection
          title="Scoring & advanced"
          subtitle="Open these panels to fine-tune scoring and less common rules."
        >
          <ScoringSettings state={props.state} decks={decks} />
          <UncommonSettings
            state={props.state}
            numDecksEffective={decksEffective}
            setBidPolicy={setBidPolicy}
            setBidReinforcementPolicy={setBidReinforcementPolicy}
            setJokerBidPolicy={setJokerBidPolicy}
            setShouldRevealKittyAtEndOfGame={setShouldRevealKittyAtEndOfGame}
            setHideThrowHaltingPlayer={setHideThrowHaltingPlayer}
            setFirstLandlordSelectionPolicy={setFirstLandlordSelectionPolicy}
            setGameStartPolicy={setGameStartPolicy}
            setGameShadowingPolicy={setGameShadowingPolicy}
            setKittyBidPolicy={setKittyBidPolicy}
            setJackVariation={setJackVariation}
            setTractorRequirements={(requirements) =>
              send({ Action: { SetTractorRequirements: requirements } })
            }
            setBombPolicy={setBombPolicy}
            setCompoundFormats={(formats) =>
              send({ Action: { SetCompoundFormats: formats } })
            }
          />
          <DifficultySettings
            state={props.state}
            setFriendSelectionPolicy={setFriendSelectionPolicy}
            setMultipleJoinPolicy={setMultipleJoinPolicy}
            setAdvancementPolicy={setAdvancementPolicy}
            setMaxRank={setMaxRank}
            setHideLandlordsPoints={setHideLandlordsPoints}
            setHidePlayedCards={setHidePlayedCards}
            setKittyPenalty={setKittyPenalty}
            setThrowPenalty={setThrowPenalty}
            setPlayTakebackPolicy={setPlayTakebackPolicy}
            setBidTakebackPolicy={setBidTakebackPolicy}
          />
          <SettingRow
            label="Game visibility"
            hint="Public games are listed for anyone to join."
            htmlFor="visibility-selector"
          >
            <SettingSelect
              id="visibility-selector"
              value={props.state.propagated.game_visibility}
              onChange={setGameVisibility}
            >
              <option value={"Unlisted"}>Unlisted</option>
              <option value={"Public"}>Public</option>
            </SettingSelect>
          </SettingRow>
        </SettingsSection>

        <SettingsSection
          title="Continuation settings"
          subtitle="Set the starting leader and your current rank."
        >
          <LandlordSelector
            players={props.state.propagated.players}
            landlordId={props.state.propagated.landlord}
            onChange={(newLandlord: number | null) =>
              send({ Action: { SetLandlord: newLandlord } })
            }
          />
          <RankSelector
            rank={currentPlayer.level}
            metaRank={currentPlayer.metalevel}
            onChangeRank={(newRank: string) =>
              send({ Action: { SetRank: newRank } })
            }
            onChangeMetaRank={(newMetaRank: number) =>
              send({ Action: { SetMetaRank: newMetaRank } })
            }
          />
        </SettingsSection>

        <SettingsSection
          title="Misc"
          subtitle="Cosmetic options and saving / loading settings."
        >
          <SettingRow
            label="Landlord label"
            hint={
              <>
                Currently:{" "}
                <span className="font-semibold text-[var(--text-primary)]">
                  {props.state.propagated.landlord_emoji !== null &&
                  props.state.propagated.landlord_emoji !== undefined &&
                  props.state.propagated.landlord_emoji !== ""
                    ? props.state.propagated.landlord_emoji
                    : "当庄"}
                </span>
              </>
            }
          >
            <SettingButton
              onClick={() => {
                setShowPicker(!showPicker);
              }}
            >
              {showPicker ? "Hide" : "Pick"}
            </SettingButton>
            <SettingButton
              disabled={props.state.propagated.landlord_emoji == null}
              onClick={() => {
                send({ Action: { SetLandlordEmoji: null } });
              }}
            >
              Default
            </SettingButton>
          </SettingRow>
          {showPicker ? (
            <React.Suspense fallback={"..."}>
              <Picker
                onEmojiClick={(ecd) => setEmoji(ecd.emoji)}
                emojiStyle={EmojiStyle.NATIVE}
              />
            </React.Suspense>
          ) : null}
          <SettingRow
            label="Setting management"
            hint="Save the current settings to your browser, or restore them."
          >
            <SettingButton
              data-tooltip-id="saveTip"
              data-tooltip-content="Save game settings"
              onClick={saveGameSettings}
            >
              Save
            </SettingButton>
            <Tooltip id="saveTip" place="top" />
            <SettingButton
              data-tooltip-id="loadTip"
              data-tooltip-content={"Load saved game settings"}
              onClick={loadGameSettings}
            >
              Load
            </SettingButton>
            <Tooltip id="loadTip" place="top" />
            <SettingButton
              data-tooltip-id="resetTip"
              data-tooltip-content="Reset game settings to defaults"
              data-ti="resetTip"
              onClick={resetGameSettings}
            >
              Reset
            </SettingButton>
            <Tooltip id="resetTip" place="top" />
          </SettingRow>
        </SettingsSection>
      </div>
    </div>
  );
};

export default Initialize;
