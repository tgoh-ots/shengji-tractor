import * as React from "react";
import {
  Settings,
  ISuitOverrides,
  DEFAULT_POINT_CARD_ICON,
  DEFAULT_TRUMP_CARD_ICON,
} from "./state/Settings";
import { CompactPicker } from "react-color";
import { useTheme } from "./ThemeProvider";
import {
  SettingsSection,
  SettingRow,
  SettingButton,
  SettingToggleRow,
} from "./SettingsWidgets";

import type { JSX } from "react";

const Picker = React.lazy(async () => await import("emoji-picker-react"));

interface IProps {
  settings: Settings;
  onChangeSettings: (settings: Settings) => void;
}

const SettingsPane = (props: IProps): JSX.Element => {
  const { settings } = props;
  const { theme, setTheme } = useTheme();
  const toggle = (partialSettings: Partial<Settings>) => (): void => {
    props.onChangeSettings({ ...props.settings, ...partialSettings });
  };

  const [link, setLink] = React.useState<string>("");

  const setChatLink = (event: React.SyntheticEvent): void => {
    event.preventDefault();
    if (link.length > 0) {
      (window as any).send({ Action: { SetChatLink: link } });
    } else {
      (window as any).send({ Action: { SetChatLink: null } });
    }
    setLink("");
  };

  return (
    <div className="settings">
      <SettingsSection
        title="Appearance"
        subtitle="Theme and how cards are drawn."
      >
        <SettingToggleRow
          name="dark-mode"
          label="Dark theme"
          checked={theme === "dark"}
          onChange={() => {
            const next = theme === "dark" ? "light" : "dark";
            setTheme(next);
            props.onChangeSettings({
              ...props.settings,
              darkMode: next === "dark",
            });
          }}
        />
        <SettingToggleRow
          name="four-color-mode"
          label="Four-color suits"
          hint="Distinct colors for ♦ and ♣."
          checked={settings.fourColor}
          onChange={toggle({ fourColor: !settings.fourColor })}
        />
        <SettingToggleRow
          name="svg-cards"
          label="Use SVG cards"
          checked={settings.svgCards}
          onChange={toggle({ svgCards: !settings.svgCards })}
        />
        <SettingToggleRow
          name="show-card-labels"
          label="Always show card labels"
          checked={settings.showCardLabels}
          onChange={toggle({ showCardLabels: !settings.showCardLabels })}
        />
        <SettingRow label="Icon on point cards" hint="Shown on 5 / 10 / K.">
          <EmojiPicker
            value={settings.pointCardIcon}
            setEmoji={(emoji) => toggle({ pointCardIcon: emoji })()}
            setDefault={toggle({ pointCardIcon: DEFAULT_POINT_CARD_ICON })}
          />
        </SettingRow>
        <SettingRow label="Icon on trump cards">
          <EmojiPicker
            value={settings.trumpCardIcon}
            setEmoji={(emoji) => toggle({ trumpCardIcon: emoji })()}
            setDefault={toggle({ trumpCardIcon: DEFAULT_TRUMP_CARD_ICON })}
          />
        </SettingRow>
        <SettingRow
          label="Suit color overrides"
          hint="Customize individual suit colors (text cards only)."
        >
          {settings.svgCards ? (
            <span className="text-sm text-[var(--text-secondary)]">
              Disabled with SVG cards
            </span>
          ) : (
            <SuitOverrides
              suitColors={settings.suitColorOverrides}
              setSuitColors={(newOverrides: ISuitOverrides) =>
                props.onChangeSettings({
                  ...props.settings,
                  suitColorOverrides: newOverrides,
                })
              }
            />
          )}
        </SettingRow>
        <SettingToggleRow
          name="disable-suit-highlights"
          label="Disable suit highlights"
          checked={settings.disableSuitHighlights}
          onChange={toggle({
            disableSuitHighlights: !settings.disableSuitHighlights,
          })}
        />
      </SettingsSection>

      <SettingsSection
        title="Hand & layout"
        subtitle="How your hand and the game board are arranged."
      >
        <SettingToggleRow
          name="reverse-card-order"
          label="Reverse card order in hand"
          checked={settings.reverseCardOrder}
          onChange={toggle({ reverseCardOrder: !settings.reverseCardOrder })}
        />
        <SettingToggleRow
          name="separate-cards-by-suit"
          label="Separate cards by effective suit"
          checked={settings.separateCardsBySuit}
          onChange={toggle({
            separateCardsBySuit: !settings.separateCardsBySuit,
          })}
        />
        <SettingToggleRow
          name="show-trick-in-player-order"
          label="Show tricks in player order"
          checked={settings.showTrickInPlayerOrder}
          onChange={toggle({
            showTrickInPlayerOrder: !settings.showTrickInPlayerOrder,
          })}
        />
        <SettingToggleRow
          name="show-points-above-game"
          label="Show points bar above the game"
          hint="Otherwise the points bar appears below."
          checked={settings.showPointsAboveGame}
          onChange={toggle({
            showPointsAboveGame: !settings.showPointsAboveGame,
          })}
        />
        <SettingToggleRow
          name="hide-chat-box"
          label="Hide chat box"
          checked={settings.hideChatBox}
          onChange={toggle({ hideChatBox: !settings.hideChatBox })}
        />
      </SettingsSection>

      <SettingsSection
        title="Gameplay"
        subtitle="Last trick, autoplay and draw behavior."
      >
        <SettingToggleRow
          name="show-last-trick"
          label="Show last trick"
          checked={settings.showLastTrick}
          onChange={toggle({ showLastTrick: !settings.showLastTrick })}
        />
        <SettingToggleRow
          name="unset-auto-play-when-winner-changes"
          label="Unset auto-play if the winner changes"
          checked={settings.unsetAutoPlayWhenWinnerChanges}
          onChange={toggle({
            unsetAutoPlayWhenWinnerChanges:
              !settings.unsetAutoPlayWhenWinnerChanges,
          })}
        />
        <SettingToggleRow
          name="beep-on-turn"
          label="Beep on your turn"
          checked={settings.beepOnTurn}
          onChange={toggle({ beepOnTurn: !settings.beepOnTurn })}
        />
        <SettingToggleRow
          name="play-sound-when-drawing-card"
          label="Play a sound when drawing a card"
          checked={settings.playDrawCardSound}
          onChange={toggle({
            playDrawCardSound: !settings.playDrawCardSound,
          })}
        />
        <SettingRow label="Autodraw speed">
          <select
            className="sj-input !min-h-[40px] w-full !py-1 sm:w-auto sm:min-w-[10rem]"
            value={
              settings.autodrawSpeedMs !== null ? settings.autodrawSpeedMs : ""
            }
            onChange={(e) =>
              toggle({ autodrawSpeedMs: parseInt(e.target.value) })()
            }
          >
            <option value="250">Default</option>
            <option value="500">Slow</option>
            <option value="10">Fast</option>
          </select>
        </SettingRow>
      </SettingsSection>

      <SettingsSection
        title="Advanced"
        subtitle="Voice-chat link, title bar and debugging."
      >
        <SettingToggleRow
          name="show-player-name"
          label="Show player name in title bar"
          checked={settings.showPlayerName}
          onChange={toggle({ showPlayerName: !settings.showPlayerName })}
        />
        <SettingToggleRow
          name="show-debug-info"
          label="Show debugging information"
          checked={settings.showDebugInfo}
          onChange={toggle({ showDebugInfo: !settings.showDebugInfo })}
        />
        <SettingRow
          label="Voice chat link"
          hint="Share a link to a voice call with the room."
        >
          <input
            type="text"
            className="sj-input !min-h-[40px] w-full !py-1 sm:w-[14rem]"
            value={link}
            onChange={(evt) => {
              evt.preventDefault();
              setLink(evt.target.value);
            }}
            placeholder="https://… link to voice chat"
          />
          <SettingButton onClick={setChatLink}>Set</SettingButton>
        </SettingRow>
      </SettingsSection>
    </div>
  );
};

const SuitOverrides = (props: {
  suitColors: ISuitOverrides;
  setSuitColors: (overrides: ISuitOverrides) => void;
}): JSX.Element => {
  const suits: Array<keyof ISuitOverrides> = ["♢", "♡", "♤", "♧", "🃟", "🃏"];
  const labels = ["♦", "♥", "♠", "♣", "LJ", "HJ"];
  return (
    <div className="flex flex-wrap items-center gap-2">
      {suits.map((suit, idx) => (
        <SuitColorPicker
          key={suit}
          suit={suit}
          label={labels[idx]}
          suitColor={props.suitColors[suit]}
          setSuitColor={(color: string) => {
            const n = { ...props.suitColors };
            n[suit] = color;
            props.setSuitColors(n);
          }}
        />
      ))}
      <SettingButton
        onClick={(evt) => {
          evt.preventDefault();
          props.setSuitColors({});
        }}
      >
        Reset
      </SettingButton>
    </div>
  );
};

const SuitColorPicker = (props: {
  suit: string;
  label: string;
  suitColor?: string;
  setSuitColor: (color: string) => void;
}): JSX.Element => {
  const [showPicker, setShowPicker] = React.useState<boolean>(false);
  return (
    <>
      <span
        className={props.suit}
        style={{
          color: props.suitColor,
          cursor: "pointer",
          fontWeight: 700,
          fontSize: "1.1rem",
          minWidth: "1.4rem",
          textAlign: "center",
        }}
        onClick={() => setShowPicker(true)}
      >
        {props.label}
      </span>
      {showPicker ? (
        <div style={{ position: "absolute", zIndex: 10 }}>
          <div
            style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0 }}
            onClick={() => setShowPicker(false)}
          />
          <CompactPicker
            color={props.suitColor}
            onChangeComplete={(c: any) => props.setSuitColor(c.hex)}
          />
        </div>
      ) : null}
    </>
  );
};

const EmojiPicker = (props: {
  value: string;
  setEmoji: (emoji: string) => void;
  setDefault: () => void;
}): JSX.Element => {
  const [showPicker, setShowPicker] = React.useState<boolean>(false);
  return (
    <div className="flex flex-wrap items-center gap-2">
      {props.value !== "" && (
        <span className="text-lg leading-none">{props.value}</span>
      )}
      <SettingButton onClick={() => setShowPicker(!showPicker)}>
        {showPicker ? "Hide" : "Pick"}
      </SettingButton>
      <SettingButton onClick={props.setDefault}>Default</SettingButton>
      {props.value !== "" && (
        <SettingButton onClick={() => props.setEmoji("")}>
          No icon
        </SettingButton>
      )}
      {showPicker && (
        <div className="w-full">
          <React.Suspense fallback={"…"}>
            <Picker
              onEmojiClick={(emoji) => {
                props.setEmoji(emoji.emoji);
              }}
            />
          </React.Suspense>
        </div>
      )}
    </div>
  );
};

export default SettingsPane;
