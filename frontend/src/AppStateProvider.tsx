import * as React from "react";
import gameStatistics, { GameStatistics } from "./state/GameStatistics";
import settings, { Settings } from "./state/Settings";
import { GameState } from "./gen-types";
import { Message } from "./ChatMessage";
import { State, combineState, noPersistence } from "./State";
import {
  stringLocalStorageState,
  numberLocalStorageState,
  localStorageState,
} from "./localStorageState";

import type { JSX } from "react";

export interface AppState {
  settings: Settings;
  gameStatistics: GameStatistics;
  connected: boolean;
  everConnected: boolean;
  reconnecting: boolean;
  roomName: string;
  name: string;
  gameState: GameState | null;
  headerMessages: string[];
  errors: string[];
  messages: Message[];
  confetti: string | null;
  changeLogLastViewed: number;
}

// Sticky room id. The room name is the key the server uses to (re)seat a player
// by name, so it must survive a page reload / network blip / tab restore — not
// just live in volatile React state. We persist it to localStorage AND keep it
// in the URL hash:
//   * On boot we PREFER the URL hash (so a shared "…/#<roomid>" link always wins
//     and a deliberate `goHome()` that clears the hash drops the room), and fall
//     back to the last-used room from localStorage when there is no hash (e.g.
//     the bare site URL was opened, or the hash was stripped in transit).
//   * On every change we mirror the value back to localStorage so the next boot
//     can recover it even if the hash is gone.
// We only treat a 16-char value as a real room id (matching the server's
// ROOM_NAME_BYTES) — anything else loads as empty so we land on JoinRoom.
const ROOM_ID_KEY = "room_name";
const roomNameState: State<string> = localStorageState(
  ROOM_ID_KEY,
  (stored: any): string => {
    const fromHash = window.location.hash.slice(1, 17);
    if (fromHash.length === 16) {
      return fromHash;
    }
    // No usable hash — recover the last room we were in, if any.
    return typeof stored === "string" && stored.length === 16 ? stored : "";
  },
  (state: string) => state,
);

const appState: State<AppState> = combineState({
  settings,
  gameStatistics,
  connected: noPersistence(() => false),
  everConnected: noPersistence(() => false),
  reconnecting: noPersistence(() => false),
  roomName: roomNameState,
  name: stringLocalStorageState("name"),
  changeLogLastViewed: numberLocalStorageState("change_log_last_viewed"),
  gameState: noPersistence<GameState | null>(() => null),
  headerMessages: noPersistence<string[]>(() => []),
  errors: noPersistence<string[]>(() => []),
  messages: noPersistence<Message[]>(() => []),
  confetti: noPersistence<string | null>(() => null),
});

interface Context {
  state: AppState;
  updateState: (newState: Partial<AppState>) => void;
}

export const AppStateContext = React.createContext<Context>({
  state: appState.loadDefault(),
  updateState: () => {},
});

export const SettingsContext = React.createContext<Settings>(
  appState.loadDefault().settings,
);

export const AppStateConsumer = AppStateContext.Consumer;

interface IProps {
  children: React.ReactNode;
}
const AppStateProvider = (props: IProps): JSX.Element => {
  const [state, setState] = React.useState<AppState>(() => {
    return appState.loadDefault();
  });
  const updateState = (newState: Partial<AppState>): void => {
    setState((s) => {
      const combined = { ...s, ...newState };
      appState.persist(s, combined);
      return combined;
    });
  };

  // If we recovered the room from localStorage (no hash was present on boot),
  // mirror it back into the URL hash so the address bar stays shareable and a
  // later `goHome()` (which clears the hash + reloads) behaves consistently.
  React.useEffect(() => {
    if (
      state.roomName.length === 16 &&
      window.location.hash.slice(1, 17) !== state.roomName
    ) {
      window.location.hash = state.roomName;
    }
    // Only on mount: subsequent room changes already sync the hash at their call
    // site (Root's setRoomName / ResetButton), and we don't want to fight a
    // deliberate hash clear.
  }, []);
  return (
    <AppStateContext.Provider value={{ state, updateState }}>
      <SettingsContext.Provider value={state.settings}>
        {props.children}
      </SettingsContext.Provider>
    </AppStateContext.Provider>
  );
};
export default AppStateProvider;
