import * as React from "react";
import classNames from "classnames";
import Errors from "./Errors";
import Initialize from "./Initialize";
import Draw from "./Draw";
import Exchange from "./Exchange";
import JoinRoom from "./JoinRoom";
import { AppStateContext } from "./AppStateProvider";
import { TimerContext } from "./TimerProvider";
import Chat from "./Chat";
import Play from "./Play";
import DebugInfo from "./DebugInfo";
import TitleHandler from "./TitleHandler";
import ResetButton from "./ResetButton";
import Toolbar from "./Toolbar";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

const Confetti = React.lazy(async () => await import("./Confetti"));

const Brand = ({ onGoHome }: { onGoHome: () => void }): JSX.Element => {
  const { t } = useTranslation();
  return (
    <h1 className="m-0 max-w-[min(100%,11rem)] text-xl font-bold leading-tight tracking-tight sm:max-w-none sm:text-3xl">
      <button
        type="button"
        onClick={onGoHome}
        title={t("brand.home")}
        aria-label={t("brand.home")}
        className="m-0 cursor-pointer border-0 bg-transparent p-0 text-left font-bold leading-tight tracking-tight text-[var(--text-on-felt)] transition-opacity hover:opacity-80 focus-visible:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent"
      >
        升级 <span className="text-[var(--accent)]">Tractor</span>
        <span className="mx-2 hidden text-[var(--text-on-felt-soft)] sm:inline">
          ·
        </span>
        <span className="hidden sm:inline">
          找朋友 <span className="text-[var(--accent)]">Finding Friends</span>
        </span>
      </button>
    </h1>
  );
};

const Root = (): JSX.Element => {
  const { state, updateState } = React.useContext(AppStateContext);
  const timerContext = React.useContext(TimerContext);
  const { t } = useTranslation();

  // Leave the current room and return to the JoinRoom landing screen.
  //
  // The server permanently binds a WebSocket to a single room (see
  // shengji_handler::handle_user_connected, which subscribes the socket to one
  // room for the life of the connection). There is no "leave room" protocol on
  // the same socket, so the clean way to drop out of a room and start fresh is
  // to clear the room from the URL hash and reload: the page boots a brand-new
  // WebSocket that lands on JoinRoom (gameState === null), with no dangling
  // socket left behind. This mirrors the existing hash-based navigation used by
  // ResetButton / Initialize.
  const goHome = (): void => {
    window.location.hash = "";
    window.location.reload();
  };

  const [previousHeaderMessages, setPreviousHeaderMessages] = React.useState<
    string[]
  >([]);
  const [showHeaderMessages, setShowHeaderMessages] = React.useState<boolean>(
    state.headerMessages.length > 0,
  );
  React.useEffect(() => {
    if (
      state.headerMessages.length > 0 &&
      (previousHeaderMessages.length !== state.headerMessages.length ||
        !previousHeaderMessages.every((m, i) => state.headerMessages[i] === m))
    ) {
      setShowHeaderMessages(true);
    } else if (state.headerMessages.length === 0) {
      setShowHeaderMessages(false);
    }
    setPreviousHeaderMessages(state.headerMessages);
  }, [state.headerMessages]);

  const headerMessages = showHeaderMessages ? (
    <div
      className="header-message"
      onClick={() => setShowHeaderMessages(false)}
    >
      {state.headerMessages.map((msg, idx) => (
        <p key={idx}>{msg}</p>
      ))}
    </div>
  ) : null;

  if (state.connected) {
    if (state.gameState === null || state.roomName.length !== 16) {
      return (
        <div className="min-h-[100dvh]">
          <Toolbar />
          {headerMessages}
          <Errors errors={state.errors} />
          <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:py-12">
            <Brand onGoHome={goHome} />
            <div className="sj-panel mt-6 p-5 sm:p-7">
              <JoinRoom
                name={state.name}
                room_name={state.roomName}
                setName={(name: string) => updateState({ name })}
                setRoomName={(roomName: string) => {
                  updateState({ roomName });
                  window.location.hash = roomName;
                }}
              />
            </div>
          </div>
          <TitleHandler playerName={state.name} />
        </div>
      );
    } else {
      return (
        <div
          className={classNames(
            "min-h-[100dvh]",
            state.settings.fourColor ? "four-color" : null,
            state.settings.showCardLabels ? "always-show-labels" : null,
            state.settings.hideChatBox ? "hide-chat-box" : null,
          )}
        >
          <Toolbar />
          {headerMessages}
          <Errors errors={state.errors} />
          {state.confetti !== null ? (
            <React.Suspense fallback={null}>
              <Confetti
                confetti={state.confetti}
                clearConfetti={() => updateState({ confetti: null })}
              />
            </React.Suspense>
          ) : null}
          <div className="sj-table-shell mx-auto w-full max-w-[1200px] px-3 pb-40 pt-3 sm:px-5">
            <div className="mb-3">
              <Brand onGoHome={goHome} />
            </div>
            <div className="game">
              {"Initialize" in state.gameState ? null : (
                <ResetButton state={state.gameState} name={state.name} />
              )}
              {"Initialize" in state.gameState ? (
                <Initialize
                  state={state.gameState.Initialize}
                  name={state.name}
                />
              ) : null}
              {"Draw" in state.gameState ? (
                <Draw
                  state={state.gameState.Draw}
                  playDrawCardSound={state.settings.playDrawCardSound}
                  autodrawSpeedMs={state.settings.autodrawSpeedMs}
                  name={state.name}
                  setTimeout={timerContext.setTimeout}
                  clearTimeout={timerContext.clearTimeout}
                />
              ) : null}
              {"Exchange" in state.gameState ? (
                <Exchange state={state.gameState.Exchange} name={state.name} />
              ) : null}
              {"Play" in state.gameState ? (
                <Play
                  playPhase={state.gameState.Play}
                  name={state.name}
                  showLastTrick={state.settings.showLastTrick}
                  unsetAutoPlayWhenWinnerChanges={
                    state.settings.unsetAutoPlayWhenWinnerChanges
                  }
                  showTrickInPlayerOrder={state.settings.showTrickInPlayerOrder}
                  beepOnTurn={state.settings.beepOnTurn}
                />
              ) : null}
              {state.settings.showDebugInfo ? <DebugInfo /> : null}
            </div>
            <Chat messages={state.messages} />
          </div>
          <TitleHandler playerName={state.name} />
        </div>
      );
    }
  } else if (state.everConnected) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-16">
        <Toolbar />
        <div className="sj-panel p-6 text-[var(--text-primary)]">
          <p>{t("app.disconnected")}</p>
        </div>
      </div>
    );
  } else {
    return (
      <div className="min-h-[100dvh]">
        <Toolbar />
        <div className="mx-auto w-full max-w-3xl px-4 py-12">
          <Brand onGoHome={goHome} />
          <div className="sj-panel mt-6 p-6 text-[var(--text-primary)]">
            <p>{t("app.welcome")}</p>
            <p>{t("app.connecting")}</p>
          </div>
        </div>
        <TitleHandler playerName={state.name} />
      </div>
    );
  }
};

export default Root;
