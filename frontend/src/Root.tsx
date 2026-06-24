import * as React from "react";
import classNames from "classnames";
import Errors from "./Errors";
import Initialize from "./Initialize";
import Draw from "./Draw";
import Exchange from "./Exchange";
import JoinRoom from "./JoinRoom";
import { AppStateContext } from "./AppStateProvider";
import { TimerContext } from "./TimerProvider";
import Credits from "./Credits";
import Chat from "./Chat";
import Play from "./Play";
import DebugInfo from "./DebugInfo";
import TitleHandler from "./TitleHandler";
import ResetButton from "./ResetButton";
import Toolbar from "./Toolbar";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

const Confetti = React.lazy(async () => await import("./Confetti"));

const Brand = (): JSX.Element => (
  <h1 className="m-0 text-2xl font-bold tracking-tight text-[var(--text-on-felt)] sm:text-3xl">
    升级 <span className="text-[var(--accent)]">Tractor</span>
    <span className="mx-2 text-[var(--text-on-felt-soft)]">·</span>
    找朋友 <span className="text-[var(--accent)]">Finding Friends</span>
  </h1>
);

const Root = (): JSX.Element => {
  const { state, updateState } = React.useContext(AppStateContext);
  const timerContext = React.useContext(TimerContext);
  const { t } = useTranslation();

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
            <Brand />
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
            <div className="mt-8 text-sm text-[var(--text-on-felt-soft)]">
              <Credits />
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
          <div className="sj-table-shell mx-auto w-full max-w-[1200px] px-3 pb-24 pt-3 sm:px-5">
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
            <div className="clear-both pt-6 text-sm text-[var(--text-on-felt-soft)]">
              <Credits />
            </div>
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
          <Brand />
          <div className="sj-panel mt-6 p-6 text-[var(--text-primary)]">
            <p>{t("app.welcome")}</p>
            <p>{t("app.connecting")}</p>
          </div>
          <div className="mt-8 text-sm text-[var(--text-on-felt-soft)]">
            <Credits />
          </div>
        </div>
        <TitleHandler playerName={state.name} />
      </div>
    );
  }
};

export default Root;
