import * as React from "react";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  onSubmit: () => void;
  playDescription: null | string;
  canSubmit: boolean;
  currentWinner: number | null;
  isCurrentPlayerTurn: boolean;
  unsetAutoPlayWhenWinnerChanges: boolean;
}

type AutoPlay = {
  observedWinner: number | null;
} | null;

const AutoPlayButton = (props: IProps): JSX.Element => {
  const {
    onSubmit,
    canSubmit,
    isCurrentPlayerTurn,
    playDescription,
    currentWinner,
    unsetAutoPlayWhenWinnerChanges,
  } = props;
  const { t } = useTranslation();

  const [autoplay, setAutoplay] = React.useState<AutoPlay | null>(null);

  React.useEffect(() => {
    if (autoplay !== null) {
      if (!canSubmit) {
        setAutoplay(null);
      } else if (
        unsetAutoPlayWhenWinnerChanges &&
        autoplay.observedWinner !== currentWinner
      ) {
        setAutoplay(null);
      } else if (isCurrentPlayerTurn) {
        setAutoplay(null);
        onSubmit();
      }
    }
  }, [
    autoplay,
    canSubmit,
    currentWinner,
    isCurrentPlayerTurn,
    unsetAutoPlayWhenWinnerChanges,
  ]);

  const handleClick = (): void => {
    if (isCurrentPlayerTurn) {
      onSubmit();
    } else if (autoplay !== null) {
      setAutoplay(null);
    } else {
      setAutoplay({ observedWinner: currentWinner });
    }
  };
  return (
    <button
      className={
        "sj-btn " + (isCurrentPlayerTurn && canSubmit ? "sj-btn-primary" : "")
      }
      onClick={handleClick}
      disabled={!canSubmit}
    >
      {isCurrentPlayerTurn
        ? `${t("play.playSelected")}${
            playDescription !== null ? " (" + playDescription + ")" : ""
          }`
        : autoplay !== null
          ? t("play.cancelAutoplay")
          : t("play.autoplay")}
    </button>
  );
};

export default AutoPlayButton;
