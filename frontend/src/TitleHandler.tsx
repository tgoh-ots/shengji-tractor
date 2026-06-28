import * as React from "react";
import { SettingsContext } from "./AppStateProvider";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

const TitleHandler = (props: { playerName?: string }): JSX.Element => {
  const settings = React.useContext(SettingsContext);
  const { t } = useTranslation();
  React.useEffect(() => {
    const title = t("app.documentTitle");
    if (
      props.playerName !== undefined &&
      props.playerName !== null &&
      settings.showPlayerName
    ) {
      document.title = `${props.playerName} | ${title}`;
    } else {
      document.title = title;
    }
  }, [props.playerName, settings.showPlayerName, t]);
  return <></>;
};

export default TitleHandler;
