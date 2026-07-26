import * as React from "react";
import { Trump } from "./gen-types";
import InlineCard from "./InlineCard";
import preloadedCards from "./preloadedCards";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  trump: Trump;
}
const TrumpE = (props: IProps): JSX.Element => {
  const { trump } = props;
  const { t } = useTranslation();
  if ("Standard" in trump) {
    const { suit, number: rank } = trump.Standard;
    // `.find` + guard: an unrecognised suit/rank pair used to index `[0]` on an
    // empty array and throw during render, blanking the game UI. Fall back to
    // naming the rank alone, which is still useful to the player.
    const card = preloadedCards.find(
      (v) => v.typ === suit && v.number === rank,
    )?.value;
    if (card === undefined) {
      return (
        <div className="trump">
          {t("trump.suitIs")} {suit} ({t("trump.rank")} {rank})
        </div>
      );
    }
    return (
      <div className="trump">
        {t("trump.suitIs")} <InlineCard card={card} /> ({t("trump.rank")} {rank}
        )
      </div>
    );
  } else if (
    trump.NoTrump.number !== undefined &&
    trump.NoTrump.number !== null
  ) {
    return (
      <div className="trump">
        {t("trump.noTrumpRank", { rank: trump.NoTrump.number })}
      </div>
    );
  } else {
    return <div className="trump">{t("trump.noTrump")}</div>;
  }
};

export default TrumpE;
