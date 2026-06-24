import * as React from "react";
import { GameMode } from "./gen-types";
import InlineCard from "./InlineCard";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  gameMode: GameMode;
  showPlayed: boolean;
}

const Friends = (props: IProps): JSX.Element => {
  const { gameMode } = props;
  const { t, lang } = useTranslation();
  if (gameMode !== "Tractor") {
    return (
      <div className="pending-friends">
        {gameMode.FindingFriends.friends.map((friend, idx) => {
          if (friend.player_id !== null) {
            return null;
          }

          if (
            friend.card === null ||
            friend.card === undefined ||
            friend.card.length === 0
          ) {
            return null;
          }
          const ordinal =
            lang === "zh"
              ? String(friend.initial_skip + 1)
              : nth(friend.initial_skip + 1);
          // Render the translated sentence, splitting on the {card} placeholder
          // so we can drop the actual card glyph inline. We deliberately do not
          // pass `card` to t() so the placeholder survives for the split.
          const template = t("friends.intro", { nth: ordinal });
          const [before, after] = template.split("{card}");
          return (
            <p key={idx}>
              {before}
              <InlineCard card={friend.card} />
              {after}{" "}
              {props.showPlayed
                ? t("friends.played", {
                    n: friend.initial_skip - friend.skip,
                  })
                : ""}
            </p>
          );
        })}
      </div>
    );
  } else {
    return <></>;
  }
};

function nth(n: number): string {
  const suffix = ["st", "nd", "rd"][
    (((((n < 0 ? -n : n) + 90) % 100) - 10) % 10) - 1
  ];
  return `${n}${suffix !== undefined ? suffix : "th"}`;
}

export default Friends;
