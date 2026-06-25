import * as React from "react";
import { SettingRow, SettingSelect } from "./SettingsWidgets";

import type { JSX } from "react";

interface IProps {
  rank: string;
  metaRank: number;
  onChangeRank: (newRank: string) => void;
  onChangeMetaRank: (newMetaRank: number) => void;
}

// prettier-ignore
const allRanks = [
  '2', '3', '4', '5', '6', '7', '8',
  '9', '10', 'J', 'Q', 'K', 'A', 'NT'
]
const RankSelector = (props: IProps): JSX.Element => {
  const [showMetaRank, setShowMetaRank] = React.useState<boolean>(false);
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    if (e.target.value !== "") {
      props.onChangeRank(e.target.value);
    }
  };
  const handleMetaChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    if (e.target.value !== "") {
      const v = parseInt(e.target.value, 10);
      props.onChangeMetaRank(v);
    }
  };

  const metaranks = [];

  if (props.metaRank > 0) {
    for (let i = 1; i <= props.metaRank + 3; i++) {
      metaranks.push(i);
    }
  } else {
    metaranks.push(props.metaRank);
    metaranks.push(1);
  }

  return (
    <SettingRow
      label="Your rank"
      hint="Tick the box to also set a meta-rank."
      htmlFor="rank-selector"
    >
      <SettingSelect
        id="rank-selector"
        className="!min-w-[5rem]"
        value={props.rank}
        onChange={handleChange}
      >
        {allRanks.map((rank) => (
          <option value={rank} key={rank}>
            {rank}
          </option>
        ))}
      </SettingSelect>
      <label
        className="flex cursor-pointer items-center gap-1.5 text-sm text-[var(--text-secondary)]"
        title="show meta-rank"
      >
        <input
          type="checkbox"
          className="h-4 w-4 accent-[var(--accent)]"
          checked={showMetaRank}
          onChange={() => setShowMetaRank(!showMetaRank)}
        />
        meta
      </label>
      {showMetaRank && (
        <SettingSelect
          className="!min-w-[5rem]"
          value={props.metaRank}
          onChange={handleMetaChange}
        >
          {metaranks.map((metarank) => (
            <option value={metarank} key={metarank}>
              {metarank}
            </option>
          ))}
        </SettingSelect>
      )}
    </SettingRow>
  );
};

export default RankSelector;
