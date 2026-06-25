import * as React from "react";
import ArrayUtils from "./util/array";
import { SettingRow, SettingSelect } from "./SettingsWidgets";

import type { JSX } from "react";

interface IProps {
  numPlayers: number;
  numDecks: number | null | undefined;
  onChange: (newNumDecks: number | null) => void;
}

const NumDecksSelector = (props: IProps): JSX.Element => {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    const newNumDecks =
      e.target.value === "" ? null : parseInt(e.target.value, 10);
    props.onChange(newNumDecks);
  };

  return (
    <SettingRow label="Number of decks" htmlFor="num-decks-selector">
      <SettingSelect
        id="num-decks-selector"
        value={props.numDecks === null ? "" : props.numDecks}
        onChange={handleChange}
      >
        <option value="">default</option>
        {ArrayUtils.range(props.numPlayers, (idx) => {
          const val = idx + 1;
          return (
            <option value={val} key={idx}>
              {val}
            </option>
          );
        })}
      </SettingSelect>
    </SettingRow>
  );
};

export default NumDecksSelector;
