import * as React from "react";
import { Player } from "./gen-types";

import type { JSX } from "react";

interface IProps {
  onKick: (playerId: number) => void;
  players: Player[];
}
const Kicker = (props: IProps): JSX.Element => {
  const [selection, setSelection] = React.useState<number | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>): void => {
    setSelection(e.target.value === "" ? null : parseInt(e.target.value, 10));
  };

  return (
    <label className="kicker flex items-center gap-2 text-sm text-[var(--text-on-felt-soft)]">
      <span>Kick player</span>
      <select
        className="sj-input !min-h-[40px] !py-1"
        value={selection === null ? "" : selection}
        onChange={handleChange}
        aria-label="Kick player"
      >
        <option value="" />
        {props.players.map((player) => (
          <option value={player.id} key={player.id}>
            {player.name}
          </option>
        ))}
      </select>
      <button
        type="button"
        className="sj-btn !min-h-[40px] !px-3 !py-1 !text-sm"
        onClick={() => {
          if (selection) {
            props.onKick(selection);
          }
        }}
        disabled={selection === null}
      >
        Kick
      </button>
    </label>
  );
};

export default Kicker;
