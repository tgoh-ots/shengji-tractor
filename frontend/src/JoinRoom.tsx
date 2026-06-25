import * as React from "react";
import { WebsocketContext } from "./WebsocketProvider";
import { TimerContext } from "./TimerProvider";
import LabeledPlay from "./LabeledPlay";
import PublicRoomsPane from "./PublicRoomsPane";
import { isWasmAvailable } from "./detectWasm";
import { useTranslation } from "./i18n";

import type { JSX } from "react";

interface IProps {
  name: string;
  room_name: string;
  setName: (name: string) => void;
  setRoomName: (name: string) => void;
}

const JoinRoom = (props: IProps): JSX.Element => {
  const [editable, setEditable] = React.useState<boolean>(false);
  const [shouldGenerate, setShouldGenerate] = React.useState<boolean>(
    props.room_name.length !== 16,
  );
  const { send } = React.useContext(WebsocketContext);
  const { setTimeout } = React.useContext(TimerContext);
  const { t, lang } = useTranslation();

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>): void =>
    props.setName(event.target.value.trim());

  const handleRoomChange = (event: React.ChangeEvent<HTMLInputElement>): void =>
    props.setRoomName(event.target.value.trim());

  const handleSubmit = (event: React.SyntheticEvent): void => {
    event.preventDefault();
    if (props.name.length > 0 && props.room_name.length === 16) {
      send({
        room_name: props.room_name,
        name: props.name,
        disable_compression: !isWasmAvailable(),
      });
    }
  };

  const editableRoomName = (
    <input
      type="text"
      className="sj-input"
      placeholder={t("join.enterRoom")}
      value={props.room_name}
      onChange={handleRoomChange}
      maxLength={16}
    />
  );
  const nonEditableRoomName = (
    <span
      className="sj-chip cursor-pointer font-mono text-base"
      title="Set the room name"
      onClick={(evt) => {
        evt.preventDefault();
        setEditable(true);
      }}
    >
      {props.room_name}
    </span>
  );

  const generateRoomName = (): void => {
    const arr = new Uint8Array(8);
    window.crypto.getRandomValues(arr);
    setShouldGenerate(false);
    props.setRoomName(
      Array.from(arr, (d) => ("0" + d.toString(16)).substr(-2)).join(""),
    );
  };

  if (shouldGenerate) {
    setTimeout(generateRoomName, 0);
  }

  return (
    <div>
      <div className="mb-4 flex justify-center">
        <LabeledPlay
          cards={["🃟", "🃟", "🃏", "🃏"]}
          trump={{ NoTrump: {} }}
          label={null}
        ></LabeledPlay>
      </div>
      <form className="join-room flex flex-col gap-4" onSubmit={handleSubmit}>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold text-[var(--text-secondary)]">
            {t("join.roomName")}
          </span>
          <span className="flex items-center gap-2">
            {editable ? editableRoomName : nonEditableRoomName}
            <button
              type="button"
              className="sj-btn sj-btn-ghost !min-h-[40px] !px-3 !text-[var(--text-primary)]"
              title={t("join.generateRoom")}
              aria-label={t("join.generateRoom")}
              onClick={() => generateRoomName()}
            >
              🎲
            </button>
          </span>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold text-[var(--text-secondary)]">
            {t("join.playerName")}
          </span>
          <input
            type="text"
            className="sj-input"
            placeholder={t("join.enterName")}
            value={props.name}
            onChange={handleChange}
            autoFocus={true}
          />
        </label>
        <button
          type="submit"
          className="sj-btn sj-btn-primary w-full"
          disabled={
            props.room_name.length !== 16 ||
            props.name.length === 0 ||
            props.name.length > 32
          }
        >
          {t("join.join")}
        </button>
      </form>
      <div className="mt-5 space-y-2 text-sm text-[var(--text-secondary)]">
        <p>{t("join.intro")}</p>
        <p>
          <a
            href={`rules.html?lang=${lang}`}
            target="_blank"
            className="text-[var(--accent)] underline"
          >
            {t("join.readRules")}
          </a>
        </p>
        <p>{t("join.shareIntro")}</p>
      </div>
      <PublicRoomsPane setRoomName={props.setRoomName} />
    </div>
  );
};

export default JoinRoom;
