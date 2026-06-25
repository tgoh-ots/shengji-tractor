import * as React from "react";
import { useEffect, useState, type JSX } from "react";

interface IProps {
  setRoomName: (name: string) => void;
}

const PublicRoomsPane = (props: IProps): JSX.Element => {
  const [publicRooms, setPublicRooms] = useState<any[]>([]);

  useEffect(() => {
    loadPublicRooms();
  }, []);
  const loadPublicRooms = (): void => {
    try {
      const fetchAsync = async (): Promise<void> => {
        const fetchResult = await fetch("public_games.json");
        const resultJSON = await fetchResult.json();
        setPublicRooms(resultJSON);
      };

      fetchAsync().catch((e) => {
        console.error(e);
      });
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <div className="mt-6 border-t border-[var(--border-subtle)] pt-5">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div>
          <h3 className="m-0 text-base font-bold tracking-tight text-[var(--text-primary)]">
            Public rooms
          </h3>
          <p className="m-0 mt-0.5 text-xs text-[var(--text-secondary)]">
            Open games anyone can join — find new friends to play with.
          </p>
        </div>
        <button
          type="button"
          onClick={loadPublicRooms}
          className="sj-btn !min-h-[44px] !px-3 !py-1 !text-sm"
        >
          Refresh
        </button>
      </div>
      {publicRooms.length === 0 ? (
        <p className="m-0 rounded-[var(--radius-xl)] border border-dashed border-[var(--border-strong)] px-3 py-4 text-center text-sm text-[var(--text-secondary)]">
          No public rooms available right now.
        </p>
      ) : (
        <ul className="m-0 flex list-none flex-col gap-2 p-0">
          {publicRooms.map((roomInfo) => (
            <li key={roomInfo.name}>
              <button
                type="button"
                onClick={() => props.setRoomName(roomInfo.name)}
                className="flex w-full items-center justify-between gap-3 rounded-[var(--radius-xl)] border border-[var(--border-subtle)] bg-[var(--surface-panel-soft)] px-3 py-2 text-left transition hover:border-[var(--accent)]"
              >
                <span className="font-mono text-sm font-semibold text-[var(--text-primary)]">
                  {roomInfo.name}
                </span>
                <span className="sj-chip">{roomInfo.num_players} players</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default PublicRoomsPane;
