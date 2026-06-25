import * as React from "react";
import ReactModal from "react-modal";
import IconButton from "./IconButton";
import Gear from "./icons/Gear";
import SettingsPane from "./SettingsPane";
import { Tooltip } from "react-tooltip";
import { Settings } from "./state/Settings";
import { AppStateContext } from "./AppStateProvider";

import type { JSX } from "react";

const contentStyle: React.CSSProperties = {
  position: "absolute",
  top: "50%",
  left: "50%",
  width: "min(640px, 92vw)",
  transform: "translate(-50%, -50%)",
  padding: "0",
};

const SettingsButton = (): JSX.Element => {
  const [modalOpen, setModalOpen] = React.useState<boolean>(false);
  const { state, updateState } = React.useContext(AppStateContext);
  return (
    <>
      <Tooltip id="settingsTip" place="top" />
      <IconButton
        onClick={() => setModalOpen(true)}
        aria-label="Settings"
        data-tooltip-id="settingsTip"
        data-tooltip-content="Change user interface settings"
      >
        <Gear width="2em" />
      </IconButton>
      <ReactModal
        isOpen={modalOpen}
        onRequestClose={() => setModalOpen(false)}
        shouldCloseOnOverlayClick
        shouldCloseOnEsc
        style={{ content: contentStyle }}
      >
        <div className="flex max-h-[85dvh] flex-col">
          <div className="flex items-start justify-between gap-3 border-b border-[var(--border-subtle)] p-4 sm:p-5">
            <div>
              <h2 className="m-0 text-lg font-bold tracking-tight text-[var(--text-primary)]">
                Interface settings
              </h2>
              <p className="m-0 mt-1 text-xs text-[var(--text-secondary)]">
                Personal display, gameplay and accessibility preferences.
              </p>
            </div>
            <button
              type="button"
              aria-label="Close"
              className="sj-btn sj-btn-ghost !min-h-[36px] !px-3 !text-[var(--text-primary)]"
              onClick={() => setModalOpen(false)}
            >
              ✕
            </button>
          </div>
          <div className="overflow-y-auto p-4 sm:p-5">
            <SettingsPane
              settings={state.settings}
              onChangeSettings={(settings: Settings) =>
                updateState({ settings })
              }
            />
          </div>
          <div className="border-t border-[var(--border-subtle)] p-3 text-right sm:p-4">
            <button
              type="button"
              className="sj-btn sj-btn-primary !min-h-[40px]"
              onClick={() => setModalOpen(false)}
            >
              Done
            </button>
          </div>
        </div>
      </ReactModal>
    </>
  );
};

export default SettingsButton;
