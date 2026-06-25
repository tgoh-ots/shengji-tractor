import * as React from "react";
import { AppStateContext } from "./AppStateProvider";
import Timeout from "./Timeout";

import type { JSX } from "react";

interface IProps {
  errors: string[];
}

const Errors = (props: IProps): JSX.Element => {
  const { updateState } = React.useContext(AppStateContext);
  if (props.errors.length === 0) {
    return <></>;
  }
  return (
    <div
      className="errors"
      role="alert"
      onClick={() => updateState({ errors: [] })}
      title="Dismiss"
    >
      <Timeout timeout={5000} callback={() => updateState({ errors: [] })} />
      <span className="errors-icon" aria-hidden="true">
        ⚠️
      </span>
      <div className="errors-body">
        {props.errors.map((err, idx) => (
          <p key={idx}>{err}</p>
        ))}
      </div>
    </div>
  );
};

export default Errors;
