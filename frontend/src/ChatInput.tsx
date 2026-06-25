import * as React from "react";
import styled from "styled-components";
import IconButton from "./IconButton";
import PaperPlane from "./icons/PaperPlane";

import type { JSX } from "react";

const ChatBox = styled.div`
  border-radius: 999px;
  border: 1px solid var(--border-subtle, #cfd6dd);
  outline: none;
  background-color: var(--surface-inset, #eee);
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 0.5em;
  padding-left: 1em;
  padding-right: 0.5em;
  margin: 0.5em;
  transition:
    border-color 120ms ease,
    box-shadow 120ms ease;
  &:focus-within {
    border-color: var(--accent, #f59e0b);
    box-shadow: 0 0 0 3px
      color-mix(in srgb, var(--accent, #f59e0b) 25%, transparent);
  }
`;
const Input = styled.input`
  outline: none;
  background-color: transparent;
  color: var(--text-primary, #121821);
  border: none;
  padding: 0.6em 0;
  font-size: 14px;
  line-height: 1.3;
  flex: 1;
  min-width: 0;
  &::placeholder {
    color: var(--text-secondary, #5a6671);
  }
`;

interface IProps {
  onSubmit: (value: string) => void;
}
const ChatInput = (props: IProps): JSX.Element => {
  const [draft, setDraft] = React.useState<string>("");
  const handleSubmit = (event: React.SyntheticEvent): void => {
    event.preventDefault();
    if (draft.length > 0) {
      props.onSubmit(draft);
      setDraft("");
    }
  };

  const disabled = draft === "";

  return (
    <form onSubmit={handleSubmit} autoComplete="off">
      <ChatBox>
        <Input
          type="text"
          placeholder="Type a message…"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
        />
        <IconButton
          type="submit"
          aria-label="Send message"
          style={{
            opacity: disabled ? 0 : 1,
            transform: disabled ? "translate(0.5em, 0)" : "none",
            color: disabled ? undefined : "var(--accent)",
          }}
          disabled={disabled}
        >
          <PaperPlane width="16px" />
        </IconButton>
      </ChatBox>
    </form>
  );
};
export default ChatInput;
