import * as React from "react";
import styled from "styled-components";

import type { JSX } from "react";

const Button = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  outline: none;
  padding: 6px;
  margin: 0;
  border: 0;
  border-radius: var(--radius-xl, 14px);
  background-color: transparent;
  cursor: pointer;
  transition:
    opacity 100ms ease-in-out,
    color 150ms ease-in-out,
    background-color 150ms ease-in-out,
    transform 100ms ease-in-out;
  color: currentColor;
  &:hover {
    background-color: rgba(255, 255, 255, 0.1);
  }
  &:active {
    transform: translateY(1px);
  }
`;

const IconButton = (
  props: React.ComponentProps<typeof Button>,
): JSX.Element => {
  return <Button className="icon-button" {...props} />;
};
export default IconButton;
