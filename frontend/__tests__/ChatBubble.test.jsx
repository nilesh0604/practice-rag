import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import ChatBubble from "../src/ChatBubble.jsx";

describe("ChatBubble", () => {
  it("renders a button with the chat-bubble test id", () => {
    render(<ChatBubble open={false} onClick={() => {}} />);
    expect(screen.getByTestId("chat-bubble")).toBeInTheDocument();
  });

  it("shows the chat icon when the panel is closed", () => {
    render(<ChatBubble open={false} onClick={() => {}} />);
    expect(screen.getByTestId("chat-bubble")).toHaveTextContent("\uD83D\uDCAC");
  });

  it("shows the close icon when the panel is open", () => {
    render(<ChatBubble open={true} onClick={() => {}} />);
    expect(screen.getByTestId("chat-bubble")).toHaveTextContent("\u2715");
  });

  it("calls onClick when clicked", () => {
    const onClick = jest.fn();
    render(<ChatBubble open={false} onClick={onClick} />);
    fireEvent.click(screen.getByTestId("chat-bubble"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("uses an accessible label that reflects the open state", () => {
    const { rerender } = render(
      <ChatBubble open={false} onClick={() => {}} />,
    );
    expect(screen.getByTestId("chat-bubble")).toHaveAttribute(
      "aria-label",
      "Open chat panel",
    );
    rerender(<ChatBubble open={true} onClick={() => {}} />);
    expect(screen.getByTestId("chat-bubble")).toHaveAttribute(
      "aria-label",
      "Close chat panel",
    );
  });

  it("exposes aria-expanded reflecting the open state", () => {
    const { rerender } = render(
      <ChatBubble open={false} onClick={() => {}} />,
    );
    expect(screen.getByTestId("chat-bubble")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    rerender(<ChatBubble open={true} onClick={() => {}} />);
    expect(screen.getByTestId("chat-bubble")).toHaveAttribute(
      "aria-expanded",
      "true",
    );
  });

  it("points aria-controls at the chat panel id", () => {
    render(<ChatBubble open={false} onClick={() => {}} />);
    expect(screen.getByTestId("chat-bubble")).toHaveAttribute(
      "aria-controls",
      "chat-widget__panel",
    );
  });
});
