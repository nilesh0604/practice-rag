import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import ChatWidget from "../src/ChatWidget.jsx";

// Mock the api module so we control SSE callbacks.
jest.mock("../src/api.js", () => ({
  streamChat: jest.fn(),
  sendFeedback: jest.fn(),
}));

const { streamChat, sendFeedback } = require("../src/api.js");

/**
 * Helper: simulate streamChat calling onToken/onSources/onMetadata/onDone.
 * Callbacks are invoked synchronously (no act() wrapping) — React 18
 * automatic batching + RTL waitFor handle the re-renders.
 */
function mockStreamChatTokens({ tokens, result, sessionId = "sess_test" }) {
  streamChat.mockImplementation(
    async ({ onToken, onSources, onMetadata, onDone }) => {
      for (const token of tokens) {
        onToken?.(token);
      }
      if (result !== undefined) {
        onSources?.({ citations: result?.citations ?? [] });
        onMetadata?.({
          session_id: result?.session_id ?? sessionId,
          confidence: result?.confidence ?? null,
          trace_id: result?.trace_id ?? null,
        });
      }
      onDone?.();
    },
  );
}

describe("ChatWidget", () => {
  beforeEach(() => {
    streamChat.mockReset();
    sendFeedback.mockReset();
  });

  it("renders the header, empty placeholder, and input", () => {
    render(<ChatWidget />);
    expect(screen.getByText("RAG Knowledge Assistant")).toBeInTheDocument();
    expect(
      screen.getByText(/Ask a question about FastAPI/),
    ).toBeInTheDocument();
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
  });

  it("sends a message and streams the response", async () => {
    mockStreamChatTokens({
      tokens: ["Hello", " world"],
      result: { answer: "Hello world", citations: [], confidence: 0.9 },
    });

    render(<ChatWidget />);
    const input = screen.getByTestId("chat-input");
    fireEvent.change(input, { target: { value: "How do I use FastAPI?" } });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByText("Hello world")).toBeInTheDocument();
    });

    expect(streamChat).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "How do I use FastAPI?",
        sessionId: null,
      }),
    );
  });

  it("renders citation chips from the result frame", async () => {
    mockStreamChatTokens({
      tokens: ["See docs"],
      result: {
        answer: "See docs",
        citations: [
          { title: "FastAPI", source_url: "https://fastapi.tiangolo.com" },
        ],
        confidence: 0.9,
      },
    });

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByText("[Source: FastAPI]")).toBeInTheDocument();
    });
  });

  it("shows low-confidence warning when confidence < 0.65", async () => {
    mockStreamChatTokens({
      tokens: ["Maybe"],
      result: { answer: "Maybe", citations: [], confidence: 0.3 },
    });

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByText(/Low confidence/)).toBeInTheDocument();
    });
  });

  it("disables input while streaming and re-enables after", async () => {
    let resolveStream;
    streamChat.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveStream = resolve;
        }),
    );

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByTestId("chat-input")).toBeDisabled();
    });

    await act(async () => {
      resolveStream();
    });

    await waitFor(() => {
      expect(screen.getByTestId("chat-input")).not.toBeDisabled();
    });
  });

  it("shows error message when streamChat throws", async () => {
    streamChat.mockRejectedValue(new Error("Network failed"));

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByText("Network failed")).toBeInTheDocument();
    });
  });

  it("sends feedback via sendFeedback on thumbs up click", async () => {
    mockStreamChatTokens({
      tokens: ["Answer"],
      result: { answer: "Answer", citations: [], confidence: 0.9 },
    });
    sendFeedback.mockResolvedValue({ recorded: true, feedback_id: 1 });

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByTestId("feedback-up-1")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("feedback-up-1"));

    await waitFor(() => {
      expect(sendFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          sessionId: "sess_test",
          messageIndex: 1,
          rating: "up",
        }),
      );
    });
  });

  it("reverts feedback state when sendFeedback fails", async () => {
    mockStreamChatTokens({
      tokens: ["Answer"],
      result: { answer: "Answer", citations: [], confidence: 0.9 },
    });
    sendFeedback.mockRejectedValue(new Error("feedback failed"));

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByTestId("feedback-up-1")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("feedback-up-1"));

    await waitFor(() => {
      expect(sendFeedback).toHaveBeenCalled();
    });

    // After failure, the button should not be active.
    await waitFor(() => {
      expect(screen.getByTestId("feedback-up-1")).not.toHaveClass(
        "feedback-btn--active",
      );
    });
  });

  it('clears messages on "New chat" click', async () => {
    mockStreamChatTokens({
      tokens: ["Answer"],
      result: { answer: "Answer", citations: [], confidence: 0.9 },
    });

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByText("Answer")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("new-chat-button"));

    await waitFor(() => {
      expect(
        screen.getByText(/Ask a question about FastAPI/),
      ).toBeInTheDocument();
    });
  });

  it("passes the session id from the result frame to subsequent requests", async () => {
    mockStreamChatTokens({
      tokens: ["First"],
      result: {
        answer: "First",
        citations: [],
        confidence: 0.9,
        session_id: "sess_abc",
      },
    });

    render(<ChatWidget />);
    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "first" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByText("First")).toBeInTheDocument();
    });

    // Second message should carry the session id.
    mockStreamChatTokens({
      tokens: ["Second"],
      result: {
        answer: "Second",
        citations: [],
        confidence: 0.9,
        session_id: "sess_abc",
      },
    });

    fireEvent.change(screen.getByTestId("chat-input"), {
      target: { value: "second" },
    });
    fireEvent.click(screen.getByTestId("send-button"));

    await waitFor(() => {
      expect(screen.getByText("Second")).toBeInTheDocument();
    });

    const secondCall = streamChat.mock.calls[1][0];
    expect(secondCall.sessionId).toBe("sess_abc");
  });
});
