import React, { useReducer, useRef, useCallback } from "react";
import MessageList from "./MessageList.jsx";
import InputBox from "./InputBox.jsx";
import ErrorBoundary from "./ErrorBoundary.jsx";
import { streamChat, sendFeedback } from "./api.js";
import { chatReducer, initialState } from "./chatReducer.js";

/**
 * ChatWidget — the main chat orchestrator.
 *
 * State is managed via a single `useReducer` (`chatReducer`) so every
 * transition is an explicit, named action (TOGGLE_PANEL, STREAM_DELTA,
 * SET_SOURCES, SET_METADATA, GUARDRAIL_REPLACEMENT, STREAM_DONE,
 * STREAM_ERROR, SET_STREAMING, SET_FEEDBACK, CLEAR_FEEDBACK,
 * ADD_USER_MESSAGE, ADD_ASSISTANT_MESSAGE, NEW_CHAT). See
 * <ref_file file="/Users/Nilesh_Shinde/iSpace/practice-rag/frontend/src/chatReducer.js" />.
 *
 * State shape:
 *  - messages: chronological list of { id, role, content, citations,
 *    confidence, streaming, error }
 *  - sessionId: current session id (null until the first response)
 *  - isStreaming: true while tokens are arriving
 *  - feedbackState: { [messageIndex]: 'up' | 'down' }
 *  - panelOpen: whether the chat panel body is expanded
 *
 * The SSE consumer (api.js streamChat) calls onToken for each delta,
 * onSources with the parsed citations, onMetadata with the session id +
 * confidence + trace id, onGuardrailReplacement when the output guardrail
 * blocks the streamed answer (swapping the visible message for the
 * refusal), and onDone when the terminal event is received.
 *
 * Per the architecture doc:
 *   App → ChatWidget → MessageList + InputBox + ErrorBoundary
 *   MessageList → CitationChip
 */
export default function ChatWidget() {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const { messages, sessionId, isStreaming, feedbackState, panelOpen } = state;
  const idCounter = useRef(0);

  const nextId = useCallback(() => {
    idCounter.current += 1;
    return `msg-${idCounter.current}`;
  }, []);

  const handleSend = useCallback(
    async (text) => {
      const userMsg = { id: nextId(), role: "user", content: text };
      const assistantMsg = {
        id: nextId(),
        role: "assistant",
        content: "",
        citations: [],
        confidence: null,
        streaming: true,
        error: null,
      };

      dispatch({ type: "ADD_USER_MESSAGE", message: userMsg });
      dispatch({ type: "ADD_ASSISTANT_MESSAGE", message: assistantMsg });
      dispatch({ type: "SET_STREAMING", value: true });

      try {
        await streamChat({
          message: text,
          sessionId,
          onToken: (token) =>
            dispatch({ type: "STREAM_DELTA", id: assistantMsg.id, token }),
          onSources: ({ citations }) =>
            dispatch({
              type: "SET_SOURCES",
              id: assistantMsg.id,
              citations,
            }),
          onMetadata: (meta) =>
            dispatch({
              type: "SET_METADATA",
              id: assistantMsg.id,
              sessionId: meta.session_id,
              confidence: meta.confidence,
            }),
          onGuardrailReplacement: ({ answer }) => {
            // The output guardrail blocked the already-streamed tokens.
            // SSE is one-way, so swap the visible message content for the
            // refusal instead of appending to it.
            dispatch({
              type: "GUARDRAIL_REPLACEMENT",
              id: assistantMsg.id,
              answer,
            });
          },
          onDone: () => dispatch({ type: "STREAM_DONE", id: assistantMsg.id }),
        });
      } catch (err) {
        dispatch({
          type: "STREAM_ERROR",
          id: assistantMsg.id,
          error: err.message || "Failed to get a response.",
        });
      } finally {
        dispatch({ type: "SET_STREAMING", value: false });
      }
    },
    [sessionId, nextId],
  );

  const handleFeedback = useCallback(
    async (messageIndex, rating) => {
      if (!sessionId) return;
      dispatch({ type: "SET_FEEDBACK", messageIndex, rating });
      try {
        await sendFeedback({
          sessionId,
          messageIndex,
          rating,
        });
      } catch (err) {
        // Revert on failure — the rating wasn't recorded.
        dispatch({ type: "CLEAR_FEEDBACK", messageIndex });
        // eslint-disable-next-line no-console
        console.error("Feedback failed:", err);
      }
    },
    [sessionId],
  );

  const handleNewChat = useCallback(() => {
    dispatch({ type: "NEW_CHAT" });
  }, []);

  const handleTogglePanel = useCallback(() => {
    dispatch({ type: "TOGGLE_PANEL" });
  }, []);

  return (
    <ErrorBoundary>
      <div className="chat-widget">
        <header className="chat-widget__header">
          <h1 className="chat-widget__title">RAG Knowledge Assistant</h1>
          <button
            className="btn btn--ghost"
            onClick={handleTogglePanel}
            aria-expanded={panelOpen}
            aria-controls="chat-widget__panel"
            data-testid="toggle-panel-button"
          >
            {panelOpen ? "Collapse" : "Expand"}
          </button>
          <button
            className="btn btn--ghost"
            onClick={handleNewChat}
            disabled={isStreaming}
            data-testid="new-chat-button"
          >
            New chat
          </button>
        </header>

        {panelOpen && (
          <div id="chat-widget__panel" className="chat-widget__panel">
            <MessageList
              messages={messages}
              sessionId={sessionId}
              onFeedback={handleFeedback}
              feedbackState={feedbackState}
            />

            <InputBox onSend={handleSend} disabled={isStreaming} />
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
}
