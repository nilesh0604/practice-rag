import React, { useEffect, useRef } from "react";
import CitationChip from "./CitationChip.jsx";

/**
 * MessageList — renders the chronological list of chat messages.
 *
 * Each assistant message can show:
 *  - citation chips (from the `event: sources` frame)
 *  - a low-confidence warning when confidence < 0.65
 *  - thumbs up/down feedback buttons (enabled after streaming completes)
 *
 * Auto-scrolls to the latest message via a ref + useEffect.
 */

const LOW_CONFIDENCE_THRESHOLD = 0.65;

export default function MessageList({
  messages,
  sessionId,
  onFeedback,
  feedbackState,
}) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="message-list message-list--empty">
        <p className="message-list__placeholder">
          Ask a question about FastAPI, Pydantic, or SQLModel to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="message-list" role="log" aria-live="polite">
      {messages.map((msg, index) => (
        <div key={msg.id} className={`message message--${msg.role}`}>
          <div className="message__bubble">
            <span className="message__role">{msg.role}</span>
            <span className="message__content">
              {msg.content || (msg.streaming ? "\u00a0" : "")}
              {msg.streaming && (
                <span className="message__cursor" aria-hidden="true">
                  |
                </span>
              )}
            </span>
          </div>

          {msg.role === "assistant" && !msg.streaming && (
            <div className="message__meta">
              {msg.citations && msg.citations.length > 0 && (
                <div className="message__citations">
                  {msg.citations.map((cite, i) => (
                    <CitationChip key={i} citation={cite} />
                  ))}
                </div>
              )}

              {typeof msg.confidence === "number" &&
                msg.confidence < LOW_CONFIDENCE_THRESHOLD && (
                  <p className="message__warning" role="status">
                    {"\u26A0"} Low confidence answer — please verify with the
                    sources.
                  </p>
                )}

              {msg.error ? (
                <p className="message__error-text">{msg.error}</p>
              ) : (
                <div className="message__feedback">
                  <button
                    className={`feedback-btn feedback-btn--up${
                      feedbackState[index] === "up"
                        ? " feedback-btn--active"
                        : ""
                    }`}
                    onClick={() => onFeedback(index, "up")}
                    disabled={!sessionId}
                    aria-label="Thumbs up"
                    data-testid={`feedback-up-${index}`}
                  >
                    {"\uD83D\uDC4D"}
                  </button>
                  <button
                    className={`feedback-btn feedback-btn--down${
                      feedbackState[index] === "down"
                        ? " feedback-btn--active"
                        : ""
                    }`}
                    onClick={() => onFeedback(index, "down")}
                    disabled={!sessionId}
                    aria-label="Thumbs down"
                    data-testid={`feedback-down-${index}`}
                  >
                    {"\uD83D\uDC4E"}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
