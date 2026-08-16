import React from "react";

/**
 * ChatBubble — the floating trigger button for the chat widget.
 *
 * Rendered `position: fixed` in the bottom-right corner of the viewport
 * (see `.chat-bubble` in `styles.css`). Clicking it toggles the chat panel
 * open/collapsed via the `TOGGLE_PANEL` action. Shows a chat icon when the
 * panel is closed and a close (×) icon when open.
 *
 * Per the architecture doc's listed component tree:
 *   ChatAssistantWidget → ChatBubble (floating trigger) → ChatPanel
 *
 * @param {object} props
 * @param {boolean} props.open - whether the chat panel is currently open
 * @param {Function} props.onClick - toggle handler (dispatches TOGGLE_PANEL)
 */
export default function ChatBubble({ open, onClick }) {
  return (
    <button
      type="button"
      className="chat-bubble"
      onClick={onClick}
      aria-label={open ? "Close chat panel" : "Open chat panel"}
      aria-expanded={open}
      aria-controls="chat-widget__panel"
      data-testid="chat-bubble"
    >
      {open ? "\u2715" : "\uD83D\uDCAC"}
    </button>
  );
}
