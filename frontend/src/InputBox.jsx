import React, { useState, useRef, useEffect } from 'react';

/**
 * InputBox — textarea + SendButton.
 *
 * Enter sends the message; Shift+Enter inserts a newline.
 * Disabled while streaming is in progress.
 *
 * Per the architecture doc:
 *   ChatWidget → InputBox → SendButton
 */
export default function InputBox({ onSend, disabled }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  // Refocus after send / disable toggle.
  useEffect(() => {
    if (!disabled) textareaRef.current?.focus();
  }, [disabled]);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="input-box">
      <textarea
        ref={textareaRef}
        className="input-box__textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Ask about FastAPI, Pydantic, or SQLModel..."
        rows={1}
        aria-label="Chat message input"
        data-testid="chat-input"
      />
      <button
        className="btn btn--send"
        onClick={handleSend}
        disabled={disabled || !text.trim()}
        aria-label="Send message"
        data-testid="send-button"
      >
        Send
      </button>
    </div>
  );
}
