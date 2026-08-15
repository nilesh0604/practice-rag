import React, { useState, useRef, useCallback } from 'react';
import MessageList from './MessageList.jsx';
import InputBox from './InputBox.jsx';
import ErrorBoundary from './ErrorBoundary.jsx';
import { streamChat, sendFeedback } from './api.js';

/**
 * ChatWidget — the main chat orchestrator.
 *
 * State:
 *  - messages: chronological list of { id, role, content, citations,
 *    confidence, streaming, error }
 *  - sessionId: current session id (null until the first response)
 *  - isStreaming: true while tokens are arriving
 *  - feedbackState: { [messageIndex]: 'up' | 'down' }
 *
 * The SSE consumer (api.js streamChat) calls onToken for each token,
 * onResult with the parsed ChatResponse (citations + confidence +
 * session_id), and onDone when [DONE] is received.
 *
 * Per the architecture doc:
 *   App → ChatWidget → MessageList + InputBox + ErrorBoundary
 *   MessageList → CitationChip
 */
export default function ChatWidget() {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [feedbackState, setFeedbackState] = useState({});
  const idCounter = useRef(0);

  const nextId = useCallback(() => {
    idCounter.current += 1;
    return `msg-${idCounter.current}`;
  }, []);

  const handleSend = useCallback(
    async (text) => {
      const userMsg = { id: nextId(), role: 'user', content: text };
      const assistantMsg = {
        id: nextId(),
        role: 'assistant',
        content: '',
        citations: [],
        confidence: null,
        streaming: true,
        error: null,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      try {
        await streamChat({
          message: text,
          sessionId,
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? { ...m, content: m.content + token }
                  : m,
              ),
            );
          },
          onResult: (result) => {
            if (result.session_id) setSessionId(result.session_id);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? {
                      ...m,
                      citations: result.citations || [],
                      confidence: result.confidence ?? null,
                    }
                  : m,
              ),
            );
          },
          onDone: () => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id ? { ...m, streaming: false } : m,
              ),
            );
          },
        });
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? {
                  ...m,
                  streaming: false,
                  error: err.message || 'Failed to get a response.',
                }
              : m,
          ),
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [sessionId, nextId],
  );

  const handleFeedback = useCallback(
    async (messageIndex, rating) => {
      if (!sessionId) return;
      setFeedbackState((prev) => ({ ...prev, [messageIndex]: rating }));
      try {
        await sendFeedback({
          sessionId,
          messageIndex,
          rating,
        });
      } catch (err) {
        // Revert on failure — the rating wasn't recorded.
        setFeedbackState((prev) => {
          const next = { ...prev };
          delete next[messageIndex];
          return next;
        });
        // eslint-disable-next-line no-console
        console.error('Feedback failed:', err);
      }
    },
    [sessionId],
  );

  const handleNewChat = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    setFeedbackState({});
    setIsStreaming(false);
  }, []);

  return (
    <ErrorBoundary>
      <div className="chat-widget">
        <header className="chat-widget__header">
          <h1 className="chat-widget__title">RAG Knowledge Assistant</h1>
          <button
            className="btn btn--ghost"
            onClick={handleNewChat}
            disabled={isStreaming}
            data-testid="new-chat-button"
          >
            New chat
          </button>
        </header>

        <MessageList
          messages={messages}
          sessionId={sessionId}
          onFeedback={handleFeedback}
          feedbackState={feedbackState}
        />

        <InputBox onSend={handleSend} disabled={isStreaming} />
      </div>
    </ErrorBoundary>
  );
}
