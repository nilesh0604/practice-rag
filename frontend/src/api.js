/**
 * API client for the practice-rag FastAPI backend.
 *
 * Exposes:
 *  - parseSseFrame — pure SSE frame parser (unit-testable without fetch)
 *  - streamChat — fetch + ReadableStream SSE consumer for POST /api/v1/chat
 *  - sendFeedback — POST /api/v1/feedback
 *  - fetchHistory — GET /api/v1/history/{session_id}
 *
 * The SSE wire format produced by the backend (Step 4):
 *
 *   data: <token>\n\n                  # one per generated token
 *   event: result\ndata: {json}\n\n    # exactly once, after the last token
 *   data: [DONE]\n\n                   # terminal sentinel
 *
 * The `event: result` frame carries the full ChatResponse JSON (session_id,
 * answer, citations, confidence) so the frontend can render citation chips
 * and the low-confidence warning without parsing streamed text.
 */

const API_BASE = '/api/v1';

/**
 * Parse a single SSE frame (text between `\n\n` separators) and dispatch
 * to the appropriate callback.
 *
 * Multi-line data is reconstructed: if a token contained a newline, the
 * backend emits `data: line1\nline2\n\n`. Non-prefixed continuation lines
 * are rejoined with `\n` so the original token is preserved.
 *
 * @param {string} frame - Raw SSE frame text (no trailing `\n\n`).
 * @param {object} callbacks - { onToken, onResult, onDone }.
 * @returns {'token'|'result'|'done'|null} frame kind, or null for empty.
 */
export function parseSseFrame(frame, callbacks = {}) {
  const lines = frame.split('\n');
  let eventType = null;
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.replace(/^data: ?/, ''));
    } else if (line.length > 0) {
      // Continuation line (newline inside a token) — rejoin.
      dataLines.push(line);
    }
  }

  const data = dataLines.join('\n');

  if (eventType === 'result') {
    callbacks.onResult?.(JSON.parse(data));
    return 'result';
  }
  if (data === '[DONE]') {
    callbacks.onDone?.();
    return 'done';
  }
  if (data.length > 0) {
    callbacks.onToken?.(data);
    return 'token';
  }
  return null;
}

/**
 * Stream a chat answer from POST /api/v1/chat via fetch + ReadableStream.
 *
 * @param {object} opts
 * @param {string} opts.message - The user's question.
 * @param {string|null} [opts.sessionId] - Existing session id or null for new.
 * @param {function} [opts.onToken] - Called with each token string.
 * @param {function} [opts.onResult] - Called with the parsed ChatResponse.
 * @param {function} [opts.onDone] - Called when [DONE] is received.
 * @returns {Promise<void>} Resolves when the stream completes.
 * @throws {Error} On non-2xx response or network failure.
 */
export async function streamChat({
  message,
  sessionId = null,
  onToken,
  onResult,
  onDone,
}) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(`Chat request failed (${response.status}): ${body}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let done = false;

  while (true) {
    const { done: streamDone, value } = await reader.read();
    if (streamDone) break;

    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const kind = parseSseFrame(frame, { onToken, onResult, onDone });
      if (kind === 'done') done = true;
    }
  }

  // Flush any trailing frame (shouldn't happen with [DONE], but be safe).
  if (!done && buffer.trim()) {
    parseSseFrame(buffer, { onToken, onResult, onDone });
  }
}

/**
 * Send thumbs up/down feedback for a message.
 *
 * @param {object} opts
 * @param {string} opts.sessionId
 * @param {number} opts.messageIndex - 0-based position in the session's
 *   chronological message list (matches FeedbackRequest.message_index).
 * @param {'up'|'down'} opts.rating
 * @param {string} [opts.comment]
 * @returns {Promise<{recorded: boolean, feedback_id: number}>}
 */
export async function sendFeedback({ sessionId, messageIndex, rating, comment }) {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message_index: messageIndex,
      rating,
      comment,
    }),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(`Feedback failed (${response.status}): ${body}`);
  }
  return response.json();
}

/**
 * Fetch the full conversation history for a session.
 *
 * @param {string} sessionId
 * @returns {Promise<{session_id: string, messages: Array}>}
 */
export async function fetchHistory(sessionId) {
  const response = await fetch(`${API_BASE}/history/${sessionId}`);
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(`History fetch failed (${response.status}): ${body}`);
  }
  return response.json();
}
