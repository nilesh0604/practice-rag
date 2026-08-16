/**
 * API client for the practice-rag FastAPI backend.
 *
 * Exposes:
 *  - parseSseFrame — pure SSE frame parser (unit-testable without fetch)
 *  - streamChat — fetch + ReadableStream SSE consumer for POST /api/v1/chat
 *  - sendFeedback — POST /api/v1/feedback
 *  - fetchHistory — GET /api/v1/history/{session_id}
 *
 * The SSE wire format produced by the backend uses named events:
 *
 *   event: delta\ndata: <token>\n\n             # one per generated token
 *   event: sources\ndata: {"citations":[...]}\n\n  # exactly once, after the last token
 *   event: metadata\ndata: {json}\n\n           # exactly once, after sources
 *   event: done\ndata: [DONE]\n\n               # terminal sentinel
 *
 * The `event: sources` frame carries the post-processed citations so the
 * frontend can render citation chips without parsing the streamed text.
 * The `event: metadata` frame carries the session id, confidence score,
 * and trace id so the frontend can set the session, render the
 * low-confidence warning, and pass the trace id back with feedback.
 */

const API_BASE = "/api/v1";

/**
 * Parse a single SSE frame (text between `\n\n` separators) and dispatch
 * to the appropriate callback.
 *
 * Multi-line data is reconstructed: if a token contained a newline, the
 * backend emits `event: delta\ndata: line1\nline2\n\n`. Non-prefixed
 * continuation lines are rejoined with `\n` so the original token is
 * preserved.
 *
 * @param {string} frame - Raw SSE frame text (no trailing `\n\n`).
 * @param {object} callbacks - { onToken, onSources, onMetadata, onDone }.
 * @returns {'delta'|'sources'|'metadata'|'done'|null} frame kind, or null for empty.
 */
export function parseSseFrame(frame, callbacks = {}) {
  const lines = frame.split("\n");
  let eventType = null;
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.replace(/^data: ?/, ""));
    } else if (line.length > 0) {
      // Continuation line (newline inside a token) — rejoin.
      dataLines.push(line);
    }
  }

  const data = dataLines.join("\n");

  if (eventType === "delta") {
    callbacks.onToken?.(data);
    return "delta";
  }
  if (eventType === "sources") {
    callbacks.onSources?.(JSON.parse(data));
    return "sources";
  }
  if (eventType === "metadata") {
    callbacks.onMetadata?.(JSON.parse(data));
    return "metadata";
  }
  if (eventType === "done" || data === "[DONE]") {
    callbacks.onDone?.();
    return "done";
  }
  if (data.length > 0) {
    // Unnamed data frame (legacy / defensive) — treat as a token.
    callbacks.onToken?.(data);
    return "delta";
  }
  return null;
}

/**
 * Stream a chat answer from POST /api/v1/chat via fetch + ReadableStream.
 *
 * @param {object} opts
 * @param {string} opts.message - The user's question.
 * @param {string|null} [opts.sessionId] - Existing session id or null for new.
 * @param {function} [opts.onToken] - Called with each token string (delta events).
 * @param {function} [opts.onSources] - Called with { citations: [...] } (sources event).
 * @param {function} [opts.onMetadata] - Called with { session_id, confidence, trace_id } (metadata event).
 * @param {function} [opts.onDone] - Called when the done event is received.
 * @returns {Promise<void>} Resolves when the stream completes.
 * @throws {Error} On non-2xx response or network failure.
 */
export async function streamChat({
  message,
  sessionId = null,
  onToken,
  onSources,
  onMetadata,
  onDone,
}) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Chat request failed (${response.status}): ${body}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done = false;

  while (true) {
    const { done: streamDone, value } = await reader.read();
    if (streamDone) break;

    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const kind = parseSseFrame(frame, {
        onToken,
        onSources,
        onMetadata,
        onDone,
      });
      if (kind === "done") done = true;
    }
  }

  // Flush any trailing frame (shouldn't happen with [DONE], but be safe).
  if (!done && buffer.trim()) {
    parseSseFrame(buffer, { onToken, onSources, onMetadata, onDone });
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
export async function sendFeedback({
  sessionId,
  messageIndex,
  rating,
  comment,
}) {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message_index: messageIndex,
      rating,
      comment,
    }),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
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
    const body = await response.text().catch(() => "");
    throw new Error(`History fetch failed (${response.status}): ${body}`);
  }
  return response.json();
}
