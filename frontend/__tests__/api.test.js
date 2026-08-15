import { parseSseFrame, streamChat, sendFeedback, fetchHistory } from '../src/api.js';

// ── parseSseFrame ──────────────────────────────────────────────

describe('parseSseFrame', () => {
  it('dispatches token frames via onToken', () => {
    const cb = { onToken: jest.fn() };
    const kind = parseSseFrame('data: Hello', cb);
    expect(kind).toBe('token');
    expect(cb.onToken).toHaveBeenCalledWith('Hello');
  });

  it('strips the "data: " prefix (with space)', () => {
    const cb = { onToken: jest.fn() };
    parseSseFrame('data: world', cb);
    expect(cb.onToken).toHaveBeenCalledWith('world');
  });

  it('handles data: without space', () => {
    const cb = { onToken: jest.fn() };
    parseSseFrame('data:nospace', cb);
    expect(cb.onToken).toHaveBeenCalledWith('nospace');
  });

  it('preserves newlines inside multi-line token frames', () => {
    const cb = { onToken: jest.fn() };
    // Backend emits: data: line1\nline2\n\n
    // After split on \n\n the frame is "data: line1\nline2"
    parseSseFrame('data: line1\nline2', cb);
    expect(cb.onToken).toHaveBeenCalledWith('line1\nline2');
  });

  it('dispatches result event frames via onResult with parsed JSON', () => {
    const payload = {
      session_id: 'sess_1',
      answer: 'Hello',
      citations: [{ title: 'Doc', source_url: 'http://example.com' }],
      confidence: 0.9,
    };
    const cb = { onResult: jest.fn() };
    const kind = parseSseFrame(`event: result\ndata: ${JSON.stringify(payload)}`, cb);
    expect(kind).toBe('result');
    expect(cb.onResult).toHaveBeenCalledWith(payload);
  });

  it('dispatches [DONE] via onDone', () => {
    const cb = { onDone: jest.fn() };
    const kind = parseSseFrame('data: [DONE]', cb);
    expect(kind).toBe('done');
    expect(cb.onDone).toHaveBeenCalled();
  });

  it('returns null for empty frames', () => {
    expect(parseSseFrame('', {})).toBeNull();
    expect(parseSseFrame('data: ', {})).toBeNull();
  });

  it('does not call onToken when not provided', () => {
    expect(() => parseSseFrame('data: hi', {})).not.toThrow();
  });
});

// ── streamChat ──────────────────────────────────────────────────

/**
 * Helper: create a mock ReadableStream that yields the given chunks.
 */
function mockReadableStream(chunks) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

function mockFetchResponse(chunks, { ok = true, status = 200 } = {}) {
  return Promise.resolve({
    ok,
    status,
    body: mockReadableStream(chunks),
    text: () => Promise.resolve('error body'),
  });
}

describe('streamChat', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it('streams tokens, result, and done in order', async () => {
    const sse = [
      'data: Hello\n\n',
      'data:  world\n\n',
      `event: result\ndata: ${JSON.stringify({
        session_id: 'sess_abc',
        answer: 'Hello world',
        citations: [],
        confidence: 0.8,
      })}\n\n`,
      'data: [DONE]\n\n',
    ];
    global.fetch.mockReturnValue(mockFetchResponse(sse));

    const tokens = [];
    let result = null;
    let doneCalled = false;

    await streamChat({
      message: 'hi',
      sessionId: null,
      onToken: (t) => tokens.push(t),
      onResult: (r) => { result = r; },
      onDone: () => { doneCalled = true; },
    });

    expect(tokens).toEqual(['Hello', ' world']);
    expect(result).toEqual({
      session_id: 'sess_abc',
      answer: 'Hello world',
      citations: [],
      confidence: 0.8,
    });
    expect(doneCalled).toBe(true);
  });

  it('handles tokens split across chunk boundaries', async () => {
    // One SSE frame split across two fetch chunks
    const sse = ['data: Hel', 'lo\n\n', 'data: [DONE]\n\n'];
    global.fetch.mockReturnValue(mockFetchResponse(sse));

    const tokens = [];
    await streamChat({ message: 'hi', onToken: (t) => tokens.push(t) });

    expect(tokens).toEqual(['Hello']);
  });

  it('sends the correct request body', async () => {
    global.fetch.mockReturnValue(mockFetchResponse(['data: [DONE]\n\n']));

    await streamChat({ message: 'test question', sessionId: 'sess_123' });

    expect(global.fetch).toHaveBeenCalledWith('/api/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: 'test question', session_id: 'sess_123' }),
    });
  });

  it('sends null session_id when not provided', async () => {
    global.fetch.mockReturnValue(mockFetchResponse(['data: [DONE]\n\n']));

    await streamChat({ message: 'test' });

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.session_id).toBeNull();
  });

  it('throws on non-2xx response', async () => {
    global.fetch.mockReturnValue(mockFetchResponse([], { ok: false, status: 404 }));

    await expect(streamChat({ message: 'hi' })).rejects.toThrow('404');
  });

  it('handles cache replay (single data frame + result + done)', async () => {
    const sse = [
      'data: Cached answer\n\n',
      `event: result\ndata: ${JSON.stringify({
        session_id: 'sess_new',
        answer: 'Cached answer',
        citations: [{ title: 'FastAPI Docs', source_url: 'http://fastapi.tiangolo.com' }],
        confidence: 0.95,
      })}\n\n`,
      'data: [DONE]\n\n',
    ];
    global.fetch.mockReturnValue(mockFetchResponse(sse));

    const tokens = [];
    let result = null;
    await streamChat({
      message: 'cached query',
      onToken: (t) => tokens.push(t),
      onResult: (r) => { result = r; },
    });

    expect(tokens).toEqual(['Cached answer']);
    expect(result.citations).toHaveLength(1);
    expect(result.citations[0].title).toBe('FastAPI Docs');
  });
});

// ── sendFeedback ────────────────────────────────────────────────

describe('sendFeedback', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it('sends POST /api/v1/feedback with correct body', async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ recorded: true, feedback_id: 1 }),
    });

    const result = await sendFeedback({
      sessionId: 'sess_1',
      messageIndex: 1,
      rating: 'up',
      comment: 'great',
    });

    expect(result).toEqual({ recorded: true, feedback_id: 1 });
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: 'sess_1',
        message_index: 1,
        rating: 'up',
        comment: 'great',
      }),
    });
  });

  it('throws on non-2xx', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 404,
      text: () => Promise.resolve('not found'),
    });

    await expect(
      sendFeedback({ sessionId: 'x', messageIndex: 0, rating: 'down' }),
    ).rejects.toThrow('404');
  });
});

// ── fetchHistory ────────────────────────────────────────────────

describe('fetchHistory', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it('GETs /api/v1/history/{id} and returns parsed JSON', async () => {
    const data = {
      session_id: 'sess_1',
      messages: [
        { role: 'user', content: 'hi', citations: null, confidence: null, created_at: '2026-01-01T00:00:00Z' },
      ],
    };
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(data),
    });

    const result = await fetchHistory('sess_1');
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/history/sess_1');
    expect(result).toEqual(data);
  });

  it('throws on non-2xx', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 404,
      text: () => Promise.resolve('not found'),
    });

    await expect(fetchHistory('missing')).rejects.toThrow('404');
  });
});
