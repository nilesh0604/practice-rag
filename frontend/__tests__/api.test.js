import {
  parseSseFrame,
  streamChat,
  sendFeedback,
  fetchHistory,
} from "../src/api.js";

// ── parseSseFrame ──────────────────────────────────────────────

describe("parseSseFrame", () => {
  it("dispatches delta event frames via onToken", () => {
    const cb = { onToken: jest.fn() };
    const kind = parseSseFrame("event: delta\ndata: Hello", cb);
    expect(kind).toBe("delta");
    expect(cb.onToken).toHaveBeenCalledWith("Hello");
  });

  it('strips the "data: " prefix (with space)', () => {
    const cb = { onToken: jest.fn() };
    parseSseFrame("event: delta\ndata: world", cb);
    expect(cb.onToken).toHaveBeenCalledWith("world");
  });

  it("handles data: without space", () => {
    const cb = { onToken: jest.fn() };
    parseSseFrame("event: delta\ndata:nospace", cb);
    expect(cb.onToken).toHaveBeenCalledWith("nospace");
  });

  it("preserves newlines inside multi-line delta frames", () => {
    const cb = { onToken: jest.fn() };
    // Backend emits: event: delta\ndata: line1\nline2\n\n
    // After split on \n\n the frame is "event: delta\ndata: line1\nline2"
    parseSseFrame("event: delta\ndata: line1\nline2", cb);
    expect(cb.onToken).toHaveBeenCalledWith("line1\nline2");
  });

  it("dispatches sources event frames via onSources with parsed JSON", () => {
    const payload = {
      citations: [{ title: "Doc", source_url: "http://example.com" }],
    };
    const cb = { onSources: jest.fn() };
    const kind = parseSseFrame(
      `event: sources\ndata: ${JSON.stringify(payload)}`,
      cb,
    );
    expect(kind).toBe("sources");
    expect(cb.onSources).toHaveBeenCalledWith(payload);
  });

  it("dispatches metadata event frames via onMetadata with parsed JSON", () => {
    const payload = {
      session_id: "sess_1",
      confidence: 0.9,
      trace_id: "lf_abc",
    };
    const cb = { onMetadata: jest.fn() };
    const kind = parseSseFrame(
      `event: metadata\ndata: ${JSON.stringify(payload)}`,
      cb,
    );
    expect(kind).toBe("metadata");
    expect(cb.onMetadata).toHaveBeenCalledWith(payload);
  });

  it("dispatches done event via onDone", () => {
    const cb = { onDone: jest.fn() };
    const kind = parseSseFrame("event: done\ndata: [DONE]", cb);
    expect(kind).toBe("done");
    expect(cb.onDone).toHaveBeenCalled();
  });

  it("returns null for empty frames", () => {
    expect(parseSseFrame("", {})).toBeNull();
    expect(parseSseFrame("data: ", {})).toBeNull();
  });

  it("does not call onToken when not provided", () => {
    expect(() => parseSseFrame("event: delta\ndata: hi", {})).not.toThrow();
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
    text: () => Promise.resolve("error body"),
  });
}

describe("streamChat", () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it("streams deltas, sources, metadata, and done in order", async () => {
    const sse = [
      "event: delta\ndata: Hello\n\n",
      "event: delta\ndata:  world\n\n",
      `event: sources\ndata: ${JSON.stringify({ citations: [] })}\n\n`,
      `event: metadata\ndata: ${JSON.stringify({
        session_id: "sess_abc",
        confidence: 0.8,
        trace_id: null,
      })}\n\n`,
      "event: done\ndata: [DONE]\n\n",
    ];
    global.fetch.mockReturnValue(mockFetchResponse(sse));

    const tokens = [];
    let sources = null;
    let metadata = null;
    let doneCalled = false;

    await streamChat({
      message: "hi",
      sessionId: null,
      onToken: (t) => tokens.push(t),
      onSources: (s) => {
        sources = s;
      },
      onMetadata: (m) => {
        metadata = m;
      },
      onDone: () => {
        doneCalled = true;
      },
    });

    expect(tokens).toEqual(["Hello", " world"]);
    expect(sources).toEqual({ citations: [] });
    expect(metadata).toEqual({
      session_id: "sess_abc",
      confidence: 0.8,
      trace_id: null,
    });
    expect(doneCalled).toBe(true);
  });

  it("handles tokens split across chunk boundaries", async () => {
    // One SSE frame split across two fetch chunks
    const sse = [
      "event: delta\ndata: Hel",
      "lo\n\n",
      "event: done\ndata: [DONE]\n\n",
    ];
    global.fetch.mockReturnValue(mockFetchResponse(sse));

    const tokens = [];
    await streamChat({ message: "hi", onToken: (t) => tokens.push(t) });

    expect(tokens).toEqual(["Hello"]);
  });

  it("sends the correct request body", async () => {
    global.fetch.mockReturnValue(
      mockFetchResponse(["event: done\ndata: [DONE]\n\n"]),
    );

    await streamChat({ message: "test question", sessionId: "sess_123" });

    expect(global.fetch).toHaveBeenCalledWith("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "test question",
        session_id: "sess_123",
      }),
    });
  });

  it("sends null session_id when not provided", async () => {
    global.fetch.mockReturnValue(
      mockFetchResponse(["event: done\ndata: [DONE]\n\n"]),
    );

    await streamChat({ message: "test" });

    const callBody = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(callBody.session_id).toBeNull();
  });

  it("throws on non-2xx response", async () => {
    global.fetch.mockReturnValue(
      mockFetchResponse([], { ok: false, status: 404 }),
    );

    await expect(streamChat({ message: "hi" })).rejects.toThrow("404");
  });

  it("handles cache replay (single delta + sources + metadata + done)", async () => {
    const sse = [
      "event: delta\ndata: Cached answer\n\n",
      `event: sources\ndata: ${JSON.stringify({
        citations: [
          { title: "FastAPI Docs", source_url: "http://fastapi.tiangolo.com" },
        ],
      })}\n\n`,
      `event: metadata\ndata: ${JSON.stringify({
        session_id: "sess_new",
        confidence: 0.95,
        trace_id: null,
      })}\n\n`,
      "event: done\ndata: [DONE]\n\n",
    ];
    global.fetch.mockReturnValue(mockFetchResponse(sse));

    const tokens = [];
    let sources = null;
    let metadata = null;
    await streamChat({
      message: "cached query",
      onToken: (t) => tokens.push(t),
      onSources: (s) => {
        sources = s;
      },
      onMetadata: (m) => {
        metadata = m;
      },
    });

    expect(tokens).toEqual(["Cached answer"]);
    expect(sources.citations).toHaveLength(1);
    expect(sources.citations[0].title).toBe("FastAPI Docs");
    expect(metadata.session_id).toBe("sess_new");
  });
});

// ── sendFeedback ────────────────────────────────────────────────

describe("sendFeedback", () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it("sends POST /api/v1/feedback with correct body", async () => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ recorded: true, feedback_id: 1 }),
    });

    const result = await sendFeedback({
      sessionId: "sess_1",
      messageIndex: 1,
      rating: "up",
      comment: "great",
    });

    expect(result).toEqual({ recorded: true, feedback_id: 1 });
    expect(global.fetch).toHaveBeenCalledWith("/api/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: "sess_1",
        message_index: 1,
        rating: "up",
        comment: "great",
      }),
    });
  });

  it("throws on non-2xx", async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 404,
      text: () => Promise.resolve("not found"),
    });

    await expect(
      sendFeedback({ sessionId: "x", messageIndex: 0, rating: "down" }),
    ).rejects.toThrow("404");
  });
});

// ── fetchHistory ────────────────────────────────────────────────

describe("fetchHistory", () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it("GETs /api/v1/history/{id} and returns parsed JSON", async () => {
    const data = {
      session_id: "sess_1",
      messages: [
        {
          role: "user",
          content: "hi",
          citations: null,
          confidence: null,
          created_at: "2026-01-01T00:00:00Z",
        },
      ],
    };
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(data),
    });

    const result = await fetchHistory("sess_1");
    expect(global.fetch).toHaveBeenCalledWith("/api/v1/history/sess_1");
    expect(result).toEqual(data);
  });

  it("throws on non-2xx", async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 404,
      text: () => Promise.resolve("not found"),
    });

    await expect(fetchHistory("missing")).rejects.toThrow("404");
  });
});
