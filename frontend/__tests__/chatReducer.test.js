import { chatReducer, initialState } from "../src/chatReducer.js";

describe("chatReducer", () => {
  it("returns state unchanged for unknown action types", () => {
    const next = chatReducer(initialState, { type: "NOPE" });
    expect(next).toBe(initialState);
  });

  it("TOGGLE_PANEL flips panelOpen", () => {
    const closed = chatReducer(initialState, { type: "TOGGLE_PANEL" });
    expect(closed.panelOpen).toBe(false);
    const reopened = chatReducer(closed, { type: "TOGGLE_PANEL" });
    expect(reopened.panelOpen).toBe(true);
  });

  it("ADD_USER_MESSAGE and ADD_ASSISTANT_MESSAGE append messages", () => {
    const user = { id: "msg-1", role: "user", content: "hi" };
    const assistant = {
      id: "msg-2",
      role: "assistant",
      content: "",
      citations: [],
      confidence: null,
      streaming: true,
      error: null,
    };
    let s = chatReducer(initialState, {
      type: "ADD_USER_MESSAGE",
      message: user,
    });
    s = chatReducer(s, {
      type: "ADD_ASSISTANT_MESSAGE",
      message: assistant,
    });
    expect(s.messages).toEqual([user, assistant]);
  });

  it("STREAM_DELTA appends a token to the target message only", () => {
    const s0 = {
      ...initialState,
      messages: [
        { id: "msg-1", role: "user", content: "hi" },
        {
          id: "msg-2",
          role: "assistant",
          content: "",
          citations: [],
          confidence: null,
          streaming: true,
          error: null,
        },
      ],
    };
    const s1 = chatReducer(s0, {
      type: "STREAM_DELTA",
      id: "msg-2",
      token: "Hello",
    });
    const s2 = chatReducer(s1, {
      type: "STREAM_DELTA",
      id: "msg-2",
      token: " world",
    });
    expect(s2.messages[1].content).toBe("Hello world");
    expect(s2.messages[0].content).toBe("hi");
  });

  it("SET_SOURCES attaches citations", () => {
    const s0 = {
      ...initialState,
      messages: [
        { id: "msg-2", role: "assistant", content: "x", citations: [] },
      ],
    };
    const s1 = chatReducer(s0, {
      type: "SET_SOURCES",
      id: "msg-2",
      citations: [{ title: "FastAPI" }],
    });
    expect(s1.messages[0].citations).toEqual([{ title: "FastAPI" }]);
  });

  it("SET_SOURCES defaults to empty array when citations missing", () => {
    const s0 = {
      ...initialState,
      messages: [{ id: "msg-2", role: "assistant", content: "x", citations: [] }],
    };
    const s1 = chatReducer(s0, { type: "SET_SOURCES", id: "msg-2" });
    expect(s1.messages[0].citations).toEqual([]);
  });

  it("SET_METADATA sets sessionId and confidence", () => {
    const s0 = {
      ...initialState,
      messages: [
        { id: "msg-2", role: "assistant", content: "x", confidence: null },
      ],
    };
    const s1 = chatReducer(s0, {
      type: "SET_METADATA",
      id: "msg-2",
      sessionId: "sess_abc",
      confidence: 0.9,
    });
    expect(s1.sessionId).toBe("sess_abc");
    expect(s1.messages[0].confidence).toBe(0.9);
  });

  it("SET_METADATA preserves existing sessionId when payload is undefined", () => {
    const s0 = { ...initialState, sessionId: "sess_abc" };
    const s1 = chatReducer(s0, {
      type: "SET_METADATA",
      id: "msg-2",
      confidence: 0.5,
    });
    expect(s1.sessionId).toBe("sess_abc");
  });

  it("GUARDRAIL_REPLACEMENT swaps streamed content for the refusal", () => {
    const s0 = {
      ...initialState,
      messages: [
        { id: "msg-2", role: "assistant", content: "harmful streamed text" },
      ],
    };
    const s1 = chatReducer(s0, {
      type: "GUARDRAIL_REPLACEMENT",
      id: "msg-2",
      answer: "I can't provide that answer.",
    });
    expect(s1.messages[0].content).toBe("I can't provide that answer.");
  });

  it("STREAM_DONE clears the streaming flag", () => {
    const s0 = {
      ...initialState,
      messages: [
        { id: "msg-2", role: "assistant", content: "x", streaming: true },
      ],
    };
    const s1 = chatReducer(s0, { type: "STREAM_DONE", id: "msg-2" });
    expect(s1.messages[0].streaming).toBe(false);
  });

  it("STREAM_ERROR sets error and clears streaming", () => {
    const s0 = {
      ...initialState,
      messages: [
        { id: "msg-2", role: "assistant", content: "x", streaming: true, error: null },
      ],
    };
    const s1 = chatReducer(s0, {
      type: "STREAM_ERROR",
      id: "msg-2",
      error: "Network failed",
    });
    expect(s1.messages[0].streaming).toBe(false);
    expect(s1.messages[0].error).toBe("Network failed");
  });

  it("SET_STREAMING sets the global flag", () => {
    const s1 = chatReducer(initialState, {
      type: "SET_STREAMING",
      value: true,
    });
    expect(s1.isStreaming).toBe(true);
    const s2 = chatReducer(s1, { type: "SET_STREAMING", value: false });
    expect(s2.isStreaming).toBe(false);
  });

  it("SET_FEEDBACK records a rating and CLEAR_FEEDBACK reverts it", () => {
    const s1 = chatReducer(initialState, {
      type: "SET_FEEDBACK",
      messageIndex: 1,
      rating: "up",
    });
    expect(s1.feedbackState[1]).toBe("up");
    const s2 = chatReducer(s1, {
      type: "CLEAR_FEEDBACK",
      messageIndex: 1,
    });
    expect(s2.feedbackState[1]).toBeUndefined();
  });

  it("NEW_CHAT resets the conversation but preserves panelOpen", () => {
    const populated = {
      messages: [{ id: "msg-1", role: "user", content: "hi" }],
      sessionId: "sess_abc",
      isStreaming: false,
      feedbackState: { 1: "up" },
      panelOpen: false,
    };
    const s1 = chatReducer(populated, { type: "NEW_CHAT" });
    expect(s1.messages).toEqual([]);
    expect(s1.sessionId).toBeNull();
    expect(s1.feedbackState).toEqual({});
    expect(s1.isStreaming).toBe(false);
    expect(s1.panelOpen).toBe(false);
  });

  it("does not mutate the previous state", () => {
    const s0 = {
      ...initialState,
      messages: [
        { id: "msg-2", role: "assistant", content: "", streaming: true },
      ],
    };
    const s1 = chatReducer(s0, {
      type: "STREAM_DELTA",
      id: "msg-2",
      token: "Hi",
    });
    expect(s0.messages[0].content).toBe("");
    expect(s1.messages).not.toBe(s0.messages);
    expect(s1.messages[0]).not.toBe(s0.messages[0]);
  });
});
