/**
 * chatReducer — central state machine for the ChatWidget.
 *
 * Replaces the previous `useState` + `useCallback` collection with a single
 * `useReducer` so every state transition is an explicit, named action
 * (matching the architecture doc's listed action types).
 *
 * Action types
 * ------------
 *  - TOGGLE_PANEL            : flip the panel open/collapsed
 *  - ADD_USER_MESSAGE        : append a user message
 *  - ADD_ASSISTANT_MESSAGE   : append a streaming assistant placeholder
 *  - STREAM_DELTA            : append a token to a streaming assistant message
 *  - SET_SOURCES             : attach citations to an assistant message
 *  - SET_METADATA            : set sessionId + per-message confidence
 *  - GUARDRAIL_REPLACEMENT   : swap streamed tokens for a refusal
 *  - STREAM_DONE             : mark an assistant message as no longer streaming
 *  - STREAM_ERROR            : mark an assistant message errored + stop streaming
 *  - SET_STREAMING           : set the global isStreaming flag
 *  - SET_FEEDBACK            : record a thumbs up/down rating for a message
 *  - CLEAR_FEEDBACK          : revert a rating (e.g. when sendFeedback fails)
 *  - NEW_CHAT                : reset the conversation (keeps panelOpen)
 *
 * State shape
 *  - messages       : chronological list of
 *                     { id, role, content, citations, confidence, streaming, error }
 *  - sessionId      : current session id (null until the first response)
 *  - isStreaming    : true while tokens are arriving
 *  - feedbackState  : { [messageIndex]: 'up' | 'down' }
 *  - panelOpen      : whether the chat panel body is expanded
 */

export const initialState = {
  messages: [],
  sessionId: null,
  isStreaming: false,
  feedbackState: {},
  panelOpen: true,
};

/**
 * Update a single message by id with an immutable patch.
 * @param {Array} messages - current messages array
 * @param {string} id - target message id
 * @param {Function} patch - (msg) => partial update object
 * @returns {Array} new messages array
 */
function patchMessage(messages, id, patch) {
  return messages.map((m) => (m.id === id ? { ...m, ...patch(m) } : m));
}

/**
 * Reducer for the ChatWidget state machine.
 * @param {object} state - current state
 * @param {object} action - { type, ...payload }
 * @returns {object} next state
 */
export function chatReducer(state, action) {
  switch (action.type) {
    case "TOGGLE_PANEL":
      return { ...state, panelOpen: !state.panelOpen };

    case "ADD_USER_MESSAGE":
      return { ...state, messages: [...state.messages, action.message] };

    case "ADD_ASSISTANT_MESSAGE":
      return { ...state, messages: [...state.messages, action.message] };

    case "STREAM_DELTA":
      return {
        ...state,
        messages: patchMessage(state.messages, action.id, (m) => ({
          content: m.content + action.token,
        })),
      };

    case "SET_SOURCES":
      return {
        ...state,
        messages: patchMessage(state.messages, action.id, () => ({
          citations: action.citations || [],
        })),
      };

    case "SET_METADATA":
      return {
        ...state,
        sessionId: action.sessionId ?? state.sessionId,
        messages: patchMessage(state.messages, action.id, () => ({
          confidence: action.confidence ?? null,
        })),
      };

    case "GUARDRAIL_REPLACEMENT":
      return {
        ...state,
        messages: patchMessage(state.messages, action.id, () => ({
          content: action.answer,
        })),
      };

    case "STREAM_DONE":
      return {
        ...state,
        messages: patchMessage(state.messages, action.id, () => ({
          streaming: false,
        })),
      };

    case "STREAM_ERROR":
      return {
        ...state,
        messages: patchMessage(state.messages, action.id, () => ({
          streaming: false,
          error: action.error,
        })),
      };

    case "SET_STREAMING":
      return { ...state, isStreaming: action.value };

    case "SET_FEEDBACK":
      return {
        ...state,
        feedbackState: {
          ...state.feedbackState,
          [action.messageIndex]: action.rating,
        },
      };

    case "CLEAR_FEEDBACK": {
      const next = { ...state.feedbackState };
      delete next[action.messageIndex];
      return { ...state, feedbackState: next };
    }

    case "NEW_CHAT":
      // Reset the conversation but preserve the panel open/collapsed state.
      return { ...initialState, panelOpen: state.panelOpen };

    default:
      return state;
  }
}
