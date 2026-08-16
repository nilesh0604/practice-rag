"""Tests for ``POST /api/v1/chat`` SSE endpoint (api/routes/chat.py).

Covers: SSE wire format (named events ``delta`` / ``sources`` /
``metadata`` / ``done``), session creation/resolution, 404 on unknown
session, cache hit replay, cache miss streaming, message persistence,
and the factored ``build_event_stream`` / ``build_replay_stream``
generators.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from schemas.chat import ChatResponse, Citation

from api.cache import LRUCache
from api.conversation import ConversationStore
from api.routes.chat import build_event_stream, build_replay_stream
from rag.post_processor import PostProcessResult


def _result(answer="Hello world", confidence=0.9, citations=None):
    return PostProcessResult(
        answer=answer,
        citations=citations or [],
        confidence=confidence,
    )


# ── build_event_stream (unit-level, no HTTP) ────────────────────────────


class TestBuildEventStream:
    def test_yields_delta_then_sources_then_metadata_then_done(self):
        orch = MagicMock()
        orch.stream_answer.return_value = iter(["A", "B", _result()])
        store = ConversationStore(":memory:")
        cache = LRUCache(max_size=4)
        sid = store.create_session()

        events = list(build_event_stream(orch, "q", "", sid, store, cache))

        assert events[0] == "event: delta\ndata: A\n\n"
        assert events[1] == "event: delta\ndata: B\n\n"
        assert events[2].startswith("event: sources\ndata: ")
        assert events[3].startswith("event: metadata\ndata: ")
        assert events[4] == "event: done\ndata: [DONE]\n\n"
        store.close()

    def test_sources_frame_carries_citations(self):
        citations = [Citation(title="FastAPI Docs", source_url="https://example.com/x")]
        orch = MagicMock()
        orch.stream_answer.return_value = iter(["tok", _result(answer="tok", citations=citations)])
        store = ConversationStore(":memory:")
        cache = LRUCache(max_size=4)
        sid = store.create_session()

        events = list(build_event_stream(orch, "q", "", sid, store, cache))
        sources_frame = next(e for e in events if e.startswith("event: sources"))
        payload = json.loads(sources_frame.split("data: ", 1)[1])
        assert len(payload["citations"]) == 1
        assert payload["citations"][0]["title"] == "FastAPI Docs"
        store.close()

    def test_metadata_frame_carries_session_confidence_trace(self):
        orch = MagicMock()
        orch.stream_answer.return_value = iter(["tok", _result(answer="tok", confidence=0.9)])
        store = ConversationStore(":memory:")
        cache = LRUCache(max_size=4)
        sid = store.create_session()

        events = list(build_event_stream(orch, "q", "", sid, store, cache))
        meta_frame = next(e for e in events if e.startswith("event: metadata"))
        payload = json.loads(meta_frame.split("data: ", 1)[1])
        assert payload["session_id"] == sid
        assert payload["confidence"] == 0.9
        # trace_id is None when Langfuse is disabled.
        assert payload["trace_id"] is None
        store.close()

    def test_user_and_assistant_messages_persisted(self):
        orch = MagicMock()
        orch.stream_answer.return_value = iter(["ans", _result(answer="ans")])
        store = ConversationStore(":memory:")
        cache = LRUCache(max_size=4)
        sid = store.create_session()

        list(build_event_stream(orch, "my question", "", sid, store, cache))
        msgs = store.get_messages(sid)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "my question"
        assert msgs[1]["content"] == "ans"
        store.close()

    def test_cached_response_stored(self):
        orch = MagicMock()
        orch.stream_answer.return_value = iter(["ans", _result(answer="ans")])
        store = ConversationStore(":memory:")
        cache = LRUCache(max_size=4)
        sid = store.create_session()

        list(build_event_stream(orch, "how do i", "", sid, store, cache))
        assert cache.get("how do i") is not None
        store.close()

    def test_empty_token_stream_still_emits_sources_metadata_done(self):
        orch = MagicMock()
        orch.stream_answer.return_value = iter([_result(answer="")])
        store = ConversationStore(":memory:")
        cache = LRUCache(max_size=4)
        sid = store.create_session()

        events = list(build_event_stream(orch, "q", "", sid, store, cache))
        # No delta frames, just sources + metadata + done
        assert events[0].startswith("event: sources")
        assert events[1].startswith("event: metadata")
        assert events[2] == "event: done\ndata: [DONE]\n\n"
        store.close()

    def test_no_result_yielded_synthesizes_empty(self):
        """If the orchestrator yields only tokens (no PostProcessResult), an
        empty result is synthesized so persistence + cache stay consistent."""
        orch = MagicMock()
        orch.stream_answer.return_value = iter(["a", "b"])
        store = ConversationStore(":memory:")
        cache = LRUCache(max_size=4)
        sid = store.create_session()

        events = list(build_event_stream(orch, "q", "", sid, store, cache))
        # The synthesized result's answer is the concatenated tokens; it is
        # persisted + cached but NOT carried on the wire (deltas already
        # delivered it). Verify persistence instead of the wire payload.
        msgs = store.get_messages(sid)
        assert msgs[-1]["content"] == "ab"
        store.close()


# ── build_replay_stream ─────────────────────────────────────────────────


class TestBuildReplayStream:
    def test_replays_answer_as_single_delta_then_sources_metadata_done(self):
        cached = ChatResponse(
            session_id="sess_old",
            answer="cached answer",
            citations=[],
            confidence=0.5,
        )
        events = list(build_replay_stream(cached, "sess_new"))
        assert events[0] == "event: delta\ndata: cached answer\n\n"
        assert events[1].startswith("event: sources")
        assert events[2].startswith("event: metadata")
        assert events[3] == "event: done\ndata: [DONE]\n\n"

    def test_replay_uses_current_session_id(self):
        cached = ChatResponse(
            session_id="sess_old",
            answer="x",
            citations=[],
            confidence=0.5,
        )
        events = list(build_replay_stream(cached, "sess_new"))
        meta_frame = next(e for e in events if e.startswith("event: metadata"))
        payload = json.loads(meta_frame.split("data: ", 1)[1])
        assert payload["session_id"] == "sess_new"


# ── HTTP endpoint (TestClient) ──────────────────────────────────────────


class TestChatEndpoint:
    def test_streams_sse_delta_events(self, client):
        with client.stream("POST", "/api/v1/chat", json={"message": "hi"}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            body = "".join(resp.iter_text())
        assert "event: delta\ndata: Hello\n\n" in body
        assert "event: delta\ndata:  world\n\n" in body
        assert "event: done\ndata: [DONE]\n\n" in body

    def test_sources_and_metadata_events_present(self, client):
        with client.stream("POST", "/api/v1/chat", json={"message": "hi"}) as resp:
            body = "".join(resp.iter_text())
        assert "event: sources" in body
        assert "event: metadata" in body
        # Both events carry a JSON data line.
        assert any(
            l.startswith("event: sources") for l in body.splitlines()
        )
        assert any(
            l.startswith("event: metadata") for l in body.splitlines()
        )

    def test_creates_session_when_omitted(self, client, mem_store):
        with client.stream("POST", "/api/v1/chat", json={"message": "hi"}) as resp:
            body = "".join(resp.iter_text())
        # The metadata frame carries the new session id.
        lines = body.splitlines()
        meta_idx = next(i for i, l in enumerate(lines) if l.startswith("event: metadata"))
        data_line = lines[meta_idx + 1]
        payload = json.loads(data_line.removeprefix("data: "))
        sid = payload["session_id"]
        assert sid.startswith("sess_")
        assert mem_store.session_exists(sid)

    def test_uses_provided_session_id(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "prior")
        with client.stream(
            "POST", "/api/v1/chat", json={"message": "hi", "session_id": sid}
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        # Two prior messages? No — one prior + new user + assistant = 3
        msgs = mem_store.get_messages(sid)
        assert len(msgs) == 3

    def test_unknown_session_id_returns_404(self, client):
        resp = client.post(
            "/api/v1/chat", json={"message": "hi", "session_id": "sess_unknown"}
        )
        assert resp.status_code == 404

    def test_cache_miss_header(self, client):
        with client.stream("POST", "/api/v1/chat", json={"message": "hi"}) as resp:
            assert resp.headers.get("x-cache") == "MISS"

    def test_cache_hit_replays_and_skips_orchestrator(
        self, client_factory, mock_orchestrator, mem_store, mem_cache
    ):
        # Pre-seed the cache with a response for the query.
        cached = ChatResponse(
            session_id="sess_seed",
            answer="cached!",
            citations=[],
            confidence=0.5,
        )
        mem_cache.put("cached query", cached)
        client = client_factory()
        with client.stream(
            "POST", "/api/v1/chat", json={"message": "cached query"}
        ) as resp:
            assert resp.headers.get("x-cache") == "HIT"
            body = "".join(resp.iter_text())
        assert "event: delta\ndata: cached!\n\n" in body
        # Orchestrator must NOT have been called on a cache hit.
        mock_orchestrator.stream_answer.assert_not_called()

    def test_empty_message_rejected(self, client):
        resp = client.post("/api/v1/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_missing_message_rejected(self, client):
        resp = client.post("/api/v1/chat", json={})
        assert resp.status_code == 422

    def test_message_too_long_rejected(self, client):
        resp = client.post("/api/v1/chat", json={"message": "x" * 4001})
        assert resp.status_code == 422

    def test_history_passed_to_orchestrator(self, client_factory, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "previous question")
        mem_store.add_message(sid, "assistant", "previous answer")
        orch = MagicMock()
        orch.stream_answer.return_value = iter(["tok", _result(answer="tok")])
        client = client_factory(orchestrator=orch)
        with client.stream(
            "POST", "/api/v1/chat", json={"message": "new q", "session_id": sid}
        ):
            pass
        _, kwargs = orch.stream_answer.call_args
        # Second positional arg is history.
        args = orch.stream_answer.call_args.args
        history = args[1] if len(args) > 1 else kwargs.get("history", "")
        assert "previous question" in history
        assert "previous answer" in history
