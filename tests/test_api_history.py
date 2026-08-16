"""Tests for ``GET /api/v1/history/{session_id}`` (api/routes/history.py)."""

from __future__ import annotations

from schemas.chat import Citation


class TestHistoryEndpoint:
    def test_returns_messages_for_session(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "hello")
        mem_store.add_message(
            sid,
            "assistant",
            "hi there",
            citations=[Citation(title="FastAPI Docs", source_url="https://example.com/x")],
            confidence=0.8,
        )
        resp = client.get(f"/api/v1/history/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "hello"
        assert body["messages"][1]["role"] == "assistant"
        assert body["messages"][1]["confidence"] == 0.8
        assert body["messages"][1]["citations"] is not None

    def test_404_for_unknown_session(self, client):
        resp = client.get("/api/v1/history/sess_unknown")
        assert resp.status_code == 404

    def test_empty_session_after_create(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "q")  # needed so session_exists is True
        resp = client.get(f"/api/v1/history/{sid}")
        assert resp.status_code == 200
        assert resp.json()["messages"] == [{"role": "user", "content": "q", "citations": None, "confidence": None, "created_at": resp.json()["messages"][0]["created_at"]}]


class TestErasureEndpoint:
    def test_erases_messages_and_feedback(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "hello")
        mem_store.add_message(sid, "assistant", "hi there")
        mem_store.add_feedback(sid, 0, "up", "great")

        resp = client.delete(f"/api/v1/history/{sid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sid
        assert body["messages_deleted"] == 2
        assert body["feedback_deleted"] == 1
        assert "erased_at" in body

        assert not mem_store.session_exists(sid)
        assert mem_store.get_messages(sid) == []
        assert mem_store.get_feedback(sid) == []

    def test_404_for_unknown_session(self, client):
        resp = client.delete("/api/v1/history/sess_unknown")
        assert resp.status_code == 404
