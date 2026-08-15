"""Tests for ``POST /api/v1/feedback`` (api/routes/feedback.py)."""

from __future__ import annotations


class TestFeedbackEndpoint:
    def test_records_up_feedback(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "q")
        mem_store.add_message(sid, "assistant", "a")
        resp = client.post(
            "/api/v1/feedback",
            json={
                "session_id": sid,
                "message_index": 1,
                "rating": "up",
                "comment": "great",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["recorded"] is True
        assert body["feedback_id"] >= 1
        fb = mem_store.get_feedback(sid)
        assert len(fb) == 1
        assert fb[0]["rating"] == "up"

    def test_records_down_without_comment(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "q")
        resp = client.post(
            "/api/v1/feedback",
            json={"session_id": sid, "message_index": 0, "rating": "down"},
        )
        assert resp.status_code == 200
        assert mem_store.get_feedback(sid)[0]["comment"] is None

    def test_404_for_unknown_session(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={"session_id": "sess_unknown", "message_index": 0, "rating": "up"},
        )
        assert resp.status_code == 404

    def test_invalid_rating_rejected(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "q")
        resp = client.post(
            "/api/v1/feedback",
            json={"session_id": sid, "message_index": 0, "rating": "sideways"},
        )
        assert resp.status_code == 422

    def test_missing_fields_rejected(self, client):
        resp = client.post("/api/v1/feedback", json={"session_id": "x"})
        assert resp.status_code == 422

    def test_comment_too_long_rejected(self, client, mem_store):
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "q")
        resp = client.post(
            "/api/v1/feedback",
            json={
                "session_id": sid,
                "message_index": 0,
                "rating": "up",
                "comment": "x" * 2001,
            },
        )
        assert resp.status_code == 422

    def test_trace_id_accepted(self, client, mem_store):
        """Step 7: feedback with a trace_id is accepted and recorded."""
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "q")
        resp = client.post(
            "/api/v1/feedback",
            json={
                "session_id": sid,
                "message_index": 0,
                "rating": "up",
                "trace_id": "trace-123",
            },
        )
        assert resp.status_code == 200

    def test_trace_id_optional(self, client, mem_store):
        """Feedback without trace_id still works (backward compatible)."""
        sid = mem_store.create_session()
        mem_store.add_message(sid, "user", "q")
        resp = client.post(
            "/api/v1/feedback",
            json={
                "session_id": sid,
                "message_index": 0,
                "rating": "down",
            },
        )
        assert resp.status_code == 200
