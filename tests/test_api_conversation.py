"""Tests for the SQLite conversation store (api/conversation.py).

Uses an in-memory database (``:memory:``) reused across calls on the same
instance so the data persists within each test.
"""

from __future__ import annotations

import pytest

from api.conversation import ConversationStore
from schemas.chat import Citation


@pytest.fixture
def store():
    s = ConversationStore(":memory:")
    yield s
    s.close()


def _citations():
    return [Citation(title="FastAPI Docs", source_url="https://example.com/x")]


class TestSession:
    def test_create_session_returns_prefixed_id(self, store):
        sid = store.create_session()
        assert sid.startswith("sess_")

    def test_new_session_does_not_exist(self, store):
        assert store.session_exists("sess_unknown") is False

    def test_session_exists_after_message(self, store):
        sid = store.create_session()
        store.add_message(sid, "user", "hi")
        assert store.session_exists(sid) is True


class TestAddMessage:
    def test_user_message_stored(self, store):
        sid = store.create_session()
        rowid = store.add_message(sid, "user", "hello")
        assert rowid >= 1
        msgs = store.get_messages(sid)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        assert msgs[0]["citations"] is None
        assert msgs[0]["confidence"] is None

    def test_assistant_message_with_citations_and_confidence(self, store):
        sid = store.create_session()
        store.add_message(
            sid, "assistant", "answer", citations=_citations(), confidence=0.8
        )
        msgs = store.get_messages(sid)
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["citations"] is not None
        assert msgs[0]["citations"][0]["title"] == "FastAPI Docs"
        assert msgs[0]["confidence"] == 0.8

    def test_chronological_order(self, store):
        sid = store.create_session()
        store.add_message(sid, "user", "first")
        store.add_message(sid, "assistant", "second")
        store.add_message(sid, "user", "third")
        msgs = store.get_messages(sid)
        assert [m["content"] for m in msgs] == ["first", "second", "third"]

    def test_invalid_role_raises(self, store):
        sid = store.create_session()
        with pytest.raises(ValueError):
            store.add_message(sid, "system", "x")

    def test_messages_isolated_per_session(self, store):
        s1 = store.create_session()
        s2 = store.create_session()
        store.add_message(s1, "user", "one")
        store.add_message(s2, "user", "two")
        assert len(store.get_messages(s1)) == 1
        assert len(store.get_messages(s2)) == 1
        assert store.get_messages(s1)[0]["content"] == "one"


class TestHistoryWindow:
    def test_empty_history_returns_empty_string(self, store):
        sid = store.create_session()
        assert store.format_history(sid) == ""

    def test_format_history_single_turn(self, store):
        sid = store.create_session()
        store.add_message(sid, "user", "What is FastAPI?")
        store.add_message(sid, "assistant", "A web framework.")
        history = store.format_history(sid)
        assert "User: What is FastAPI?" in history
        assert "Assistant: A web framework." in history

    def test_window_truncates_to_last_n_turns(self, store):
        sid = store.create_session()
        for i in range(15):
            store.add_message(sid, "user", f"q{i}")
            store.add_message(sid, "assistant", f"a{i}")
        # turns=2 → last 4 messages
        msgs = store.get_history(sid, turns=2)
        assert len(msgs) == 4
        assert msgs[0]["content"] == "q13"

    def test_window_starts_on_user_turn(self, store):
        sid = store.create_session()
        for i in range(6):
            store.add_message(sid, "user", f"q{i}")
            store.add_message(sid, "assistant", f"a{i}")
        # turns=1 → last 2 messages; slice of last 2 starts on "q5" (user)
        msgs = store.get_history(sid, turns=1)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "q5"

    def test_window_drops_leading_assistant(self, store):
        sid = store.create_session()
        # 3 turns = 6 messages; window of 2 turns = last 4 = [a1, q2, a2, ...]
        # actually last 4 = [q2,a2,q3?...] — let's build 3 turns and request 1 turn
        for i in range(3):
            store.add_message(sid, "user", f"q{i}")
            store.add_message(sid, "assistant", f"a{i}")
        # turns=1 → last 2 messages = [q2, a2], starts on user
        msgs = store.get_history(sid, turns=1)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"

    def test_trailing_unpaired_user_included(self, store):
        sid = store.create_session()
        store.add_message(sid, "user", "q0")
        store.add_message(sid, "assistant", "a0")
        store.add_message(sid, "user", "q1")  # unpaired
        msgs = store.get_history(sid, turns=1)
        # last 2 = [a0, q1] → leading assistant dropped → [q1]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "q1"

    def test_turns_zero_returns_empty(self, store):
        sid = store.create_session()
        store.add_message(sid, "user", "q")
        assert store.get_history(sid, turns=0) == []


class TestFeedback:
    def test_add_and_get_feedback(self, store):
        sid = store.create_session()
        store.add_message(sid, "user", "q")
        store.add_message(sid, "assistant", "a")
        fid = store.add_feedback(sid, message_index=1, rating="up", comment="great")
        assert fid >= 1
        fb = store.get_feedback(sid)
        assert len(fb) == 1
        assert fb[0]["rating"] == "up"
        assert fb[0]["comment"] == "great"
        assert fb[0]["message_index"] == 1

    def test_invalid_rating_raises(self, store):
        sid = store.create_session()
        with pytest.raises(ValueError):
            store.add_feedback(sid, 0, "sideways")

    def test_feedback_isolated_per_session(self, store):
        s1 = store.create_session()
        s2 = store.create_session()
        store.add_message(s1, "user", "q")
        store.add_feedback(s1, 0, "up")
        store.add_message(s2, "user", "q")
        store.add_feedback(s2, 0, "down")
        assert len(store.get_feedback(s1)) == 1
        assert len(store.get_feedback(s2)) == 1
        assert store.get_feedback(s1)[0]["rating"] == "up"
