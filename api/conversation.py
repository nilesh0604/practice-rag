"""Conversation manager — SQLite session store with a last-10-turns history window.

Persists per-session chat messages (user + assistant) and feedback to a
SQLite file. The RAG orchestrator receives a *formatted* history string
built from the most recent ``HISTORY_WINDOW`` turns so the generator's
system prompt can carry conversational context without unbounded growth.

Design notes (from the architecture doc):

- **State:** SQLite file (the doc's "State → SQLite file" scaling path).
  For a single user this is more than enough; the swap to Postgres/Redis
  is documented as the cloud equivalent.
- **History window:** the last 10 turns (a turn = one user + one assistant
  message) are formatted as ``User: ...\\nAssistant: ...``. Older turns are
  dropped from the prompt context. Full LLM-based history *summarization*
  is listed in the build order but is deferred — for a local 3B model the
  extra round-trip adds latency and risk; the fixed window is the
  pragmatic default and is documented as a simplification.
- **Thread safety:** a single connection with ``check_same_thread=False``
  guarded by a ``threading.Lock`` (FastAPI runs sync endpoints in a
  threadpool). Tests construct the store with ``":memory:"`` and reuse the
  instance so the in-memory database persists across calls.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from schemas.chat import Citation

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH: str = os.getenv("CHAT_DB_PATH", "data/chat.db")
"""Default SQLite path. Overridable via the ``CHAT_DB_PATH`` env var."""

HISTORY_WINDOW: int = 10
"""Number of most-recent *turns* (user+assistant pairs) included in the
formatted history string passed to the generator. Matches the architecture
doc's "last-10-turns window"."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,            -- 'user' | 'assistant'
    content     TEXT    NOT NULL,
    citations   TEXT,                        -- JSON list, nullable
    confidence  REAL,                         -- nullable (assistant only)
    created_at  TEXT    NOT NULL             -- ISO-8601 UTC
);
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    message_index INTEGER NOT NULL,
    rating        TEXT    NOT NULL,          -- 'up' | 'down'
    comment       TEXT,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
"""


class ConversationStore:
    """SQLite-backed conversation history + feedback store."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        # check_same_thread=False because FastAPI calls sync endpoints from
        # a threadpool; access is serialized by _lock.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── sessions ───────────────────────────────────────────────────────

    def create_session(self) -> str:
        """Create a new session and return its id.

        The id is ``sess_<UTC timestamp>_<8-char uuid hex>`` — sortable,
        human-readable, and collision-resistant for a single-user project.
        """
        import uuid

        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        session_id = f"sess_{ts}_{uuid.uuid4().hex[:8]}"
        logger.debug("Created session %s", session_id)
        return session_id

    def session_exists(self, session_id: str) -> bool:
        """True if at least one message has been recorded for the session."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                (session_id,),
            )
            return cur.fetchone() is not None

    # ── messages ───────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[Citation] | None = None,
        confidence: float | None = None,
    ) -> int:
        """Append a message to a session. Returns the new row id.

        ``role`` must be ``"user"`` or ``"assistant"``. Citations are
        serialized to JSON; confidence is stored only for assistant turns.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        citations_json = (
            json.dumps([c.model_dump(mode="json") for c in citations])
            if citations
            else None
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO messages (session_id, role, content, citations, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, citations_json, confidence, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return all messages for a session in chronological order.

        Each dict has: ``id``, ``role``, ``content``, ``citations`` (list of
        Citation dicts or None), ``confidence`` (float or None),
        ``created_at``.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content, citations, confidence, created_at "
                "FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [self._row_to_message(r) for r in rows]

    def get_history(self, session_id: str, turns: int = HISTORY_WINDOW) -> list[dict[str, Any]]:
        """Return the last ``turns`` turns (user+assistant pairs) for a session.

        A "turn" is one user message followed by one assistant message. The
        window is expressed in turns, so ``turns=10`` returns up to 20
        messages. If the session has an unpaired trailing user message it is
        included.
        """
        if turns < 1:
            return []
        all_msgs = self.get_messages(session_id)
        # Take the last (turns * 2) messages, then trim to a turn boundary.
        window = all_msgs[-(turns * 2) :]
        # If we sliced into the middle of a turn (started on an assistant
        # message), drop the leading assistant message so the window starts
        # on a user turn.
        if window and window[0]["role"] == "assistant":
            window = window[1:]
        return window

    def format_history(self, session_id: str, turns: int = HISTORY_WINDOW) -> str:
        """Format the recent conversation as a string for the system prompt.

        Returns ``"User: ...\\nAssistant: ...\\n..."`` or the empty string
        when the session has no prior messages (a fresh session).
        """
        msgs = self.get_history(session_id, turns=turns)
        if not msgs:
            return ""
        lines: list[str] = []
        for m in msgs:
            label = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{label}: {m['content']}")
        return "\n".join(lines)

    # ── feedback ───────────────────────────────────────────────────────

    def add_feedback(
        self,
        session_id: str,
        message_index: int,
        rating: str,
        comment: str | None = None,
    ) -> int:
        """Record a thumbs up/down rating for a message in a session.

        ``message_index`` is the 0-based position of the rated message within
        the session's chronological message list (matching
        ``FeedbackRequest.message_index``). Returns the new feedback row id.
        """
        if rating not in ("up", "down"):
            raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO feedback (session_id, message_index, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, message_index, rating, comment, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_feedback(self, session_id: str) -> list[dict[str, Any]]:
        """Return all feedback entries for a session in chronological order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, message_index, rating, comment, created_at "
                "FROM feedback WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── lifecycle ──────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
        citations_raw = row["citations"]
        citations = json.loads(citations_raw) if citations_raw else None
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "citations": citations,
            "confidence": row["confidence"],
            "created_at": row["created_at"],
        }
