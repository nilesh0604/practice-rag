"""Shared pytest fixtures for the API (Step 4) tests.

Provides a FastAPI ``TestClient`` wired with mocked dependencies so the
HTTP tests never touch Ollama, Qdrant, or disk. Each test can override the
mocked collaborators (orchestrator, store, cache) via the returned helpers.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.cache import LRUCache
from api.conversation import ConversationStore
from api.deps import (
    get_cache,
    get_conversation_store,
    get_corpus_dir,
    get_orchestrator,
)
from api.main import create_app
from rag.post_processor import PostProcessResult


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    """A MagicMock orchestrator whose ``stream_answer`` yields two tokens + a result."""
    orch = MagicMock()
    result = PostProcessResult(
        answer="Hello world",
        citations=[],
        confidence=0.9,
    )
    orch.stream_answer.return_value = iter(["Hello", " world", result])
    return orch


@pytest.fixture
def mem_store() -> Iterator[ConversationStore]:
    """An in-memory SQLite conversation store (closed after the test)."""
    store = ConversationStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def mem_cache() -> LRUCache:
    return LRUCache(max_size=8)


@pytest.fixture
def client_factory(mock_orchestrator, mem_store, mem_cache):
    """Build a TestClient with the mocked deps installed.

    Returns a callable so a test can rebuild the client after mutating the
    mocks (e.g. changing ``stream_answer`` return value).
    """

    def _make(orchestrator=mock_orchestrator, store=mem_store, cache=mem_cache) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator
        app.dependency_overrides[get_conversation_store] = lambda: store
        app.dependency_overrides[get_cache] = lambda: cache
        app.dependency_overrides[get_corpus_dir] = lambda: "data/corpus"
        return TestClient(app)

    return _make


@pytest.fixture
def client(client_factory) -> TestClient:
    """A default TestClient with the standard mocked deps."""
    return client_factory()
