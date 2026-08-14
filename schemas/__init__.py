"""Shared Pydantic contracts — the seam between the ingestion pipeline and the serving layer.

Both the offline LangChain ingestion pipeline and the online FastAPI server
import from this package, so it is the single source of truth for the data
shapes that cross the Qdrant collection boundary and the HTTP boundary.
"""

from schemas.chat import ChatRequest, ChatResponse, Citation
from schemas.documents import DocumentChunk, RetrievedDoc
from schemas.feedback import FeedbackRequest

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "DocumentChunk",
    "FeedbackRequest",
    "RetrievedDoc",
]
