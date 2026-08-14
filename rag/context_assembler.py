"""Context assembler — formats retrieved chunks into the system-prompt CONTEXT block.

Takes the list of ``RetrievedDoc`` objects returned by the hybrid retriever and
produces a numbered, delimited text block that the generator injects into the
system prompt's ``{context}`` placeholder (see the architecture doc's system
prompt template).

Each chunk is rendered as::

    [1] Title — section (optional)
    <chunk content>

    [2] Title — section (optional)
    <chunk content>

The numbering lets the generator reference sources by position and makes the
context block easy to inspect during debugging.
"""

from __future__ import annotations

from schemas.documents import RetrievedDoc

DEFAULT_MAX_CHUNKS: int = 5
"""Maximum chunks to include in the context. Matches the retriever's top-K=5."""

DEFAULT_MAX_CHARS: int = 12000
"""Soft character cap on the assembled context to avoid overflowing the 8K
token context window (≈ 4 chars/token → ~3000 tokens of context headroom
after the system prompt, history, and the 512-token generation budget)."""

EMPTY_CONTEXT: str = "(No relevant context found.)"
"""Placeholder shown to the generator when no chunks were retrieved, so the
LLM can trigger its "I don't have enough information" rule."""


class ContextAssembler:
    """Formats retrieved docs into the CONTEXT block for the system prompt."""

    def __init__(
        self,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def assemble(self, docs: list[RetrievedDoc]) -> str:
        """Build the CONTEXT string from retrieved docs.

        Truncates to ``max_chunks`` and stops adding chunks once the soft
        ``max_chars`` cap is reached (the partially-added chunk is kept so the
        generator always sees complete chunks rather than cut-off text).
        """
        if not docs:
            return EMPTY_CONTEXT

        blocks: list[str] = []
        total = 0
        for i, doc in enumerate(docs[: self.max_chunks], start=1):
            header = self._format_header(doc, i)
            body = doc.content.strip()
            block = f"{header}\n{body}"
            if total + len(block) > self.max_chars and blocks:
                break
            blocks.append(block)
            total += len(block)

        return "\n\n".join(blocks)

    @staticmethod
    def _format_header(doc: RetrievedDoc, index: int) -> str:
        """Render the per-chunk header line: ``[n] Title`` or ``[n] Title — section``."""
        if doc.section:
            return f"[{index}] {doc.title} — {doc.section}"
        return f"[{index}] {doc.title}"
