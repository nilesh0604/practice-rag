import React from "react";

/**
 * CitationChip — renders `[Source: title]` as a clickable external link with
 * an optional snippet, relevance score, and last-modified date.
 *
 * Matches the SSE `event: sources` frame's `citations` array shape:
 *   { title, source_url, snippet?, relevanceScore?, lastModified? }
 * Older citations without the optional fields still render correctly.
 */
function formatDate(value) {
  if (!value) return null;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export default function CitationChip({ citation }) {
  const href = citation.source_url || citation.sourceUrl || "#";
  const snippet = citation.snippet;
  const score = citation.relevanceScore ?? citation.relevance_score;
  const lastMod = formatDate(citation.lastModified ?? citation.last_modified);

  return (
    <a
      className="citation-chip"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      data-testid="citation-chip"
    >
      <span className="citation-chip__title">[Source: {citation.title}]</span>
      {snippet && <span className="citation-chip__snippet">{snippet}</span>}
      <span className="citation-chip__meta">
        {typeof score === "number" && (
          <span className="citation-chip__score" data-testid="citation-score">
            {(score * 100).toFixed(0)}% match
          </span>
        )}
        {lastMod && (
          <span className="citation-chip__date" data-testid="citation-date">
            Updated {lastMod}
          </span>
        )}
      </span>
    </a>
  );
}
