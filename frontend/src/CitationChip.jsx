import React from 'react';

/**
 * CitationChip — renders `[Source: title]` as a clickable external link.
 *
 * Matches the architecture doc's `CitationChip` component and the SSE
 * `event: result` frame's `citations` array shape:
 *   { title: string, source_url: string }
 */
export default function CitationChip({ citation }) {
  const href = citation.source_url || citation.sourceUrl || '#';
  return (
    <a
      className="citation-chip"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      data-testid="citation-chip"
    >
      [Source: {citation.title}]
    </a>
  );
}
