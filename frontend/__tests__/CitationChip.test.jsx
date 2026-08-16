import React from "react";
import { render, screen } from "@testing-library/react";
import CitationChip from "../src/CitationChip.jsx";

describe("CitationChip", () => {
  it("renders [Source: title] as a link with the source_url", () => {
    render(
      <CitationChip
        citation={{
          title: "FastAPI Docs",
          source_url: "https://fastapi.tiangolo.com",
        }}
      />,
    );
    const link = screen.getByTestId("citation-chip");
    expect(link).toHaveTextContent("[Source: FastAPI Docs]");
    expect(link).toHaveAttribute("href", "https://fastapi.tiangolo.com");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("falls back to # when no source_url", () => {
    render(<CitationChip citation={{ title: "No URL" }} />);
    const link = screen.getByTestId("citation-chip");
    expect(link).toHaveAttribute("href", "#");
  });

  it("supports camelCase sourceUrl key", () => {
    render(
      <CitationChip
        citation={{ title: "Doc", sourceUrl: "https://example.com" }}
      />,
    );
    expect(screen.getByTestId("citation-chip")).toHaveAttribute(
      "href",
      "https://example.com",
    );
  });

  it("renders snippet when provided", () => {
    render(
      <CitationChip
        citation={{
          title: "FastAPI Docs",
          source_url: "https://fastapi.tiangolo.com",
          snippet: "Use Depends for dependency injection.",
        }}
      />,
    );
    expect(
      screen.getByText("Use Depends for dependency injection."),
    ).toBeInTheDocument();
  });

  it("renders relevance score as percentage", () => {
    render(
      <CitationChip
        citation={{
          title: "Doc",
          source_url: "https://example.com",
          relevanceScore: 0.834,
        }}
      />,
    );
    expect(screen.getByTestId("citation-score")).toHaveTextContent("83% match");
  });

  it("renders last-modified date when provided", () => {
    render(
      <CitationChip
        citation={{
          title: "Doc",
          source_url: "https://example.com",
          lastModified: "2026-01-15T00:00:00Z",
        }}
      />,
    );
    expect(screen.getByTestId("citation-date")).toHaveTextContent(/Updated/);
  });

  it("omits score and date when not provided", () => {
    render(
      <CitationChip
        citation={{ title: "Doc", source_url: "https://example.com" }}
      />,
    );
    expect(screen.queryByTestId("citation-score")).not.toBeInTheDocument();
    expect(screen.queryByTestId("citation-date")).not.toBeInTheDocument();
  });
});
