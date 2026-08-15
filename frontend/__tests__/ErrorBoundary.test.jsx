import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import ErrorBoundary from "../src/ErrorBoundary.jsx";

// A component that throws on render to trigger the boundary.
function Boom({ shouldThrow }) {
  if (shouldThrow) throw new Error("kaboom");
  return <div data-testid="ok-child">OK</div>;
}

describe("ErrorBoundary", () => {
  // Suppress console.error noise from React's error handling in tests.
  let consoleErrorSpy;
  beforeEach(() => {
    consoleErrorSpy = jest.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    consoleErrorSpy.mockRestore();
  });

  it("renders children when no error", () => {
    render(
      <ErrorBoundary>
        <Boom shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("ok-child")).toBeInTheDocument();
  });

  it("renders fallback UI when a child throws", () => {
    render(
      <ErrorBoundary>
        <Boom shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
    expect(screen.getByText("kaboom")).toBeInTheDocument();
  });

  it('recovers when "Try again" is clicked', () => {
    const { rerender } = render(
      <ErrorBoundary>
        <Boom shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();

    // First swap the child to a non-throwing version (the boundary is
    // still in error state, so the fallback stays visible).
    rerender(
      <ErrorBoundary>
        <Boom shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();

    // Now click "Try again" — the boundary resets and renders the
    // (now non-throwing) child.
    fireEvent.click(screen.getByText("Try again"));
    expect(screen.getByTestId("ok-child")).toBeInTheDocument();
  });
});
