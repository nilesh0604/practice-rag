import React from "react";
import { render, screen } from "@testing-library/react";
import App from "../src/App.jsx";

describe("App", () => {
  it("renders the ChatWidget with the title", () => {
    render(<App />);
    expect(screen.getByText("RAG Knowledge Assistant")).toBeInTheDocument();
  });

  it("renders the chat input", () => {
    render(<App />);
    expect(screen.getByTestId("chat-input")).toBeInTheDocument();
  });
});
