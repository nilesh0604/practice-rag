import React from "react";
import ChatWidget from "./ChatWidget.jsx";

/**
 * Root component — renders the ChatWidget.
 *
 * Per the architecture doc's component tree:
 *   App → ChatWidget → MessageList + InputBox + ErrorBoundary
 */
export default function App() {
  return (
    <div className="app">
      <ChatWidget />
    </div>
  );
}
