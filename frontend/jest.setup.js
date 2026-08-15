import "@testing-library/jest-dom";

// jsdom does not implement Element.scrollIntoView — mock it so the
// MessageList auto-scroll useEffect doesn't throw.
window.HTMLElement.prototype.scrollIntoView = jest.fn();

// jsdom (used by jest-environment-jsdom) does not provide TextEncoder,
// TextDecoder, or ReadableStream globally. The api.test.js streamChat
// mock needs them to build ReadableStream chunks.
if (typeof global.TextEncoder === "undefined") {
  global.TextEncoder = require("util").TextEncoder;
}
if (typeof global.TextDecoder === "undefined") {
  global.TextDecoder = require("util").TextDecoder;
}
if (typeof global.ReadableStream === "undefined") {
  const { ReadableStream } = require("node:stream/web");
  global.ReadableStream = ReadableStream;
}
