// jest.setup.js
const { TextEncoder, TextDecoder } = require("util");

global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

const $ = require("jquery");

// Mock the tooltip method
$.fn.tooltip = jest.fn().mockReturnValue({
  on: jest.fn(),
  tooltip: jest.fn(),
});
