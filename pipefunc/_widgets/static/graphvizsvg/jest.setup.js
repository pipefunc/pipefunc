// jest.setup.js
require("jquery-mousewheel");
require("jquery-color");

const { TextEncoder, TextDecoder } = require("util");

global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

const $ = require("jquery");

// Mock the tooltip method
$.fn.tooltip = jest.fn().mockReturnValue({
  on: jest.fn(),
});
