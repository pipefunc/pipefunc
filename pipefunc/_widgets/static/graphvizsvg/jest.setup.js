// jest.setup.js
const $ = require('jquery');
require('jquery-mousewheel');
require('jquery-color');

global.$ = $;
global.jQuery = $;

const { TextEncoder, TextDecoder } = require('util');

global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;
