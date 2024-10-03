import $ from 'jquery';
import 'jquery-mousewheel';
import 'jquery-color';

// Make jQuery and its plugins globally available in the test environment
global.$ = $;
global.jQuery = $;

const { TextEncoder, TextDecoder } = require('util');

global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;
