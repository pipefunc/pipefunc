// plugin.js
// jQuery plugin definition

import $ from "jquery";
import GraphvizSvg from "./graphvizsvg";

function Plugin(option) {
  return this.each(function () {
    const $this = $(this);
    let data = $this.data("graphviz.svg");
    const options = typeof option === "object" && option;

    if (!data && /destroy/.test(option)) return;
    if (!data) $this.data("graphviz.svg", (data = new GraphvizSvg(this, options)));
    if (typeof option === "string") data[option]();
  });
}

const old = $.fn.graphviz;

$.fn.graphviz = Plugin;
$.fn.graphviz.Constructor = GraphvizSvg;

// No conflict
$.fn.graphviz.noConflict = function () {
  $.fn.graphviz = old;
  return this;
};
