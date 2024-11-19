// graphvizSvg.js
import jQuery from "jquery";
if (typeof window !== "undefined") {
  window.jQuery = window.$ = jQuery;
}
import "bootstrap";
import { setup } from "./setup";
import { ColorUtil } from "./color";
import { colorElement, restoreElement, highlight, tooltip, bringToFront, sendToBack } from "./styling";

class GraphvizSvg {
  static VERSION = "1.0.1";
  static GVPT_2_PX = 32.5;

  static DEFAULTS = {
    url: null,
    svg: null,
    shrink: "0.125pt",
    tooltips: {
      init($graph) {
        const $a = $(this);
        $a.tooltip({
          container: $graph,
          placement: "left",
          animation: false,
          viewport: null,
        }).on("hide.bs.tooltip", function () {
          // keep them visible even if you accidentally mouse over
          if ($a.attr("data-tooltip-keepvisible")) {
            return false;
          }
        });
      },
      show() {
        const $a = $(this);
        $a.attr("data-tooltip-keepvisible", true);
        $a.tooltip("show");
      },
      hide() {
        const $a = $(this);
        $a.removeAttr("data-tooltip-keepvisible");
        $a.tooltip("hide");
      },
      update() {
        const $this = $(this);
        if ($this.attr("data-tooltip-keepvisible")) {
          $this.tooltip("show");
        }
      },
    },
    highlight: {
      selected(col) {
        return col;
      },
      unselected(col, bg) {
        return ColorUtil.transition(col, bg, 0.9);
      },
    },
    ready: null,
  };

  constructor(element, options) {
    this.type = null;
    this.options = null;
    this.enabled = null;
    this.$element = null;

    this.init("graphviz.svg", element, options);
  }

  init(type, element, options) {
    this.enabled = true;
    this.type = type;
    this.$element = $(element);
    this.options = this.getOptions(options);

    if (this.options.url) {
      $.get(
        this.options.url,
        null,
        (data) => {
          const svg = $("svg", data);
          this.$element.html(document.adoptNode(svg[0]));
          setup(this);
        },
        "xml"
      );
    } else {
      if (this.options.svg) {
        this.$element.html(this.options.svg);
      }
      setup(this);
    }
  }

  setup() {
    return setup(this);
  }

  getDefaults() {
    return GraphvizSvg.DEFAULTS;
  }

  getOptions(options) {
    options = $.extend({}, this.getDefaults(), this.$element.data(), options);

    if (options.shrink) {
      if (typeof options.shrink !== "object") {
        options.shrink = {
          x: options.shrink,
          y: options.shrink,
        };
      }
      options.shrink.x = this.convertToPx(options.shrink.x);
      options.shrink.y = this.convertToPx(options.shrink.y);
    }
    return options;
  }

  convertToPx(val) {
    let retval = val;
    if (typeof val === "string") {
      let end = val.length;
      let factor = 1.0;
      if (val.endsWith("px")) {
        end -= 2;
      } else if (val.endsWith("pt")) {
        end -= 2;
        factor = GraphvizSvg.GVPT_2_PX;
      }
      retval = parseFloat(val.substring(0, end)) * factor;
    }
    return retval;
  }

  findEdge(nodeName, testEdge, $retval) {
    const retval = [];
    for (const name in this._edgesByName) {
      const match = testEdge(nodeName, name);
      if (match) {
        if ($retval) {
          $retval.push(this._edgesByName[name]);
        }
        retval.push(match);
      }
    }
    return retval;
  }

  findLinked(node, includeEdges, testEdge, $retval) {
    const $node = $(node);
    let $edges = null;
    if (includeEdges) {
      $edges = $retval;
    }
    const names = this.findEdge($node.attr("data-name"), testEdge, $edges);
    names.forEach((name) => {
      const n = this._nodesByName[name];
      if (!$retval.is(n)) {
        $retval.push(n);
        this.findLinked(n, includeEdges, testEdge, $retval);
      }
    });
  }

  highlight($nodesEdges, tooltips) {
    return highlight($nodesEdges, tooltips, this);
  }

  colorElement($el, getColor) {
    return colorElement($el, getColor, this);
  }

  restoreElement($el) {
    return restoreElement($el, this);
  }

  tooltip($elements, show) {
    return tooltip($elements, show, this);
  }

  bringToFront($el) {
    return bringToFront($el, this);
  }

  sendToBack($el) {
    return sendToBack($el, this);
  }


  // Public methods
  nodes() {
    return this.$nodes;
  }

  edges() {
    return this.$edges;
  }

  nodesByName() {
    return this._nodesByName;
  }

  edgesByName() {
    return this._edgesByName;
  }

  linkedTo(node, includeEdges) {
    const $retval = $();
    this.findLinked(
      node,
      includeEdges,
      (nodeName, edgeName) => {
        let other = null;
        const connection = edgeName.split("->");
        if (
          connection.length > 1 &&
          (connection[1] === nodeName || connection[1].startsWith(nodeName + ":"))
        ) {
          return connection[0].split(":")[0];
        }
        return other;
      },
      $retval
    );
    return $retval;
  }

  linkedFrom(node, includeEdges) {
    const $retval = $();
    this.findLinked(
      node,
      includeEdges,
      (nodeName, edgeName) => {
        let other = null;
        const connection = edgeName.split("->");
        if (
          connection.length > 1 &&
          (connection[0] === nodeName || connection[0].startsWith(nodeName + ":"))
        ) {
          return connection[1].split(":")[0];
        }
        return other;
      },
      $retval
    );
    return $retval;
  }

  linked(node, includeEdges) {
    const $retval = $();
    $retval.push(node); // Add the original node
    const fromNodes = this.linkedFrom(node, includeEdges);
    const toNodes = this.linkedTo(node, includeEdges);
    return $retval.add(fromNodes).add(toNodes);
  }

  destroy() {
    this.$element.off(`.${this.type}`).removeData(this.type);
  }
}

// jQuery plugin definition
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

export default GraphvizSvg;
