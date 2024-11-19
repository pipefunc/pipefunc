// graphvizsvg.js
import $ from "jquery";
import { setup } from "./setup";
import { ColorUtil, convertToPx } from "./utils";
import { linkedTo, linkedFrom, linked, findEdge, findLinked } from "./graph";
import {
  colorElement,
  restoreElement,
  highlight,
  tooltip,
  bringToFront,
  sendToBack,
} from "./styling";

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
      options.shrink.x = convertToPx(options.shrink.x, GraphvizSvg.GVPT_2_PX);
      options.shrink.y = convertToPx(options.shrink.y, GraphvizSvg.GVPT_2_PX);
    }
    return options;
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
    return linkedTo(node, includeEdges, this);
  }

  linkedFrom(node, includeEdges) {
    return linkedFrom(node, includeEdges, this);
  }

  linked(node, includeEdges) {
    return linked(node, includeEdges, this);
  }

  findEdge(nodeName, testEdge, $retval) {
    return findEdge(nodeName, testEdge, $retval, this);
  }

  findLinked(node, includeEdges, testEdge, $retval) {
    return findLinked(node, includeEdges, testEdge, $retval, this);
  }

  destroy() {
    this.$element.off(`.${this.type}`).removeData(this.type);
  }
}

export default GraphvizSvg;
