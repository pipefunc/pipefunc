// graphvizSvg.js
import jQuery from "jquery";
if (typeof window !== "undefined") {
  window.jQuery = window.$ = jQuery;
}
import 'jquery-mousewheel';
import 'jquery-color';
import 'bootstrap';

class GraphvizSvg {
  static VERSION = '1.0.1';
  static GVPT_2_PX = 32.5;

  static DEFAULTS = {
    url: null,
    svg: null,
    shrink: '0.125pt',
    tooltips: {
      init($graph) {
        const $a = $(this);
        $a.tooltip({
          container: $graph,
          placement: 'left',
          animation: false,
          viewport: null,
        }).on('hide.bs.tooltip', function () {
          // keep them visible even if you accidentally mouse over
          if ($a.attr('data-tooltip-keepvisible')) {
            return false;
          }
        });
      },
      show() {
        const $a = $(this);
        $a.attr('data-tooltip-keepvisible', true);
        $a.tooltip('show');
      },
      hide() {
        const $a = $(this);
        $a.removeAttr('data-tooltip-keepvisible');
        $a.tooltip('hide');
      },
      update() {
        const $this = $(this);
        if ($this.attr('data-tooltip-keepvisible')) {
          $this.tooltip('show');
        }
      },
    },
    zoom: true,
    highlight: {
      selected(col) {
        return col;
      },
      unselected(col, bg) {
        return $.Color(col).transition(bg, 0.9);
      },
    },
    ready: null,
  };

  constructor(element, options) {
    this.type = null;
    this.options = null;
    this.enabled = null;
    this.$element = null;

    this.init('graphviz.svg', element, options);
  }

  init(type, element, options) {
    this.enabled = true;
    this.type = type;
    this.$element = $(element);
    this.options = this.getOptions(options);

    if (this.options.url) {
      $.get(this.options.url, null, (data) => {
        const svg = $('svg', data);
        this.$element.html(document.adoptNode(svg[0]));
        this.setup();
      }, 'xml');
    } else {
      if (this.options.svg) {
        this.$element.html(this.options.svg);
      }
      this.setup();
    }
  }

  getDefaults() {
    return GraphvizSvg.DEFAULTS;
  }

  getOptions(options) {
    options = $.extend({}, this.getDefaults(), this.$element.data(), options);

    if (options.shrink) {
      if (typeof options.shrink !== 'object') {
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

  setup() {
    const options = this.options;

    // Save key elements in the graph for easy access
    const $svg = $(this.$element.children('svg'));
    const $graph = $svg.children('g:first');
    this.$svg = $svg;
    this.$graph = $graph;
    this.$background = $graph.children('polygon:first'); // might not exist
    this.$nodes = $graph.children('.node');
    this.$edges = $graph.children('.edge');
    this._nodesByName = {};
    this._edgesByName = {};

    // Add top-level class and copy background color to element
    this.$element.addClass('graphviz-svg');
    if (this.$background.length) {
      this.$element.css('background', this.$background.attr('fill'));
    }

    // Setup all the nodes and edges
    this.$nodes.each((_, el) => this.setupNodesEdges($(el), true));
    this.$edges.each((_, el) => this.setupNodesEdges($(el), false));

    // Remove the graph title element
    const $title = this.$graph.children('title');
    this.$graph.attr('data-name', $title.text());
    $title.remove();

    if (options.zoom) {
      this.setupZoom();
    }

    // Notify when setup is complete
    if (options.ready) {
      options.ready.call(this);
    }
  }

  setupNodesEdges($el, isNode) {
    const options = this.options;
    const that = this;  // Store reference to GraphvizSvg instance

    // Save the colors of the paths, ellipses, and polygons
    $el.find('polygon, ellipse, path').each((_, elem) => {
      const $this = $(elem);
      // Save original colors
      $this.data('graphviz.svg.color', {
        fill: $this.attr('fill'),
        stroke: $this.attr('stroke'),
      });

      // Shrink it if it's a node
      if (isNode && options.shrink) {
        this.scaleNode($this);
      }
    });

    // Save the node name and check for user comments
    const $title = $el.children('title');
    if ($title.length) {
      // Remove any compass points
      const title = $title.text().replace(/:[snew][ew]?/g, '');
      $el.attr('data-name', title);
      $title.remove();
      if (isNode) {
        this._nodesByName[title] = $el[0];
      } else {
        this._edgesByName[title] = $el[0];
      }
      // Check for user-added comments
      let previousSibling = $el[0].previousSibling;
      while (previousSibling && previousSibling.nodeType !== 8) {
        previousSibling = previousSibling.previousSibling;
      }
      if (previousSibling && previousSibling.nodeType === 8) {
        const htmlDecode = (input) => {
          const e = document.createElement('div');
          e.innerHTML = input;
          return e.childNodes[0].nodeValue;
        };
        const value = htmlDecode(previousSibling.nodeValue.trim());
        if (value !== title) {
          // User-added comment
          $el.attr('data-comment', value);
        }
      }
    }

    // Remove namespace from a[xlink:title]
    $el.find('a').filter(function () { return $(this).attr('xlink:title'); }).each(function () {
      const $a = $(this);
      $a.attr('title', $a.attr('xlink:title'));
      $a.removeAttr('xlink:title');
      if (options.tooltips) {
        options.tooltips.init.call(this, that.$element);
      }
    });
  }

  setupZoom() {
    this.zoom = {
      width: this.$svg.attr('width'),
      height: this.$svg.attr('height'),
      percentage: null,
    };
    this.scaleView(100.0);
    this.$element.on('mousewheel', (evt) => {
      if (evt.shiftKey) {
        let percentage = this.zoom.percentage;
        percentage -= evt.deltaY * evt.deltaFactor;
        if (percentage < 100.0) {
          percentage = 100.0;
        }
        // Get pointer offset in view
        const dx = evt.pageX - this.$svg.offset().left;
        const dy = evt.pageY - this.$svg.offset().top;
        const rx = dx / this.$svg.width();
        const ry = dy / this.$svg.height();

        // Offset within frame ($element)
        const px = evt.pageX - this.$element.offset().left;
        const py = evt.pageY - this.$element.offset().top;

        this.scaleView(percentage);
        // Scroll so pointer is still in the same place
        this.$element.scrollLeft((rx * this.$svg.width()) + 0.5 - px);
        this.$element.scrollTop((ry * this.$svg.height()) + 0.5 - py);
        return false; // Stop propagation
      }
    });
  }

  scaleView(percentage) {
    this.$svg.attr('width', `${percentage}%`);
    this.$svg.attr('height', `${percentage}%`);
    this.zoom.percentage = percentage;
    // Update tooltip position
    const $everything = this.$nodes.add(this.$edges);
    $everything.children('a[title]').each((_, el) => {
      this.options.tooltips.update.call(el);
    });
  }

  scaleNode($node) {
    const dx = this.options.shrink.x;
    const dy = this.options.shrink.y;
    const tagName = $node.prop('tagName');
    if (tagName === 'ellipse') {
      $node.attr('rx', parseFloat($node.attr('rx')) - dx);
      $node.attr('ry', parseFloat($node.attr('ry')) - dy);
    } else if (tagName === 'polygon') {
      // Scale manually
      const bbox = $node[0].getBBox();
      const cx = bbox.x + (bbox.width / 2);
      const cy = bbox.y + (bbox.height / 2);
      const pts = $node.attr('points').trim().split(' ');
      const points = pts.map((pt) => {
        const [xStr, yStr] = pt.split(',');
        const ox = parseFloat(xStr);
        const oy = parseFloat(yStr);
        const newX = (((cx - ox) / (bbox.width / 2) * dx) + ox).toFixed(2);
        const newY = (((cy - oy) / (bbox.height / 2) * dy) + oy).toFixed(2);
        return `${newX},${newY}`;
      }).join(' ');
      $node.attr('points', points);
    }
  }

  convertToPx(val) {
    let retval = val;
    if (typeof val === 'string') {
      let end = val.length;
      let factor = 1.0;
      if (val.endsWith('px')) {
        end -= 2;
      } else if (val.endsWith('pt')) {
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
    const names = this.findEdge($node.attr('data-name'), testEdge, $edges);
    names.forEach((name) => {
      const n = this._nodesByName[name];
      if (!$retval.is(n)) {
        $retval.push(n);
        this.findLinked(n, includeEdges, testEdge, $retval);
      }
    });
  }

  colorElement($el, getColor) {
    const bg = this.$element.css('background');
    $el.find('polygon, ellipse, path').each((_, elem) => {
      const $this = $(elem);
      const color = $this.data('graphviz.svg.color');
      if (color.fill && $this.prop('tagName') !== 'path') {
        $this.attr('fill', getColor(color.fill, bg));
      }
      if (color.stroke) {
        $this.attr('stroke', getColor(color.stroke, bg));
      }
    });
  }

  restoreElement($el) {
    $el.find('polygon, ellipse, path').each((_, elem) => {
      const $this = $(elem);
      const color = $this.data('graphviz.svg.color');
      if (color.fill) {
        $this.attr('fill', color.fill);
      }
      if (color.stroke) {
        $this.attr('stroke', color.stroke);
      }
    });
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
    const nodeName = $(node).attr('data-name');

    // Only find direct incoming connections
    this.findLinked(node, includeEdges, (_, edgeName) => {
        const match = edgeName.match(new RegExp(`^(.*)->${nodeName}`));
        return match ? match[1] : null;
    }, $retval);

    return $retval;
  }
  linkedFrom(node, includeEdges) {
    const $retval = $();
    const nodeName = $(node).attr('data-name');

    // Only find direct outgoing connections
    this.findLinked(node, includeEdges, (_, edgeName) => {
        const match = edgeName.match(new RegExp(`^${nodeName}->(.*)`));
        return match ? match[1] : null;
    }, $retval);

    return $retval;
  }

  linked(node, includeEdges) {
    const $retval = $();
    $retval.push(node);  // Add the original node
    const fromNodes = this.linkedFrom(node, includeEdges);
    const toNodes = this.linkedTo(node, includeEdges);
    return $retval.add(fromNodes).add(toNodes);
  }

  tooltip($elements, show) {
    const options = this.options;
    $elements.each(function () {
      $(this).find('a[title]').each(function () {
        if (show) {
          options.tooltips.show.call(this);
        } else {
          options.tooltips.hide.call(this);
        }
      });
    });
  }

  bringToFront($elements) {
    $elements.detach().appendTo(this.$graph);
  }

  sendToBack($elements) {
    if (this.$background.length) {
      $elements.insertAfter(this.$background);
    } else {
      $elements.detach().prependTo(this.$graph);
    }
  }

  highlight($nodesEdges, tooltips) {
    const options = this.options;
    const $everything = this.$nodes.add(this.$edges);
    if ($nodesEdges && $nodesEdges.length > 0) {
      // Dim all other elements
      $everything.not($nodesEdges).each((_, el) => {
        this.colorElement($(el), options.highlight.unselected);
        this.tooltip($(el));
      });
      $nodesEdges.each((_, el) => {
        this.colorElement($(el), options.highlight.selected);
      });
      if (tooltips) {
        this.tooltip($nodesEdges, true);
      }
    } else {
      // Restore all elements
      $everything.each((_, el) => {
        this.restoreElement($(el));
      });
      this.tooltip($everything);
    }
  }

  destroy() {
    this.$element.off(`.${this.type}`).removeData(this.type);
  }
}

// jQuery plugin definition
function Plugin(option) {
  return this.each(function () {
    const $this = $(this);
    let data = $this.data('graphviz.svg');
    const options = typeof option === 'object' && option;

    if (!data && /destroy/.test(option)) return;
    if (!data) $this.data('graphviz.svg', (data = new GraphvizSvg(this, options)));
    if (typeof option === 'string') data[option]();
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
