// setup.js
import $ from 'jquery';

export function setup(context) {
  const options = context.options;

  // Save key elements in the graph for easy access
  const $svg = $(context.$element.children("svg"));
  const $graph = $svg.children("g:first");
  context.$svg = $svg;
  context.$graph = $graph;
  context.$background = $graph.children("polygon:first"); // might not exist
  context.$nodes = $graph.children(".node");
  context.$edges = $graph.children(".edge");
  context._nodesByName = {};
  context._edgesByName = {};

  // Add top-level class and copy background color to element
  context.$element.addClass("graphviz-svg");
  if (context.$background.length) {
    context.$element.css("background", context.$background.attr("fill"));
  }

  // Setup all the nodes and edges
  context.$nodes.each((_, el) => _setupNodesEdges($(el), true, context));
  context.$edges.each((_, el) => _setupNodesEdges($(el), false, context));

  // Remove the graph title element
  const $title = context.$graph.children("title");
  context.$graph.attr("data-name", $title.text());
  $title.remove();

  // Notify when setup is complete
  if (options.ready) {
    options.ready.call(context);
  }
}

function _setupNodesEdges($el, isNode, context) {
  const options = context.options;

  // Save the colors of the paths, ellipses, and polygons
  $el.find("polygon, ellipse, path").each((_, elem) => {
    const $element = $(elem);
    // Save original colors
    $element.data("graphviz.svg.color", {
      fill: $element.attr("fill"),
      stroke: $element.attr("stroke"),
    });

    // Shrink it if it's a node
    if (isNode && options.shrink) {
      _scaleNode($element, context);
    }
  });

  // Save the node name and check for user comments
  const $title = $el.children("title");
  if ($title.length) {
    // Remove any compass points
    const title = $title.text().replace(/:[snew][ew]?/g, "");
    $el.attr("data-name", title);
    $title.remove();
    if (isNode) {
      context._nodesByName[title] = $el[0];
    } else {
      context._edgesByName[title] = $el[0];
    }
    // Check for user-added comments
    let previousSibling = $el[0].previousSibling;
    while (previousSibling && previousSibling.nodeType !== 8) {
      previousSibling = previousSibling.previousSibling;
    }
    if (previousSibling && previousSibling.nodeType === 8) {
      const htmlDecode = (input) => {
        const e = document.createElement("div");
        e.innerHTML = input;
        return e.childNodes[0].nodeValue;
      };
      const value = htmlDecode(previousSibling.nodeValue.trim());
      if (value !== title) {
        // User-added comment
        $el.attr("data-comment", value);
      }
    }
  }

  // Remove namespace from a[xlink:title]
  $el
    .find("a")
    .filter(function () {
      return $(this).attr("xlink:title");
    })
    .each(function () {
      const $a = $(this);
      $a.attr("title", $a.attr("xlink:title"));
      $a.removeAttr("xlink:title");
      if (options.tooltips) {
        options.tooltips.init.call(this, context.$element);
      }
    });
}

function _scaleNode($node, context) {
  const dx = context.options.shrink.x;
  const dy = context.options.shrink.y;
  const tagName = $node.prop("tagName");
  if (tagName === "ellipse") {
    $node.attr("rx", parseFloat($node.attr("rx")) - dx);
    $node.attr("ry", parseFloat($node.attr("ry")) - dy);
  } else if (tagName === "polygon") {
    // Scale manually
    const bbox = $node[0].getBBox();
    const cx = bbox.x + bbox.width / 2;
    const cy = bbox.y + bbox.height / 2;
    const pts = $node.attr("points").trim().split(" ");
    const points = pts
      .map((pt) => {
        const [xStr, yStr] = pt.split(",");
        const ox = parseFloat(xStr);
        const oy = parseFloat(yStr);
        const newX = (((cx - ox) / (bbox.width / 2)) * dx + ox).toFixed(2);
        const newY = (((cy - oy) / (bbox.height / 2)) * dy + oy).toFixed(2);
        return `${newX},${newY}`;
      })
      .join(" ");
    $node.attr("points", points);
  }
}
