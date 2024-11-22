// widget.js
/*
 * WASM Loading Strategy
 * --------------------
 * The widget needs to use the Graphviz WASM binary, which is typically loaded
 * as a separate file. To work in Jupyter notebooks without a separate server,
 * we:
 *
 * 1. Import the WASM binary directly and let our esbuild plugin embed it
 * 2. Override the global fetch function to intercept WASM file requests
 * 3. Return our embedded binary instead of trying to fetch the file
 * 4. Disable web worker mode to avoid complications with WASM loading
 *
 * This ensures that:
 * - The WASM binary is available immediately
 * - No external files need to be loaded
 * - The widget works in any Jupyter environment
 * - We avoid duplicate WASM loading
 */

import * as d3 from "d3";
import "graphvizsvg";
import { graphviz as d3graphviz } from "d3-graphviz";

// Import the WASM binary that's now embedded in our bundle
import wasmBinary from "../node_modules/@hpcc-js/wasm/dist/graphvizlib.wasm";

function getLegendElements(graphvizInstance, $) {
  const legendNodes = [];
  const legendEdges = [];

  graphvizInstance.nodes().each(function () {
    const $node = $(this);
    if ($node.attr("data-name").startsWith("legend_")) {
      legendNodes.push($node[0]);
    }
  });

  graphvizInstance.edges().each(function () {
    const $edge = $(this);
    if ($edge.attr("data-name").startsWith("legend_")) {
      legendEdges.push($edge[0]);
    }
  });
  return { legendNodes: $(legendNodes), legendEdges: $(legendEdges) };
}

function findEdges(text, searchFunction, graphvizInstance, $) {
  const $set = $();
  graphvizInstance.edges().each((index, edge) => {
    if (edge.textContent && searchFunction(text, edge.textContent)) {
      $set.push(edge);
    }
  });
  return $set;
}

function findNodes(text, searchFunction, nodeName, nodeLabel, graphvizInstance, $) {
  const $set = $();
  const nodes = graphvizInstance.nodesByName();

  for (const [nodeID, node] of Object.entries(nodes)) {
    if (
      (nodeName && searchFunction(text, nodeID)) ||
      (nodeLabel && node.textContent && searchFunction(text, node.textContent))
    ) {
      $set.push(node);
    }
  }
  return $set;
}

function search(text, searchObject, graphvizInstance, $) {
  let searchFunction;

  switch (searchObject.type) {
    case "exact":
      searchFunction = (search, str) => str.trim() === search.trim();
      break;
    case "included":
      searchFunction = (search, str) => {
        const searchStr = searchObject.case === "insensitive" ? search.toLowerCase() : search;
        const valStr = searchObject.case === "insensitive" ? str.toLowerCase() : str;
        return valStr.indexOf(searchStr) !== -1;
      };
      break;
    case "regex":
      searchFunction = (search, str) => {
        const regex = new RegExp(search, searchObject.case === "insensitive" ? "i" : undefined);
        return !!str.trim().match(regex);
      };
      break;
  }

  let $edges = $();
  if (searchObject.edgeLabel) {
    $edges = findEdges(text, searchFunction, graphvizInstance, $);
  }

  let $nodes = $();
  if (searchObject.nodeLabel || searchObject.nodeName) {
    $nodes = findNodes(
      text,
      searchFunction,
      searchObject.nodeName,
      searchObject.nodeLabel,
      graphvizInstance,
      $
    );
  }
  return { nodes: $nodes, edges: $edges };
}

function getConnectedNodes(nodeSet, mode, graphvizInstance) {
  let resultSet = $().add(nodeSet);
  const nodes = graphvizInstance.nodesByName();

  nodeSet.each((i, el) => {
    if (mode === "single") {
      resultSet = resultSet.add(el);
    } else if (el.className.baseVal === "edge") {
      const [startNode, endNode] = $(el).data("name").split("->");
      if ((mode === "bidirectional" || mode === "upstream") && startNode) {
        resultSet = resultSet
          .add(nodes[startNode])
          .add(graphvizInstance.linkedTo(nodes[startNode], true));
      }
      if ((mode === "bidirectional" || mode === "downstream") && endNode) {
        resultSet = resultSet
          .add(nodes[endNode])
          .add(graphvizInstance.linkedFrom(nodes[endNode], true));
      }
    } else {
      if (mode === "bidirectional" || mode === "upstream") {
        resultSet = resultSet.add(graphvizInstance.linkedTo(el, true));
      }
      if (mode === "bidirectional" || mode === "downstream") {
        resultSet = resultSet.add(graphvizInstance.linkedFrom(el, true));
      }
    }
  });
  return resultSet;
}

function highlightSelection(graphvizInstance, currentSelection, $) {
  let highlightedNodes = $();
  let highlightedEdges = $();

  currentSelection.forEach((selection) => {
    const nodes = getConnectedNodes(selection.set, selection.direction, graphvizInstance);
    highlightedNodes = highlightedNodes.add(nodes);
  });

  const { legendNodes, legendEdges } = getLegendElements(graphvizInstance, $);
  highlightedNodes = highlightedNodes.add(legendNodes);
  highlightedEdges = highlightedEdges.add(legendEdges);

  graphvizInstance.highlight(highlightedNodes, highlightedEdges);
}

function handleGraphvizSvgEvents(graphvizInstance, $, currentSelection, getSelectedDirection) {
  // Add hover event listeners for edges
  console.log("graphvizInstance", graphvizInstance);
  graphvizInstance.edges().each(function () {
    const $edge = $(this);

    // Store the original stroke width, with a fallback to "1"
    const originalStroke = $edge.attr("stroke-width") || "1";
    $edge.data("original-stroke", originalStroke);

    $edge.on("mouseenter", function () {
      // Highlight edge by making the stroke width thicker
      $(this).find("path").attr("stroke-width", "3");
      // Highlight edge label by making the text visible
      $(this).find("text").attr("fill", "black");
    });

    $edge.on("mouseleave", function () {
      // Revert edge highlight by restoring the original stroke color
      const originalStroke = $(this).data("original-stroke");
      $(this).find("path").attr("stroke-width", originalStroke);
      // Revert edge label highlight by making the text transparent
      $(this).find("text").attr("fill", "transparent");
    });
  });

  // Add event listeners for nodes
  graphvizInstance.nodes().click(function (event) {
    const nodeSet = $().add(this);
    const selectionObject = {
      set: nodeSet,
      direction: getSelectedDirection(),
    };
    if (event.ctrlKey || event.metaKey || event.shiftKey) {
      currentSelection.push(selectionObject);
    } else {
      currentSelection.splice(0, currentSelection.length, selectionObject);
    }

    highlightSelection(graphvizInstance, currentSelection, $);
  });

  // Add a keydown event listener for escape key to reset highlights
  $(document).keydown(function (event) {
    if (event.keyCode === 27) {
      // Escape key
      graphvizInstance.highlight();
    }
  });
}

async function initialize({ model }) {}

async function render({ model, el }) {
  // Override fetch to return our embedded WASM binary
  const originalFetch = window.fetch;
  window.fetch = function (url, options) {
    if (url.toString().includes("graphvizlib.wasm")) {
      console.log("Intercepted WASM fetch");
      return Promise.resolve(new Response(wasmBinary));
    }
    return originalFetch(url, options);
  };

  el.innerHTML = '<div id="graph" style="text-align: center;"></div>';

  // Ensure the DOM is fully rendered before initializing Graphviz
  await new Promise((resolve) => {
    $(resolve);
  });

  const d3graphvizInstance = d3graphviz("#graph", { useWorker: false }); // Important: disable worker to use our embedded binary;

  // Wait for initialization
  await new Promise((resolve) => {
    d3graphvizInstance.on("initEnd", resolve);
  });

  const currentSelection = [];

  let selectedDirection = model.get("selected_direction") || "bidirectional";

  const searchObject = {
    type: model.get("search_type") || "included",
    case: model.get("case_sensitive") ? "sensitive" : "insensitive",
    nodeName: true,
    nodeLabel: true,
    edgeLabel: true,
  };

  let graphvizInstance;

  // Initialize GraphvizSvg first
  $("#graph").graphviz({
    shrink: null,
    zoom: false,
    ready: function () {
      graphvizInstance = this;
      handleGraphvizSvgEvents(graphvizInstance, $, currentSelection, () => selectedDirection);
    },
  });

  const renderGraph = (dotSource) => {
    const transition = d3.transition("graphTransition").ease(d3.easeLinear).delay(0).duration(500);

    d3graphvizInstance
      .engine("dot")
      .fade(true)
      .transition(transition)
      .tweenPaths(true)
      .tweenShapes(true)
      .zoomScaleExtent([0, Infinity])
      .zoom(true)
      .renderDot(dotSource)
      .fit(true)
      .on("end", function () {
        // This is the key line that reconnects d3 and GraphvizSvg
        // Calls the jquery.graphviz.svg setup directly
        $("#graph").data("graphviz.svg").setup(); // Re-setup after rendering
      });
  };

  const resetGraph = () => {
    d3graphvizInstance.resetZoom();
    graphvizInstance.highlight();
    currentSelection.length = 0;
  };

  const updateDirection = (newDirection) => {
    selectedDirection = newDirection;
    resetGraph();
  };

  const searchAndHighlight = (query) => {
    const searchResults = search(query, searchObject, graphvizInstance, $);
    const { legendNodes, legendEdges } = getLegendElements(graphvizInstance, $);
    const nodesToHighlight = searchResults.nodes.add(legendNodes);
    const edgesToHighlight = searchResults.edges.add(legendEdges);
    graphvizInstance.highlight(nodesToHighlight, edgesToHighlight);
  };

  model.on("change:search_type", () => {
    searchObject.type = model.get("search_type");
  });

  model.on("change:case_sensitive", () => {
    searchObject.case = model.get("case_sensitive") ? "sensitive" : "insensitive";
  });

  model.on("change:dot_source", () => {
    renderGraph(model.get("dot_source"));
  });

  model.on("change:selected_direction", () => {
    updateDirection(model.get("selected_direction"));
  });

  model.on("msg:custom", (msg) => {
    if (msg.action === "reset_zoom") {
      resetGraph();
    } else if (msg.action === "search") {
      searchAndHighlight(msg.query);
    }
  });

  renderGraph(model.get("dot_source"));
}

export default { initialize, render };
