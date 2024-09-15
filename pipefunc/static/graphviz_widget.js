function loadScript(url) {
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
        document.head.append(script);
    });
}

async function render({ model, el }) {
    // Load scripts in order
    await loadScript("https://unpkg.com/jquery@3.6.1/dist/jquery.min.js");
    await loadScript("https://unpkg.com/jquery-mousewheel@3.1.13/jquery.mousewheel.js");
    await loadScript("https://unpkg.com/jquery-color@2.2.0/dist/jquery.color.js");
    await loadScript("https://unpkg.com/d3@7.6.1/dist/d3.min.js");
    await loadScript("https://cdn.jsdelivr.net/gh/mountainstorm/jquery.graphviz.svg@master/js/jquery.graphviz.svg.js");
    await loadScript("https://unpkg.com/@hpcc-js/wasm@1.18.0/dist/index.min.js");
    await loadScript("https://unpkg.com/d3-graphviz@4.4.0/build/d3-graphviz.min.js");

    const $ = window.jQuery;
    const d3 = window.d3;

    // Prepare the graph container
    el.innerHTML = '<div id="graph" style="text-align: center;"></div>';
    const graphEl = $(el).find('#graph');

    // Initialize a d3-graphviz renderer instance
    var graphviz = d3.select("#graph").graphviz();

    // Configuration for transitions in rendering the graph
    var d3Config = {
        transitionDelay: 0,
        transitionDuration: 500
    };

    // Variable for storing the selected graph rendering engine
    var selectedEngine = "dot";

    // Object for saving the current GraphVizSVG
    var graphVizObject;

    // Variable for storing the selected direction for highlighting
    var selectedDirection = model.get("selected_direction") || "bidirectional";

    // Array holding the current selections
    var currentSelection = [];

    // Search-related variables
    const searchObject = {
        type: "included",
        case: "insensitive",
        nodeName: true,
        nodeLabel: true,
        edgeLabel: true,
    };

    // Main search function to find nodes and edges
    function search(text, mode = "highlight", options = {}) {
        const opt = {
            type: options.type || "included",
            case: options.case || "insensitive",
            nodeName: options.nodeName !== false,
            nodeLabel: options.nodeLabel !== false,
            edgeLabel: options.edgeLabel !== false,
        };

        let searchFunction;
        if (opt.type === "included") {
            searchFunction = (search, str) => {
                const searchStr = opt.case === "insensitive" ? search.toLowerCase() : search;
                const valStr = opt.case === "insensitive" ? str.toLowerCase() : str;
                return valStr.indexOf(searchStr) !== -1;
            };
        } else if (opt.type === "regex") {
            searchFunction = (search, str) => {
                const regex = new RegExp(search, opt.case === "insensitive" ? "i" : undefined);
                return regex.test(str);
            };
        }

        const $nodes = opt.nodeLabel || opt.nodeName ? findNodes(text, searchFunction, opt.nodeName, opt.nodeLabel) : $();
        const $edges = opt.edgeLabel ? findEdges(text, searchFunction) : $();

        return { nodes: $nodes, edges: $edges };
    }

    // Function to find edges matching the search criteria
    function findEdges(text, searchFunction) {
        const $set = $();
        graphVizObject.edges().each((index, edge) => {
            if (edge.textContent && searchFunction(text, edge.textContent)) {
                $set.push(edge);
            }
        });
        return $set;
    }

    // Function to find nodes matching the search criteria
    function findNodes(text, searchFunction, nodeName = true, nodeLabel = true) {
        const $set = $();
        const nodes = graphVizObject.nodesByName();

        for (const [nodeID, node] of Object.entries(nodes)) {
            if ((nodeName && searchFunction(text, nodeID)) ||
                (nodeLabel && node.textContent && searchFunction(text, node.textContent))) {
                $set.push(node);
            }
        }
        return $set;
    }

    // Function to highlight selected nodes and their connected nodes
    function highlightSelection() {
        let highlightedNodes = $();
        currentSelection.forEach(selection => {
            const nodes = getConnectedNodes(selection.set, selection.direction);
            highlightedNodes = highlightedNodes.add(nodes);
        });
        graphVizObject.highlight(highlightedNodes, true);
    }

    // Function to retrieve nodes connected in the specified direction
    function getConnectedNodes(nodeSet, mode = "bidirectional") {
        let resultSet = $().add(nodeSet);
        const nodes = graphVizObject.nodesByName();

        nodeSet.each((i, el) => {
            if (el.className.baseVal === "edge") {
                const [startNode, endNode] = $(el).data("name").split("->");
                if ((mode === "bidirectional" || mode === "upstream") && startNode) {
                    resultSet = resultSet.add(nodes[startNode]).add(graphVizObject.linkedTo(nodes[startNode], true));
                }
                if ((mode === "bidirectional" || mode === "downstream") && endNode) {
                    resultSet = resultSet.add(nodes[endNode]).add(graphVizObject.linkedFrom(nodes[endNode], true));
                }
            } else {
                if (mode === "bidirectional" || mode === "upstream") {
                    resultSet = resultSet.add(graphVizObject.linkedTo(el, true));
                }
                if (mode === "bidirectional" || mode === "downstream") {
                    resultSet = resultSet.add(graphVizObject.linkedFrom(el, true));
                }
            }
        });
        return resultSet;
    }

    // Function to reset the graph zoom and selection highlights
    function resetGraph() {
        graphviz.resetZoom();
        graphVizObject.highlight(); // Reset node selection on reset
        currentSelection = [];
    }

    // Function to update the selected direction for highlighting
    function updateDirection(newDirection) {
        selectedDirection = newDirection;
        resetGraph();
    }

    // Main function to render the graph from DOT source
    function render(dotSource) {
        var transition = d3.transition("graphTransition")
            .ease(d3.easeLinear)
            .delay(d3Config.transitionDelay)
            .duration(d3Config.transitionDuration);

        graphviz
            .engine(selectedEngine)
            .fade(true)
            .transition(transition)
            .tweenPaths(true)
            .tweenShapes(true)
            .zoomScaleExtent([0, Infinity])
            .zoom(true)
            .renderDot(dotSource)
            .on("end", function () {
                // Calls the jquery.graphviz.svg setup directly
                $('#graph').data('graphviz.svg').setup();  // Re-setup after rendering
            });
    }

    // Document ready function
    $(document).ready(function () {
        // Initialize the GraphVizSVG object from jquery.graphviz.svg.js
        $("#graph").graphviz({
            shrink: null,
            zoom: false,
            ready: function () {
                graphVizObject = this;

                // Event listener for node clicks to handle selection
                graphVizObject.nodes().click(function (event) {
                    const nodeSet = $().add(this);
                    const selectionObject = {
                        set: nodeSet,
                        direction: selectedDirection
                    };

                    // If CMD, CTRL, or SHIFT is pressed, add to the selection
                    if (event.ctrlKey || event.metaKey || event.shiftKey) {
                        currentSelection.push(selectionObject);
                    } else {
                        currentSelection = [selectionObject];
                    }

                    highlightSelection();
                });
                // Event listener for pressing the escape key to cancel highlights
                $(document).keydown(function (event) {
                    if (event.keyCode === 27) {
                        graphVizObject.highlight();
                    }
                });
            }
        });
    });

    // Function to search nodes and edges and highlight results
    function searchAndHighlight(query) {
        const searchResults = search(query, "highlight", searchObject);
        graphVizObject.highlight(searchResults.nodes, searchResults.edges);
    }

    // Event listeners for `anywidget` events
    model.on("change:dot_source", () => {
        render(model.get("dot_source"));
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

    render(model.get("dot_source"));
}
export default { render };
