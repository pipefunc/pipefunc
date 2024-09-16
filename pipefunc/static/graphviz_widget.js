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
    await loadScript(
        "https://cdn.jsdelivr.net/gh/mountainstorm/jquery.graphviz.svg@master/js/jquery.graphviz.svg.js"
    );
    await loadScript("https://unpkg.com/@hpcc-js/wasm@1.18.0/dist/index.min.js");
    await loadScript("https://unpkg.com/d3-graphviz@4.4.0/build/d3-graphviz.min.js");

    const $ = window.jQuery;
    const d3 = window.d3;

    // Prepare the graph container
    el.innerHTML = '<div id="graph" style="text-align: center;"></div>';

    // Initialize a d3-graphviz renderer instance
    var graphviz = d3.select("#graph").graphviz();

    // Configuration for transitions in rendering the graph
    var d3Config = {
        transitionDelay: 0,
        transitionDuration: 500,
    };

    // Variable for storing the selected graph rendering engine
    var selectedEngine = "dot";

    // Object for saving the current GraphVizSVG
    var GraphvizSvg;

    // Variable for storing the selected direction for highlighting
    var selectedDirection = model.get("selected_direction") || "bidirectional";

    // Array holding the current selections
    var currentSelection = [];

    // Search-related variables
    const searchObject = {
        type: model.get("search_type") || "included",
        case: model.get("case_sensitive") ? "sensitive" : "insensitive",
        nodeName: true,
        nodeLabel: true,
        edgeLabel: true,
    };

    // Listener for changes in search type
    model.on("change:search_type", () => {
        searchObject.type = model.get("search_type");
    });

    // Listener for changes in case sensitivity
    model.on("change:case_sensitive", () => {
        searchObject.case = model.get("case_sensitive") ? "sensitive" : "insensitive";
    });

    // Main search function to find nodes and edges
    function search(text) {
        let searchFunction;

        if (searchObject.type === "exact") {
            searchFunction = (search, str) => str.trim() === search.trim();
        } else if (searchObject.type === "included") {
            searchFunction = (search, str) => {
                const searchStr =
                    searchObject.case === "insensitive" ? search.toLowerCase() : search;
                const valStr = searchObject.case === "insensitive" ? str.toLowerCase() : str;
                return valStr.indexOf(searchStr) !== -1;
            };
        } else if (searchObject.type === "regex") {
            searchFunction = (search, str) => {
                const regex = new RegExp(
                    search,
                    searchObject.case === "insensitive" ? "i" : undefined
                );
                return !!str.trim().match(regex);
            };
        }

        let $edges = $();
        if (searchObject.edgeLabel) {
            $edges = findEdges(text, searchFunction);
        }

        let $nodes = $();
        if (searchObject.nodeLabel || searchObject.nodeName) {
            $nodes = findNodes(text, searchFunction, searchObject.nodeName, searchObject.nodeLabel);
        }

        return { nodes: $nodes, edges: $edges };
    }
    // Function to find edges matching the search criteria
    function findEdges(text, searchFunction) {
        const $set = $();
        GraphvizSvg.edges().each((index, edge) => {
            if (edge.textContent && searchFunction(text, edge.textContent)) {
                $set.push(edge);
            }
        });
        return $set;
    }

    // Function to find nodes matching the search criteria
    function findNodes(text, searchFunction, nodeName = true, nodeLabel = true) {
        const $set = $();
        const nodes = GraphvizSvg.nodesByName();

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

    // Function to highlight selected nodes and their connected nodes
    function highlightSelection() {
        let highlightedNodes = $();
        currentSelection.forEach((selection) => {
            const nodes = getConnectedNodes(selection.set, selection.direction);
            highlightedNodes = highlightedNodes.add(nodes);
        });
        GraphvizSvg.highlight(highlightedNodes, true);
    }

    // Function to retrieve nodes connected in the specified direction
    function getConnectedNodes(nodeSet, mode = "bidirectional") {
        let resultSet = $().add(nodeSet);
        const nodes = GraphvizSvg.nodesByName();

        nodeSet.each((i, el) => {
            if (el.className.baseVal === "edge") {
                const [startNode, endNode] = $(el).data("name").split("->");
                if ((mode === "bidirectional" || mode === "upstream") && startNode) {
                    resultSet = resultSet
                        .add(nodes[startNode])
                        .add(GraphvizSvg.linkedTo(nodes[startNode], true));
                }
                if ((mode === "bidirectional" || mode === "downstream") && endNode) {
                    resultSet = resultSet
                        .add(nodes[endNode])
                        .add(GraphvizSvg.linkedFrom(nodes[endNode], true));
                }
            } else {
                if (mode === "bidirectional" || mode === "upstream") {
                    resultSet = resultSet.add(GraphvizSvg.linkedTo(el, true));
                }
                if (mode === "bidirectional" || mode === "downstream") {
                    resultSet = resultSet.add(GraphvizSvg.linkedFrom(el, true));
                }
            }
        });
        return resultSet;
    }

    // Function to reset the graph zoom and selection highlights
    function resetGraph() {
        graphviz.resetZoom();
        GraphvizSvg.highlight(); // Reset node selection on reset
        currentSelection = [];
    }

    // Function to update the selected direction for highlighting
    function updateDirection(newDirection) {
        selectedDirection = newDirection;
        resetGraph();
    }

    // Main function to render the graph from DOT source
    function render(dotSource) {
        var transition = d3
            .transition("graphTransition")
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
                $("#graph").data("graphviz.svg").setup(); // Re-setup after rendering
            });
    }

    // Document ready function
    $(document).ready(function () {
        // Initialize the GraphVizSVG object from jquery.graphviz.svg.js
        $("#graph").graphviz({
            shrink: null,
            zoom: false,
            ready: function () {
                GraphvizSvg = this;

                // Event listener for node clicks to handle selection
                GraphvizSvg.nodes().click(function (event) {
                    const nodeSet = $().add(this);
                    const selectionObject = {
                        set: nodeSet,
                        direction: selectedDirection,
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
                        GraphvizSvg.highlight();
                    }
                });
            },
        });
    });

    // Function to search nodes and edges and highlight results
    function searchAndHighlight(query) {
        const searchResults = search(query);
        GraphvizSvg.highlight(searchResults.nodes, searchResults.edges);
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
