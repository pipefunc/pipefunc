function loadScript(url) {
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.onload = resolve;
        script.onerror = () => reject(new Error(`Failed to load script: ${url}`));
        document.head.append(script);
    });
}

async function loadAllScripts() {
    const scripts = [
        "https://unpkg.com/jquery@3.6.1/dist/jquery.min.js",
        "https://unpkg.com/jquery-mousewheel@3.1.13/jquery.mousewheel.js",
        "https://unpkg.com/jquery-color@2.2.0/dist/jquery.color.js",
        "https://unpkg.com/d3@7.6.1/dist/d3.min.js",
        "https://cdn.jsdelivr.net/gh/mountainstorm/jquery.graphviz.svg@master/js/jquery.graphviz.svg.js",
        "https://unpkg.com/@hpcc-js/wasm@1.18.0/dist/index.min.js",
        "https://unpkg.com/d3-graphviz@4.4.0/build/d3-graphviz.min.js",
    ];

    for (const script of scripts) {
        await loadScript(script);
    }
}

function getLegendElements(GraphvizSvg, $) {
    const legendNodes = [];
    const legendEdges = [];

    GraphvizSvg.nodes().each(function () {
        const $node = $(this);
        if ($node.attr("data-name").startsWith("legend_")) {
            legendNodes.push($node[0]);
        }
    });

    GraphvizSvg.edges().each(function () {
        const $edge = $(this);
        if ($edge.attr("data-name").startsWith("legend_")) {
            legendEdges.push($edge[0]);
        }
    });
    return { legendNodes: $(legendNodes), legendEdges: $(legendEdges) };
}

function findEdges(text, searchFunction, GraphvizSvg, $) {
    const $set = $();
    GraphvizSvg.edges().each((index, edge) => {
        if (edge.textContent && searchFunction(text, edge.textContent)) {
            $set.push(edge);
        }
    });
    return $set;
}

function findNodes(text, searchFunction, nodeName, nodeLabel, GraphvizSvg, $) {
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

function search(text, searchObject, GraphvizSvg, $) {
    let searchFunction;

    switch (searchObject.type) {
        case "exact":
            searchFunction = (search, str) => str.trim() === search.trim();
            break;
        case "included":
            searchFunction = (search, str) => {
                const searchStr =
                    searchObject.case === "insensitive" ? search.toLowerCase() : search;
                const valStr = searchObject.case === "insensitive" ? str.toLowerCase() : str;
                return valStr.indexOf(searchStr) !== -1;
            };
            break;
        case "regex":
            searchFunction = (search, str) => {
                const regex = new RegExp(
                    search,
                    searchObject.case === "insensitive" ? "i" : undefined
                );
                return !!str.trim().match(regex);
            };
            break;
    }

    let $edges = $();
    if (searchObject.edgeLabel) {
        $edges = findEdges(text, searchFunction, GraphvizSvg, $);
    }

    let $nodes = $();
    if (searchObject.nodeLabel || searchObject.nodeName) {
        $nodes = findNodes(
            text,
            searchFunction,
            searchObject.nodeName,
            searchObject.nodeLabel,
            GraphvizSvg,
            $
        );
    }
    return { nodes: $nodes, edges: $edges };
}

function getConnectedNodes(nodeSet, mode, GraphvizSvg) {
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

function highlightSelection(GraphvizSvg, currentSelection, $) {
    let highlightedNodes = $();
    let highlightedEdges = $();

    currentSelection.forEach((selection) => {
        const nodes = getConnectedNodes(selection.set, selection.direction, GraphvizSvg);
        highlightedNodes = highlightedNodes.add(nodes);
    });

    const { legendNodes, legendEdges } = getLegendElements(GraphvizSvg, $);
    highlightedNodes = highlightedNodes.add(legendNodes);
    highlightedEdges = highlightedEdges.add(legendEdges);

    GraphvizSvg.highlight(highlightedNodes, highlightedEdges);
}

async function render({ model, el }) {
    await loadAllScripts();

    const $ = window.jQuery;
    const d3 = window.d3;

    el.innerHTML = '<div id="graph" style="text-align: center;"></div>';

    const d3graphviz = d3.select("#graph").graphviz();
    let GraphvizSvg;
    const currentSelection = [];
    let selectedDirection = model.get("selected_direction") || "bidirectional";

    const searchObject = {
        type: model.get("search_type") || "included",
        case: model.get("case_sensitive") ? "sensitive" : "insensitive",
        nodeName: true,
        nodeLabel: true,
        edgeLabel: true,
    };

    const renderGraph = (dotSource) => {
        const transition = d3
            .transition("graphTransition")
            .ease(d3.easeLinear)
            .delay(0)
            .duration(500);

        d3graphviz
            .engine("dot")
            .fade(true)
            .transition(transition)
            .tweenPaths(true)
            .tweenShapes(true)
            .zoomScaleExtent([0, Infinity])
            .zoom(true)
            .renderDot(dotSource)
            .on("end", function () {
                $("#graph").data("graphviz.svg").setup();
            });
    };

    const resetGraph = () => {
        d3graphviz.resetZoom();
        GraphvizSvg.highlight();
        currentSelection.length = 0;
    };

    const updateDirection = (newDirection) => {
        selectedDirection = newDirection;
        resetGraph();
    };

    const searchAndHighlight = (query) => {
        const searchResults = search(query, searchObject, GraphvizSvg, $);
        const { legendNodes, legendEdges } = getLegendElements(GraphvizSvg, $);
        const nodesToHighlight = searchResults.nodes.add(legendNodes);
        const edgesToHighlight = searchResults.edges.add(legendEdges);
        GraphvizSvg.highlight(nodesToHighlight, edgesToHighlight);
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

    $(document).ready(function () {
        $("#graph").graphviz({
            shrink: null,
            zoom: false,
            ready: function () {
                GraphvizSvg = this;

                // Add hover event listeners for edges
                GraphvizSvg.edges().each(function () {
                    const $edge = $(this);

                    // Store the original color if not already stored
                    $edge.data("original-stroke", $edge.find("path").attr("stroke-width"));

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

                GraphvizSvg.nodes().click(function (event) {
                    const nodeSet = $().add(this);
                    const selectionObject = {
                        set: nodeSet,
                        direction: selectedDirection,
                    };

                    if (event.ctrlKey || event.metaKey || event.shiftKey) {
                        currentSelection.push(selectionObject);
                    } else {
                        currentSelection.splice(0, currentSelection.length, selectionObject);
                    }

                    highlightSelection(GraphvizSvg, currentSelection, $);
                });

                $(document).keydown(function (event) {
                    if (event.keyCode === 27) {
                        // Escape key
                        GraphvizSvg.highlight();
                    }
                });
            },
        });
    });

    renderGraph(model.get("dot_source"));
}

export default { render };
