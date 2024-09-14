from typing import Any

import anywidget
import ipywidgets as widgets
import traitlets


class PipeFuncGraphWidget(anywidget.AnyWidget):
    """A widget for rendering a graphviz graph using d3-graphviz.

    Example:
    -------
    >>> dot_string = "digraph { a -> b; b -> c; c -> a; }"
    >>> pipe_func_graph_widget = PipeFuncGraphWidget(dot_source=dot_string)
    >>> pipe_func_graph_widget

    """

    _esm = """
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

        var graphviz = d3.select("#graph").graphviz();

        var d3Config = {
            transitionDelay: 0,
            transitionDuration: 500
        };

        var selectedEngine = "dot"; // Hardcoded to "dot"
        var graphVizObject;
        var selectedDirection = model.get("selected_direction") || "bidirectional";
        var currentSelection = [];

        function highlightSelection() {
            let highlightedNodes = $();
            currentSelection.forEach(selection => {
                const nodes = getConnectedNodes(selection.set, selection.direction);
                highlightedNodes = highlightedNodes.add(nodes);
            });
            graphVizObject.highlight(highlightedNodes, true);
        }

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

        function resetGraph() {
            graphviz.resetZoom();
            graphVizObject.highlight();
            currentSelection = [];
        }

        function updateDirection(newDirection) {
            selectedDirection = newDirection;
            resetGraph();
        }

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
                    $('#graph').data('graphviz.svg').setup();
                });
        }

        $(document).ready(function () {
            $("#graph").graphviz({
                shrink: null,
                zoom: false,
                ready: function () {
                    graphVizObject = this;

                    graphVizObject.nodes().click(function (event) {
                        const nodeSet = $().add(this);
                        const selectionObject = {
                            set: nodeSet,
                            direction: selectedDirection
                        };

                        if (event.ctrlKey || event.metaKey || event.shiftKey) {
                            currentSelection.push(selectionObject);
                        } else {
                            currentSelection = [selectionObject];
                        }

                        highlightSelection();
                    });

                    $(document).keydown(function (event) {
                        if (event.keyCode === 27) {
                            graphVizObject.highlight();
                        }
                    });
                }
            });
        });

        model.on("change:dot_source", () => {
            render(model.get("dot_source"));
        });

        model.on("change:selected_direction", () => {
            updateDirection(model.get("selected_direction"));
        });

        model.on("msg:custom", (msg) => {
            if (msg.action === "reset_zoom") {
                resetGraph();
            }
        });

        render(model.get("dot_source"));
    }
    export default { render };
    """

    _css = """
    #graph {
        margin: auto;
    }
    """

    dot_source = traitlets.Unicode("").tag(sync=True)
    selected_direction = traitlets.Unicode("bidirectional").tag(sync=True)


def graph_widget(dot_string: str = "digraph { a -> b; b -> c; c -> a; }") -> widgets.VBox:
    pipe_func_graph_widget = PipeFuncGraphWidget(dot_source=dot_string)
    reset_button = widgets.Button(description="Reset Zoom")
    direction_selector = widgets.Dropdown(
        options=["bidirectional", "downstream", "upstream", "single"],
        value="bidirectional",
        description="Direction:",
    )

    # Define button actions
    def reset_graph(_: Any) -> None:
        pipe_func_graph_widget.send({"action": "reset_zoom"})

    def update_direction(change: dict) -> None:
        pipe_func_graph_widget.selected_direction = change["new"]

    reset_button.on_click(reset_graph)
    direction_selector.observe(update_direction, names="value")

    # Display widgets
    return widgets.VBox([reset_button, direction_selector, pipe_func_graph_widget])
