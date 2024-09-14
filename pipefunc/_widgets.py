import anywidget
import traitlets


class PipeFuncGraphWidget(anywidget.AnyWidget):
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

        // jQuery, d3, and other dependencies should now be available globally
        const $ = window.jQuery;
        const d3 = window.d3;

        // Prepare the graph container
        const container = document.createElement('div');
        container.id = "graph";
        container.style.textAlign = "center";
        el.appendChild(container);

        // Initialize Graphviz with extra functionalities
        var graphvizInstance = d3.select("#graph").graphviz();

        var currentSelection = [];
        var selectedDirection = "bidirectional";
        var graphVizObject;

        // Function to highlight nodes
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
            graphvizInstance.resetZoom();
            graphVizObject.highlight();
            currentSelection = [];
        }

        // Main function to render the graph from DOT source
        function renderGraph(dotSource) {
            var transition = d3.transition("graphTransition")
                .ease(d3.easeLinear)
                .delay(0)
                .duration(500);

            graphvizInstance
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

        // Listen for changes to the dotSource and render it
        model.on("change:dotSource", () => {
            const dotSource = model.get("dotSource");
            renderGraph(dotSource);
        });

        // Initial render
        renderGraph(model.get("dotSource"));
    }
    export default { render };
    """

    _css = """
    #graph {
        margin: auto;
    }
    """

    dotSource = traitlets.Unicode("").tag(sync=True)


# Example usage:
dot_string = "digraph { a -> b; b -> c; c -> a; }"
pipe_func_graph_widget = PipeFuncGraphWidget(dotSource=dot_string)
pipe_func_graph_widget
