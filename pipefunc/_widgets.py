from IPython.display import display, HTML
from string import Template

def render_graphviz(dot_content, selected_engine='dot', vs_code_message=None):
    if vs_code_message is None:
        vs_code_message = {}

    html_template = Template('''
    <!DOCTYPE html>
    <meta charset="utf-8">
    <body>
    <div id="graph" style="text-align: center;"></div>
    <div id="faulttoolbar" style="display: none;">Error: <span id="faultmessage"></span></div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/d3@5.16.0/dist/d3.min.js" onload="loadHpccWasm()"></script>
    <script>
    function addScript(src, onloadFunction) {
        var script = document.createElement('script');
        script.src = src;
        script.onload = onloadFunction;
        document.body.appendChild(script);
    }

    function loadHpccWasm() {
        addScript("https://unpkg.com/@hpcc-js/wasm@0.3.11/dist/index.min.js", loadD3Graphviz);
    }

    function loadD3Graphviz() {
        addScript("https://unpkg.com/d3-graphviz@3.0.5/build/d3-graphviz.js", loadJqueryGraphvizSvg);
    }

    function loadJqueryGraphvizSvg() {
        addScript("https://cdn.jsdelivr.net/gh/mountainstorm/jquery.graphviz.svg@master/js/jquery.graphviz.svg.js", renderGraph);
    }

    function renderGraph() {
        var transition = d3.transition("startTransition")
            .ease(d3.easeLinear)
            .delay(100)
            .duration(1000);

        var graphviz = d3.select("#graph").graphviz()
            .engine('$selected_engine')
            .fade(true)
            .transition(transition)
            .tweenPaths(true)
            .tweenShapes(true)
            .zoomScaleExtent([0, Infinity])
            .zoom(true)
            .onerror(function (err) {
                document.getElementById('faultmessage').innerHTML = err;
                document.getElementById('faulttoolbar').style.display = 'block';
                console.log(err);
            })
            .renderDot(`$dot_content`)
            .on("end", function () {
                // Additional JavaScript code to run after the graph is rendered
                console.log("Graph rendering completed.");
                if ('$vs_code_message' && '$vs_code_message'.search) {
                    var searchText = '$vs_code_message'.search.text || '$vs_code_message'.search || '';
                    if (typeof searchText === 'string' && searchText.length > 0) {
                        console.log("Search text:", searchText);
                    }
                }
                // Resetup using jQuery
                $$('#graph').data('graphviz.svg').setup();
            });
    }
    </script>
    </body>
    ''')

    html_content = html_template.substitute(
        dot_content=dot_content,
        selected_engine=selected_engine,
        vs_code_message=str(vs_code_message)
    )
    display(HTML(html_content))

# Example usage:
render_graphviz('digraph {a -> b}')