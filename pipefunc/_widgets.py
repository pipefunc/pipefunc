from string import Template

from IPython.display import HTML, display


def render_graphviz(dot_content):
    html_template = Template("""
    <!DOCTYPE html>
    <meta charset="utf-8">
    <body>
    <div id="graph" style="text-align: center;"></div>
    <script src="https://unpkg.com/d3@5.16.0/dist/d3.min.js" onload="loadNext()"></script>
    <script>
    function loadNext() {
        var script1 = document.createElement('script');
        script1.src = "https://unpkg.com/@hpcc-js/wasm@0.3.11/dist/index.min.js";
        script1.onload = function() {
            var script2 = document.createElement('script');
            script2.src = "https://unpkg.com/d3-graphviz@3.0.5/build/d3-graphviz.js";
            script2.onload = function() {
                d3.select("#graph").graphviz().renderDot(`$dot_content`);
            };
            document.body.appendChild(script2);
        };
        document.body.appendChild(script1);
    }
    </script>
    </body>
    """)

    html_content = html_template.substitute(dot_content=dot_content)
    display(HTML(html_content))


# Example usage:
render_graphviz("digraph {a -> b}")
