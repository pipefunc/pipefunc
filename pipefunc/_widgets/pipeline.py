import getpass
import importlib.metadata
import sys
from importlib.util import find_spec
from typing import Any

import ipywidgets as widgets
from IPython.display import display

from pipefunc import PipeFunc, Pipeline
from pipefunc import __file__ as package_root

DEPENDENCIES = [
    "networkx",
    "psutil",
    "cloudpickle",
    "numpy",
    "adaptive",
    "adaptive_scheduler",
    "xarray",
    "pandas",
    "zarr",
    "ipywidgets",
    "matplotlib",
    "pygraphviz",
    "graphviz",
    "holoviews",
    "bokeh",
    "jupyter_bokeh",
]


class PipelineWidget:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self._initialize_widgets()
        self._bind_events()
        self._configure_tab_display()

    def _initialize_widgets(self) -> None:
        self.info_button = self._create_button(
            "Show Pipeline Info",
            "info",
            "Click to view pipeline information",
        )
        self.visualize_button = self._create_button(
            "Visualize Pipeline",
            "primary",
            "Click to visualize the pipeline",
        )
        self.visualize_type = widgets.Dropdown(
            options=["graphviz", "matplotlib", "holoviews"],
            value="graphviz",
            description="Type:",
        )
        self.fig_width = widgets.IntSlider(value=10, min=1, max=50, step=1, description="Width:")
        self.fig_height = widgets.IntSlider(value=10, min=1, max=50, step=1, description="Height:")
        # Because graphviz is default, hide the sliders
        self.fig_width.layout.display = self.fig_height.layout.display = "none"
        self.info_output_display = widgets.Output()
        self.visualize_output_display = widgets.Output()
        self.package_output_display = widgets.Output()
        self.package_button = self._create_button(
            "Show Package Info",
            "primary",
            "Click to view package information",
        )

    def _create_button(self, description: str, style: str, tooltip: str) -> widgets.Button:
        return widgets.Button(description=description, button_style=style, tooltip=tooltip)

    def _bind_events(self) -> None:
        self.info_button.on_click(self._show_pipeline_info)
        self.visualize_button.on_click(self._visualize_pipeline)
        self.fig_width.observe(self._on_size_change, names="value")
        self.fig_height.observe(self._on_size_change, names="value")
        self.visualize_type.observe(self._on_visualize_type_change, names="value")
        self.package_button.on_click(self._show_package_info)

    def _configure_tab_display(self) -> None:
        self.tab = widgets.Tab()
        self.tab.children = [
            self._create_info_tab(),
            self._create_visualize_tab(),
            self._create_package_info_tab(),
        ]
        self.tab.set_title(0, "Info")
        self.tab.set_title(1, "Visualize")
        self.tab.set_title(2, "Package Info")

    def _create_package_info_tab(self) -> widgets.VBox:
        return self._create_vbox(
            "Package Information",
            self.package_button,
            self.package_output_display,
        )

    def _create_info_tab(self) -> widgets.VBox:
        return self._create_vbox("Pipeline Information", self.info_button, self.info_output_display)

    def _create_vbox(
        self,
        title: str,
        button: widgets.Button,
        output: widgets.Output,
    ) -> widgets.VBox:
        return widgets.VBox(
            [widgets.HTML(f"<h2>{title}</h2>"), button, widgets.HTML("<hr>"), output],
        )

    def _create_visualize_tab(self) -> widgets.VBox:
        return widgets.VBox(
            [
                widgets.HTML("<h2>Visualize Pipeline</h2>"),
                self.visualize_type,
                self.fig_width,
                self.fig_height,
                widgets.HTML("<hr>"),
                self.visualize_button,
                self.visualize_output_display,
            ],
        )

    def _show_pipeline_info(
        self,
        _button: widgets.Button | None = None,
    ) -> None:  # pragma: no cover
        with self.info_output_display:
            if self.info_output_display.outputs:
                self.info_output_display.clear_output()
                self.info_button.description = "Show Pipeline Info"
                self.info_button.button_style = "info"
            else:
                self._display_pipeline_info()

    def _display_pipeline_info(self) -> None:  # pragma: no cover
        html_content = self._pipeline_info_html()
        self.info_output_display.clear_output(wait=True)
        display(widgets.HTML(html_content))
        self.info_button.description = "Hide Pipeline Info"
        self.info_button.button_style = "warning"

    def _pipeline_info_html(self) -> str:
        html_content = _build_legend() + "".join(
            self._create_function_info_html(func) for func in self.pipeline.functions
        )
        return html_content + _build_combined_pipefunc_info_table(self.pipeline.functions)

    def _create_function_info_html(self, func: PipeFunc) -> str:
        headers = ["Parameter", "Type", "Default Value", "Returned By"]
        root_args = set(self.pipeline.topological_generations.root_args)
        defaults = self.pipeline.defaults
        rows = [
            _format_param(
                param,
                func.parameter_annotations.get(param, "—"),
                param in root_args,
                defaults.get(param),
                self.pipeline.output_to_func.get(param, "—"),
            )
            for param in func.parameters
        ]

        html_content = (
            f"<h4><b>Function:</b> {_format_code(func.__name__)}</h4>{_build_table(headers, rows)}"
        )
        if func.output_annotation:
            html_content += (
                "<h4>Outputs</h4><ul>"
                + "".join(
                    f"<li class='output-type'><b>{output_name}</b>: {output_type.__name__}</li>"
                    for output_name, output_type in func.output_annotation.items()
                )
                + "</ul>"
            )
        return html_content + "<hr>"

    def _visualize_pipeline(self, _button: widgets.Button | None = None) -> None:
        if self.visualize_type.value == "holoviews":
            import holoviews as hv

            hv.extension("bokeh")
        with self.visualize_output_display:
            self.visualize_output_display.clear_output(wait=True)
            self._render_visualization()

    def _render_visualization(self) -> None:
        if self.visualize_type.value == "matplotlib":
            figsize = (self.fig_width.value, self.fig_height.value)
            display(widgets.HTML("<h3>Pipeline Visualization (Matplotlib)</h3>"))
            display(self.pipeline.visualize_matplotlib(figsize=figsize))
        elif self.visualize_type.value == "graphviz":
            display(widgets.HTML("<h3>Pipeline Visualization (Graphviz)</h3>"))
            display(self.pipeline.visualize_graphviz(return_type="html"))
        else:
            import panel as pn

            display(widgets.HTML("<h3>Pipeline Visualization (HoloViews)</h3>"))
            hv_output = self.pipeline.visualize_holoviews()
            display(pn.ipywidget(hv_output))

    def _on_size_change(self, _change: dict) -> None:
        if self.visualize_type.value == "matplotlib":
            self._visualize_pipeline()

    def _on_visualize_type_change(self, change: dict) -> None:
        display_visibility = "" if change["new"] == "matplotlib" else "none"
        self.fig_width.layout.display = self.fig_height.layout.display = display_visibility
        self._visualize_pipeline()

    def _show_package_info(self, _button: widgets.Button | None = None) -> None:  # pragma: no cover
        with self.package_output_display:
            self.package_output_display.clear_output(wait=True)
            display(widgets.HTML(self._package_info_html()))

    def _package_info_html(self) -> str:
        pipefunc_version = _get_installed_version("pipefunc") or "Unknown"
        location: str = package_root
        user = getpass.getuser()
        info_table = self._create_package_info_table(pipefunc_version, location, user)
        dependencies_table = self._create_dependencies_table()
        return f"<h3>PipeFunc Package Details</h3>{info_table}<h3>Dependencies</h3>{dependencies_table}"

    def _create_package_info_table(self, version: str, location: str, user: str) -> str:
        v = sys.version_info
        return _build_table(
            ["Attribute", "Value"],
            [
                ["Version", version],
                ["Installed Location", location],
                ["Current User", user],
                ["Python Version", f"{v.major}.{v.minor}.{v.micro}"],
            ],
        )

    def _create_dependencies_table(self) -> str:
        rows = [[dep, _check_dependency(dep)] for dep in sorted(DEPENDENCIES)]
        return _build_table(["Dependency", "Status"], rows)

    def display(self) -> None:  # pragma: no cover
        """Displays the widget in the notebook."""
        display(self.tab)


def _get_installed_version(module_name: str) -> str | None:
    try:
        return importlib.metadata.version(module_name)
    except ModuleNotFoundError:  # pragma: no cover
        return None
    except AttributeError:  # pragma: no cover
        return "unknown"


def _check_dependency(dependency: str) -> str:
    version = _get_installed_version(dependency) if find_spec(dependency) else None
    return f"✅ {version}" if version else "❌"


def _build_table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = _build_row(headers, "th")
    rows_html = "".join(_build_row(row) for row in rows)
    return f"<table><thead>{header_html}</thead><tbody>{rows_html}</tbody></table>"


def _build_row(cells: list[str], cell_tag: str = "td") -> str:
    return f"<tr>{''.join(f'<{cell_tag}>{cell}</{cell_tag}>' for cell in cells)}</tr>"


def _format_param(
    param: str,
    param_type: Any,
    is_root: bool,  # noqa: FBT001
    default_value: str | None,
    returned_by: PipeFunc | str,
) -> list[str]:
    param_html = (
        f"<span style='color: #e74c3c; font-weight: bold;'>{param}</span>" if is_root else param
    )
    default_html = (
        f"<span style='color: #2ecc71;'>{default_value}</span>"
        if default_value is not None
        else "—"
    )
    returned_by = (
        _format_code(returned_by.__name__) if isinstance(returned_by, PipeFunc) else returned_by
    )
    param_type = _format_code(param_type.__name__) if param_type != "—" else param_type
    return [param_html, param_type, default_html, returned_by]


def _build_legend() -> str:
    return """
        <div class="legend" style="font-size: 14px; margin-bottom: 15px;">
            <b>Legend:</b>
            <ul>
                <li><span style="color: #e74c3c; font-weight: bold;">Bold & Red</span> parameters are pipeline root arguments (root_args).</li>
                <li>Parameters with <span style="color: #2ecc71;">green text</span> have default values.</li>
            </ul>
        </div>
    """


def _build_combined_pipefunc_info_table(functions: list[PipeFunc]) -> str:
    headers = ["Attribute"] + [func.__name__ for func in functions]
    attributes = ["Output name", "Cache", "Profile", "MapSpec", "Resources", "Debug"]
    rows = [
        [attribute]
        + [
            format_attribute_value(getattr(func, attribute.lower().replace(" ", "_"), None))
            for func in functions
        ]
        for attribute in attributes
    ]

    return _build_table(headers, rows)


def format_attribute_value(value: Any) -> str:
    if isinstance(value, bool):
        return "✅" if value else "❌"
    if isinstance(value, int | str):
        return str(value)
    return "—" if value is None else str(value)


def _format_code(text: str) -> str:
    return f"<code>{text}</code>"
