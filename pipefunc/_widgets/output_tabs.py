from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Generator


_BASE_CSS = """
<style id="output-tabs-base-css">
.lm-TabBar-tab {
    transition: background-color 0.3s ease, border-left 0.3s ease;
}

@keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}
</style>
"""


class OutputTabs:
    """A ``ipywidgets.Tab`` widget that contains ``ipywidgets.Output`` widgets."""

    def __init__(self, num_outputs: int) -> None:
        from ipywidgets import Output, Tab

        self.outputs: list[Output] = [Output() for _ in range(num_outputs)]
        self._visible_outputs: dict[Output, bool] = dict.fromkeys(self.outputs, False)
        self.tab: Tab = Tab(children=[])
        self._tab_statuses: dict[int, str] = {}

    def display(self) -> None:
        """Display the ``ipywidgets.Tab`` widget."""
        import IPython.display

        html = IPython.display.HTML(_BASE_CSS)
        IPython.display.display(html, self.tab)

    def show_output(self, index: int) -> None:
        """Show the output at the given index."""
        self._visible_outputs[self.outputs[index]] = True
        self._sync()

    def hide_output(self, index: int) -> None:
        """Hide the output at the given index."""
        self._visible_outputs[self.outputs[index]] = False
        self._sync()

    def _sync(self) -> None:
        children = [output for output in self.outputs if self._visible_outputs[output]]
        self.tab.children = children
        for i_tab, output in enumerate(self.tab.children):
            index = self.outputs.index(output)
            self.tab.set_title(i_tab, str(index))

    def set_tab_status(self, index: int, status: Literal["running", "completed", "failed"]) -> None:
        """Sets the tab status using dynamic CSS targeting the specific tab."""
        if index >= len(self.tab.titles):
            return

        # Store status
        self._tab_statuses[index] = status

        # Update the title with status symbol
        current_title = self.tab.titles[index]
        clean_title = current_title.replace("●", "").replace("✓", "").replace("✗", "").strip()

        if status == "running":
            self.tab.set_title(index, f"● {clean_title}")
        elif status == "completed":
            self.tab.set_title(index, f"✓ {clean_title}")
        elif status == "failed":
            self.tab.set_title(index, f"✗ {clean_title}")

        # Inject specific CSS for this tab using nth-child selector
        self._inject_tab_specific_css(index, status)

    def _inject_tab_specific_css(self, tab_index: int, status: str) -> None:
        """Inject CSS that targets a specific tab by its position."""
        from IPython.display import HTML, display

        # Remove previous CSS for this tab
        display(HTML(f'<style id="tab-status-{tab_index}"></style>'))

        # Create CSS that targets the specific tab by position
        css_selector = f".lm-TabBar-content li:nth-child({tab_index + 1})"

        if status == "running":
            css = f"""
            <style id="tab-status-{tab_index}">
            {css_selector} {{
                background: var(--jp-warn-color3) !important;
                animation: pulse 2s infinite;
            }}
            {css_selector}.lm-mod-current {{
                background: var(--jp-layout-color1) !important;
                border-left: 3px solid var(--jp-warn-color1) !important;
                animation: pulse 2s infinite;
            }}
            </style>
            """
        elif status == "completed":
            css = f"""
            <style id="tab-status-{tab_index}">
            {css_selector} {{
                background: var(--jp-success-color3) !important;
            }}
            {css_selector}.lm-mod-current {{
                background: var(--jp-layout-color1) !important;
                border-left: 3px solid var(--jp-success-color1) !important;
            }}
            </style>
            """
        elif status == "failed":
            css = f"""
            <style id="tab-status-{tab_index}">
            {css_selector} {{
                background: var(--jp-error-color3) !important;
            }}
            {css_selector}.lm-mod-current {{
                background: var(--jp-layout-color1) !important;
                border-left: 3px solid var(--jp-error-color1) !important;
            }}
            </style>
            """

        display(HTML(css))

    @contextmanager
    def output_context(self, index: int) -> Generator[None, None, None]:
        """Context manager to show the output at the given index."""
        if not 0 <= index < len(self.outputs):
            msg = f"Index {index} out of range for {len(self.outputs)} outputs"
            raise IndexError(msg)
        with self.outputs[index]:
            yield
        self.show_output(index)
