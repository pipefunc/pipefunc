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

/* Tab status styles using nth-child selectors and widget state classes */
"""

_BASE_CSS_TEMPLATE = """
/* Tab {i} styles */
.tab-{i}-running .lm-TabBar-content li:nth-child({nth_child}) {{
    background: var(--jp-warn-color3) !important;
    animation: pulse 2s infinite;
}}
.tab-{i}-running .lm-TabBar-content li:nth-child({nth_child}).lm-mod-current {{
    background: var(--jp-layout-color1) !important;
    border-left: 3px solid var(--jp-warn-color1) !important;
    animation: pulse 2s infinite;
}}
.tab-{i}-completed .lm-TabBar-content li:nth-child({nth_child}) {{
    background: var(--jp-success-color3) !important;
}}
.tab-{i}-completed .lm-TabBar-content li:nth-child({nth_child}).lm-mod-current {{
    background: var(--jp-layout-color1) !important;
    border-left: 3px solid var(--jp-success-color1) !important;
}}
.tab-{i}-failed .lm-TabBar-content li:nth-child({nth_child}) {{
    background: var(--jp-error-color3) !important;
}}
.tab-{i}-failed .lm-TabBar-content li:nth-child({nth_child}).lm-mod-current {{
    background: var(--jp-layout-color1) !important;
    border-left: 3px solid var(--jp-error-color1) !important;
}}
"""


def _generate_tab_css(num_outputs: int) -> str:
    """Generate CSS for tab status styling based on the number of outputs."""
    base_css = _BASE_CSS
    for i in range(num_outputs):
        nth_child = i + 1  # CSS nth-child is 1-indexed
        base_css += _BASE_CSS_TEMPLATE.format(i=i, nth_child=nth_child)
    return base_css


class OutputTabs:
    """A ``ipywidgets.Tab`` widget that contains ``ipywidgets.Output`` widgets."""

    def __init__(self, num_outputs: int) -> None:
        from ipywidgets import Output, Tab

        self.outputs: list[Output] = [Output() for _ in range(num_outputs)]
        self._visible_outputs: dict[Output, bool] = dict.fromkeys(self.outputs, False)
        self.tab: Tab = Tab(children=[])
        self._tab_statuses: dict[int, str] = {}
        self._num_outputs = num_outputs

    def display(self) -> None:
        """Display the ``ipywidgets.Tab`` widget."""
        import IPython.display

        # Generate CSS dynamically based on number of outputs
        css = _generate_tab_css(self._num_outputs)
        html = IPython.display.HTML(css)
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
            if not self.tab.titles[i_tab]:
                self.tab.set_title(i_tab, str(index))

    def set_tab_status(self, index: int, status: Literal["running", "completed", "failed"]) -> None:
        """Sets the tab status using CSS classes on the widget itself."""
        if index >= len(self.tab.titles):
            return

        # Store status
        old_status = self._tab_statuses.get(index)
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

        # Update CSS classes on the widget
        self._update_tab_classes(index, old_status, status)

    def _update_tab_classes(self, index: int, old_status: str | None, new_status: str) -> None:
        """Update CSS classes on the Tab widget to control individual tab styling."""
        # Remove old status class if it exists
        if old_status:
            old_class = f"tab-{index}-{old_status}"
            self.tab.remove_class(old_class)

        # Add new status class
        new_class = f"tab-{index}-{new_status}"
        self.tab.add_class(new_class)

    @contextmanager
    def output_context(self, index: int) -> Generator[None, None, None]:
        """Context manager to show the output at the given index."""
        if not 0 <= index < len(self.outputs):
            msg = f"Index {index} out of range for {len(self.outputs)} outputs"
            raise IndexError(msg)
        with self.outputs[index]:
            yield
        self.show_output(index)
