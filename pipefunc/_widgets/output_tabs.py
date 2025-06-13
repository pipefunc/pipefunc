from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Generator

# Status symbols mapping
_STATUS_SYMBOLS = {"running": "●", "completed": "✓", "failed": "✗"}


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
        from IPython.display import HTML, display

        css = _generate_tab_css(len(self.outputs))
        display(HTML(css), self.tab)

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
        # Update stored status and CSS classes
        old_status = self._tab_statuses.get(index)
        self._tab_statuses[index] = status
        self._update_tab_classes(index, old_status, status)

        # Update title with status symbol
        current_title = self.tab.titles[index]
        # Remove any existing status symbols
        for symbol in _STATUS_SYMBOLS.values():
            current_title = current_title.replace(symbol, "").strip()

        new_title = f"{_STATUS_SYMBOLS[status]} {current_title}"
        self.tab.set_title(index, new_title)

    def _update_tab_classes(self, index: int, old_status: str | None, new_status: str) -> None:
        """Update CSS classes on the Tab widget to control individual tab styling."""
        if old_status:
            self.tab.remove_class(f"tab-{index}-{old_status}")
        self.tab.add_class(f"tab-{index}-{new_status}")

    @contextmanager
    def output_context(self, index: int) -> Generator[None, None, None]:
        """Context manager to show the output at the given index."""
        if not 0 <= index < len(self.outputs):
            msg = f"Index {index} out of range for {len(self.outputs)} outputs"
            raise IndexError(msg)
        with self.outputs[index]:
            yield
        self.show_output(index)


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

_TAB_CSS_TEMPLATE = """
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
    css_parts = [_BASE_CSS]
    css_parts.extend(_TAB_CSS_TEMPLATE.format(i=i, nth_child=i + 1) for i in range(num_outputs))
    css_parts.append("</style>")
    return "".join(css_parts)
