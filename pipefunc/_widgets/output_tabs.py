from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Generator

# Status symbols mapping
_STATUS_SYMBOLS = {"running": "●", "completed": "✓", "failed": "✗"}


class OutputTabs:
    """A ``ipywidgets.Tab`` widget that contains ``ipywidgets.Output`` widgets."""

    def __init__(self, num_outputs: int, max_completed_tabs: int | None = None) -> None:
        from ipywidgets import Output, Tab

        self.outputs: list[Output] = [Output() for _ in range(num_outputs)]
        self._visible_outputs: dict[Output, bool] = dict.fromkeys(self.outputs, False)
        self.tab: Tab = Tab(children=[])
        self._tab_statuses: dict[int, str] = {}
        self._max_completed_tabs = max_completed_tabs
        self._completed_indices: dict[int, None] = {}

    def display(self) -> None:
        """Display the ``ipywidgets.Tab`` widget."""
        from IPython.display import HTML, display

        css = _generate_tab_css(len(self.outputs))
        display(HTML(css), self.tab)

    def show_output(self, index_output: int) -> None:
        """Show the output at the given index_output."""
        self._visible_outputs[self.outputs[index_output]] = True
        self._sync()

    def hide_output(self, index_output: int) -> None:
        """Hide the output at the given index_output."""
        self._visible_outputs[self.outputs[index_output]] = False
        self._sync()

    def _sync(self) -> None:
        self._enforce_max_completed_tabs()
        children = [output for output in self.outputs if self._visible_outputs[output]]
        self.tab.children = children
        for index_tab, output in enumerate(self.tab.children):
            index = self.outputs.index(output)
            status = self._tab_statuses.get(index)
            new_title = f"{_STATUS_SYMBOLS[status]} {index}" if status else str(index)
            if self.tab.get_title(index_tab) != new_title:
                self.tab.set_title(index_tab, new_title)

    def set_tab_status(
        self,
        index_output: int,
        status: Literal["running", "completed", "failed"],
    ) -> None:
        """Sets the tab status using CSS classes on the widget itself."""
        index_tab = self.tab.children.index(self.outputs[index_output])
        old_status = self._tab_statuses.get(index_tab)
        old_class = f"tab-{index_tab}-{old_status}"
        new_class = f"tab-{index_tab}-{status}"
        if old_class:
            self.tab.remove_class(old_class)
        self.tab.add_class(new_class)
        self._tab_statuses[index_tab] = status
        if status == "completed" and index_output not in self._completed_indices:
            self._completed_indices[index_output] = None
        self._sync()

    def _enforce_max_completed_tabs(self) -> None:
        """Hide oldest completed tabs if more than ``max_completed_tabs`` are visible."""
        if self._max_completed_tabs is None:
            return

        visible_completed = [
            i for i in self._completed_indices if self._visible_outputs[self.outputs[i]]
        ]

        if len(visible_completed) > self._max_completed_tabs:
            num_to_hide = len(visible_completed) - self._max_completed_tabs
            for i in visible_completed[:num_to_hide]:
                self._visible_outputs[self.outputs[i]] = False

    @contextmanager
    def output_context(self, index_output: int) -> Generator[None, None, None]:
        """Context manager to show the output at the given index_output."""
        if not 0 <= index_output < len(self.outputs):
            msg = f"Index {index_output} out of range for {len(self.outputs)} outputs"
            raise IndexError(msg)
        with self.outputs[index_output]:
            yield
        self.show_output(index_output)


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
