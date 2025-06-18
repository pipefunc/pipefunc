from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Generator

    from ipywidgets import Output

_STATUS_SYMBOLS = {"running": "●", "completed": "✓", "failed": "✗"}


def _output() -> Output:
    from ipywidgets import Output

    return Output()


@dataclass
class _OutputTab:
    """A simple data class to hold the state of a single output tab."""

    index: int
    widget: Output = field(default_factory=_output)
    visible: bool = False
    status: str | None = None
    completion_order: int | None = None

    @property
    def title(self) -> str:
        return f"{_STATUS_SYMBOLS[self.status]} {self.index}" if self.status else str(self.index)


class OutputTabs:
    """A ``ipywidgets.Tab`` widget that contains ``ipywidgets.Output`` widgets."""

    def __init__(self, num_outputs: int, max_completed_tabs: int | None = None) -> None:
        from ipywidgets import Tab

        self._tabs: list[_OutputTab] = [_OutputTab(i) for i in range(num_outputs)]
        self.tab: Tab = Tab(children=[])
        self._max_completed_tabs = max_completed_tabs
        self._completion_counter = count()

    def output(self, index_output: int) -> Output:
        """Get the `ipywidgets.Output` widget at the given index_output."""
        if not 0 <= index_output < len(self._tabs):
            msg = f"Index {index_output} out of range for {len(self._tabs)} outputs"
            raise IndexError(msg)
        return self._tabs[index_output].widget

    def display(self) -> None:
        """Display the ``ipywidgets.Tab`` widget."""
        from IPython.display import HTML, display

        css = _generate_tab_css(len(self._tabs))
        display(HTML(css))
        display(self.tab)

    def show_output(self, index_output: int) -> None:
        """Show the output at the given index_output."""
        self._tabs[index_output].visible = True
        self._sync()

    def hide_output(self, index_output: int) -> None:
        """Hide the output at the given index_output."""
        self._tabs[index_output].visible = False
        self._sync()

    def _sync(self) -> None:
        current_index = self.tab.selected_index
        current_output = self.tab.children[current_index] if current_index is not None else None
        self._enforce_max_completed_tabs()
        visible_tabs = [tab for tab in self._tabs if tab.visible]
        classes = [f"tab-{i}-{tab.status}" for i, tab in enumerate(visible_tabs) if tab.status]
        titles = [tab.title for tab in visible_tabs]
        children = tuple(tab.widget for tab in visible_tabs)

        if not children:  # pragma: no cover
            new_index = None
        elif current_output in children:
            new_index = children.index(current_output)
        else:
            new_index = 0

        self.tab.children, self.tab._dom_classes, self.tab.titles, self.tab.selected_index = (
            children,
            classes,
            titles,
            new_index,
        )

    def set_tab_status(
        self,
        index_output: int,
        status: Literal["running", "completed", "failed"],
    ) -> None:
        """Sets the tab status using CSS classes on the widget itself."""
        tab = self._tabs[index_output]
        tab.status = status
        if status == "completed" and tab.completion_order is None:
            tab.completion_order = next(self._completion_counter)
        self._sync()

    def _enforce_max_completed_tabs(self) -> None:
        """Hide oldest completed tabs if more than ``max_completed_tabs`` are visible."""
        if self._max_completed_tabs is None:
            return

        visible_completed = [t for t in self._tabs if t.visible and t.status == "completed"]
        if len(visible_completed) > self._max_completed_tabs:
            # Sort by completion order to find the oldest
            visible_completed.sort(key=lambda t: t.completion_order)  # type: ignore[arg-type,return-value]
            num_to_hide = len(visible_completed) - self._max_completed_tabs
            for tab_to_hide in visible_completed[:num_to_hide]:
                tab_to_hide.visible = False

    @contextmanager
    def output_context(self, index_output: int) -> Generator[None, None, None]:
        """Context manager to show the output at the given index_output."""
        output = self.output(index_output)
        with output:
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
