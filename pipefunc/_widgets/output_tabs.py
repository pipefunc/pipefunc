from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


class OutputTabs:
    """A ``ipywidgets.Tab`` widget that contains ``ipywidgets.Output`` widgets."""

    def __init__(self, num_outputs: int) -> None:
        from ipywidgets import Output, Tab

        self.outputs: list[Output] = [Output() for _ in range(num_outputs)]
        self._visible_outputs: dict[Output, bool] = dict.fromkeys(self.outputs, False)
        self.tab: Tab = Tab(children=[])

    def display(self) -> None:
        """Display the ``ipywidgets.Tab`` widget."""
        import IPython

        IPython.display.display(self.tab)

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

    @contextmanager
    def output_context(self, index: int) -> Generator[None, None, None]:
        """Context manager to show the output at the given index."""
        if not 0 <= index < len(self.outputs):
            msg = f"Index {index} out of range for {len(self.outputs)} outputs"
            raise IndexError(msg)
        with self.outputs[index]:
            yield
        self.show_output(index)
