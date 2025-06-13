import pytest

from pipefunc._widgets.output_tabs import OutputTabs


def test_output_tabs_widget() -> None:
    tabs = OutputTabs(3)
    tabs.display()
    tabs.show_output(0)
    tabs.show_output(1)
    tabs.show_output(2)
    tabs.hide_output(1)
    tabs.hide_output(2)
    with tabs.output_context(0):
        print("Hello")
    with tabs.output_context(1):
        print("World")
    with tabs.output_context(2):
        print("!")

    with pytest.raises(IndexError, match="Index 42 out of range"), tabs.output_context(42):
        pass
