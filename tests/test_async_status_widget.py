from __future__ import annotations

import importlib.util
import types

import pytest

has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
has_ipython = importlib.util.find_spec("IPython") is not None


@pytest.mark.skipif(
    not (has_ipywidgets and has_ipython),
    reason="requires ipywidgets and IPython",
)
def test_async_status_widget_auto_display(monkeypatch):
    from pipefunc._widgets.async_status_widget import AsyncTaskStatusWidget

    displayed: list[object] = []

    def record_display(obj):
        displayed.append(obj)

    monkeypatch.setattr("IPython.display.display", record_display)

    widget = AsyncTaskStatusWidget(display=True)

    assert displayed
    assert displayed[0] is widget.widget


def test_maybe_async_task_status_widget_missing_ipywidgets(monkeypatch, capsys):
    from pipefunc._widgets import helpers

    monkeypatch.setattr(helpers, "has_ipywidgets", False)
    monkeypatch.setattr(helpers, "is_running_in_ipynb", lambda: True)

    result = helpers.maybe_async_task_status_widget(task=None)  # type: ignore[arg-type]

    captured = capsys.readouterr()
    assert "Install with: pip install ipywidgets" in captured.out
    assert result is None


def test_show_hide_widget():
    from pipefunc._widgets import helpers

    widget = types.SimpleNamespace(layout=types.SimpleNamespace(display="none"))

    helpers.show(widget)
    assert widget.layout.display == "block"

    helpers.hide(widget)
    assert widget.layout.display == "none"
