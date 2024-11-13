from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import IPython.display
import ipywidgets as widgets

from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._progress import Status


def _span(class_name: str, value: str) -> str:
    return f'<span class="{class_name}">{value}</span>'


def _create_button(
    description: str,
    button_style: str,
    icon: str,
    on_click: Callable[[Any], None],
) -> widgets.Button:
    button = widgets.Button(description=description, button_style=button_style, icon=icon)
    button.on_click(on_click)
    return button


def _create_progress_bar(name: OUTPUT_TYPE, progress: float) -> widgets.FloatProgress:
    return widgets.FloatProgress(
        value=progress,
        max=1.0,
        description=", ".join(at_least_tuple(name)),
        layout={"width": "95%"},
        bar_style="info",
        style={"description_width": "150px"},
    )


def _create_html_label(class_name: str, initial_value: str) -> widgets.HTML:
    return widgets.HTML(value=_span(class_name, initial_value))


class ProgressTracker:
    """Class to track progress and display it with ipywidgets."""

    def __init__(
        self,
        progress_dict: dict[OUTPUT_TYPE, Status],
        task: asyncio.Task[Any] | None = None,
        *,
        target_progress_change: float = 0.05,
        auto_update: bool = True,
        display: bool = True,
        in_async: bool = True,
    ) -> None:
        self.task: asyncio.Task[None] | None = task
        self.progress_dict: dict[OUTPUT_TYPE, Status] = progress_dict
        self.target_progress_change: float = target_progress_change
        self.auto_update: bool = auto_update
        self.auto_update_task: asyncio.Task | None = None
        self.in_async: bool = in_async
        self.last_update_time: float = 0.0
        self._min_auto_update_interval: float = 0.1
        self._max_auto_update_interval: float = 10.0
        self._first_auto_update_interval: float = 1.0
        self._sync_update_interval: float = 0.1
        self.progress_bars: dict[OUTPUT_TYPE, widgets.FloatProgress] = {}
        self.labels: dict[OUTPUT_TYPE, dict[OUTPUT_TYPE, widgets.HTML]] = {}
        self.buttons: dict[OUTPUT_TYPE, widgets.Button] = {
            "update": _create_button(
                description="Update Progress",
                button_style="info",
                icon="refresh",
                on_click=self.update_progress,
            ),
            "toggle_auto_update": _create_button(
                description="Start Auto-Update",
                button_style="success",
                icon="refresh",
                on_click=self._toggle_auto_update,
            ),
            "cancel": _create_button(
                description="Cancel Calculation",
                button_style="danger",
                icon="stop",
                on_click=self._cancel_calculation,
            ),
        }
        for name, status in self.progress_dict.items():
            self.progress_bars[name] = _create_progress_bar(name, status.progress)
            self.labels[name] = {
                "percentage": _create_html_label("percent-label", f"{status.progress * 100:.1f}%"),
                "estimated_time": _create_html_label(
                    "estimate-label",
                    "Elapsed: 0.00 sec | ETA: Calculating...",
                ),
                "speed": _create_html_label("speed-label", "Speed: Calculating..."),
            }
        self.auto_update_interval_label = _create_html_label(
            "interval-label",
            "Auto-update every: N/A",
        )
        self._initial_update_period: float = 30.0
        self._initial_max_update_interval: float = 1.0
        self.start_time: float = 0.0
        if display:
            self.display()
        if self.task is not None:
            self._set_auto_update(auto_update)

    def attach_task(self, task: asyncio.Task[Any]) -> None:
        """Attach a new task to the progress tracker."""
        self.task = task
        self._set_auto_update(self.auto_update)

    def update_progress(self, _: Any = None, *, force: bool = False) -> None:
        """Update the progress values and labels."""
        if not self.in_async and not force:
            assert self.task is None
            # If not in asyncio, `update_progress` is called after each iteration,
            # so, we throttle the updates to avoid excessive updates.
            now = time.monotonic()
            if now - self.last_update_time < self._sync_update_interval:
                return
            self.last_update_time = time.monotonic()

        for name, status in self.progress_dict.items():
            if status.progress == 0:
                continue
            progress_bar = self.progress_bars[name]
            progress_bar.value = status.progress
            if status.progress >= 1.0:
                progress_bar.bar_style = "success"
                progress_bar.remove_class("animated-progress")
                progress_bar.add_class("completed-progress")
            else:
                progress_bar.remove_class("completed-progress")
                progress_bar.add_class("animated-progress")
            self._update_labels(name, status)
        if self._all_completed():
            self._mark_completed()

    def _update_labels(self, name: OUTPUT_TYPE, status: Status) -> None:
        assert status.progress > 0
        labels = self.labels[name]
        iterations_label = f"✓ {status.n_completed:,} | ⏳ {status.n_left:,}"
        labels["percentage"].value = _span(
            "percent-label",
            f"{status.progress * 100:.1f}% | {iterations_label}",
        )
        elapsed_time = status.elapsed_time()
        if status.end_time is not None:
            eta = "Completed"
        else:
            estimated_time_left = (1.0 - status.progress) * (elapsed_time / status.progress)
            eta = f"ETA: {estimated_time_left:.2f} sec"
        speed = f"{status.n_completed / elapsed_time:,.2f}" if elapsed_time > 0 else "∞"
        labels["speed"].value = _span("speed-label", f"Speed: {speed} iterations/sec")
        labels["estimated_time"].value = _span(
            "estimate-label",
            f"Elapsed: {elapsed_time:.2f} sec | {eta}",
        )

    def _calculate_adaptive_interval_with_previous(self) -> float:
        """Calculate a dynamic interval based on progress changes for all resources."""
        min_interval = self._min_auto_update_interval
        max_interval = self._max_auto_update_interval
        shortest_interval = max_interval
        current_time = time.monotonic()
        for status in self.progress_dict.values():
            if status.progress <= 0 or status.progress >= 1:
                continue
            assert status.start_time is not None
            elapsed_time = current_time - status.start_time
            progress_rate = status.progress / elapsed_time
            estimated_time_for_target = self.target_progress_change / progress_rate
            # Estimate time for target progress change
            shortest_interval = min(shortest_interval, estimated_time_for_target)
        return min(max(shortest_interval, min_interval), max_interval)

    async def _auto_update_progress(self) -> None:
        """Periodically update the progress."""
        self.start_time = time.monotonic()
        while self.auto_update:
            self.update_progress()
            current_time = time.monotonic()
            elapsed_since_start = current_time - self.start_time

            new_interval = self._calculate_adaptive_interval_with_previous()
            if elapsed_since_start <= self._initial_update_period:
                new_interval = min(new_interval, self._initial_max_update_interval)

            if self._all_completed():
                break

            self.auto_update_interval_label.value = _span(
                "interval-label",
                f"Auto-update every: {new_interval:.2f} sec",
            )
            await asyncio.sleep(new_interval)

    def _all_completed(self) -> bool:
        return all(status.progress >= 1.0 for status in self.progress_dict.values())

    def _mark_completed(self) -> None:
        if self.auto_update:
            self._toggle_auto_update()
        self.auto_update_interval_label.value = _span("interval-label", "Completed all tasks 🎉")
        for button in self.buttons.values():
            button.disabled = True

    def _toggle_auto_update(self, _: Any = None) -> None:
        """Toggle the auto-update feature on or off."""
        self._set_auto_update(not self.auto_update)

    def _set_auto_update(self, value: bool) -> None:  # noqa: FBT001
        """Set the auto-update feature to the given value."""
        self.auto_update = value
        self.buttons["toggle_auto_update"].description = (
            "Stop Auto-Update" if self.auto_update else "Start Auto-Update"
        )
        self.buttons["toggle_auto_update"].button_style = (
            "danger" if self.auto_update else "success"
        )
        if self.auto_update:
            self.auto_update_task = asyncio.create_task(self._auto_update_progress())
        elif self.auto_update_task is not None:
            self.auto_update_task.cancel()
            self.auto_update_task = None

    def _cancel_calculation(self, _: Any) -> None:
        """Cancel the ongoing calculation."""
        assert self.task is not None
        self.task.cancel()
        self.update_progress()  # Update progress one last time
        if self.auto_update:
            self._toggle_auto_update()
        for button in self.buttons.values():
            button.disabled = True
        for progress_bar in self.progress_bars.values():
            if progress_bar.value < 1.0:
                progress_bar.bar_style = "danger"
                progress_bar.remove_class("animated-progress")
                progress_bar.add_class("completed-progress")
        self.auto_update_interval_label.value = _span("interval-label", "Calculation cancelled ❌")

    def _widgets(self) -> widgets.VBox:
        """Display the progress widgets with styles."""
        progress_containers = []
        for name in self.progress_dict:
            labels = self.labels[name]
            labels_box = widgets.HBox(
                [labels["percentage"], labels["estimated_time"], labels["speed"]],
                layout=widgets.Layout(justify_content="space-between"),
            )
            container = widgets.VBox(
                [self.progress_bars[name], labels_box],
                layout=widgets.Layout(border="1px solid #999999", margin="2px 0", padding="2px"),
            )
            container.add_class("container")
            progress_containers.append(container)

        buttons = self.buttons
        button_box = widgets.HBox(
            [buttons["update"], buttons["toggle_auto_update"], buttons["cancel"]],
            layout=widgets.Layout(justify_content="center"),
        )
        parts = (
            [*progress_containers, button_box, self.auto_update_interval_label]
            if self.task
            else progress_containers
        )
        return widgets.VBox(parts, layout=widgets.Layout(max_width="700px"))

    def display(self) -> None:
        style = """
        <style>
            .progress {
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .progress-bar {
                border-radius: 5px;
                transition: width 0.4s ease;
                background-image: linear-gradient(
                    -45deg,
                    rgba(255, 255, 255, 0.15) 25%,
                    transparent 25%,
                    transparent 50%,
                    rgba(255, 255, 255, 0.15) 50%,
                    rgba(255, 255, 255, 0.15) 75%,
                    transparent 75%,
                    transparent
                );
                background-size: 40px 40px;
            }
            .animated-progress .progress-bar {
                animation: stripes 1s linear infinite;
            }
            .completed-progress .progress-bar {
                animation: none;
            }
            @keyframes stripes {
                0% {
                    background-position: 0 0;
                }
                100% {
                    background-position: 40px 0;
                }
            }
            .container {
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .percent-label {
                margin-left: 10px;
                font-weight: bold;
                color: #3366cc;
            }
            .estimate-label {
                font-style: italic;
                color: #666;
            }
            .speed-label {
                font-weight: bold;
                color: #009900;
            }
            .interval-label {
                font-weight: bold;
                color: #990000;
            }
            .widget-label {
                margin-bottom: 5px;
                color: #333;
                font-family: monospace;
            }
            .widget-button {
                margin-top: 5px;
            }
        </style>
        """
        IPython.display.display(IPython.display.HTML(style))
        IPython.display.display(self._widgets())
