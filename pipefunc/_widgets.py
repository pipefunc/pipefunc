import asyncio
import time
from pathlib import Path
from typing import Any, NamedTuple

import ipywidgets as widgets
from IPython.display import HTML, display

from pipefunc._utils import prod
from pipefunc.map import StorageBase


class Status(NamedTuple):
    left: int
    total: int
    percentage: float

    @property
    def done(self) -> int:
        return self.total - self.left

    @property
    def is_done(self) -> int:
        return self.total - self.left


def progress(r: Path | StorageBase) -> Status:
    if isinstance(r, Path):
        return Status(left=0 if r.exists() else 1, total=1, percentage=1.0)
    mask = r.mask
    left = mask.data.sum()
    total = prod(mask.shape)
    return Status(left=left, total=total, percentage=1 - left / total)


def span(class_name: str, value: str) -> str:
    return f'<span class="{class_name}">{value}</span>'


def create_button(description: str, button_style: str, icon: str) -> widgets.Button:
    return widgets.Button(
        description=description,
        button_style=button_style,
        icon=icon,
    )


def create_progress_bar(name: str, progress: float) -> widgets.FloatProgress:
    return widgets.FloatProgress(
        value=progress,
        max=1.0,
        description=name,
        layout={"width": "95%", "class_": "animated-progress"},
        bar_style="info",
        style={"description_width": "150px"},
    )


def create_html_label(class_name: str, initial_value: str) -> widgets.HTML:
    return widgets.HTML(value=span(class_name, initial_value))


class ProgressTracker:
    """Class to track progress and display it with ipywidgets."""

    def __init__(
        self,
        resource_manager: Any,
        *,
        target_progress_change: float = 0.1,
        auto_update: bool = True,
    ) -> None:
        self.resource_manager: Any = resource_manager
        self.target_progress_change: float = target_progress_change
        self.progress: dict[str, float] = {name: 0.0 for name in self.resource_manager.store}
        self.start_times: dict[str, float | None] = {name: None for name in self.progress}
        self.done_times: dict[str, float | None] = {name: None for name in self.progress}
        self.auto_update: bool = False
        self.auto_update_task: asyncio.Task | None = None
        self.first_update: bool = True

        # Initialize widgets for progress tracking
        self.progress_bars: dict[str, widgets.FloatProgress] = {}
        self.labels: dict[str, dict[str, widgets.HTML]] = {}

        # Create control buttons
        self.buttons: dict[str, widgets.Button] = {
            "update": create_button("Update Progress", "info", "refresh"),
            "toggle_auto_update": create_button(
                "Start Auto-Update",
                "success",
                "refresh",
            ),
            "cancel": create_button("Cancel Calculation", "danger", "stop"),
        }
        self.buttons["update"].on_click(self.update_progress)
        self.buttons["toggle_auto_update"].on_click(self.toggle_auto_update)
        self.buttons["cancel"].on_click(self.cancel_calculation)

        # Create progress widgets
        for name, progress in self.progress.items():
            self.progress_bars[name] = create_progress_bar(name, progress)
            self.labels[name] = {
                "percentage": create_html_label(
                    "percent-label",
                    f"{progress * 100:.1f}%",
                ),
                "estimated_time": create_html_label(
                    "estimate-label",
                    "Elapsed: 0.00 sec | ETA: calculating...",
                ),
                "speed": create_html_label(
                    "speed-label",
                    "Speed: calculating...",
                ),
            }

        # Create auto-update label
        self.auto_update_interval_label = create_html_label(
            "interval-label",
            "Auto-update every: N/A",
        )

        self._display_widgets()

        if auto_update:
            self.toggle_auto_update(None)

    def update_progress(self, _: Any) -> None:
        """Update the progress values and labels."""
        current_time = time.time()
        for name, store in self.resource_manager.store.items():
            if self.done_times[name] is not None:
                continue
            status = progress(store)
            current_progress = status.percentage
            if self.start_times[name] is None and current_progress > 0:
                self.start_times[name] = current_time
            if self.done_times[name] is None and current_progress >= 1.0:
                self.done_times[name] = current_time
            self.progress[name] = current_progress
            progress_bar = self.progress_bars[name]
            progress_bar.value = current_progress
            if current_progress >= 1.0:
                progress_bar.bar_style = "success"
                progress_bar.remove_class("animated-progress")
                progress_bar.add_class("completed-progress")
            else:
                progress_bar.remove_class("completed-progress")
                progress_bar.add_class("animated-progress")
            self._update_labels(name, current_time, status)

    def _update_labels(
        self,
        name: str,
        current_time: float,
        status: Status,
    ) -> None:
        label_dict = self.labels[name]
        iterations_label = f"✓ {status.done:,} | ⏳ {status.left:,}"
        label_dict["percentage"].value = span(
            "percent-label",
            f"{status.percentage * 100:.1f}% | {iterations_label}",
        )
        start_time = self.start_times[name]
        if start_time is None:
            return
        end_time = self.done_times[name]
        elapsed_time = (end_time or current_time) - start_time
        if end_time is not None:
            eta = "Completed"
        else:
            estimated_time_left = (1.0 - status.percentage) * (elapsed_time / status.percentage)
            eta = f"ETA: {estimated_time_left:.2f} sec"
        speed = f"{status.done / elapsed_time:,.2f}" if elapsed_time > 0 else "∞"
        label_dict["speed"].value = span("speed-label", f"Speed: {speed} iterations/sec")
        label_dict["estimated_time"].value = span(
            "estimate-label",
            f"Elapsed: {elapsed_time:.2f} sec | {eta}",
        )

    def _calculate_adaptive_interval_with_previous(self) -> float:
        """Calculate a dynamic interval based on progress changes for all resources."""
        min_interval = 0.1  # minimum interval to avoid extremely rapid updates
        max_interval = 10.0  # maximum interval to prevent excessively slow updates
        shortest_interval = max_interval
        current_time = time.time()
        for name, current_progress in self.progress.items():
            if current_progress <= 0 or current_progress >= 1:
                continue
            start_time = self.start_times[name]
            assert start_time is not None
            elapsed_time = current_time - start_time
            progress_rate = current_progress / elapsed_time
            estimated_time_for_target = self.target_progress_change / progress_rate
            # Estimate time for target progress change
            shortest_interval = min(shortest_interval, estimated_time_for_target)
        return min(max(shortest_interval, min_interval), max_interval)

    async def _auto_update_progress(self) -> None:
        """Periodically update the progress."""
        while self.auto_update:
            self.update_progress(None)
            if self.first_update:
                new_interval = 1.0
                self.first_update = False
            else:
                new_interval = self._calculate_adaptive_interval_with_previous()

            # Update interval display
            if not self.first_update:
                self.auto_update_interval_label.value = span(
                    "interval-label",
                    f"Auto-update every: {new_interval:.2f} sec",
                )

            # Check if all tasks are completed
            if all(progress >= 1.0 for progress in self.progress.values()):
                self.toggle_auto_update(None)
                self.auto_update_interval_label.value = span(
                    "interval-label",
                    "Auto-update every: N/A",
                )
                for button in self.buttons.values():
                    button.disabled = True
                break
            await asyncio.sleep(new_interval)

    def toggle_auto_update(self, _: Any) -> None:
        """Toggle the auto-update feature on or off."""
        self.auto_update = not self.auto_update
        self.buttons["toggle_auto_update"].description = (
            "Stop Auto-Update" if self.auto_update else "Start Auto-Update"
        )
        self.buttons["toggle_auto_update"].button_style = (
            "danger" if self.auto_update else "success"
        )
        if self.auto_update:
            self.first_update = True
            self.auto_update_task = asyncio.create_task(self._auto_update_progress())
        elif self.auto_update_task is not None:
            self.auto_update_task.cancel()
            self.auto_update_task = None

    def cancel_calculation(self, _: Any) -> None:
        """Cancel the ongoing calculation."""
        if self.resource_manager.task is not None:
            self.resource_manager.task.cancel()

            # Update progress one last time
            self.update_progress(None)

            # Disable auto-update
            if self.auto_update:
                self.toggle_auto_update(None)

            # Disable all buttons
            for button in self.buttons.values():
                button.disabled = True

            # Stop animation and set bar style to danger for in-progress bars
            for progress_bar in self.progress_bars.values():
                if progress_bar.value < 1.0:
                    progress_bar.bar_style = "danger"
                    progress_bar.remove_class("animated-progress")
                    progress_bar.add_class("completed-progress")

            self.auto_update_interval_label.value = span(
                "interval-label",
                "Calculation cancelled",
            )

    def _display_widgets(self) -> None:
        """Display the progress widgets with styles."""
        style = """
        <style>
            .progress {
                border-radius: 5px;
            }
            .animated-progress .progress-bar, .completed-progress .progress-bar {
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
                border-radius: 5px;
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
                margin-top: 10px;
            }
        </style>
        """
        # Create individual progress containers for each item in progress
        progress_containers = []
        for name in self.progress:
            # Create a horizontal box for labels
            labels = self.labels[name]
            labels_box = widgets.HBox(
                [labels["percentage"], labels["estimated_time"], labels["speed"]],
                layout=widgets.Layout(justify_content="space-between"),
            )
            # Create a vertical box for the progress bar and labels
            container = widgets.VBox(
                [self.progress_bars[name], labels_box],
                layout=widgets.Layout(class_="progress"),
            )
            progress_containers.append(container)

        # Create the main vertical box layout
        buttons = self.buttons
        button_box = widgets.HBox(
            [buttons["update"], buttons["toggle_auto_update"], buttons["cancel"]],
            layout=widgets.Layout(class_="widget-button", justify_content="center"),
        )
        progress_layout = widgets.VBox(
            [*progress_containers, button_box, self.auto_update_interval_label],
            layout=widgets.Layout(max_width="600px"),
        )

        display(HTML(style))
        display(progress_layout)
