import asyncio
import time
from typing import Any, TypeAlias

import ipywidgets as widgets
from IPython.display import HTML, display

from pipefunc._utils import at_least_tuple
from pipefunc.map._run import _Status

_OUTPUT_TYPE: TypeAlias = str | tuple[str, ...]


def span(class_name: str, value: str) -> str:
    return f'<span class="{class_name}">{value}</span>'


def create_button(description: str, button_style: str, icon: str) -> widgets.Button:
    return widgets.Button(description=description, button_style=button_style, icon=icon)


def create_progress_bar(name: _OUTPUT_TYPE, progress: float) -> widgets.FloatProgress:
    return widgets.FloatProgress(
        value=progress,
        max=1.0,
        description=", ".join(at_least_tuple(name)),
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
        progress_dict: dict[_OUTPUT_TYPE, _Status],
        map_task: asyncio.Task[None] | None = None,
        *,
        target_progress_change: float = 0.1,
        auto_update: bool = True,
    ) -> None:
        self.map_task: asyncio.Task[None] | None = map_task
        self.progress_dict: dict[_OUTPUT_TYPE, _Status] = progress_dict
        self.target_progress_change: float = target_progress_change
        self.auto_update: bool = False
        self.auto_update_task: asyncio.Task | None = None
        self.first_update: bool = True

        # Initialize widgets for progress tracking
        self.progress_bars: dict[_OUTPUT_TYPE, widgets.FloatProgress] = {}
        self.labels: dict[_OUTPUT_TYPE, dict[_OUTPUT_TYPE, widgets.HTML]] = {}

        # Create control buttons
        self.buttons: dict[_OUTPUT_TYPE, widgets.Button] = {
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
        for name, status in self.progress_dict.items():
            self.progress_bars[name] = create_progress_bar(name, status.progress)
            self.labels[name] = {
                "percentage": create_html_label("percent-label", f"{status.progress * 100:.1f}%"),
                "estimated_time": create_html_label(
                    "estimate-label",
                    "Elapsed: 0.00 sec | ETA: calculating...",
                ),
                "speed": create_html_label("speed-label", "Speed: calculating..."),
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
        if self.all_completed():
            self.mark_completed()

    def _update_labels(
        self,
        name: _OUTPUT_TYPE,
        status: _Status,
    ) -> None:
        assert status.progress > 0
        label_dict = self.labels[name]
        iterations_label = f"✓ {status.n_completed:,} | ⏳ {status.n_left:,}"
        label_dict["percentage"].value = span(
            "percent-label",
            f"{status.progress * 100:.1f}% | {iterations_label}",
        )
        start_time = status.start_time
        if start_time is None:
            return
        elapsed_time = status.elapsed_time()
        if status.end_time is not None:
            eta = "Completed"
        elif status.progress == 0:
            eta = "Calculating..."
        else:
            estimated_time_left = (1.0 - status.progress) * (elapsed_time / status.progress)
            eta = f"ETA: {estimated_time_left:.2f} sec"
        speed = f"{status.n_completed / elapsed_time:,.2f}" if elapsed_time > 0 else "∞"
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
        while self.auto_update:
            self.update_progress(None)
            if self.first_update:
                new_interval = 1.0
                self.first_update = False
            else:
                new_interval = self._calculate_adaptive_interval_with_previous()

            # Check if all tasks are completed
            if self.all_completed():
                break

            # Update interval display
            if not self.first_update:
                self.auto_update_interval_label.value = span(
                    "interval-label",
                    f"Auto-update every: {new_interval:.2f} sec",
                )

            await asyncio.sleep(new_interval)

    def all_completed(self) -> bool:
        return all(status.progress >= 1.0 for status in self.progress_dict.values())

    def mark_completed(self) -> None:
        if self.auto_update:
            self.toggle_auto_update(None)
        self.auto_update_interval_label.value = span(
            "interval-label",
            "Completed all tasks 🎉",
        )
        for button in self.buttons.values():
            button.disabled = True

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
        if self.map_task is not None:
            self.map_task.cancel()

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
        for name in self.progress_dict:
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
