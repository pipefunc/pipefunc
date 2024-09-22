import asyncio
import time
from pathlib import Path
from typing import Any, NamedTuple

import ipywidgets as widgets
from IPython.display import HTML, display

from pipefunc._utils import prod
from pipefunc.map import StorageBase


class Status(NamedTuple):
    done: bool
    total: int
    percentage: float


def progress(r: Path | StorageBase) -> Status:
    if isinstance(r, Path):
        return Status(done=r.exists(), total=1, percentage=1.0)
    mask = r.mask
    left = mask.data.sum()
    total = prod(mask.shape)
    return Status(done=left == 0, total=total, percentage=1 - left / total)


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
        self.progress_dict: dict[str, float] = {name: 0.0 for name in self.resource_manager.store}
        self.start_times: dict[str, float | None] = {name: None for name in self.progress_dict}
        self.done_times: dict[str, float | None] = {name: None for name in self.progress_dict}
        self.auto_update: bool = False
        self.progress_bars: dict[str, widgets.FloatProgress] = {}
        self.estimated_time_labels: dict[str, widgets.HTML] = {}
        self.percentage_labels: dict[str, widgets.HTML] = {}
        self.speed_labels: dict[str, widgets.HTML] = {}
        self.auto_update_interval_label: widgets.HTML = widgets.HTML(
            value="Auto-update every: Calculating...",
        )
        self.update_button: widgets.Button
        self.toggle_auto_update_button: widgets.Button
        self.cancel_button: widgets.Button
        self.auto_update_task: asyncio.Task | None = None
        self.first_update: bool = True

        self._setup_widgets()
        self._display_widgets()

        if auto_update:
            self.toggle_auto_update(None)

    def _calculate_progress(self, resource: Any) -> Status:
        """Calculate the progress for a given resource."""
        return progress(resource)

    def _setup_widgets(self) -> None:
        """Initialize widgets for progress tracking."""
        self.update_button = widgets.Button(
            description="Update Progress",
            button_style="info",
            icon="refresh",
        )
        self.toggle_auto_update_button = widgets.Button(
            description="Start Auto-Update",
            button_style="success",
            icon="refresh",
        )
        self.cancel_button = widgets.Button(
            description="Cancel Calculation",
            button_style="danger",
            icon="stop",
        )

        for name in self.progress_dict:
            progress = self.progress_dict[name]
            self.progress_bars[name] = widgets.FloatProgress(
                value=progress,
                max=1.0,
                layout={"width": "95%"},
                bar_style="info",
                style={"description_width": "150px"},
            )
            self.percentage_labels[name] = widgets.HTML(
                value=f'<span class="percent-label">{progress * 100:.1f}%</span>',
            )
            self.estimated_time_labels[name] = widgets.HTML(
                value='<span class="estimate-label">Estimated time left: calculating...</span>',
            )
            self.speed_labels[name] = widgets.HTML(
                value='<span class="speed-label">Speed: calculating...</span>',
            )

            self.progress_bars[name].add_class("animated-progress")
        self.auto_update_interval_label = widgets.HTML(
            value='<span class="interval-label">Auto-update every: N/A</span>',
        )
        self.update_button.on_click(self.update_progress)
        self.toggle_auto_update_button.on_click(self.toggle_auto_update)
        self.cancel_button.on_click(self.cancel_calculation)

    def update_progress(self, _: Any) -> None:
        """Update the progress values and labels."""
        current_time = time.time()

        for name, store in self.resource_manager.store.items():
            t = time.time()
            if self.done_times[name] is not None:
                continue
            status = self._calculate_progress(store)
            print(f"Calculated progress for {name} in {(time.time() - t)*1e6:.2f} μs")
            current_progress = status.percentage
            if self.start_times[name] is None and current_progress > 0:
                self.start_times[name] = time.time()
            if self.done_times[name] is None and current_progress >= 1.0:
                self.done_times[name] = time.time()
            self.progress_dict[name] = current_progress

            # Update the progress bar
            progress_bar = self.progress_bars[name]
            progress_bar.value = current_progress

            # Update animation class based on progress
            if current_progress >= 1.0:
                progress_bar.bar_style = "success"
                progress_bar.remove_class("animated-progress")
                progress_bar.add_class("completed-progress")
            else:
                progress_bar.remove_class("completed-progress")
                progress_bar.add_class("animated-progress")

            # Update percentage label
            iterations_done = int(status.total * current_progress)
            iterations_left = status.total - iterations_done
            iterations_label = f"✓ {iterations_done:,} | ⏳ {iterations_left:,}"
            label = f'<span class="percent-label">{current_progress * 100:.1f}% | {iterations_label}</span>'
            self.percentage_labels[name].value = label

            # Update elapsed time and estimate
            start_time = self.start_times[name]
            end_time = self.done_times[name]
            if start_time is not None:
                if end_time is not None:
                    elapsed_time = end_time - start_time
                    label = f'<span class="estimate-label">Elapsed time: {elapsed_time:.2f} sec | Completed</span>'
                else:
                    elapsed_time = current_time - start_time
                    estimated_time_left = (
                        (1.0 - current_progress) * (elapsed_time / current_progress)
                        if current_progress > 0
                        else float("inf")
                    )
                    label = f'<span class="estimate-label">Elapsed: {elapsed_time:.2f} sec | ETA: {estimated_time_left:.2f} sec</span>'

                # Calculate and update speed
                speed = iterations_done / elapsed_time if elapsed_time > 0 else float("inf")
                speed_label = f'<span class="speed-label">Speed: {speed:,.2f} iterations/sec</span>'
                self.speed_labels[name].value = speed_label
            else:
                label = (
                    '<span class="estimate-label">Elapsed: 0.00 sec | ETA: calculating...</span>'
                )
            self.estimated_time_labels[name].value = label

            # Update description accurately
            progress_bar.description = f"{name}"

        print(f"Updated progress at {current_time}")

    def _calculate_adaptive_interval_with_previous(self) -> float:
        """Calculate a dynamic interval based on progress changes for all resources."""
        min_interval = 0.1  # minimum interval to avoid extremely rapid updates
        max_interval = 10.0  # maximum interval to prevent excessively slow updates
        shortest_interval = max_interval

        current_time = time.time()
        print(f"Current progress dict: {self.progress_dict}")
        for name, current_progress in self.progress_dict.items():
            if current_progress <= 0 or current_progress >= 1:
                continue
            start_time = self.start_times[name]
            assert start_time is not None
            elapsed_time = current_time - start_time
            progress_rate = current_progress / elapsed_time
            estimated_time_for_target = self.target_progress_change / progress_rate

            print(
                f"Resource: {name}, Current Progress: {current_progress:.4f}, "
                f"Elapsed Time: {elapsed_time:.4f}, "
                f"Progress Rate: {progress_rate:.4f} per second, "
                f"Estimated Time for Target: {estimated_time_for_target:.2f} seconds",
            )

            # Estimate time for target progress change
            shortest_interval = min(shortest_interval, estimated_time_for_target)

        calculated_interval = min(max(shortest_interval, min_interval), max_interval)
        print(f"Calculated Interval: {calculated_interval:.2f}")
        return calculated_interval

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
                self.auto_update_interval_label.value = f"Auto-update every: {new_interval:.2f} sec"
                print(f"Auto-update interval: {new_interval:.2f} sec")

            # Check if all tasks are completed
            if all(progress >= 1.0 for progress in self.progress_dict.values()):
                self.toggle_auto_update(None)
                self.auto_update_interval_label.value = "Auto-update every: N/A"
                break
            await asyncio.sleep(new_interval)

    def toggle_auto_update(self, _: Any) -> None:
        """Toggle the auto-update feature on or off."""
        self.auto_update = not self.auto_update
        self.toggle_auto_update_button.description = (
            "Stop Auto-Update" if self.auto_update else "Start Auto-Update"
        )
        self.toggle_auto_update_button.button_style = "danger" if self.auto_update else "success"
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
            print("Calculation cancelled.")
        else:
            print("No ongoing calculation to cancel.")

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
        # Create individual progress containers for each item in progress_dict
        progress_containers = []
        for name in self.progress_dict:
            progress_bar = self.progress_bars[name]
            percentage_label = self.percentage_labels[name]
            estimated_time_label = self.estimated_time_labels[name]
            speed_label = self.speed_labels[name]

            # Create a horizontal box for labels
            labels_box = widgets.HBox(
                [percentage_label, estimated_time_label, speed_label],
                layout=widgets.Layout(justify_content="space-between"),
            )

            # Create a vertical box for the progress bar and labels
            container = widgets.VBox(
                [progress_bar, labels_box],
                layout=widgets.Layout(class_="progress"),
            )

            progress_containers.append(container)

        # Create the main vertical box layout
        progress_layout = widgets.VBox(
            [
                *progress_containers,
                widgets.HBox(
                    [self.update_button, self.toggle_auto_update_button, self.cancel_button],
                    layout=widgets.Layout(class_="widget-button", justify_content="center"),
                ),
                self.auto_update_interval_label,
            ],
            layout=widgets.Layout(max_width="600px"),
        )

        display(HTML(style))
        display(progress_layout)
