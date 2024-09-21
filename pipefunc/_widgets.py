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
        self.progress_dict: dict[str, float] = {
            name: self._calculate_progress(store).percentage
            for name, store in self.resource_manager.store.items()
        }
        self.start_times: dict[str, float | None] = {name: None for name in self.progress_dict}
        self.done_times: dict[str, float | None] = {name: None for name in self.progress_dict}
        self.auto_update: bool = False
        self.progress_bars: dict[str, widgets.FloatProgress] = {}
        self.estimated_time_labels: dict[str, widgets.HTML] = {}
        self.percentage_labels: dict[str, widgets.HTML] = {}
        self.auto_update_interval_label: widgets.HTML = widgets.HTML(
            value="Auto-update every: Calculating...",
        )
        self.update_button: widgets.Button
        self.toggle_auto_update_button: widgets.Button
        self.auto_update_task: asyncio.Task | None = None

        self._setup_widgets()
        self._display_widgets()

        self.last_update_times: dict[str, float | None] = {
            name: None for name in self.progress_dict
        }
        self.last_progress: dict[str, float] = {name: 0.0 for name in self.progress_dict}
        if auto_update:
            self.toggle_auto_update(None)

    def _calculate_progress(self, resource: Any) -> Status:
        """Calculate the progress for a given resource."""
        return progress(resource)

    def _setup_widgets(self) -> None:
        """Initialize widgets for progress tracking."""
        self.update_button = widgets.Button(description="Update Progress")
        self.toggle_auto_update_button = widgets.Button(description="Start Auto-Update")

        for i, name in enumerate(self.progress_dict):
            color_class = "row-even" if i % 2 == 0 else "row-odd"
            progress = self.progress_dict[name]
            self.progress_bars[name] = widgets.FloatProgress(
                value=progress,
                max=1.0,
                layout={"width": "600px"},
                bar_style="info" if progress < 1 else "success",
            )
            self.percentage_labels[name] = widgets.HTML(
                value=f'<span class="percent-label">{progress * 100:.1f}%</span>',
            )
            self.estimated_time_labels[name] = widgets.HTML(
                value='<span class="estimate-label">Estimated time left: calculating...</span>',
            )

            self.progress_bars[name].add_class(color_class)  # Use add_class method
        self.auto_update_interval_label = widgets.HTML(
            value='<span class="interval-label">Auto-update every: N/A</span>',
        )
        self.update_button.on_click(self.update_progress)
        self.toggle_auto_update_button.on_click(self.toggle_auto_update)

    def update_progress(self, _: Any) -> None:
        """Update the progress values and labels."""
        current_time = time.time()

        for name, store in self.resource_manager.store.items():
            t = time.time()
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

            # Update percentage label
            label = f'<span class="percent-label">{current_progress * 100:.1f}%</span>'
            self.percentage_labels[name].value = label

            iterations_done = int(status.total * current_progress)
            iterations_left = status.total - iterations_done
            iterations_label = f"✅ {iterations_done} | ⏰ {iterations_left}"

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
                    label = f'<span class="estimate-label">Elapsed time: {elapsed_time:.2f} sec | Estimated time left: {estimated_time_left:.2f} sec</span>'
            else:
                label = '<span class="estimate-label">Elapsed time: 0.00 sec | Estimated time left: calculating...</span>'
            self.estimated_time_labels[name].value = label
            # Update last known progress and times after calculations
            self.last_update_times[name] = current_time
            self.last_progress[name] = current_progress

            # Update description accurately
            progress_bar.description = f"{name} {iterations_label}"

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

            # Calculate interval based on the previous state
            new_interval = self._calculate_adaptive_interval_with_previous()

            # Update interval display
            self.auto_update_interval_label.value = f"Auto-update every: {new_interval:.2f} sec"
            print(f"Auto-update interval: {new_interval:.2f} sec")

            # Update last known progress and times after calculations
            current_time = time.time()
            for name in self.progress_dict:
                self.last_update_times[name] = current_time
                self.last_progress[name] = self.progress_dict[name]

            await asyncio.sleep(new_interval)

    def toggle_auto_update(self, _: Any) -> None:
        """Toggle the auto-update feature on or off."""
        self.auto_update = not self.auto_update
        self.toggle_auto_update_button.description = (
            "Stop Auto-Update" if self.auto_update else "Start Auto-Update"
        )
        if self.auto_update:
            self.auto_update_task = asyncio.create_task(self._auto_update_progress())
        else:
            assert self.auto_update_task is not None
            self.auto_update_task.cancel()
            self.auto_update_task = None

    def _display_widgets(self) -> None:
        """Display the progress widgets with styles."""
        style = """
        <style>
            .progress-container {
                margin-bottom: 10px;
                padding: 5px;
                border: 1px solid lightgray;
                border-radius: 5px;
            }
            .row-even {
                background-color: #f8f8f8;
            }
            .row-odd {
                background-color: #ffffff;
            }
            .percent-label {
                margin-left: 10px;
                font-weight: bold;
            }
            .estimate-label {
                font-style: italic;
                color: grey;
            }
            .interval-label {
                font-weight: bold;
                color: darkblue;
            }
        </style>
        """
        # Create individual progress containers for each item in progress_dict
        progress_containers = []
        for name in self.progress_dict:
            progress_bar = self.progress_bars[name]
            percentage_label = self.percentage_labels[name]
            estimated_time_label = self.estimated_time_labels[name]

            # Create a horizontal box for labels
            labels_box = widgets.HBox(
                [percentage_label, estimated_time_label],
                layout=widgets.Layout(justify_content="space-between"),
            )

            # Create a vertical box for the progress bar and labels
            container = widgets.VBox(
                [progress_bar, labels_box],
                layout=widgets.Layout(class_="progress-container"),
            )

            progress_containers.append(container)

        # Create the main vertical box layout
        progress_layout = widgets.VBox(
            [
                *progress_containers,
                self.update_button,
                self.toggle_auto_update_button,
                self.auto_update_interval_label,
            ],
            layout=widgets.Layout(max_width="800px"),
        )

        display(HTML(style))
        display(progress_layout)
