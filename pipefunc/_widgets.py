import asyncio
import time
from pathlib import Path
from typing import NamedTuple

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

    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.progress_dict: dict[str, float] = {
            name: self._calculate_progress(store).percentage
            for name, store in self.resource_manager.store.items()
        }
        self.start_times: dict[str, float | None] = {name: None for name in self.progress_dict}
        self.auto_update = False
        # Initialize widgets
        self.progress_bars = {}
        self.estimated_time_labels = {}
        self.percentage_labels = {}

        self.auto_update_interval_label = widgets.HTML(value="Auto-update every: Calculating...")

        self._setup_widgets()
        self._display_widgets()

        self.last_update_times: dict[str, float | None] = {
            name: None for name in self.progress_dict
        }
        self.last_progress: dict[str, float] = {name: 0.0 for name in self.progress_dict}

        self.toggle_auto_update(None)

    def _calculate_progress(self, resource) -> Status:
        """Calculate the progress for a given resource."""
        return progress(resource)

    def _setup_widgets(self):
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

    def update_progress(self, _):
        """Update the progress values and labels."""
        current_time = time.time()

        for name, store in self.resource_manager.store.items():
            status = self._calculate_progress(store)
            current_progress = status.percentage
            self.progress_dict[name] = current_progress

            # Update the progress bar
            progress_bar = self.progress_bars[name]
            progress_bar.value = current_progress

            # Update percentage label
            self.percentage_labels[
                name
            ].value = f'<span class="percent-label">{current_progress * 100:.1f}%</span>'

            iterations_done = int(status.total * current_progress)
            iterations_left = status.total - iterations_done
            iterations_label = f"✅ {iterations_done} | ⏰ {iterations_left}"

            # Update elapsed time and estimate
            if self.start_times[name] is not None:
                elapsed_time = current_time - self.start_times[name]
                estimated_time_left = (
                    (1.0 - current_progress) * (elapsed_time / current_progress)
                    if current_progress > 0
                    else float("inf")
                )
                self.estimated_time_labels[
                    name
                ].value = f'<span class="estimate-label">Estimated time left: {estimated_time_left:.2f} sec</span>'
            else:
                self.estimated_time_labels[
                    name
                ].value = '<span class="estimate-label">Estimated time left: calculating...</span>'

            # Update last known progress and times after calculations
            self.last_update_times[name] = current_time
            self.last_progress[name] = current_progress

            # Update description accurately
            progress_bar.description = f"{name} {iterations_label}"

        print(f"Updated progress at {current_time}")

    def _calculate_adaptive_interval_with_previous(
        self,
        previous_progress_dict: dict[str, float],
    ) -> float:
        """Calculate a dynamic interval based on progress changes for all resources."""
        min_interval = 0.1  # minimum interval to avoid extremely rapid updates
        max_interval = 3.0  # maximum interval to prevent excessively slow updates
        shortest_interval = max_interval

        current_time = time.time()

        for name, current_progress in self.progress_dict.items():
            if self.last_update_times[name] is None or current_progress <= 0:
                continue

            elapsed_time = current_time - self.last_update_times[name]
            progress_change = current_progress - previous_progress_dict[name]

            print(
                f"Resource: {name}, Current Progress: {current_progress:.4f}, "
                f"Previous Progress: {previous_progress_dict[name]:.4f}, "
                f"Elapsed Time: {elapsed_time:.4f}",
            )

            if progress_change > 0:
                estimated_time_for_5_percent = elapsed_time / (progress_change / 0.2)
                shortest_interval = min(shortest_interval, estimated_time_for_5_percent)

        calculated_interval = min(max(shortest_interval, min_interval), max_interval)
        print(f"Calculated Interval: {calculated_interval:.2f}")
        return calculated_interval

    async def _auto_update_progress(self, interval: float = 1.0):
        """Periodically update the progress."""
        while self.auto_update:
            previous_progress_dict = self.progress_dict.copy()

            self.update_progress(None)

            # Calculate interval based on the previous state
            new_interval = self._calculate_adaptive_interval_with_previous(previous_progress_dict)

            # Update interval display
            self.auto_update_interval_label.value = f"Auto-update every: {new_interval:.2f} sec"
            print(f"Auto-update interval: {new_interval:.2f} sec")

            # Update last known progress and times after calculations
            current_time = time.time()
            for name in self.progress_dict:
                self.last_update_times[name] = current_time
                self.last_progress[name] = self.progress_dict[name]

            await asyncio.sleep(new_interval)

    def toggle_auto_update(self, _):
        """Toggle the auto-update feature on or off."""
        self.auto_update = not self.auto_update
        self.toggle_auto_update_button.description = (
            "Stop Auto-Update" if self.auto_update else "Start Auto-Update"
        )
        if self.auto_update:
            asyncio.create_task(self._auto_update_progress(interval=1.0))

    def _display_widgets(self):
        """Display the progress widgets with styles."""

    def _display_widgets(self):
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

        progress_layout = widgets.VBox(
            [
                *[
                    widgets.VBox(
                        [
                            self.progress_bars[name],
                            widgets.HBox(
                                [self.percentage_labels[name], self.estimated_time_labels[name]],
                                layout=widgets.Layout(justify_content="space-between"),
                            ),
                        ],
                        layout=widgets.Layout(class_="progress-container"),
                    )
                    for name in self.progress_dict
                ],
                self.update_button,
                self.toggle_auto_update_button,
                self.auto_update_interval_label,  # Display interval label
            ],
            layout=widgets.Layout(max_width="800px"),
        )

        display(HTML(style))
        display(progress_layout)
