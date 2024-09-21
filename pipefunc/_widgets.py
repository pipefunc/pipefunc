import asyncio
import time
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML, display

from pipefunc._utils import prod
from pipefunc.map import StorageBase


def progress(r: Path | StorageBase) -> float:
    if isinstance(r, Path):
        r = r.exists()
        return 1.0
    mask = r.mask
    left = mask.data.sum()
    total = prod(mask.shape)
    return 1 - left / total


class ProgressTracker:
    """Class to track progress and display it with ipywidgets."""

    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.progress_dict: dict[str, float] = {
            name: self._calculate_progress(store)
            for name, store in self.resource_manager.store.items()
        }
        self.start_times: dict[str, float | None] = {name: None for name in self.progress_dict}
        self.progress_bars = {}
        self.estimated_time_labels = {}
        self.percentage_labels = {}
        self.auto_update = False

        self._setup_widgets()
        self._display_widgets()

    def _calculate_progress(self, resource) -> float:
        """Calculate the progress for a given resource."""
        return progress(resource)

    async def _auto_update_progress(self, interval: float):
        """Periodically update the progress."""
        while self.auto_update:
            self.update_progress(None)
            await asyncio.sleep(interval)

    def _setup_widgets(self):
        """Initialize widgets for progress tracking."""
        self.update_button = widgets.Button(description="Update Progress")
        self.toggle_auto_update_button = widgets.Button(description="Start Auto-Update")

        for i, name in enumerate(self.progress_dict):
            color_class = "row-even" if i % 2 == 0 else "row-odd"
            self.progress_bars[name] = widgets.FloatProgress(
                value=self.progress_dict[name],
                max=1.0,
                description=name,
            )
            self.percentage_labels[name] = widgets.HTML(
                value=f'<span class="percent-label">{self.progress_dict[name] * 100:.1f}%</span>',
            )
            self.estimated_time_labels[name] = widgets.HTML(
                value='<span class="estimate-label">Estimated time left: calculating...</span>',
            )

            self.progress_bars[name].layout.class_ = f"progress-container {color_class}"

        self.update_button.on_click(self.update_progress)
        self.toggle_auto_update_button.on_click(self.toggle_auto_update)

    def update_progress(self, _):
        """Update the progress values and labels."""
        self.progress_dict = {
            name: self._calculate_progress(store)
            for name, store in self.resource_manager.store.items()
        }

        current_time = time.time()

        for name, progress_bar in self.progress_bars.items():
            current_progress = self.progress_dict[name]

            if self.start_times[name] is None and current_progress > 0:
                self.start_times[name] = current_time

            progress_bar.value = current_progress
            self.percentage_labels[
                name
            ].value = f'<span class="percent-label">{current_progress * 100:.1f}%</span>'

            if self.start_times[name] is not None:
                elapsed_time = current_time - self.start_times[name]
                estimated_time_left = (1.0 - current_progress) * (elapsed_time / current_progress)
                self.estimated_time_labels[
                    name
                ].value = f'<span class="estimate-label">Estimated time left: {estimated_time_left:.2f} sec</span>'
            else:
                self.estimated_time_labels[
                    name
                ].value = '<span class="estimate-label">Estimated time left: calculating...</span>'

    def toggle_auto_update(self, _):
        """Toggle the auto-update feature on or off."""
        self.auto_update = not self.auto_update
        self.toggle_auto_update_button.description = (
            "Stop Auto-Update" if self.auto_update else "Start Auto-Update"
        )
        if self.auto_update:
            asyncio.create_task(self._auto_update_progress(interval=2.0))

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
            ],
            layout=widgets.Layout(max_width="800px"),
        )

        display(HTML(style))
        display(progress_layout)
