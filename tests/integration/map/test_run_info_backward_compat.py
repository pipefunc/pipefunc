from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map._run_info import RunInfo

if TYPE_CHECKING:
    from pathlib import Path


def test_run_info_loads_missing_error_handling(tmp_path: Path) -> None:
    """Simulate loading a run_info.json generated before error_handling existed."""

    @pipefunc(output_name="double")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([(double_it, "x[i] -> double[i]")])
    inputs = {"x": np.arange(3)}
    pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")

    run_info_path = RunInfo.path(tmp_path)
    with run_info_path.open() as f:
        data = json.load(f)

    data.pop("error_handling", None)
    assert "error_handling" not in data
    with run_info_path.open("w") as f:
        json.dump(data, f, indent=4)

    run_info = RunInfo.load(tmp_path)

    assert run_info.error_handling == "raise"
