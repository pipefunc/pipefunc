from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._run_info import RunInfo
from pipefunc.typing import Array


@pytest.mark.parametrize("cwd_is_different", [False, True])
def test_run_info_load_handles_relative_paths(tmp_path, cwd_is_different):
    """Reproduce the regression from #893 where RunInfo.load mis-resolves paths.

    The pipeline stores ``input_paths`` in ``run_info.json`` as relative strings like
    ``"adaptive_1d/run_folder_0.0/inputs/x.cloudpickle"``.  After the path
    refactor, RunInfo.load now joins those with ``run_folder.parent``, which breaks
    when both pieces include the run-folder prefix.  This test captures that
    behaviour for both the original working directory and a changed one.
    """

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double(x: int) -> int:
        return x * 2

    @pipefunc(output_name="total")
    def total(y: Array[int]) -> int:
        return sum(y)

    pipeline = Pipeline([double, total])

    run_folder = Path("adaptive_1d/run_folder_0.0")

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        pipeline.map({"x": [1, 2, 3]}, run_folder=run_folder, parallel=False)

        run_folder_abs = run_folder.resolve()

        if cwd_is_different:
            # Mimic the ReadTheDocs failure: load from *outside* the run folder root.
            os.chdir(tmp_path / "..")

        info = RunInfo.load(run_folder_abs)

        assert info.run_folder.name == "run_folder_0.0"
        values = info.inputs["x"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        assert values == [1, 2, 3]
    finally:
        os.chdir(original_cwd)
