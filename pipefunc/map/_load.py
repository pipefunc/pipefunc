from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipefunc._utils import requires

from ._run import _load_from_store, _maybe_load_array
from ._run_info import RunInfo

if TYPE_CHECKING:
    import xarray as xr


def load_outputs(*output_names: str, run_folder: str | Path) -> Any:
    """Load the outputs of a run."""
    run_folder = Path(run_folder)
    run_info = RunInfo.load(run_folder)
    store = run_info.init_store()
    outputs = [_load_from_store(output_name, store).value for output_name in output_names]
    outputs = [_maybe_load_array(o) for o in outputs]
    return outputs[0] if len(output_names) == 1 else outputs


def load_xarray_dataset(
    *output_name: str,
    run_folder: str | Path,
    load_intermediate: bool = True,
) -> xr.Dataset:
    """Load the output(s) of a `pipeline.map` as an `xarray.Dataset`.

    Parameters
    ----------
    output_name
        The names of the outputs to load. If empty, all outputs are loaded.
    run_folder
        The folder where the pipeline run was stored.
    load_intermediate
        Whether to load intermediate outputs as coordinates.

    Returns
    -------
        An `xarray.Dataset` containing the outputs of the pipeline run.

    """
    requires("xarray", reason="load_xarray_dataset", extras="xarray")
    from .xarray import load_xarray_dataset

    run_info = RunInfo.load(run_folder)
    return load_xarray_dataset(
        run_info.mapspecs,
        run_info.inputs,
        run_folder=run_folder,
        output_names=output_name,  # type: ignore[arg-type]
        load_intermediate=load_intermediate,
    )
