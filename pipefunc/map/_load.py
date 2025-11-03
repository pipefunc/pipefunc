from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pipefunc._utils import (
    at_least_tuple,
    load,
    pandas_to_polars,
    requires,
)
from pipefunc.helpers import FileValue

from ._result import DirectValue
from ._run_info import RunInfo
from ._storage_array._base import StorageBase

if TYPE_CHECKING:
    import xarray as xr

    from pipefunc._pipeline._types import OUTPUT_TYPE

    from ._result import StoreType


def load_outputs(*output_names: str, run_folder: str | Path) -> Any:
    """Load the outputs of a run.

    Allows loading outputs even from partial runs that failed.
    Map results are returned as `numpy.masked_array`s where missing values
    are masked.

    Parameters
    ----------
    output_names
        The names of the outputs to load. If empty, no outputs are loaded.
    run_folder
        The ``run_folder`` used in ``pipeline.map`` or ``pipeline.map_async``.

    See Also
    --------
    load_all_outputs
        For loading all outputs.

    """
    run_info = RunInfo.load(run_folder)
    store = run_info.init_store()
    outputs = [_load_from_store(output_name, store).value for output_name in output_names]
    outputs = [maybe_load_data(o) for o in outputs]
    return outputs[0] if len(output_names) == 1 else outputs


def load_all_outputs(run_folder: str | Path) -> dict[str, Any]:
    """Load all outputs of a run.

    Allows loading outputs even from partial runs that failed.
    Map results are returned as `numpy.masked_array`s where missing values
    are masked.

    Parameters
    ----------
    run_folder
        The ``run_folder`` used in ``pipeline.map`` or ``pipeline.map_async``.

    See Also
    --------
    load_outputs
        For loading specific outputs.

    """
    run_info = RunInfo.load(run_folder)
    output_names = sorted(run_info.all_output_names)
    outputs = load_outputs(*output_names, run_folder=run_folder)
    if len(output_names) == 1:
        return {output_names[0]: outputs}
    return dict(zip(output_names, outputs))


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
        Whether to load intermediate outputs.

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
        output_names=output_name or None,  # type: ignore[arg-type]
        load_intermediate=load_intermediate,
    )


def load_dataframe(
    *output_name: str,
    run_folder: str | Path,
    load_intermediate: bool = True,
    backend: Literal["pandas", "polars"] = "pandas",
) -> Any:
    """Load the output(s) of a `pipeline.map` as a DataFrame.

    Parameters
    ----------
    output_name
        The names of the outputs to load. If empty, all outputs are loaded.
    run_folder
        The folder where the pipeline run was stored.
    load_intermediate
        Whether to load intermediate outputs.
    backend
        DataFrame library to use when constructing the result. Defaults to ``"pandas"``.

    Returns
    -------
        A DataFrame from the selected backend containing the outputs of the pipeline run.

    """
    from .xarray import xarray_dataset_to_dataframe

    ds = load_xarray_dataset(
        *output_name,
        run_folder=run_folder,
        load_intermediate=load_intermediate,
    )
    df = xarray_dataset_to_dataframe(ds)
    if backend == "pandas":
        return df
    if backend == "polars":
        return pandas_to_polars(df)
    msg = f"Unknown backend '{backend}'. Expected 'pandas' or 'polars'."  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def maybe_load_data(x: Any) -> Any:
    """Load data if it is a `FileValue` or `StorageBase`."""
    if isinstance(x, StorageBase):
        return x.to_array()
    if isinstance(x, FileValue):
        return x.load()
    return x


class _StoredValue(NamedTuple):
    value: Any
    exists: bool


def _load_from_store(
    output_name: OUTPUT_TYPE,
    store: dict[str, StoreType],
    *,
    return_output: bool = True,
) -> _StoredValue:
    outputs: list[Any] = []
    all_exist = True

    for name in at_least_tuple(output_name):
        storage = store[name]
        if isinstance(storage, StorageBase):
            outputs.append(storage)
        elif isinstance(storage, Path):
            if storage.is_file():
                outputs.append(load(storage) if return_output else None)
            else:
                all_exist = False
                outputs.append(None)
        else:
            assert isinstance(storage, DirectValue)
            if storage.exists():
                outputs.append(storage.value)
            else:
                all_exist = False
                outputs.append(None)

    if not return_output:
        outputs = None  # type: ignore[assignment]
    elif len(outputs) == 1:
        outputs = outputs[0]

    return _StoredValue(outputs, all_exist)
