"""Provides `xarray` integration for `pipefunc`."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from pipefunc._utils import infer_shape

from ._load import load_outputs, maybe_load_data
from ._mapspec import MapSpec, mapspec_axes, trace_dependencies
from ._run_info import RunInfo

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from pipefunc import Pipeline

    from ._result import ResultDict


def load_xarray(
    output_name: str,
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    run_folder: str | Path,
    *,
    load_intermediate: bool = True,
) -> xr.DataArray:
    """Load and represent the data as an `xarray.DataArray`."""
    return _xarray(
        output_name,
        mapspecs,
        inputs,
        data_loader=partial(_data_loader, run_folder=run_folder),  # type: ignore[arg-type]
        load_intermediate=load_intermediate,
    )


def load_xarray_dataset(
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    *,
    run_folder: str | Path,
    output_names: list[str] | None = None,
    load_intermediate: bool = True,
) -> xr.Dataset:
    """Load the xarray dataset."""
    if not output_names:
        run_info = RunInfo.load(run_folder)
        output_names = sorted(run_info.all_output_names)
    return _xarray_dataset(
        mapspecs,
        inputs,
        data_loader=partial(_data_loader, run_folder=run_folder),  # type: ignore[arg-type]
        output_names=output_names,
        load_intermediate=load_intermediate,
    )


def xarray_dataset_from_results(
    inputs: dict[str, Any],
    results: ResultDict,
    pipeline: Pipeline,
    *,
    load_intermediate: bool = True,
) -> xr.Dataset:
    """Load the xarray dataset from the results as returned by `pipefunc.Pipeline.map`."""
    mapspecs = pipeline.mapspecs()
    output_names = sorted(results.keys())
    return _xarray_dataset(
        mapspecs,
        inputs,
        data_loader=partial(_data_loader, data=results),
        output_names=output_names,
        load_intermediate=load_intermediate,
    )


def _data_loader(
    output_name: str,
    *,
    run_folder: Path | None = None,
    data: ResultDict | None = None,
) -> Any:
    if data is not None:
        assert data is not None
        return data[output_name].output
    assert run_folder is not None
    return load_outputs(output_name, run_folder=run_folder)


def _xarray(
    output_name: str,
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    data_loader: Callable[[str], Any],
    *,
    load_intermediate: bool = True,
) -> xr.DataArray:
    """Load and represent the data as an `xarray.DataArray`."""
    all_dependencies = trace_dependencies(mapspecs)
    target_dependencies = all_dependencies.get(output_name, {})
    axes_mapping = mapspec_axes(mapspecs)
    coord_mapping: dict[tuple[str, ...], dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list),
    )
    dims: set[str] = set()
    for name, axes in target_dependencies.items():
        dims.update(axes)
        if name in inputs:
            array = inputs[name]
            array = maybe_load_data(array)
        elif load_intermediate:
            array = data_loader(name)
        else:
            continue

        array = _maybe_to_array(array)
        array = _reshape_if_needed(array, name, axes_mapping)
        if axes == axes_mapping[name]:
            coord_mapping[axes][name].append(array)

    coords = {}
    for axes, dct in coord_mapping.items():
        if len(dct) == 1:
            name, (array,) = next(iter(dct.items()))
        else:
            names = list(dct.keys())
            name = ":".join(names)
            arrays = list(itertools.chain.from_iterable(dct.values()))
            first = arrays[0]
            if isinstance(first, np.ndarray) and first.ndim > 1:  # not supported in pandas
                shape = first.shape
                array = np.empty(shape, dtype=object)
                for i in np.ndindex(shape):
                    array[i] = tuple(arr[i] for arr in arrays)
            else:
                array = _create_multiindex(arrays, names=names)
        coords[name] = (axes, array)

    data = data_loader(output_name)
    data = _maybe_to_array(data)
    data = _reshape_if_needed(data, output_name, axes_mapping)

    return xr.DataArray(data, coords=coords, dims=axes_mapping[output_name], name=output_name)


def _maybe_to_array(x: Any) -> np.ndarray | Any:
    """Convert an iterable to an array."""
    if isinstance(x, np.ndarray):
        return x
    shape = infer_shape(x)
    if shape == ():
        return x
    arr = np.empty(shape, dtype=object)
    arr[:] = x
    return arr


def _reshape_if_needed(array: Any, name: str, axes_mapping: dict[str, tuple[str, ...]]) -> Any:
    """Reshape N-D array to match mapspec dimensionality using object arrays."""
    dims = axes_mapping.get(name)
    if not isinstance(array, np.ndarray) or not dims or array.ndim <= len(dims):
        return array
    expected_shape = array.shape[: len(dims)]
    new_array = np.empty(expected_shape, dtype=object)
    for index in np.ndindex(expected_shape):
        new_array[index] = array[index]
    return new_array


def _xarray_dataset(
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    *,
    data_loader: Callable[[str], Any],
    output_names: list[str],
    load_intermediate: bool = True,
) -> xr.Dataset:
    """Load the xarray dataset."""
    mapspec_output_names = [n for ms in mapspecs for n in ms.output_names if n in output_names]
    single_output_names = [n for n in output_names if n not in mapspec_output_names]
    data_arrays = {
        name: _xarray(name, mapspecs, inputs, data_loader, load_intermediate=load_intermediate)
        for name in mapspec_output_names
    }
    all_coords = {coord for data in data_arrays.values() for coord in data.coords}
    # Remove the DataArrays that are already appear in other DataArrays' coords
    to_merge = [v for k, v in data_arrays.items() if k not in all_coords]
    ds = xr.merge(to_merge, compat="override")
    for name in single_output_names:
        array = data_loader(name)
        array = _maybe_to_array(array)
        if isinstance(array, np.ndarray):
            # Wrap in DimensionlessArray to avoid xarray trying to interpret
            # the data and requiring dimensions, resulting in an error
            ds[name] = ((), DimensionlessArray(array))
        else:
            ds[name] = ((), array)
    return ds


@dataclass
class DimensionlessArray:
    """A class to represent an array without dimensions."""

    arr: np.ndarray


def _split_tuple_columns(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    tuple_columns = [col for col in df.columns if ":" in col]
    for col in tuple_columns:
        new_col_names = col.split(":")
        for i, new_col in enumerate(new_col_names):
            result_df[new_col] = df[col].apply(lambda x: x[i])  # noqa: B023
        result_df = result_df.drop(col, axis=1)
    return result_df


def xarray_dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an xarray dataset to a pandas dataframe."""
    if not ds.coords:
        # Return a single row dataframe if there are no coordinates
        data = {}
        for data_var, value in ds.data_vars.items():
            val = value.data
            # Unwrap 0D numpy arrays
            if isinstance(val, np.ndarray) and val.ndim == 0:
                val = val.item()
            # Unwrap DimensionlessArray
            if isinstance(val, DimensionlessArray):
                val = val.arr
            data[data_var] = [val]
        return pd.DataFrame(data)
    df = ds.to_dataframe().reset_index(drop=True)
    # Identify if a column is a DimensionlessArray
    for col in df.columns:
        if isinstance(df[col].iloc[0], DimensionlessArray):
            df[col] = df[col].apply(lambda x: x.arr)
    return _split_tuple_columns(df)


def _create_multiindex(
    arrays: list[Any],
    *,
    names: list[str],
) -> pd.MultiIndex:
    """Create a pandas MultiIndex, with a fallback for unhashable types.

    Attempts to use `pandas.MultiIndex.from_arrays`, which is fast but requires
    hashable elements. If that fails with a `TypeError`, it falls back to assuming
    that items within each array are unique.
    """
    try:
        return pd.MultiIndex.from_arrays(arrays, names=names)
    except TypeError:
        # This path assumes that items within each array are unique.
        codes = [range(len(arr)) for arr in arrays]
        levels = [pd.Index(arr) for arr in arrays]
        return pd.MultiIndex(
            levels=levels,
            codes=codes,
            names=names,
            verify_integrity=False,
        )
