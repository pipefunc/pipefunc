"""Provides `xarray` integration for `pipefunc`."""

from __future__ import annotations

import itertools
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from pipefunc.map import MapSpec, load_outputs
from pipefunc.map._mapspec import mapspec_axes, trace_dependencies
from pipefunc.map._run_info import RunInfo

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from pipefunc import Pipeline
    from pipefunc.map._run import Result


def _data_loader(
    output_name: str,
    *,
    run_folder: Path | None = None,
    data: dict[str, Result] | None = None,
) -> Any:
    if data is not None:
        assert data is not None
        return data[output_name].output
    assert run_folder is not None
    return load_outputs(output_name, run_folder=run_folder)


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


def _xarray(
    output_name: str,
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    data_loader: Callable[[str], Any],
    *,
    load_intermediate: bool = True,
) -> xr.DataArray:
    """Load and represent the data as an `xarray.DataArray`."""
    data = data_loader(output_name)
    all_dependencies = trace_dependencies(mapspecs)
    target_dependencies = all_dependencies.get(output_name, {})
    axes_mapping = mapspec_axes(mapspecs)
    coord_mapping: dict[tuple[str, ...], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list),
    )
    dims: set[str] = set()
    for name, axes in target_dependencies.items():
        dims.update(axes)
        if name in inputs:
            array = inputs[name]
        elif load_intermediate:
            array = data_loader(name)
        else:
            continue

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
            array = pd.MultiIndex.from_arrays(arrays, names=names)
        coords[name] = (axes, array)

    return xr.DataArray(data, coords=coords, dims=axes_mapping[output_name], name=output_name)


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
        name: _xarray(
            name,
            mapspecs,
            inputs,
            data_loader,
            load_intermediate=load_intermediate,
        )
        for name in mapspec_output_names
    }
    all_coords = {coord for data in data_arrays.values() for coord in data.coords}
    # Remove the DataArrays that are already appear in other DataArrays' coords
    to_merge = [v for k, v in data_arrays.items() if k not in all_coords]
    ds = xr.merge(to_merge, compat="override")
    for name in single_output_names:
        array = data_loader(name)
        ds[name] = array
    return ds


def xarray_dataset_from_results(
    inputs: dict[str, Any],
    results: OrderedDict[str, Result],
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
