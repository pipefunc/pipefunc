"""Provides xarray integration for pipefunc."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from pipefunc.map import MapSpec, load_outputs
from pipefunc.map._mapspec import mapspec_axes, trace_dependencies

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def load_xarray(
    output_name: str,
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    run_folder: str | Path,
) -> xr.DataArray:
    """Load and represent the data as an `xarray.DataArray`."""
    data = load_outputs(output_name, run_folder=run_folder)
    all_dependencies = trace_dependencies(mapspecs)
    target_dependencies = all_dependencies.get(output_name, {})
    axes_mapping = mapspec_axes(mapspecs)
    coord_mapping = defaultdict(lambda: defaultdict(list))
    dims = set()
    for name, axes in target_dependencies.items():
        array = inputs[name] if name in inputs else load_outputs(name, run_folder=run_folder)
        if axes == axes_mapping[name]:
            coord_mapping[axes][name].append(array)
            dims.update(axes)
        else:
            dims.update(axes)

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

    return xr.DataArray(data, coords=coords, dims=axes_mapping[output_name])


def load_xarray_dataset(
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    *,
    run_folder: str | Path,
    output_names: list[str] | None = None,
) -> xr.Dataset:
    """Load the xarray dataset."""
    if output_names is None:
        output_names = [name for ms in mapspecs for name in ms.output_names]
    data_vars = {
        name: load_xarray(name, mapspecs, inputs, run_folder=run_folder) for name in output_names
    }
    return xr.Dataset(data_vars)
