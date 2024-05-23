"""Provides xarray integration for pipefunc."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from pipefunc.map import MapSpec, load_outputs
from pipefunc.map._mapspec import trace_dependencies

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import numpy as np


def to_xarray(
    output_data: np.ndarray,
    output_name: str,
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    *,
    run_folder: str | Path | None = None,
    use_intermediate: bool = False,
    verbose: bool = True,
) -> xr.DataArray:
    """Convert the output data to an `xarray.DataArray`."""
    # Find the target mapspec
    target_mapspec = next(ms for ms in mapspecs if output_name in ms.output_names)

    # Trace dependencies
    all_dependencies = trace_dependencies(mapspecs)
    target_dependencies = all_dependencies[target_mapspec.output_names[0]]

    coords = {}

    # Gather the coordinates based on traced dependencies
    for axis, input_names in target_dependencies.items():
        if len(input_names) > 1:
            arrays = []
            for name in input_names:
                if name in inputs:
                    arrays.append(inputs[name])
                elif use_intermediate and run_folder:
                    arrays.append(load_outputs(name, run_folder=run_folder))
                else:
                    break
            else:
                multiindex = pd.MultiIndex.from_arrays(arrays, names=input_names)
                coords[axis] = (":".join(input_names), multiindex)
                continue

        input_name = input_names[0]  # Unpack the single-element list
        if input_name in inputs:
            coords[axis] = (input_name, inputs[input_name])
        elif use_intermediate and run_folder:
            coord_data = load_outputs(input_name, run_folder=run_folder)
            coords[axis] = (input_name, coord_data)
        elif verbose:
            msg = (
                f"Input '{input_name}' not found in `inputs` and `intermediate_results=False`,"
                f" so coordinate with `{axis=}` cannot be created."
            )
            print(msg)

    # Create MultiIndexes for each combined axis
    multiindex_coords = dict(coords.values())

    # Rename the axis names
    axis_name_mapping = {axis: coords[axis][0] for axis in coords}
    axis_names = [axis_name_mapping.get(name, name) for name in target_mapspec.output_indices]
    return xr.DataArray(output_data, dims=axis_names, coords=multiindex_coords, name=output_name)


def load_xarray_dataset(
    mapspecs: list[MapSpec],
    inputs: dict[str, Any],
    *,
    run_folder: str | Path,
    output_names: list[str] | None = None,
    use_intermediate: bool = True,
) -> xr.Dataset:
    """Load the xarray dataset."""
    if output_names is None:
        output_names = [name for ms in mapspecs for name in ms.output_names]
    data = [load_outputs(name, run_folder=run_folder) for name in output_names]
    return xr.Dataset(
        {
            name: to_xarray(
                output_data,
                name,
                mapspecs,
                inputs,
                run_folder=run_folder,
                use_intermediate=use_intermediate,
            )
            for name, output_data in zip(output_names, data)
        },
    )
