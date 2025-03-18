from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from pipefunc._utils import requires

from ._storage_array._base import StorageBase

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from pipefunc import Pipeline


class _Missing: ...


class DirectValue:
    __slots__ = ["value"]

    def __init__(self, value: Any | type[_Missing] = _Missing) -> None:
        self.value = value

    def exists(self) -> bool:
        return self.value is not _Missing


StoreType: TypeAlias = StorageBase | Path | DirectValue


@dataclass
class Result:
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any
    store: StoreType


MAX_RESULT_LENGTH = 10_000


class ResultDict(dict[str, Result]):
    def __init__(
        self,
        *args: Any,
        _pipeline_: Pipeline | None = None,
        _inputs_: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._pipeline = _pipeline_
        self._inputs = _inputs_
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        text = super().__repr__()
        if len(text) > MAX_RESULT_LENGTH:
            actual_length = len(text)
            msg = (
                f"⚠️  ResultDict is too large to display completely "
                f"({actual_length:,} characters, truncated at {MAX_RESULT_LENGTH:,} characters).\n"
                "     To view the full contents, use `dict(result_dict)` "
                "or access individual items by their keys."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            return text[:MAX_RESULT_LENGTH] + "..."
        return text

    def to_xarray(self, *, load_intermediate: bool = True) -> xr.Dataset:
        """Convert the results to an `xarray.Dataset`."""
        if self._pipeline is None or self._inputs is None:
            msg = (
                "The `to_xarray` method can only be used when the `ResultDict` was created"
                " by calling `pipefunc.Pipeline.map` or `pipefunc.Pipeline.map_async`."
            )
            raise ValueError(msg)
        requires("xarray", reason="to_xarray", extras="xarray")
        from .xarray import xarray_dataset_from_results

        return xarray_dataset_from_results(
            self._inputs,
            self,
            self._pipeline,
            load_intermediate=load_intermediate,
        )

    def to_dataframe(self, *, load_intermediate: bool = True) -> pd.DataFrame:
        """Convert the results to a `pandas.DataFrame`."""
        ds = self.to_xarray(load_intermediate=load_intermediate)  # ensures xarray is installed
        from .xarray import xarray_dataset_to_dataframe

        return xarray_dataset_to_dataframe(ds)
