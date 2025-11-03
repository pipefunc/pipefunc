from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

from pipefunc._utils import pandas_to_polars, requires

from ._storage_array._base import StorageBase

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
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

    def type_cast(self, *, inplace: bool = True) -> ResultDict:
        """Casts NumPy array dtypes using pipeline annotations.

        Applies `ndarray.astype()` to array outputs from `MapSpec` operations,
        using the numeric/boolean type annotation (e.g., `np.int64`, `float`)
        from the original pipeline. Primarily used to convert object arrays
        from `pipeline.map` results.

        Parameters
        ----------
        inplace : bool, optional
            If True (default), modifies this `ResultDict` in place.
            If False, returns a modified copy.

        Returns
        -------
        ResultDict
            The `ResultDict` with dtypes cast.

        Raises
        ------
        ValueError
            If the `ResultDict` was not created by `pipeline.map`.
        TypeError, ValueError
            If `ndarray.astype()` fails during casting.

        """
        if self._pipeline is None:  # pragma: no cover
            msg = "ResultDict was not created by Pipeline.map"
            raise ValueError(msg)
        result = self.copy() if not inplace else self  # shallow copy
        for output_name, annotation in self._pipeline.output_annotations.items():
            if output_name not in self._pipeline.mapspec_names:
                continue  # not an array
            if _is_np_subdtype(annotation):
                if not inplace:  # avoid modifying the original if inplace=False
                    result[output_name] = copy.deepcopy(result[output_name])

                try:
                    casted_array = result[output_name].output.astype(annotation)
                except (TypeError, ValueError) as e:
                    warnings.warn(
                        f"Could not cast output '{output_name}' to {annotation}"
                        f" due to error: {e}. Likely due to an incorrect type annotation."
                        " Leaving as original dtype.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    result[output_name].output = casted_array
        return result

    def copy(self) -> ResultDict:
        """Return a shallow copy of the ResultDict."""
        new_dict = ResultDict(self)
        new_dict._pipeline = self._pipeline
        new_dict._inputs = self._inputs
        return new_dict

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

    def to_xarray(self, *, load_intermediate: bool = True, type_cast: bool = True) -> xr.Dataset:
        """Convert the results to an `xarray.Dataset`.

        If ``type_cast`` is ``True``, the object ``numpy.ndarray``s are type cast to the
        types specified in the annotations.
        """
        if self._pipeline is None or self._inputs is None:
            msg = (
                "The `to_xarray` method can only be used when the `ResultDict` was created"
                " by calling `pipefunc.Pipeline.map` or `pipefunc.Pipeline.map_async`."
            )
            raise ValueError(msg)
        requires("xarray", reason="to_xarray", extras="xarray")
        from .xarray import xarray_dataset_from_results

        results = self.type_cast(inplace=False) if type_cast else self

        return xarray_dataset_from_results(
            self._inputs,
            results,
            self._pipeline,
            load_intermediate=load_intermediate,
        )

    def to_dataframe(
        self,
        *,
        load_intermediate: bool = True,
        type_cast: bool = True,
        backend: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """Convert the results to a DataFrame.

        Parameters
        ----------
        load_intermediate
            Whether to load intermediate outputs.
        type_cast
            If ``True``, apply dtype casting based on annotations.
        backend
            DataFrame library to use. Defaults to ``"pandas"``.

        """
        ds = self.to_xarray(  # ensures xarray is installed
            load_intermediate=load_intermediate,
            type_cast=type_cast,
        )
        from .xarray import xarray_dataset_to_dataframe

        df = xarray_dataset_to_dataframe(ds)
        if backend == "pandas":
            return df
        if backend == "polars":
            return pandas_to_polars(df)
        msg = f"Unknown backend '{backend}'. Expected 'pandas' or 'polars'."  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover


def _is_np_subdtype(annotation: Any) -> bool:
    try:
        return (
            np.issubdtype(annotation, np.integer)
            or np.issubdtype(annotation, np.floating)
            or np.issubdtype(annotation, np.bool_)
        )
    except TypeError:
        return False
