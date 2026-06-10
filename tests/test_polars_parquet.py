"""Tests for Parquet serialization and `pl.LazyFrame` support (issue #879)."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TC002, needed at runtime to resolve `np.ndarray` annotations
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._utils import PARQUET_MAGIC, dump, is_parquet_file, load
from pipefunc.map import load_outputs
from pipefunc.typing import is_type_compatible

has_polars = importlib.util.find_spec("polars") is not None
pytestmark = pytest.mark.skipif(not has_polars, reason="polars not installed")

if has_polars:
    import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


def test_dump_dataframe_as_parquet(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "df.cloudpickle"
    dump(df, path)
    assert path.read_bytes()[:4] == PARQUET_MAGIC
    assert is_parquet_file(path)
    loaded = load(path)
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(df)


def test_dump_non_dataframe_still_pickles(tmp_path: Path) -> None:
    path = tmp_path / "obj.cloudpickle"
    dump({"a": 1}, path)
    assert not is_parquet_file(path)
    assert load(path) == {"a": 1}


def test_dump_falls_back_to_pickle_on_parquet_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pl.DataFrame({"a": [1, 2]})

    def fail(*args: Any, **kwargs: Any) -> None:
        msg = "boom"
        raise ValueError(msg)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail)
    path = tmp_path / "df.cloudpickle"
    dump(df, path)
    assert path.read_bytes()[:4] != PARQUET_MAGIC
    loaded = load(path)
    assert isinstance(loaded, pl.DataFrame)
    assert loaded.equals(df)


def test_load_with_cache(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2]})
    path = tmp_path / "df.cloudpickle"
    dump(df, path)
    assert load(path, cache=True).equals(df)


def test_file_array_with_dataframes(tmp_path: Path) -> None:
    from pipefunc.map._storage_array._file import FileArray

    arr = FileArray(tmp_path / "arr", shape=(2,))
    arr.dump((0,), pl.DataFrame({"a": [1]}))
    arr.dump((1,), pl.DataFrame({"a": [2]}))
    assert is_parquet_file(arr._index_to_file(0))
    element = arr[0,]
    assert isinstance(element, pl.DataFrame)
    assert element["a"].to_list() == [1]
    # `to_array` exercises the threaded `_load_all` byte-sniffing path
    full = arr.to_array()
    assert all(isinstance(x, pl.DataFrame) for x in full)


def test_dataframe_to_lazyframe_type_compatible() -> None:
    assert is_type_compatible(pl.DataFrame, pl.LazyFrame)
    assert not is_type_compatible(pl.LazyFrame, pl.DataFrame)
    assert not is_type_compatible(int, pl.LazyFrame)


def test_map_lazyframe_input_scans_parquet(tmp_path: Path) -> None:
    @pipefunc(output_name="df")
    def make_df() -> pl.DataFrame:
        return pl.DataFrame({"a": [1, 2, 3]})

    @pipefunc(output_name="total")
    def consume(df: pl.LazyFrame) -> int:
        assert isinstance(df, pl.LazyFrame)
        # The plan must be a Parquet scan, not an in-memory DataFrame
        assert "DF" not in df.explain(optimized=False)
        return df.select(pl.col("a").sum()).collect().item()

    pipeline = Pipeline([make_df, consume])  # validates type annotations
    result = pipeline.map({}, run_folder=tmp_path, parallel=False, show_progress=False)
    assert result["total"].output == 6
    df_path = tmp_path / "outputs" / "df.cloudpickle"
    assert is_parquet_file(df_path)
    loaded = load_outputs("df", run_folder=tmp_path)
    assert isinstance(loaded, pl.DataFrame)
    assert loaded["a"].to_list() == [1, 2, 3]


def test_map_lazyframe_input_without_run_folder() -> None:
    @pipefunc(output_name="df")
    def make_df() -> pl.DataFrame:
        return pl.DataFrame({"a": [1, 2, 3]})

    @pipefunc(output_name="total")
    def consume(df: pl.LazyFrame) -> int:
        assert isinstance(df, pl.LazyFrame)
        return df.select(pl.col("a").sum()).collect().item()

    pipeline = Pipeline([make_df, consume])
    result = pipeline.map({}, parallel=False, show_progress=False, storage="dict")
    assert result["total"].output == 6


def test_map_elementwise_lazyframe(tmp_path: Path) -> None:
    @pipefunc(output_name="df", mapspec="x[i] -> df[i]")
    def make_df(x: int) -> pl.DataFrame:
        return pl.DataFrame({"a": [x, x * 2]})

    @pipefunc(output_name="total", mapspec="df[i] -> total[i]")
    def consume(df: pl.LazyFrame) -> int:
        assert isinstance(df, pl.LazyFrame)
        return df.select(pl.col("a").sum()).collect().item()

    pipeline = Pipeline([make_df, consume])
    result = pipeline.map(
        {"x": [1, 10]},
        run_folder=tmp_path,
        parallel=True,
        show_progress=False,
    )
    assert result["total"].output.tolist() == [3, 30]


def test_map_reduction_keeps_dataframes(tmp_path: Path) -> None:
    @pipefunc(output_name="df", mapspec="x[i] -> df[i]")
    def make_df(x: int) -> pl.DataFrame:
        return pl.DataFrame({"a": [x]})

    @pipefunc(output_name="n")
    def reduce_all(df: np.ndarray) -> int:
        assert all(isinstance(d, pl.DataFrame) for d in df)
        return len(df)

    pipeline = Pipeline([make_df, reduce_all])
    result = pipeline.map(
        {"x": [1, 10]},
        run_folder=tmp_path,
        parallel=False,
        show_progress=False,
    )
    assert result["n"].output == 2


def test_run_lazyframe_input() -> None:
    @pipefunc(output_name="df")
    def make_df() -> pl.DataFrame:
        return pl.DataFrame({"a": [1, 2, 3]})

    @pipefunc(output_name="total")
    def consume(df: pl.LazyFrame) -> int:
        assert isinstance(df, pl.LazyFrame)
        return df.select(pl.col("a").sum()).collect().item()

    pipeline = Pipeline([make_df, consume])
    assert pipeline.run("total", kwargs={}) == 6


def test_run_lazyframe_from_input_kwarg() -> None:
    @pipefunc(output_name="total")
    def consume(df: pl.LazyFrame) -> int:
        assert isinstance(df, pl.LazyFrame)
        return df.select(pl.col("a").sum()).collect().item()

    pipeline = Pipeline([consume])
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert pipeline.run("total", kwargs={"df": df}) == 6


def test_map_lazyframe_from_input_kwarg(tmp_path: Path) -> None:
    @pipefunc(output_name="total")
    def consume(df: pl.LazyFrame) -> int:
        assert isinstance(df, pl.LazyFrame)
        return df.select(pl.col("a").sum()).collect().item()

    pipeline = Pipeline([consume])
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = pipeline.map({"df": df}, run_folder=tmp_path, parallel=False, show_progress=False)
    assert result["total"].output == 6


def test_lazyframe_passthrough() -> None:
    @pipefunc(output_name="lf")
    def make_lf() -> pl.LazyFrame:
        return pl.DataFrame({"a": [1, 2, 3]}).lazy()

    @pipefunc(output_name="total")
    def consume(lf: pl.LazyFrame) -> int:
        assert isinstance(lf, pl.LazyFrame)
        return lf.select(pl.col("a").sum()).collect().item()

    pipeline = Pipeline([make_lf, consume])
    result = pipeline.map({}, parallel=False, show_progress=False, storage="dict")
    assert result["total"].output == 6


def test_to_hashable_lazyframe() -> None:
    from pipefunc.cache import to_hashable

    lf = pl.DataFrame({"a": [1, 2]}).lazy()
    key = to_hashable(lf)
    assert hash(key) == hash(to_hashable(pl.DataFrame({"a": [1, 2]}).lazy()))


def test_helpers_when_polars_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipefunc._utils import is_lazyframe_annotation

    monkeypatch.delitem(sys.modules, "polars")
    assert not is_lazyframe_annotation(pl.LazyFrame)
    assert not is_type_compatible(pl.DataFrame, pl.LazyFrame)
