from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs

if TYPE_CHECKING:
    from pathlib import Path


def _irregular_pipeline() -> Pipeline:
    @pipefunc(output_name="words", mapspec="text[i] -> words[i, j*]")
    def split(text: str) -> list[str]:
        return text.split()

    @pipefunc(output_name="lengths", mapspec="words[i, j*] -> lengths[i, j*]")
    def lengths(words: str) -> int:
        return len(words)

    return Pipeline([split, lengths])


def test_pipeline_irregular_dict_storage() -> None:
    pipeline = _irregular_pipeline()
    inputs = {"text": ["hello world", "pipefunc", "two words"]}

    result = pipeline.map(
        inputs=inputs,
        storage="dict",
        parallel=False,
        internal_shapes={"words": (3,), "lengths": (3,)},
    )

    words = result["words"].output
    assert isinstance(words, np.ma.MaskedArray)
    assert words.shape == (3, 3)
    words_mask = np.asarray(np.ma.getmaskarray(words), dtype=bool)
    assert words_mask[0, 2]

    dataset = result.to_xarray(type_cast=False)
    assert "_mask" in dataset["words"].attrs

    type_cast = result.type_cast(inplace=False)
    cast_lengths = type_cast["lengths"].output
    assert isinstance(cast_lengths, np.ndarray)


def test_pipeline_irregular_file_storage(tmp_path: Path) -> None:
    pipeline = _irregular_pipeline()
    inputs = {"text": ["short", "much longer phrase"]}
    run_folder = tmp_path / "run"

    result = pipeline.map(
        inputs=inputs,
        storage="file_array",
        run_folder=run_folder,
        parallel=False,
        internal_shapes={"words": (4,), "lengths": (4,)},
        show_progress=True,
    )

    lengths = result["lengths"].output
    assert isinstance(lengths, np.ma.MaskedArray)
    lengths_mask = np.asarray(np.ma.getmaskarray(lengths), dtype=bool)
    assert lengths_mask[0, 1]

    from_disk = load_outputs("words", run_folder=run_folder)
    assert isinstance(from_disk, np.ma.MaskedArray)
    assert from_disk.shape[1] == 4
