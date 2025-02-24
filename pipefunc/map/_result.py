from __future__ import annotations

import warnings
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from ._storage_array._base import StorageBase


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


class ResultDict(UserDict[str, Result]):
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
