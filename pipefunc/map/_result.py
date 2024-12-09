from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, TypeAlias

from ._storage_array._base import StorageBase


class _Missing: ...


class DirectValue:
    __slots__ = ["value"]

    def __init__(self, value: Any | type[_Missing] = _Missing) -> None:
        self.value = value

    def exists(self) -> bool:
        return self.value is not _Missing


StoreType: TypeAlias = StorageBase | Path | DirectValue


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any
    store: StoreType
