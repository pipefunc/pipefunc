from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from pathlib import Path

    from ._storage_array._base import StorageBase


class _Missing: ...


class DirectValue:
    __slots__ = ["value"]

    def __init__(self, value: Any | type[_Missing] = _Missing) -> None:
        self.value = value

    def exists(self) -> bool:
        return self.value is not _Missing


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any
    store: StorageBase | Path | DirectValue
