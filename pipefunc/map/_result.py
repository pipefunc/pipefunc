from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

from ._shapes import external_shape_from_mask, internal_shape_from_mask, shape_is_resolved
from ._storage_array._base import StorageBase

if TYPE_CHECKING:
    from ._types import ShapeTuple


class _Missing: ...


class DirectValue:
    __slots__ = ["value"]

    def __init__(self, value: Any | type[_Missing] = _Missing) -> None:
        self.value = value

    def exists(self) -> bool:
        return self.value is not _Missing


def _maybe_array_path(output_name: str, run_folder: Path | None) -> Path | None:
    if run_folder is None:
        return None
    assert isinstance(output_name, str)
    return run_folder / "outputs" / output_name


@dataclass
class LazyStorage:
    """Object that can generate a StorageBase instance if its shape is resolved."""

    output_name: str
    shape: ShapeTuple
    shape_mask: tuple[bool, ...]
    storage_class: type[StorageBase]
    run_folder: Path | None

    def evaluate(self) -> StorageBase:
        if not shape_is_resolved(self.shape):
            msg = "Cannot evaluate lazy store with unresolved shape."
            raise ValueError(msg)
        path = _maybe_array_path(self.output_name, self.run_folder)
        external_shape = external_shape_from_mask(self.shape, self.shape_mask)
        internal_shape = internal_shape_from_mask(self.shape, self.shape_mask)
        return self.storage_class(path, external_shape, internal_shape, self.shape_mask)

    def maybe_evaluate(self) -> StorageBase | LazyStorage:
        if shape_is_resolved(self.shape):
            return self.evaluate()
        return self


StoreType: TypeAlias = StorageBase | LazyStorage | Path | DirectValue


class Result(NamedTuple):
    function: str
    kwargs: dict[str, Any]
    output_name: str
    output: Any
    store: StoreType
