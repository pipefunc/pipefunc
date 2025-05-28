from __future__ import annotations

from typing import Literal, TypeAlias

ShapeTuple: TypeAlias = tuple[int | Literal["?"], ...]
UserShapeDict: TypeAlias = dict[str, int | Literal["?"] | ShapeTuple]
ShapeDict: TypeAlias = dict[str, ShapeTuple]
Index: TypeAlias = int | tuple[int, ...]
Indices: TypeAlias = list[int] | list[tuple[int, ...]]
