from typing import TypeAlias

ShapeTuple: TypeAlias = tuple[int | str, ...]
UserShapeDict: TypeAlias = dict[str, int | str | ShapeTuple]
ShapeDict: TypeAlias = dict[str, ShapeTuple]
