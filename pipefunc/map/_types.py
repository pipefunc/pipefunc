from typing import TypeAlias

ShapeTuple: TypeAlias = tuple[int, ...]
UserShapeDict: TypeAlias = dict[str, int | ShapeTuple]
ShapeDict: TypeAlias = dict[str, ShapeTuple]
