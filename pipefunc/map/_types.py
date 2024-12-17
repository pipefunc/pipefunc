from typing import Literal, TypeAlias

ShapeTuple: TypeAlias = tuple[int | Literal["?"], ...]
UserShapeDict: TypeAlias = dict[str, int | Literal["?"] | ShapeTuple]
ShapeDict: TypeAlias = dict[str, ShapeTuple]
