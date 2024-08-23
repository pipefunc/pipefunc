from types import UnionType
from typing import Any, Union, get_args, get_origin

import numpy as np
import numpy.typing as npt


def check_identical_or_any(incoming_type: type, required_type: type) -> bool:
    """Check if types are identical or if either type is Any."""
    return incoming_type == required_type or Any in (incoming_type, required_type)


def _all_types_compatible(args1: tuple[type, ...], args2: tuple[type, ...]) -> bool:
    """Helper function to check if all types in args1 are compatible with any type in args2."""
    return all(any(are_types_compatible(t1, t2) for t2 in args2) for t1 in args1)


def handle_union_types(incoming_type: type, required_type: type) -> bool | None:
    """Handle compatibility logic for Union types with directional consideration."""
    if (isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union) and (
        isinstance(required_type, UnionType) or get_origin(required_type) == Union
    ):
        incoming_type_args = get_args(incoming_type)
        required_type_args = get_args(required_type)

        # The incoming type must be fully compatible with the required type
        # Each element in the incoming union must be compatible with the required type
        return _all_types_compatible(incoming_type_args, required_type_args)

    if isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union:
        # Each part of the incoming union must be fully compatible with the required type
        return all(are_types_compatible(t, required_type) for t in get_args(incoming_type))

    if isinstance(required_type, UnionType) or get_origin(required_type) == Union:
        # The incoming type only needs to match one part of the required union
        return any(are_types_compatible(incoming_type, t) for t in get_args(required_type))

    return None  # Indicate that this logic didn't apply


def handle_generic_types(incoming_type: type, required_type: type) -> bool | None:
    """Handle compatibility logic for generic types like List, Dict, etc."""
    incoming_origin = get_origin(incoming_type)
    required_origin = get_origin(required_type)
    if incoming_origin and required_origin:
        if incoming_origin != required_origin:
            return False
        return all(
            are_types_compatible(t1, t2)
            for t1, t2 in zip(get_args(incoming_type), get_args(required_type))
        )

    return None  # Indicate that this logic didn't apply


def handle_numpy_array_types(incoming_type: type, required_type: type) -> bool | None:
    """Handle complex NumPy array types and ArrayLike."""
    incoming_origin = get_origin(incoming_type)
    required_origin = get_origin(required_type)

    if npt.NDArray in (incoming_origin, required_origin):
        return all(
            are_types_compatible(t1, t2)
            for t1, t2 in zip(get_args(incoming_type), get_args(required_type))
        )

    return None  # Indicate that this logic didn't apply


def handle_numpy_scalar_types(incoming_type: type, required_type: type) -> bool | None:
    """Handle compatibility logic for NumPy scalar types."""
    if issubclass(incoming_type, np.generic) and issubclass(required_type, np.generic):
        return np.can_cast(incoming_type, required_type)

    return None  # Indicate that this logic didn't apply


def check_subclass_relationship(incoming_type: type, required_type: type) -> bool | None:
    """Check subclass relationships between types."""
    if isinstance(incoming_type, type) and isinstance(required_type, type):
        return issubclass(incoming_type, required_type) or issubclass(required_type, incoming_type)

    return None  # Indicate that this logic didn't apply


def are_types_compatible(incoming_type: type, required_type: type) -> bool:  # noqa: PLR0911
    """Main function that combines all checks for type compatibility."""
    if check_identical_or_any(incoming_type, required_type):
        return True

    if (result := handle_union_types(incoming_type, required_type)) is not None:
        return result

    if (result := handle_generic_types(incoming_type, required_type)) is not None:
        return result

    if (result := handle_numpy_array_types(incoming_type, required_type)) is not None:
        return result

    if (result := handle_numpy_scalar_types(incoming_type, required_type)) is not None:
        return result

    if (result := check_subclass_relationship(incoming_type, required_type)) is not None:
        return result

    return False
