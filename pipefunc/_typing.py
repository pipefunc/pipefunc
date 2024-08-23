from types import UnionType
from typing import Any, Union, get_args, get_origin

import numpy as np
import numpy.typing as npt


def check_identical_or_any(type1, type2):
    """Check if types are identical or if either type is Any."""
    return type1 == type2 or Any in (type1, type2)


def _all_types_compatible(args1, args2, strict=False):
    """Helper function to check if all types in args1 are compatible with any type in args2.
    If strict is True, it ensures that all types in args2 are also compatible with args1.
    """
    result = all(any(are_types_compatible(t1, t2) for t2 in args2) for t1 in args1)
    if strict:
        return result and all(any(are_types_compatible(t2, t1) for t1 in args1) for t2 in args2)
    return result


def handle_union_types(incoming_type, required_type):
    """Handle compatibility logic for Union types with directional consideration."""
    if (isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union) and (
        isinstance(required_type, UnionType) or get_origin(required_type) == Union
    ):
        incoming_type_args = get_args(incoming_type)
        required_type_args = get_args(required_type)

        # Enforce directionality: incoming must be compatible with required, but not necessarily the reverse
        return _all_types_compatible(
            incoming_type_args,
            required_type_args,
        ) and not _all_types_compatible(required_type_args, incoming_type_args)

    if isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union:
        return any(are_types_compatible(t, required_type) for t in get_args(incoming_type))

    if isinstance(required_type, UnionType) or get_origin(required_type) == Union:
        # Here, we make sure the incoming type can match one of the required types
        return _all_types_compatible([incoming_type], get_args(required_type))

    return None  # Indicate that this logic didn't apply


def handle_generic_types(type1, type2):
    """Handle compatibility logic for generic types like List, Dict, etc."""
    origin1, origin2 = get_origin(type1), get_origin(type2)
    if origin1 and origin2:
        if origin1 != origin2:
            return False
        return all(are_types_compatible(t1, t2) for t1, t2 in zip(get_args(type1), get_args(type2)))

    return None  # Indicate that this logic didn't apply


def handle_numpy_array_types(type1, type2):
    """Handle complex NumPy array types and ArrayLike."""
    origin1, origin2 = get_origin(type1), get_origin(type2)

    if npt.NDArray in (origin1, origin2):
        return all(are_types_compatible(t1, t2) for t1, t2 in zip(get_args(type1), get_args(type2)))

    if npt.ArrayLike in (type1, type2):
        return True

    return None  # Indicate that this logic didn't apply


def handle_numpy_scalar_types(type1, type2):
    """Handle compatibility logic for NumPy scalar types."""
    if (
        isinstance(type1, type)
        and isinstance(type2, type)
        and issubclass(type1, np.generic)
        and issubclass(type2, np.generic)
    ):
        return np.can_cast(type1, type2)

    return None  # Indicate that this logic didn't apply


def check_subclass_relationship(type1, type2):
    """Check subclass relationships between types."""
    if isinstance(type1, type) and isinstance(type2, type):
        return issubclass(type1, type2) or issubclass(type2, type1)

    return False


def are_types_compatible(type1, type2):
    """Main function that combines all checks for type compatibility."""
    if check_identical_or_any(type1, type2):
        return True

    if result := handle_union_types(type1, type2):
        return result

    if result := handle_generic_types(type1, type2):
        return result

    if result := handle_numpy_array_types(type1, type2):
        return result

    if result := handle_numpy_scalar_types(type1, type2):
        return result

    return check_subclass_relationship(type1, type2)
