from types import UnionType
from typing import Any, Union, get_args, get_origin


def check_identical_or_any(incoming_type: type[Any], required_type: type[Any]) -> bool:
    """Check if types are identical or if required_type is Any."""
    return incoming_type == required_type or required_type is Any


def _all_types_compatible(
    incoming_args: tuple[type[Any], ...],
    required_args: tuple[type[Any], ...],
) -> bool:
    """Helper function to check if all incoming types are compatible with any required type."""
    return all(any(is_type_compatible(t1, t2) for t2 in required_args) for t1 in incoming_args)


def handle_union_types(incoming_type: type[Any], required_type: type[Any]) -> bool | None:
    """Handle compatibility logic for Union types with directional consideration."""
    if (isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union) and (
        isinstance(required_type, UnionType) or get_origin(required_type) == Union
    ):
        incoming_type_args = get_args(incoming_type)
        required_type_args = get_args(required_type)
        return _all_types_compatible(incoming_type_args, required_type_args)

    if isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union:
        return all(is_type_compatible(t, required_type) for t in get_args(incoming_type))

    if isinstance(required_type, UnionType) or get_origin(required_type) == Union:
        return any(is_type_compatible(incoming_type, t) for t in get_args(required_type))

    return None


def handle_generic_types(incoming_type: type[Any], required_type: type[Any]) -> bool | None:
    incoming_origin = get_origin(incoming_type) or incoming_type
    required_origin = get_origin(required_type) or required_type

    if incoming_origin and required_origin:
        if isinstance(incoming_origin, type) and isinstance(required_origin, type):
            if not issubclass(incoming_origin, required_origin):
                return False
        elif incoming_origin != required_origin:
            return False

        incoming_args = get_args(incoming_type)
        required_args = get_args(required_type)

        # If required_type has no arguments, it's compatible with any parameterized version
        if not required_args:
            return True

        # If incoming_type has no arguments but required_type does, they're not compatible
        if not incoming_args and required_args:
            return False

        # If both have arguments, check compatibility of each argument
        return all(is_type_compatible(t1, t2) for t1, t2 in zip(incoming_args, required_args))
    return None


def is_type_compatible(incoming_type: type[Any], required_type: type[Any]) -> bool:
    """Check if the incoming type is compatible with the required type."""
    if check_identical_or_any(incoming_type, required_type):
        return True
    if (result := handle_union_types(incoming_type, required_type)) is not None:
        return result
    if (result := handle_generic_types(incoming_type, required_type)) is not None:
        return result
    return False
