import sys
from types import UnionType
from typing import Any, ForwardRef, NamedTuple, Union, get_args, get_origin


class NoAnnotation:
    pass


class TypeCheckMemo(NamedTuple):
    globals: dict[str, Any]
    locals: dict[str, Any]
    self_type: type | None = None


def _evaluate_forwardref(ref: ForwardRef, memo: TypeCheckMemo) -> Any:
    """Evaluate a forward reference using the provided memo."""
    kw = {} if sys.version_info < (3, 13) else {"self_type": memo.self_type}
    return ref._evaluate(memo.globals, memo.locals, recursive_guard=frozenset(), **kw)


def _resolve_type(type_: Any, memo: TypeCheckMemo) -> Any:
    """Resolve forward references in a type hint."""
    if isinstance(type_, str):
        return _evaluate_forwardref(ForwardRef(type_), memo)
    if isinstance(type_, ForwardRef):
        return _evaluate_forwardref(type_, memo)
    origin = get_origin(type_)
    if origin in {Union, UnionType}:  # Handle both Union and new | syntax
        args = get_args(type_)
        resolved_args = tuple(_resolve_type(arg, memo) for arg in args)
        return Union[resolved_args]  # noqa: UP007
    if origin:
        args = get_args(type_)
        resolved_args = tuple(_resolve_type(arg, memo) for arg in args)
        return origin[resolved_args]  # Ensure correct subscripting for generic types
    return type_


def _check_identical_or_any(incoming_type: type[Any], required_type: type[Any]) -> bool:
    """Check if types are identical or if required_type is Any."""
    return (
        incoming_type == required_type
        or required_type is Any
        or incoming_type is NoAnnotation
        or required_type is NoAnnotation
    )


def _all_types_compatible(
    incoming_args: tuple[Any, ...],
    required_args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> bool:
    """Helper function to check if all incoming types are compatible with any required type."""
    return all(
        any(is_type_compatible(t1, t2, memo) for t2 in required_args) for t1 in incoming_args
    )


def _handle_union_types(
    incoming_type: type[Any],
    required_type: type[Any],
    memo: TypeCheckMemo,
) -> bool | None:
    """Handle compatibility logic for Union types with directional consideration."""
    if (isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union) and (
        isinstance(required_type, UnionType) or get_origin(required_type) == Union
    ):
        incoming_type_args = get_args(incoming_type)
        required_type_args = get_args(required_type)
        return _all_types_compatible(incoming_type_args, required_type_args, memo)

    if isinstance(incoming_type, UnionType) or get_origin(incoming_type) == Union:
        return all(is_type_compatible(t, required_type, memo) for t in get_args(incoming_type))

    if isinstance(required_type, UnionType) or get_origin(required_type) == Union:
        return any(is_type_compatible(incoming_type, t, memo) for t in get_args(required_type))

    return None


def _handle_generic_types(
    incoming_type: type[Any],
    required_type: type[Any],
    memo: TypeCheckMemo,
) -> bool | None:
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

        if not required_args or not incoming_args:
            return True
        # If both have arguments, check compatibility of each argument
        return all(is_type_compatible(t1, t2, memo) for t1, t2 in zip(incoming_args, required_args))
    return None


def is_type_compatible(
    incoming_type: type[Any],
    required_type: type[Any],
    memo: TypeCheckMemo | None = None,
) -> bool:
    """Check if the incoming type is compatible with the required type, resolving forward references."""
    if memo is None:  # for testing purposes
        memo = TypeCheckMemo(globals={}, locals={})
    incoming_type = _resolve_type(incoming_type, memo)
    required_type = _resolve_type(required_type, memo)

    if _check_identical_or_any(incoming_type, required_type):
        return True
    if (result := _handle_union_types(incoming_type, required_type, memo)) is not None:
        return result
    if (result := _handle_generic_types(incoming_type, required_type, memo)) is not None:
        return result
    return False
