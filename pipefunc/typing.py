"""Custom type hinting utilities for pipefunc."""

import sys
from collections.abc import Callable
from types import GenericAlias, UnionType
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Generic,
    NamedTuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np


class NoAnnotation:
    """Marker class for missing type annotations."""


T = TypeVar("T")


class ArrayElementType(Generic[T]):
    """Marker class for the element type of an annotated numpy array."""


class Array(Generic[T], np.ndarray[Any, np.dtype[np.object_]]):
    """Annotated numpy array type hint with element type."""

    # NOTE: Ideally we would do something like this:
    # `Array = Annotated[np.ndarray[Any, np.dtype[object]], ArrayElementType[T]]`
    # however, Annotated doesn't support generics in metadata, see:
    # https://discuss.python.org/t/generics-in-metadata-of-annotated/62059
    def __class_getitem__(cls, item: T) -> Any:
        """Return an annotated numpy array with the provided element type."""
        return Annotated[
            np.ndarray[Any, np.dtype[np.object_]],
            ArrayElementType[item],  # type: ignore[valid-type]
        ]


class TypeCheckMemo(NamedTuple):
    """Named tuple to store memoization data for type checking."""

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
    if (isinstance(incoming_type, UnionType) or get_origin(incoming_type) is Union) and (
        isinstance(required_type, UnionType) or get_origin(required_type) is Union
    ):
        incoming_type_args = get_args(incoming_type)
        required_type_args = get_args(required_type)
        return _all_types_compatible(incoming_type_args, required_type_args, memo)

    if isinstance(incoming_type, UnionType) or get_origin(incoming_type) is Union:
        return all(is_type_compatible(t, required_type, memo) for t in get_args(incoming_type))

    if isinstance(required_type, UnionType) or get_origin(required_type) is Union:
        return any(is_type_compatible(incoming_type, t, memo) for t in get_args(required_type))

    return None


def _extract_array_element_type(metadata: list[Any]) -> Any | None:
    """Extract the ArrayElementType from the metadata if it exists."""
    return next((get_args(t)[0] for t in metadata if get_origin(t) is ArrayElementType), None)


def _compare_annotated_types(
    incoming_type: type[Any],
    required_type: type[Any],
    memo: TypeCheckMemo,
) -> bool:
    """Compare Annotated types including metadata."""
    incoming_primary, *incoming_metadata = get_args(incoming_type)
    required_primary, *required_metadata = get_args(required_type)

    # Recursively check the primary types
    if not is_type_compatible(incoming_primary, required_primary, memo):
        return False

    # Compare metadata (extras)
    incoming_array_element_type = _extract_array_element_type(incoming_metadata)
    required_array_element_type = _extract_array_element_type(required_metadata)
    if incoming_array_element_type is not None and required_array_element_type is not None:
        return is_type_compatible(incoming_array_element_type, required_array_element_type, memo)
    return True


def _compare_single_annotated_type(
    annotated_type: type[Any],
    other_type: type[Any],
    memo: TypeCheckMemo,
) -> bool:
    """Handle cases where only one of the types is Annotated."""
    primary_type, *_ = get_args(annotated_type)
    return is_type_compatible(primary_type, other_type, memo)


def _compare_generic_type_origins(
    incoming_origin: type[Any],
    required_origin: type[Any],
) -> bool:
    """Compare the origins of generic types for compatibility."""
    if isinstance(incoming_origin, type) and isinstance(required_origin, type):
        return issubclass(incoming_origin, required_origin)
    return incoming_origin == required_origin


def _compare_generic_type_args(
    incoming_args: tuple[Any, ...],
    required_args: tuple[Any, ...],
    memo: TypeCheckMemo,
) -> bool:
    """Compare the arguments of generic types for compatibility."""
    if not required_args or not incoming_args:
        return True
    return all(is_type_compatible(t1, t2, memo) for t1, t2 in zip(incoming_args, required_args))


def _handle_generic_types(
    incoming_type: type[Any],
    required_type: type[Any],
    memo: TypeCheckMemo,
) -> bool | None:
    incoming_origin = get_origin(incoming_type) or incoming_type
    required_origin = get_origin(required_type) or required_type

    # Handle Annotated types
    if incoming_origin is Annotated and required_origin is Annotated:
        return _compare_annotated_types(incoming_type, required_type, memo)
    if incoming_origin is Annotated:
        return _compare_single_annotated_type(incoming_type, required_type, memo)
    if required_origin is Annotated:
        return _compare_single_annotated_type(required_type, incoming_type, memo)

    # Handle generic types
    if incoming_origin and required_origin:
        if not _compare_generic_type_origins(incoming_origin, required_origin):
            return False
        incoming_args = get_args(incoming_type)
        required_args = get_args(required_type)
        return _compare_generic_type_args(incoming_args, required_args, memo)

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


def is_object_array_type(tp: Any) -> bool:
    """Check if the given type is similar to `Array[T]`.

    2. `Annotated[numpy.ndarray[Any, numpy.dtype[numpy.object_]], T]`
    3. `numpy.ndarray[Any, numpy.dtype[numpy.object_]]`
    """
    if get_origin(tp) is np.ndarray:
        # Base case: directly an np.ndarray[Any, np.dtype[np.object_]]
        return get_args(tp) == (Any, np.dtype[np.object_])
    if get_origin(tp) is Annotated:
        # Recursive case: strip the Annotated and check the first argument
        array_type, _ = get_args(tp)
        return is_object_array_type(array_type)

    return False


class Unresolvable:
    """Class to represent an unresolvable type hint."""

    def __init__(self, type_str: str) -> None:
        """Initialize the Unresolvable instance."""
        self.type_str = type_str

    def __repr__(self) -> str:
        """Return a string representation of the Unresolvable instance."""
        return f"Unresolvable[{self.type_str}]"

    def __eq__(self, other: object) -> bool:
        """Check equality between two Unresolvable instances."""
        if isinstance(other, Unresolvable):
            return self.type_str == other.type_str
        return False

    def __hash__(self) -> int:
        """Return a hash of the Unresolvable instance."""
        return hash(self.type_str)


def safe_get_type_hints(func: Callable[..., Any]) -> dict[str, Any]:  # noqa: PLR0912
    """Safely get type hints for a function, resolving forward references."""
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001
        hints = func.__annotations__

    resolved_hints = {}
    for arg, hint in hints.items():
        if isinstance(hint, str):
            # String annotations (from __future__ import annotations)
            try:
                resolved_hints[arg] = eval(hint, func.__globals__)  # noqa: S307
            except NameError:
                resolved_hints[arg] = Unresolvable(hint)
        elif isinstance(hint, ForwardRef):
            # Forward references
            try:
                resolved_hints[arg] = hint._evaluate(func.__globals__, None)
            except NameError:
                resolved_hints[arg] = Unresolvable(hint.__forward_arg__)
        elif isinstance(hint, GenericAlias):
            # Handle generics, recursively resolve arguments
            origin = hint.__origin__
            args = hint.__args__
            resolved_args = []
            for arg_type in args:
                try:
                    if isinstance(arg_type, ForwardRef):
                        resolved_args.append(arg_type._evaluate(func.__globals__, None))
                    else:
                        resolved_args.append(arg_type)
                except NameError:  # noqa: PERF203
                    resolved_args.append(Unresolvable(str(arg_type)))
            resolved_hints[arg] = origin[tuple(resolved_args)]
        elif hint is type(None):
            resolved_hints[arg] = None
        elif isinstance(hint, GenericAlias) and origin is Union:
            # Handle Union types
            resolved_args = []
            for arg_type in hint.__args__:
                if isinstance(arg_type, ForwardRef):
                    try:
                        resolved_args.append(arg_type._evaluate(func.__globals__, None))
                    except NameError:
                        resolved_args.append(Unresolvable(arg_type.__forward_arg__))
                else:
                    resolved_args.append(arg_type)
            resolved_hints[arg] = Union[tuple(resolved_args)]  # noqa: UP007
        else:
            resolved_hints[arg] = hint

    return resolved_hints
