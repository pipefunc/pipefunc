from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
    get_type_hints,
)

import numpy as np
import numpy.typing as npt
import pytest

from pipefunc.typing import (
    Array,
    ArrayElementType,
    NoAnnotation,
    TypeCheckMemo,
    Unresolvable,
    is_object_array_type,
    is_type_compatible,
    safe_get_type_hints,
)

NoneType = type(None)


def test_are_types_compatible_standard():
    assert is_type_compatible(list[int], list[int])
    assert not is_type_compatible(list[int], list[float])
    assert not is_type_compatible(int | str, Union[str, float])  # noqa: UP007
    assert not is_type_compatible(int | str, str | float)
    assert not is_type_compatible(int, float)
    assert not is_type_compatible(Any, int)
    assert is_type_compatible(int, Any)
    assert is_type_compatible(dict[int, dict[str, str]], dict[int, dict[str, Any]])
    assert is_type_compatible(dict[int, str], dict)
    assert is_type_compatible(dict, dict[int, str])
    assert is_type_compatible(dict[int, str], Annotated[dict[int, str], float])


def test_are_types_compatible_union():
    assert not is_type_compatible(int | str, str)
    assert not is_type_compatible(int | str, int)
    assert is_type_compatible(int, int | str)
    assert is_type_compatible(str, int | str)
    assert not is_type_compatible(int | str, float)
    assert is_type_compatible(dict[int, str], dict[int, str | int])


def test_are_types_compatible_numpy():
    assert is_type_compatible(npt.NDArray[np.int64], npt.NDArray[np.int_])
    assert not is_type_compatible(npt.NDArray[np.float32], npt.NDArray[np.int64])
    assert not is_type_compatible(np.int32, np.int64)
    assert not is_type_compatible(np.float64, np.int32)
    assert is_type_compatible(npt.NDArray, npt.NDArray[Any])
    assert is_type_compatible(npt.NDArray[Any], npt.NDArray[Any])
    assert is_type_compatible(npt.NDArray[np.int32], npt.NDArray[Any])
    # npt.NDArray without arguments turns into numpy.ndarray[typing.Any, numpy.dtype[+_ScalarType_co]]
    assert not is_type_compatible(npt.NDArray[Any], npt.NDArray)


def test_are_types_compatible_standard_edge_cases():
    # Test with nested lists
    assert is_type_compatible(list[list[int]], list[list[int]])
    assert not is_type_compatible(list[list[int]], list[list[float]])
    assert is_type_compatible(list[list[int]], list[list[Any]])

    # Test with generic containers and Any
    assert not is_type_compatible(list[Any], list[int])
    assert is_type_compatible(list[int], list[Any])
    assert is_type_compatible(dict[int, str], dict[Any, Any])
    assert is_type_compatible(dict[int, str], dict[Any, Any])
    assert not is_type_compatible(dict[int, str], dict[int, float])
    assert not is_type_compatible(dict[int, str], list[int])  # Different container types


def test_are_types_compatible_union_edge_cases():
    # Test with more complex unions
    # Same elements, different order
    assert is_type_compatible(int | str | float, str | float | int)
    # Subset of a larger union
    assert is_type_compatible(int | str, int | str | float)
    # Completely different sets
    assert not is_type_compatible(int | str, float | complex)

    # Test with deeply nested unions
    assert is_type_compatible(Union[int | str, float], Union[str, int, float])  # noqa: UP007
    assert not is_type_compatible(Union[int | str, float], Union[float, complex])  # noqa: UP007

    # Test union with Any
    assert is_type_compatible(int, int | complex)
    assert is_type_compatible(complex, int | complex)
    assert is_type_compatible(int | str, Any)


def test_are_types_compatible_numpy_edge_cases():
    # Test NDArray with different type parameters
    assert is_type_compatible(npt.NDArray[np.int32], npt.NDArray[np.int32])
    assert not is_type_compatible(npt.NDArray[np.float64], npt.NDArray[np.int32])
    # Any in NDArray should be compatible
    assert is_type_compatible(npt.NDArray[np.int32], npt.NDArray[Any])

    # Test with numpy.generic (base class for all NumPy scalars)
    assert is_type_compatible(np.int64, np.generic)
    assert is_type_compatible(np.float64, np.generic)

    # Test scalar types compatibility
    # Can cast between NumPy integer types
    assert is_type_compatible(np.int64, np.int_)
    # Cannot cast from float to integer directly
    assert not is_type_compatible(np.float64, np.int32)


def test_directionality_union():
    # Test directional compatibility
    # Broader incoming type is not compatible with narrower required type
    assert not is_type_compatible(int | str, str)
    # Narrower incoming type is compatible with broader required type
    assert is_type_compatible(str, int | str)

    # Test specific type vs. Union
    # Compatible
    assert is_type_compatible(int, int | str)
    # Not compatible, broader incoming type isn't allowed
    assert not is_type_compatible(int | str, int)


def test_abc_compatibility():
    assert is_type_compatible(list[int], Sequence[int])
    assert is_type_compatible(list[int], Sequence[int])
    assert is_type_compatible(dict[str, int], Mapping[str, int])
    assert is_type_compatible(dict[str, int], Mapping[str, int])
    assert not is_type_compatible(Sequence[int], list[int])
    assert not is_type_compatible(Mapping[str, int], dict[str, int])


def test_generic_type_edge_cases():
    assert is_type_compatible(list, Sequence)
    assert is_type_compatible(list, Sequence)
    assert is_type_compatible(list[int], list)
    assert is_type_compatible(list, list[int])
    assert is_type_compatible(dict[str, int], dict)
    assert is_type_compatible(dict, dict[str, int])


def test_check_none():
    assert is_type_compatible(None, None)
    assert is_type_compatible(None, Any)
    assert not is_type_compatible(int, None)
    assert not is_type_compatible(None, int)
    assert not is_type_compatible(None, str)


def test_no_annotation():
    assert is_type_compatible(Any, NoAnnotation)
    assert is_type_compatible(NoAnnotation, Any)
    assert is_type_compatible(NoAnnotation, NoAnnotation)
    assert is_type_compatible(int, NoAnnotation)
    assert is_type_compatible(NoAnnotation, int)


def test_forward_refs():
    class Node:
        def __init__(self, value: int, children: list[Node]):
            self.value = value
            self.children = children or []

    # Case: Automatically capture the current frame for memo initialization
    frame = sys._getframe(0)
    memo = TypeCheckMemo(frame.f_globals, frame.f_locals)

    # Case: Test with forward references using string annotations
    assert is_type_compatible("Node", Node, memo)
    assert is_type_compatible("list[Node]", list[Node], memo)

    # Case: Test with ForwardRef
    node_ref = ForwardRef("Node")
    assert is_type_compatible(node_ref, Node, memo)

    # Case: Test nested forward references
    nested_ref = ForwardRef("list[Node]")
    assert is_type_compatible(nested_ref, list[Node], memo)
    assert not is_type_compatible(nested_ref, tuple[Node, ...], memo)


def test_array_type_alias():
    assert is_type_compatible(Array[int], np.ndarray[Any, np.dtype[np.object_]])
    assert is_type_compatible(Array[int], np.ndarray[Any, np.dtype(np.object_)])
    assert is_type_compatible(Array[int], np.ndarray[Any, np.dtype(object)])
    assert not is_type_compatible(Array[int], np.ndarray[Any, np.dtype[np.int_]])

    ObjArray: TypeAlias = np.ndarray[Any, np.dtype[np.object_]]  # ObjArray
    # Test with Annotated and ArrayElementType
    assert is_type_compatible(Array[int], Annotated[ObjArray, ArrayElementType[int]])
    assert not is_type_compatible(Array[int], Annotated[ObjArray, ArrayElementType[float]])
    # Here float is not wrapped in ArrayElementType, so the metadata
    # is ignored and therefore compatible
    assert is_type_compatible(Array[int], Annotated[ObjArray, float])


@pytest.mark.parametrize(
    ("tp", "expected"),
    [
        # Case: `Array`
        (Array[int], True),
        # Case: `numpy.ndarray[typing.Any, numpy.dtype[numpy.object_]]`
        (np.ndarray[Any, np.dtype[np.object_]], True),
        # Case: Annotated type that doesn't match expected structure
        (Annotated[np.ndarray[Any, np.dtype[np.float64]], ArrayElementType[int]], False),
        # Case: numpy.ndarray with different dtype
        (np.ndarray[Any, np.dtype[np.float64]], False),
        # Case: Completely unrelated type (e.g., int)
        (int, False),
        # Case: Annotated type that is not related to numpy ndarray
        (Annotated[int, ArrayElementType[int]], False),
    ],
)
def test_is_array_type(tp, expected):
    assert is_object_array_type(tp) == expected


@pytest.mark.parametrize(
    "x",
    [
        Array[int],  # Valid Annotated type
        np.ndarray[Any, np.dtype[np.object_]],  # Valid plain ndarray
        Annotated[
            np.ndarray[Any, np.dtype[np.float64]],
            ArrayElementType[int],
        ],  # Non-matching Annotated
        np.ndarray[Any, np.dtype[np.float64]],  # Non-matching ndarray
        int,  # Unrelated type
        Annotated[int, ArrayElementType[int]],  # Annotated, but not ndarray
    ],
)
def test_is_valid_array_type(x):
    expected_result = x in [
        Array[int],  # Valid Annotated type
        np.ndarray[Any, np.dtype[np.object_]],  # Valid plain ndarray
    ]
    assert is_object_array_type(x) == expected_result


def test_compare_annotated_types_with_different_primary_types():
    # Case: Annotated types with different primary types
    # This should trigger the uncovered lines where primary types are compared and found to be incompatible
    AnnotatedType1 = Annotated[np.ndarray[Any, np.dtype[np.object_]], ArrayElementType[int]]  # noqa: N806
    AnnotatedType2 = Annotated[list[int], ArrayElementType[int]]  # noqa: N806

    # Since np.ndarray and list[int] are different primary types, this should return False
    assert not is_type_compatible(AnnotatedType1, AnnotatedType2)


def test_compatible_types_with_multiple_annotated_fields():
    # Case: Annotated types with multiple annotated fields
    # This should not care about other metadata than ArrayElementType
    AnnotatedType1 = Annotated[np.ndarray[Any, np.dtype[np.object_]], ArrayElementType[int], float]  # noqa: N806
    AnnotatedType2 = Annotated[np.ndarray[Any, np.dtype[np.object_]], ArrayElementType[int], int]  # noqa: N806
    assert is_type_compatible(AnnotatedType1, AnnotatedType2)

    AnnotatedType1 = Annotated[np.ndarray[Any, np.dtype[np.object_]], ArrayElementType[int], float]  # noqa: N806
    AnnotatedType2 = Annotated[np.ndarray[Any, np.dtype[np.object_]], int, ArrayElementType[int]]  # noqa: N806
    assert is_type_compatible(AnnotatedType1, AnnotatedType2)


def test_is_type_compatible_with_generics():
    T = TypeVar("T")
    S = TypeVar("S", str, int)  # Constrained TypeVar
    N = TypeVar("N", bound=Number)  # Bounded TypeVar

    # Original tests
    assert is_type_compatible(list[str], T)
    assert is_type_compatible(list[str], list[T])
    assert is_type_compatible(list[T], list[T])
    assert is_type_compatible(list[list[str]], list[list[T]])
    assert not is_type_compatible(list[list[str]], list[tuple[T]])
    assert not is_type_compatible(list[str], tuple[T])

    # Tests with constrained TypeVar
    assert is_type_compatible(str, S)
    assert is_type_compatible(int, S)
    assert not is_type_compatible(float, S)
    assert is_type_compatible(list[str], list[S])
    assert not is_type_compatible(list[float], list[S])

    # Tests with bounded TypeVar
    assert is_type_compatible(int, N)
    assert is_type_compatible(float, N)
    assert not is_type_compatible(str, N)
    assert is_type_compatible(list[int], list[N])
    assert not is_type_compatible(list[str], list[N])

    # More complex nested structures
    assert is_type_compatible(dict[str, list[int]], dict[T, list[N]])
    assert not is_type_compatible(dict[str, list[str]], dict[T, list[N]])
    assert is_type_compatible(tuple[list[int], dict[str, float]], tuple[list[N], dict[S, N]])
    assert not is_type_compatible(tuple[list[int], dict[str, str]], tuple[list[N], dict[S, N]])

    # Union types with TypeVars
    assert is_type_compatible(Union[int, str], Union[T, S])  # noqa: UP007
    assert is_type_compatible(Union[int, float], Union[N, T])  # noqa: UP007
    assert is_type_compatible(Union[int, str], Union[N, T])  # noqa: UP007

    # Nested TypeVars
    R = TypeVar("R", bound=Sequence[T])
    assert is_type_compatible(list[list[int]], R)
    assert is_type_compatible(tuple[list[str], list[int]], tuple[R, R])
    assert is_type_compatible(list[dict[str, int]], R)
    Q = TypeVar("Q", bound=Sequence[S])
    assert not is_type_compatible(list[dict[str, int]], Q)

    # TypeVar with Any
    A = TypeVar("A", bound=Any)
    assert is_type_compatible(int, A)
    assert is_type_compatible(str, A)
    assert is_type_compatible(list[int], A)
    assert is_type_compatible(dict[str, float], A)

    # Multiple TypeVars
    M = TypeVar("M")
    K = TypeVar("K")
    assert is_type_compatible(dict[str, int], dict[M, K])
    assert is_type_compatible(dict[int, list[str]], dict[M, list[K]])
    assert not is_type_compatible(dict[int, tuple[str, int]], dict[M, list[K]])


def test_is_type_compatible_with_generics_incoming_generic():
    # TODO: We need to properly handle incoming generic types
    # by resolving the generic type parameters. For now, we just
    # skip the check and return True if the incoming type is a TypeVar.
    T = TypeVar("T")
    assert is_type_compatible(T, list[str])
    assert is_type_compatible(list[T], list[str])
    assert is_type_compatible(list[T], list[T])


def test_is_type_compatible_with_unresolvable():
    with pytest.warns(UserWarning, match="Unresolvable type"):
        assert is_type_compatible(Unresolvable("UndefinedType"), int)
        assert is_type_compatible(int, Unresolvable("UndefinedType"))


def test_safe_get_type_hints_basic_types():
    def func(a: int, b: str) -> None:
        pass

    expected = {
        "a": int,
        "b": str,
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected
    assert safe_get_type_hints(func) == get_type_hints(func)


def test_safe_get_type_hints_forward_ref():
    def func(a: UndefinedType) -> None:  # type: ignore[name-defined]  # noqa: F821
        pass

    expected = {
        "a": Unresolvable("UndefinedType"),
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected


def test_safe_get_type_hints_generic_type():
    def func(
        a: list[int],
        b: str | None,
    ) -> None:
        pass

    expected = {
        "a": list[int],
        "b": Optional[str],  # noqa: UP007
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected


def test_safe_get_type_hints_mixed_resolved_unresolved():
    def func(
        a: UndefinedType,  # type: ignore[name-defined]  # noqa: F821
        b: int | UndefinedType,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        pass

    expected = {
        "a": Unresolvable("UndefinedType"),
        "b": Unresolvable("int | UndefinedType"),
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected


def test_safe_get_type_hints_unresolvable_generic():
    def func(
        a: list[UndefinedType],  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        pass

    expected = {
        "a": Unresolvable("list[UndefinedType]"),
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected


def test_safe_get_type_hints_exception_handling():
    def func(
        a: undefined.variable,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        pass

    expected = {
        "a": Unresolvable("undefined.variable"),
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected


def test_safe_get_type_hints_eval_fallback():
    global SomeType
    SomeType = int

    def func(a: SomeType) -> None:  # type: ignore[name-defined]
        pass

    expected = {
        "a": SomeType,
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected


def test_safe_get_type_hints_complex_generic():
    def func(
        a: list[int] | UndefinedType,  # type: ignore[name-defined] # noqa: F821
        b: list[int | str],
    ) -> None:
        pass

    expected = {
        "a": Unresolvable("list[int] | UndefinedType"),
        "b": list[int | str],
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected
    assert str(expected["a"]) == "Unresolvable[list[int] | UndefinedType]"


def test_safe_get_type_hints_no_annotations():
    def func(a, b):
        pass

    expected = {}
    assert safe_get_type_hints(func) == expected


def test_unresolvable_equality():
    a = Unresolvable("A")
    a2 = Unresolvable("A")
    b = Unresolvable("B")
    assert a == a2
    assert a != b
    assert a != 1


def test_safe_get_type_hints_with_annotated():
    def func(a: Annotated[int, str]) -> None:
        pass

    expected1 = {
        "a": int,
        "return": NoneType,
    }
    expected2 = {
        "a": Annotated[int, str],
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected1
    assert safe_get_type_hints(func, include_extras=True) == expected2


def test_safe_get_type_hints_with_class():
    class MyClass:
        x: int

        def __init__(self, a: int, b: str) -> None:
            self.a = a
            self.b = b

    expected = {
        "a": int,
        "b": str,
        "return": NoneType,
    }
    assert safe_get_type_hints(MyClass.__init__) == expected
    assert safe_get_type_hints(MyClass.__init__) == get_type_hints(MyClass.__init__)
