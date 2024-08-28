from __future__ import annotations

from pipefunc.typing import safe_get_type_hints


def test_safe_get_type_hints_with_string_annotations():
    def func(a: int, b: str) -> None:  # noqa: ARG001
        pass

    expected = {
        "a": int,
        "b": str,
        "return": None,
    }
    assert safe_get_type_hints(func) == expected
