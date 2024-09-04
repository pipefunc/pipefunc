from __future__ import annotations

from typing import get_type_hints

from pipefunc.typing import safe_get_type_hints

NoneType = type(None)


def test_safe_get_type_hints_with_string_annotations():
    def func(a: int, b: str) -> None:
        pass

    expected = {
        "a": int,
        "b": str,
        "return": NoneType,
    }
    assert safe_get_type_hints(func) == expected
    assert safe_get_type_hints(func) == get_type_hints(func)
