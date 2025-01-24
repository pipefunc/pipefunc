import pytest

from pipefunc.map._result import ResultDict


class LongRepresentation:
    def __repr__(self) -> str:
        return "a" * 50_000


def test_truncating_result_dict_repr() -> None:
    result = ResultDict()
    assert repr(result) == "{}"
    result["a"] = LongRepresentation()  # type: ignore[assignment]
    with pytest.warns(UserWarning, match="ResultDict is too large to display completely"):
        text = repr(result)
    assert text.startswith(
        "{'a': aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
