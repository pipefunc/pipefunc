import pytest

from pipefunc._utils import extract_docstrings


def my_google_style_function(a: int, b: str = "hello", c: float = 3.14):
    """
    This is my function.

    Args:
        a: The first parameter.
        b: The second parameter.
        c: The third parameter.
    """


def my_numpy_style_function(a: int, b: str = "hello", c: float = 3.14):
    """
    This is my function.

    Parameters
    ----------
    a : int
        The first parameter.
    b : str, optional
        The second parameter.
    c : float, optional
        The third parameter.
    """


def my_numpydoc_style_function_no_types(a: int, b: str = "hello", c: float = 3.14):
    """
    This is my function.

    Parameters
    ----------
    a
        The first parameter.
    b
        The second parameter.
    c
        The third parameter.
    """


def my_sphinx_style_function(a: int, b: str = "hello", c: float = 3.14):
    """
    This is my function.

    :param a: The first parameter.
    :param b: The second parameter.
    :param c: The third parameter.
    """


@pytest.mark.parametrize(
    ("function", "style"),
    [
        (my_google_style_function, "google"),
        (my_numpy_style_function, "numpy"),
        (my_numpydoc_style_function_no_types, "numpy"),
        (my_sphinx_style_function, "sphinx"),
    ],
)
def test_extract_docstring(function, style) -> None:
    doc = extract_docstrings(function, docstring_parser=style)
    expected_parameters = {
        "a": "The first parameter.",
        "b": "The second parameter.",
        "c": "The third parameter.",
    }
    assert doc.parameters == expected_parameters
    expected_description = "This is my function."
    assert doc.description == expected_description
    doc_auto = extract_docstrings(function)  # default is "auto"
    assert doc_auto.parameters == expected_parameters
    assert doc_auto.description == expected_description


def test_exception_wrong_parser() -> None:
    with pytest.raises(ValueError, match="Invalid docstring parser"):
        extract_docstrings(
            my_google_style_function,
            docstring_parser="wrong",  # type: ignore[arg-type]
        )
