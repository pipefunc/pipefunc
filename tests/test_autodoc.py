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
    docstring = extract_docstrings(function, docstring_parser=style)
    expected = {
        "a": "The first parameter.",
        "b": "The second parameter.",
        "c": "The third parameter.",
    }
    assert docstring == expected
    docstring_auto = extract_docstrings(function)
    assert docstring == docstring_auto
