import pytest

from pipefunc._utils import extract_docstrings


def my_google_style_function(a: int, b: str = "hello", c: float = 3.14):
    """
    This is my function.

    Args:
        a: The first parameter.
        b: The second parameter.
        c: The third parameter.

    Returns:
        Description of the return value.
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

    Returns
    -------
    int
        Description of the return value.
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

    Returns
    -------
        Description of the return value.
    """


def my_sphinx_style_function(a: int, b: str = "hello", c: float = 3.14):
    """
    This is my function.

    :param a: The first parameter.
    :param b: The second parameter.
    :param c: The third parameter.

    :return: Description of the return value.
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
    expected_description = "This is my function."
    expected_parameters = {
        "a": "The first parameter.",
        "b": "The second parameter.",
        "c": "The third parameter.",
    }
    expected_return = "Description of the return value."
    assert doc.parameters == expected_parameters
    assert doc.description == expected_description
    assert doc.returns == expected_return
    doc_auto = extract_docstrings(function)  # default is "auto"
    assert doc_auto.parameters == expected_parameters
    assert doc_auto.description == expected_description
    assert doc_auto.returns == expected_return


def test_exception_wrong_parser() -> None:
    with pytest.raises(ValueError, match="Invalid docstring parser"):
        extract_docstrings(
            my_google_style_function,
            docstring_parser="wrong",  # type: ignore[arg-type]
        )
