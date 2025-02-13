from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._pipeline._autodoc import PipelineDocumentation, format_pipeline_docs
from pipefunc._utils import parse_function_docstring
from pipefunc.typing import NoAnnotation

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


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
    doc = parse_function_docstring(function, docstring_parser=style)
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
    doc_auto = parse_function_docstring(function)  # default is "auto"
    assert doc_auto.parameters == expected_parameters
    assert doc_auto.description == expected_description
    assert doc_auto.returns == expected_return


def test_exception_wrong_parser() -> None:
    with pytest.raises(ValueError, match="Invalid docstring parser"):
        parse_function_docstring(
            my_google_style_function,
            docstring_parser="wrong",  # type: ignore[arg-type]
        )


@pytest.fixture
def pipeline() -> Pipeline:
    @pipefunc(output_name="c")
    def f1(a: int, b: int) -> int:  # Sphinx style
        """This is function f1.

        :param a: Parameter a.
        :param b: Parameter b.

        :return: Description of the return
        """
        return a + b

    @pipefunc(output_name="d")
    def f2(b: int, c: int, x: int = 1) -> int:  # Numpy style
        """This is function f2.

        Parameters
        ----------
        b : int
            Parameter b.
        c : int
            Parameter c.
        x : int, optional
            Parameter x.

        Returns
        -------
        int
            Description of the return
        """
        return b * c * x

    @pipefunc(output_name="e")
    def f3(c: int, d: int, x: int = 1) -> int:  # Google style
        """This is function f3.

        Args:
            c: Parameter c.
            d: Parameter d.
            x: Parameter x.

        Returns:
            Description of the return
        """
        return c * d * x

    return Pipeline([f1, f2, f3])


def test_print_doc_default(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(order="alphabetical")
    captured = capsys.readouterr()
    assert "Function Output Descriptions" in captured.out
    assert "Parameters" in captured.out
    assert "Return Values" in captured.out


def test_print_doc_no_borders(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(borders=False)
    captured = capsys.readouterr()
    assert "─" not in captured.out
    assert "│" not in captured.out
    assert "┓" not in captured.out


def test_print_doc_borders(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(borders=True)
    captured = capsys.readouterr()
    assert "─" in captured.out
    assert "│" in captured.out
    assert "┓" in captured.out


def test_print_doc_skip_optional(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(skip_optional=True)
    captured = capsys.readouterr()
    assert "x" not in captured.out


def test_print_doc_no_skip_optional(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(skip_optional=False)
    captured = capsys.readouterr()
    assert " x " in captured.out


def test_print_doc_skip_intermediate(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(
        skip_intermediate=True,
        description_table=False,
        parameters_table=True,
        returns_table=False,
    )
    captured = capsys.readouterr()
    assert " c " not in captured.out
    assert " d " not in captured.out


def test_print_doc_no_skip_intermediate(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(
        skip_intermediate=False,
        description_table=False,
        parameters_table=True,
        returns_table=False,
    )
    captured = capsys.readouterr()
    assert " c " in captured.out
    assert " d " in captured.out


def test_print_doc_no_tables(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(
        description_table=False,
        parameters_table=False,
        returns_table=False,
    )
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_doc_description_table_only(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(
        description_table=True,
        parameters_table=False,
        returns_table=False,
    )
    captured = capsys.readouterr()
    assert "Function Output Descriptions" in captured.out
    assert "Parameters" not in captured.out
    assert "Return Values" not in captured.out


def test_print_doc_parameters_table_only(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(
        description_table=False,
        parameters_table=True,
        returns_table=False,
    )
    captured = capsys.readouterr()
    assert "Function Output Descriptions" not in captured.out
    assert "Parameters" in captured.out
    assert "Return Values" not in captured.out


def test_print_doc_returns_table_only(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline.print_documentation(
        description_table=False,
        parameters_table=False,
        returns_table=True,
    )
    captured = capsys.readouterr()
    assert "Function Output Descriptions" not in captured.out
    assert "Parameters" not in captured.out
    assert "Return Values" in captured.out


def test_format_pipeline_docs_no_print(pipeline: Pipeline) -> None:
    from rich.table import Table

    tables = format_pipeline_docs(PipelineDocumentation.from_pipeline(pipeline), print_table=False)
    assert tables is not None
    assert len(tables) == 3
    assert all(isinstance(table, Table) for table in tables)


def test_pipeline_doc_return_type_annotation(pipeline: Pipeline) -> None:
    @pipefunc(output_name="c")
    def f1(a: int, b: int) -> list[int]:
        return [a + b]

    pipeline = pipeline.copy(validate_type_annotations=False)
    pipeline.replace(f1)
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.r_annotations["c"] == list[int]


def test_pipeline_doc_return_type_annotation_multiple_outputs(
    pipeline: Pipeline,
    capsys: CaptureFixture,
) -> None:
    @pipefunc(output_name=("c", "d"))
    def f_multiple(a: int, b: int) -> tuple[list[int], set[str]]:
        """Yolo.

        Args:
            a: Parameter a.
            b: Parameter b.

        Returns:
            Description of the return.
        """

        return [a + b], {str(a + b)}

    @pipefunc(output_name="e")
    def g_single(c: list[int], d: set[str]) -> int:
        """Foo.

        :param c: Parameter c.
        :param d: Parameter d.

        :return: Description of the return.
        """
        return 1

    pipeline = Pipeline([f_multiple, g_single])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.r_annotations["c"] == list[int]
    assert doc.r_annotations["d"] == set[str]
    pipeline.print_documentation()
    assert "Description of the return. (type c: list) (type d: set)" in capsys.readouterr().out


def test_pipeline_doc_return_type_annotation_multiple_outputs_same_type(
    pipeline: Pipeline,
) -> None:
    @pipefunc(output_name=("c", "d"))
    def f_multiple(a: int, b: int) -> tuple[int, int]:
        return a + b, a + b

    pipeline = Pipeline([f_multiple])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.r_annotations["c"] is int
    assert doc.r_annotations["d"] is int
    pipeline.print_documentation()  # just to check no errors


def test_pipeline_doc_return_type_annotation_no_type(pipeline: Pipeline) -> None:
    @pipefunc(output_name=("c", "d"))
    def f_multiple(a: int, b: int):
        return a + b, a + b

    pipeline = Pipeline([f_multiple])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert "c" not in doc.p_annotations
    assert "d" not in doc.p_annotations
    pipeline.print_documentation()  # just to check no errors


def test_pipeline_doc_parameter_annotation_multiple_outputs(pipeline: Pipeline) -> None:
    @pipefunc(output_name=("c", "d"))
    def f_multiple(a: int, b: str):
        return a, b

    pipeline = Pipeline([f_multiple])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.p_annotations["a"] is int
    assert doc.p_annotations["b"] is str


def test_pipeline_doc_parameter_annotation_no_type(pipeline: Pipeline) -> None:
    @pipefunc(output_name=("c", "d"))
    def f_multiple(a, b):
        return a, b

    pipeline = Pipeline([f_multiple])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert "a" not in doc.p_annotations
    assert "b" not in doc.p_annotations


def test_pipeline_doc_parameter_annotation_conflicting_types(pipeline: Pipeline) -> None:
    @pipefunc(output_name="c")
    def f1(a: int):
        return a

    @pipefunc(output_name="d")
    def f2(a: str):
        return a

    pipeline = Pipeline([f1, f2])
    with pytest.warns(
        UserWarning,
        match="Conflicting annotations for parameter",
    ):
        PipelineDocumentation.from_pipeline(pipeline)


def test_pipeline_doc_descriptions_empty_dict_when_no_descriptions(pipeline: Pipeline) -> None:
    # Remove the descriptions from the functions
    for f in pipeline.functions:
        f.func.__doc__ = ""
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.descriptions == {"c": "—", "d": "—", "e": "—"}


def test_pipeline_doc_defaults(pipeline: Pipeline) -> None:
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.defaults == {"x": 1}


def test_pipeline_doc_root_args(pipeline: Pipeline) -> None:
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert set(doc.root_args) == {"a", "b", "x"}


def test_pipeline_doc_descriptions(pipeline: Pipeline) -> None:
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.descriptions == {
        "c": "This is function f1.",
        "d": "This is function f2.",
        "e": "This is function f3.",
    }


def test_pipeline_doc_parameters(pipeline: Pipeline) -> None:
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.parameters == {
        "a": ["Parameter a."],
        "b": ["Parameter b."],
        "c": ["Parameter c."],
        "x": ["Parameter x."],
        "d": ["Parameter d."],
    }


def test_pipeline_doc_returns(pipeline: Pipeline) -> None:
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.returns == {
        "c": "Description of the return",
        "d": "Description of the return",
        "e": "Description of the return",
    }


def test_pipeline_doc_annotations(pipeline: Pipeline) -> None:
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.p_annotations == {
        "a": int,
        "b": int,
        "c": int,
        "d": int,
        "x": int,
    }
    assert doc.r_annotations == {
        "c": int,
        "d": int,
        "e": int,
    }


def test_autodoc_no_docstring(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    for f in pipeline.functions:
        f.func.__doc__ = ""
    pipeline.print_documentation()
    captured = capsys.readouterr()
    assert "(type: int)" in captured.out


def test_autodoc_missing_parameter_and_return(capsys: CaptureFixture) -> None:
    @pipefunc(output_name="c")
    def f1(a: int, b: int) -> float:
        """This is function f1.

        :param a: Parameter a.
        """
        return (a + b) / 2

    pipeline = Pipeline([f1])

    pipeline.print_documentation()
    captured = capsys.readouterr()
    assert "— (type: int)" in captured.out  # missing parameter b
    assert "— (type: float)" in captured.out  # missing return value
    assert "This is function f1." in captured.out
    assert "Parameter a." in captured.out


def test_renames(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline = pipeline.copy()
    pipeline.update_renames({"a": "alpha", "b": "beta", "e": "epsilon"})
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert "alpha" in doc.parameters
    assert "beta" in doc.parameters
    assert "a" not in doc.parameters
    assert "b" not in doc.parameters
    assert "epsilon" in doc.returns
    assert "e" not in doc.returns
    pipeline.print_documentation()
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" in out
    assert "epsilon" in out


def test_no_annotations(pipeline: Pipeline, capsys: CaptureFixture) -> None:
    pipeline = pipeline.copy()

    @pipefunc(output_name="c")
    def f1(a, b):  # Sphinx style
        """This is function f1.

        :param a: Parameter a.
        """
        return a + b

    pipeline.replace(f1)
    pipeline.update_defaults({"b": 2})
    pipeline.print_documentation()
    captured = capsys.readouterr()
    assert "— (type: NoAnnotation)" in captured.out
    assert "Parameter b. (default: 2) (type: int)" in captured.out


def test_bound_parameter_is_not_in_docstring(
    pipeline: Pipeline,
    capsys: CaptureFixture,
) -> None:
    @pipefunc(output_name="c", bound={"b": 10})
    def f1(a: int, b: int = 2) -> int:
        """This is function f1.

        :param a: Parameter a.
        :param b: Parameter b.
        """
        return a + b

    pipeline = Pipeline([f1])
    info = pipeline.info()
    assert info is not None
    inputs = info["inputs"]
    assert "a" in inputs
    assert "b" not in inputs
    pipeline.print_documentation()
    captured = capsys.readouterr()
    assert "Parameter a" in captured.out
    assert "Parameter b" not in captured.out


def test_same_parameter_in_multiple_functions_different_doc(
    capsys: CaptureFixture,
) -> None:
    @pipefunc(output_name="c")
    def f1(a: int, b: int) -> int:
        """This is function f1.

        :param a: Foo a.
        :param b: Parameter b.
        """
        return a + b

    @pipefunc(output_name="d")
    def f2(a: int, c: int) -> int:
        """This is function f2.

        :param a: Bar a.
        :param c: Parameter c.
        """
        return a * c

    pipeline = Pipeline([f1, f2])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert len(doc.parameters["a"]) == 2
    pipeline.print_documentation()
    out = capsys.readouterr().out
    assert "1. Foo a." in out
    assert "2. Bar a." in out


def test_output_annotation_multiple_outputs(pipeline: Pipeline) -> None:
    # Here we cannot extract the type annotation because of the output_picker
    @pipefunc(output_name=("c", "d"), output_picker=lambda x, key: x[key])
    def f_multiple(a: int, b: int) -> dict[str, int]:
        """Yolo.

        Args:
            a: Parameter a.
            b: Parameter b.

        Returns:
            Description of the return.
        """

        return {"c": a + b, "d": a - b}

    pipeline = Pipeline([f_multiple])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.r_annotations["c"] == NoAnnotation
    assert doc.r_annotations["d"] == NoAnnotation


def test_dataclass_docstring(capsys: CaptureFixture) -> None:
    @dataclass
    class Foo:
        """
        This is a Foo class.

        :param a: The first parameter.
        :param b: The second parameter.
        """

        a: int
        b: str

    pipeline = Pipeline([PipeFunc(Foo, "foo")])
    doc = PipelineDocumentation.from_pipeline(pipeline)
    assert doc.parameters == {"a": ["The first parameter."], "b": ["The second parameter."]}
    pipeline.print_documentation()
    out = capsys.readouterr().out
    assert "This is a Foo class." in out
    assert "— (type: Foo)" in out  # returns section
