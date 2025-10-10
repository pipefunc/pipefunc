import inspect
from typing import Any

from pipefunc import NestedPipeFunc, PipeFunc, pipefunc
from pipefunc.typing import NoAnnotation


# --- Helper Functions ---
def simple_func(a: int, b: str = "default") -> float:
    """A simple function for testing."""
    return float(a) + len(b)


def multi_output_func(x: int) -> tuple[int, str]:
    """Function returning multiple outputs."""
    return x * 2, f"Value is {x}"


class CallableClass:
    """A callable class for testing."""

    def __call__(self, data: dict, factor: int = 2) -> dict:
        return {k: v * factor for k, v in data.items()}


# --- Tests for PipeFunc.__signature__ ---


def test_pipefunc_basic_signature() -> None:
    """Test the signature of a basic PipeFunc."""
    pf = PipeFunc(simple_func, output_name="c")
    sig = pf.__signature__

    assert isinstance(sig, inspect.Signature)
    assert list(sig.parameters.keys()) == ["a", "b"]

    # Parameter 'a'
    param_a = sig.parameters["a"]
    assert param_a.name == "a"
    assert param_a.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert param_a.default == inspect.Parameter.empty
    assert param_a.annotation is int

    # Parameter 'b'
    param_b = sig.parameters["b"]
    assert param_b.name == "b"
    assert param_b.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert param_b.default == "default"
    assert param_b.annotation is str

    # Return annotation
    assert sig.return_annotation is float


def test_pipefunc_signature_with_renames() -> None:
    """Test signature reflects renamed parameters."""
    pf = PipeFunc(simple_func, output_name="c", renames={"a": "alpha", "b": "beta"})
    sig = pf.__signature__

    assert list(sig.parameters.keys()) == ["alpha", "beta"]
    assert sig.parameters["alpha"].annotation is int
    assert sig.parameters["beta"].annotation is str
    assert sig.parameters["beta"].default == "default"
    assert sig.return_annotation is float


def test_pipefunc_signature_with_defaults_override() -> None:
    """Test signature reflects overridden defaults."""
    pf = PipeFunc(simple_func, output_name="c", defaults={"b": "new_default"})
    sig = pf.__signature__

    assert sig.parameters["b"].default == "new_default"

    # Update defaults and check again
    pf.update_defaults({"b": "updated_default"}, overwrite=True)
    sig_updated = pf.__signature__
    assert sig_updated.parameters["b"].default == "updated_default"


def test_pipefunc_signature_with_bound_params() -> None:
    """Test signature excludes bound parameters."""
    pf = PipeFunc(simple_func, output_name="c", bound={"b": "bound_value"})
    sig = pf.__signature__

    # 'b' should not be in the signature parameters as it's bound
    assert list(sig.parameters.keys()) == ["a"]
    assert sig.parameters["a"].annotation is int
    assert sig.return_annotation is float


def test_pipefunc_signature_with_multiple_outputs() -> None:
    """Test signature return annotation for tuple output."""
    pf = PipeFunc(multi_output_func, output_name=("double", "message"))
    sig = pf.__signature__

    assert list(sig.parameters.keys()) == ["x"]
    assert sig.parameters["x"].annotation is int
    # The return annotation should be Tuple[int, str]
    assert sig.return_annotation == tuple[int, str]


def test_pipefunc_signature_with_output_picker() -> None:
    """Test signature return annotation when output_picker is used."""

    def pick_first(output: Any, name: str) -> Any:
        return output[0]

    pf = PipeFunc(
        multi_output_func,
        output_name="double",  # Only asking for one output via picker
        output_picker=pick_first,
    )
    sig = pf.__signature__

    # Return annotation should be empty as the picker's return type isn't known
    assert sig.return_annotation == inspect.Parameter.empty


def test_pipefunc_signature_updates_on_change() -> None:
    """Test that the signature is updated after modifying the PipeFunc."""
    pf = PipeFunc(simple_func, output_name="c")
    sig1 = pf.__signature__
    assert list(sig1.parameters.keys()) == ["a", "b"]
    assert sig1.parameters["b"].default == "default"

    # Apply changes
    pf.update_renames({"a": "alpha"})
    pf.update_defaults({"b": "new_default"})
    pf.update_bound({"alpha": 123})  # Bind the newly named 'a'

    # Check updated signature
    sig2 = pf.__signature__
    assert list(sig2.parameters.keys()) == ["b"]  # 'alpha' is now bound
    assert sig2.parameters["b"].default == "new_default"
    assert sig2.return_annotation is float


def test_pipefunc_signature_with_scope() -> None:
    """Test signature reflects scoped parameter names."""
    pf = PipeFunc(simple_func, output_name="c")
    pf.update_scope("my_scope", inputs={"a", "b"}, outputs={"c"})
    sig = pf.__signature__

    assert list(sig.parameters.keys()) == ["my_scope.a", "my_scope.b"]
    assert sig.parameters["my_scope.a"].annotation is int
    assert sig.parameters["my_scope.b"].annotation is str
    assert sig.parameters["my_scope.b"].default == "default"
    assert sig.return_annotation is float  # Output scope doesn't affect return annotation type


def test_pipefunc_signature_callable_class() -> None:
    """Test signature inference for a callable class instance."""
    instance = CallableClass()
    pf = PipeFunc(instance, output_name="result")
    sig = pf.__signature__

    assert list(sig.parameters.keys()) == ["data", "factor"]
    assert sig.parameters["data"].annotation is dict
    assert sig.parameters["factor"].annotation is int
    assert sig.parameters["factor"].default == 2
    assert sig.return_annotation is dict


# --- Tests for NestedPipeFunc.__signature__ ---


def test_nestedpipefunc_signature() -> None:
    """Test the signature of a NestedPipeFunc."""

    @pipefunc("intermediate")
    def func1(x: int, y: str = "y") -> float:
        return float(x + len(y))

    @pipefunc("final")
    def func2(intermediate: float, z: bool = True) -> str:  # noqa: FBT002
        return f"{intermediate}_{z}"

    nf = NestedPipeFunc([func1, func2], ("intermediate", "final"))
    sig = nf.__signature__

    # Parameters should be the external inputs (x, y, z), excluding intermediates
    assert list(sig.parameters.keys()) == ["x", "y", "z"]

    # Check parameter details
    assert sig.parameters["x"].annotation is int
    assert sig.parameters["x"].default == inspect.Parameter.empty
    assert sig.parameters["y"].annotation is str
    assert sig.parameters["y"].default == "y"
    assert sig.parameters["z"].annotation is bool
    assert sig.parameters["z"].default is True

    # Check return annotation (combines outputs)
    assert nf.output_name == ("intermediate", "final")
    assert sig.return_annotation == tuple[float, str]


def test_nestedpipefunc_signature_with_renames_and_defaults() -> None:
    """Test NestedPipeFunc signature with renames and defaults."""

    @pipefunc("intermediate")
    def func1(x: int, y: str = "y") -> float:
        return float(x + len(y))

    @pipefunc("final")
    def func2(intermediate: float, z: bool = True) -> str:  # noqa: FBT002
        return f"{intermediate}_{z}"

    # Rename an input and an output, provide new default
    nf = NestedPipeFunc(
        [func1, func2],
        renames={"x": "input_x", "final": "result"},
    )
    nf.update_defaults({"z": False})  # Override default for z
    assert nf.output_name == ("intermediate", "result")

    sig = nf.__signature__

    # Check parameters
    assert list(sig.parameters.keys()) == ["input_x", "y", "z"]
    assert sig.parameters["input_x"].annotation is int
    assert sig.parameters["y"].default == "y"
    assert sig.parameters["z"].default is False  # Updated default

    # Check return annotation (includes renamed output)
    assert nf.output_name == ("intermediate", "result")
    assert sig.return_annotation == tuple[float, str]


def test_nestedpipefunc_signature_with_bound() -> None:
    """Test NestedPipeFunc signature excludes bound parameters."""

    @pipefunc("intermediate")
    def func1(x: int, y: str = "y") -> float:
        return float(x + len(y))

    @pipefunc("final")
    def func2(intermediate: float, z: bool = True) -> str:  # noqa: FBT002
        return f"{intermediate}_{z}"

    # Bind one of the inputs
    nf = NestedPipeFunc([func1, func2], bound={"y": "bound_y_value"})
    sig = nf.__signature__

    # 'y' should be excluded from the signature
    assert list(sig.parameters.keys()) == ["x", "z"]
    assert sig.parameters["x"].annotation is int
    assert sig.parameters["z"].default is True

    # Check return annotation
    assert nf.output_name == ("intermediate", "final")
    assert sig.return_annotation == tuple[float, str]


def test_pipefunc_signature_no_annotation() -> None:
    """Test signature when type annotations are missing."""

    def no_anno_func(a, b):
        return a + b

    pf = PipeFunc(no_anno_func, output_name="c")
    sig = pf.__signature__

    assert list(sig.parameters.keys()) == ["a", "b"]
    assert sig.parameters["a"].annotation == inspect.Parameter.empty
    assert sig.parameters["b"].annotation == inspect.Parameter.empty
    assert sig.return_annotation == inspect.Parameter.empty


def test_pipefunc_signature_partial_annotation() -> None:
    """Test signature with partial type annotations."""

    def partial_anno_func(a: int, b) -> str:
        return str(a + len(str(b)))

    pf = PipeFunc(partial_anno_func, output_name="c")
    sig = pf.__signature__

    assert list(sig.parameters.keys()) == ["a", "b"]
    assert sig.parameters["a"].annotation is int
    assert sig.parameters["b"].annotation == inspect.Parameter.empty
    assert sig.return_annotation is str


def test_pipefunc_signature_with_renamed_output_tuple() -> None:
    """Test signature when elements of a tuple output are renamed."""
    pf = PipeFunc(
        multi_output_func,
        output_name=("val1", "val2"),
        renames={"val2": "message_renamed"},
    )
    sig = pf.__signature__

    assert pf.output_name == ("val1", "message_renamed")
    # The signature's return annotation reflects the types based on the *original*
    # function's return type hint, before the output name renaming.
    assert sig.return_annotation == tuple[int, str]


def test_scoped_parameter():
    @pipefunc(
        output_name="test",
        renames={"input": "y.input2"},
    )
    def test(input: int = 1) -> Any:  # noqa: A002
        return input

    parameter = test.__signature__.parameters["y.input2"]
    assert parameter.annotation is int
    assert parameter.default == 1


def test_pipefunc_signature_with_multiple_output_names_and_one_no_annotation():
    @pipefunc(output_name="a")
    def f() -> int:
        return 1

    @pipefunc(output_name="b")
    def g(a: int):
        return f"hello_{a}"

    nf = NestedPipeFunc([f, g], output_name=("a", "b"))
    assert nf.output_annotation == {"a": int, "b": NoAnnotation}
    sig = nf.__signature__
    assert sig.return_annotation is inspect.Parameter.empty
