from __future__ import annotations

import importlib.util

import pytest

from pipefunc import PipeFunc, Pipeline, VariantPipeline, pipefunc
from pipefunc._variant_pipeline import is_identical_pipefunc

has_psutil = importlib.util.find_spec("psutil") is not None


def test_variant_pipeline_single_group() -> None:
    # Define functions with variants and variant groups
    @pipefunc(output_name="c", variant="add")
    def f(a, b):
        print("Running f (add)")
        return a + b

    @pipefunc(output_name="c", variant="sub")
    def f_alt(a, b):
        print("Running f_alt (sub)")
        return a - b

    # Function without a variant
    @pipefunc(output_name="d")
    def g(b, c, x=1):
        print("Running g")
        return b * c * x

    # Create a VariantPipeline with a default variant
    pipeline = VariantPipeline([f, f_alt, g], default_variant="add")
    pipeline_add = pipeline.with_variant()  # Default variant
    assert isinstance(pipeline_add, Pipeline)
    assert len(pipeline_add.functions) == 2
    pipeline_sub = pipeline.with_variant(select="sub")
    assert isinstance(pipeline_sub, Pipeline)
    assert len(pipeline_sub.functions) == 2
    assert pipeline_add(a=1, b=2) == 2 * (1 + 2) * 1
    assert pipeline_sub(a=1, b=2) == 2 * (1 - 2) * 1

    # Test invariant with `from_pipelines`
    pipelines = VariantPipeline.from_pipelines(("add", pipeline_add), ("sub", pipeline_sub))
    pipeline_add2 = pipelines.with_variant(select="add")
    pipeline_sub2 = pipelines.with_variant(select="sub")
    assert isinstance(pipeline_add2, Pipeline)
    assert isinstance(pipeline_sub2, Pipeline)
    assert pipeline_add2(a=1, b=2) == 2 * (1 + 2) * 1
    assert pipeline_sub2(a=1, b=2) == 2 * (1 - 2) * 1


def test_variant_pipeline_multiple_groups() -> None:
    # Define functions with variants and variant groups
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        print("Running f (add)")
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        print("Running f_alt (sub)")
        return a - b

    @pipefunc(output_name="d", variant_group="op2", variant="mul")
    def g(b, c, x=3):
        print("Running g (mul)")
        return b * c * x

    @pipefunc(output_name="d", variant_group="op2", variant="div")
    def g_alt(b, c, x=3):
        print("Running g_alt (div)")
        return b * c / x

    # Function without a variant
    @pipefunc(output_name="e")
    def h(c, d, x=3):
        print("Running h")
        return c * d * x

    # Create a VariantPipeline with a default variant
    pipeline = VariantPipeline([f, f_alt, g, g_alt, h], default_variant="add")

    pipeline_add = pipeline.with_variant(select={"op1": "add", "op2": "mul"})
    pipeline_sub = pipeline.with_variant(select={"op1": "sub", "op2": "div"})
    assert isinstance(pipeline_add, Pipeline)
    assert isinstance(pipeline_sub, Pipeline)
    assert pipeline_sub(a=1, b=2) == (2 * (1 - 2) / 3) * (1 - 2) * 3
    assert pipeline_add(a=1, b=2) == (2 * (1 + 2) * 3) * (1 + 2) * 3


@pytest.mark.skipif(not has_psutil, reason="psutil not installed")
def test_lazy_debug_profile_cache() -> None:
    # We just check that these parameters are passed through to the Pipeline
    # No need to test the actual functionality of these parameters,
    # as they are tested in test_pipeline.py
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline(
        [f, f_alt],
        lazy=True,
        debug=True,
        profile=True,
        cache_type="simple",
        cache_kwargs={"arg": "val"},
    )
    with pytest.raises(ValueError, match="No variant selected and no default variant provided"):
        pipeline.with_variant()
    pipeline_add = pipeline.with_variant(select="add")

    assert isinstance(pipeline_add, Pipeline)
    assert pipeline_add.lazy
    assert pipeline_add.debug
    assert pipeline_add.profile
    assert pipeline_add._cache_type == "simple"
    assert pipeline_add._cache_kwargs == {"arg": "val"}
    assert pipeline_add(a=1, b=2).evaluate() == 3


def test_validate_type_annotations() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a: int, b: int) -> int:
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a: int, b: int) -> str:
        return str(a - b)

    pipeline = VariantPipeline([f, f_alt], validate_type_annotations=False)
    pipeline_add = pipeline.with_variant(select="add")
    assert isinstance(pipeline_add, Pipeline)
    assert not pipeline_add.validate_type_annotations

    # Now test that the Pipeline does raise a TypeError when validate_type_annotations=True
    # with an invalid type annotation
    @pipefunc(output_name="d")
    def g(c: str):  # Incorrect type annotation
        return c

    pipeline = VariantPipeline([f, f_alt, g], validate_type_annotations=True)
    with pytest.raises(TypeError, match="Inconsistent type annotations"):
        pipeline.with_variant(select="add")


def test_error_handling_ambiguous_variant() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op2", variant="add")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline([f, f_alt])

    with pytest.raises(ValueError, match="Ambiguous variant"):
        pipeline.with_variant(select="add")


def test_error_handling_with_variant() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    pipeline = VariantPipeline([f, f_alt])

    with pytest.raises(ValueError, match="Unknown variant"):
        pipeline.with_variant(select="mul")

    with pytest.raises(TypeError, match="Invalid variant type"):
        pipeline.with_variant(select=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown variant group"):
        pipeline.with_variant(select={"op2": "add"})

    with pytest.raises(ValueError, match="Unknown variant"):
        pipeline.with_variant(select={"op1": "mul"})


def test_variants_mapping_and_inverse() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    @pipefunc(output_name="d", variant_group="op2", variant="mul")
    def g(b, c):
        return b * c

    # Function without a variant
    @pipefunc(output_name="e")
    def h(c, d):
        return c * d

    pipeline = VariantPipeline([f, f_alt, g, h])
    assert pipeline.variants_mapping() == {
        "op1": {"add", "sub"},
        "op2": {"mul"},
    }

    assert pipeline._variants_mapping_inverse() == {
        "add": {"op1"},
        "sub": {"op1"},
        "mul": {"op2"},
    }


def test_copy_method() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    pipeline = VariantPipeline([f], debug=True, profile=False, cache_type="simple")
    pipeline_copy = pipeline.copy(debug=False, profile=True, cache_type="lru")

    assert pipeline_copy.debug is False
    assert pipeline_copy.profile is True
    assert pipeline_copy.cache_type == "lru"
    assert pipeline_copy.functions == pipeline.functions
    assert pipeline_copy.variants_mapping() == pipeline.variants_mapping()


def test_getattr_for_pipeline_attributes() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    pipeline = VariantPipeline([f])
    with pytest.raises(
        AttributeError,
        match="This is a `VariantPipeline`, not a `Pipeline`",
    ):
        _ = pipeline.map  # type: ignore[attr-defined]

    with pytest.raises(AttributeError, match="'VariantPipeline' object has no attribute"):
        _ = pipeline.doesnotexist  # type: ignore[attr-defined]


def test_default_variant() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    @pipefunc(output_name="d", variant_group="op2", variant="mul")
    def g(b, c):
        return b * c

    @pipefunc(output_name="d", variant_group="op2", variant="div")
    def g_alt(b, c):
        return b / c

    # Test with a single default variant
    pipeline = VariantPipeline([f, f_alt, g, g_alt], default_variant="add")
    pipeline_add = pipeline.with_variant()
    assert {func.variant for func in pipeline_add.functions} == {"add", "mul", "div"}

    # Test with a default variant per group
    pipeline = VariantPipeline(
        [f, f_alt, g, g_alt],
        default_variant={"op1": "sub", "op2": "div"},
    )
    pipeline_sub_div = pipeline.with_variant()
    assert [func.variant for func in pipeline_sub_div.functions] == ["sub", "div"]


def test_variant_pipeline_with_no_variant_group() -> None:
    @pipefunc(output_name="c", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant="sub")
    def f_alt(a, b):
        return a - b

    @pipefunc(output_name="d")  # This function has no variant group or variant
    def g(b, c):
        return b * c

    pipeline = VariantPipeline([f, f_alt, g], default_variant="add")
    pipeline_add = pipeline.with_variant()
    assert isinstance(pipeline_add, Pipeline)
    assert len(pipeline_add.functions) == 2
    assert pipeline_add(a=1, b=2) == 6  # (1 + 2) * 2

    pipeline_sub = pipeline.with_variant(select="sub")
    assert isinstance(pipeline_sub, Pipeline)
    assert len(pipeline_sub.functions) == 2
    assert pipeline_sub(a=1, b=2) == -2  # (1 - 2) * 2


def test_variant_pipeline_copy_with_different_functions() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d", variant_group="op2", variant="mul")
    def g(b, c):
        return b * c

    pipeline = VariantPipeline([f, g], default_variant={"op1": "add", "op2": "mul"})

    # Create new functions to replace the existing ones in the copy
    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    @pipefunc(output_name="d", variant_group="op2", variant="div")
    def g_alt(b, c):
        return b / c

    # Copy the pipeline with new functions
    pipeline_copy = pipeline.copy(functions=[f_alt, g_alt])

    # Check that the copied pipeline has the new functions
    assert pipeline_copy.functions == [f_alt, g_alt]

    # Check that the original pipeline is unchanged
    assert pipeline.functions == [f, g]

    # Test the copied pipeline with a variant selection
    pipeline_sub_div = pipeline_copy.with_variant(
        select={"op1": "sub", "op2": "div"},
    )
    assert isinstance(pipeline_sub_div, Pipeline)
    assert pipeline_sub_div(a=4, b=2) == 1.0  # (4 - 2) / 2


def test_variant_pipeline_copy_with_different_default_variant() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    @pipefunc(output_name="d", variant_group="op2", variant="mul")
    def g(b, c):
        return b * c

    @pipefunc(output_name="d", variant_group="op2", variant="div")
    def g_alt(b, c):
        return b / c

    pipeline = VariantPipeline([f, f_alt, g, g_alt], default_variant="add")

    # Copy the pipeline with a different default_variant
    pipeline_copy = pipeline.copy(default_variant={"op1": "sub", "op2": "div"})

    # Check that the copied pipeline has the new default_variant
    assert pipeline_copy.default_variant == {"op1": "sub", "op2": "div"}

    # Check that the original pipeline is unchanged
    assert pipeline.default_variant == "add"

    # Test the copied pipeline with the new default_variant
    pipeline_sub_div = pipeline_copy.with_variant()
    assert isinstance(pipeline_sub_div, Pipeline)
    assert pipeline_sub_div(a=4, b=2) == 1.0  # (4 - 2) / 2

    # Test the original pipeline with its default_variant
    assert pipeline.variants_mapping() == {"op1": {"add", "sub"}, "op2": {"mul", "div"}}
    pipeline_add = pipeline.with_variant()
    assert isinstance(pipeline_add, VariantPipeline)
    assert pipeline_add.variants_mapping() == {"op1": {"add"}, "op2": {"mul", "div"}}


def test_variant_pipeline_with_variant_as_input_to_another_function() -> None:
    @pipefunc(output_name="c", variant_group="op", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op", variant="sub")
    def f_alt(a, b):
        return a - b

    @pipefunc(output_name="d")
    def g(c):
        return c * 2

    pipeline = VariantPipeline([f, f_alt, g], default_variant="add")

    # Test with the default variant
    pipeline_add = pipeline.with_variant()
    assert isinstance(pipeline_add, Pipeline)
    assert pipeline_add(a=1, b=2) == 6  # (1 + 2) * 2

    # Test with a selected variant
    pipeline_sub = pipeline.with_variant(select={"op": "sub"})
    assert isinstance(pipeline_sub, Pipeline)
    assert pipeline_sub(a=1, b=2) == -2  # (1 - 2) * 2


def test_pipefunc_with_variant_group_but_no_variant_error() -> None:
    def f(a, b):
        return a + b

    with pytest.raises(
        ValueError,
        match="`variant_group='add'` cannot be set without a corresponding `variant`",
    ):
        PipeFunc(f, output_name="c", variant_group="add")


def test_variant_pipeline_with_no_variants() -> None:
    @pipefunc(output_name="c")
    def f(a, b):
        return a + b

    @pipefunc(output_name="d")
    def g(b, c):
        return b * c

    with pytest.raises(
        ValueError,
        match="No variants found in the pipeline. Use a regular `Pipeline` instead.",
    ):
        VariantPipeline([f, g])


def test_variant_pipeline_with_only_some_functions_having_variants() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    @pipefunc(output_name="d")  # This function does not have a variant
    def g(b, c):
        return b * c

    pipeline = VariantPipeline([f, f_alt, g], default_variant="add")

    # Test with the default variant
    pipeline_add = pipeline.with_variant()
    assert isinstance(pipeline_add, Pipeline)
    assert pipeline_add(a=1, b=2) == 6  # (1 + 2) * 2

    # Test with a selected variant
    pipeline_sub = pipeline.with_variant(select={"op1": "sub"})
    assert isinstance(pipeline_sub, Pipeline)
    assert pipeline_sub(a=1, b=2) == -2  # (1 - 2) * 2


def test_variant_pipeline_with_default_variant_not_in_functions() -> None:
    @pipefunc(output_name="c", variant_group="op1", variant="add")
    def f(a, b):
        return a + b

    @pipefunc(output_name="c", variant_group="op1", variant="sub")
    def f_alt(a, b):
        return a - b

    # The default variant "mul" is not in any of the functions
    pipeline = VariantPipeline([f, f_alt], default_variant="mul")

    with pytest.raises(ValueError, match="Unknown variant"):
        pipeline.with_variant()


def test_from_pipelines_basic() -> None:
    @pipefunc(output_name="x")
    def f(a, b):
        return a + b

    @pipefunc(output_name="y")
    def g(x, c):
        return x * c

    pipeline1 = Pipeline([f, g])
    pipeline2 = Pipeline([f, g.copy(func=lambda x, c: x / c)])
    variant_pipeline = VariantPipeline.from_pipelines(
        ("add_mul", pipeline1),
        ("add_div", pipeline2),
    )
    add_mul_pipeline = variant_pipeline.with_variant(select="add_mul")
    add_div_pipeline = variant_pipeline.with_variant(select="add_div")
    assert isinstance(add_mul_pipeline, Pipeline)
    assert isinstance(add_div_pipeline, Pipeline)
    assert add_mul_pipeline(a=1, b=2, c=3) == 9  # (1 + 2) * 3
    assert add_div_pipeline(a=1, b=2, c=3) == 1.0  # (1 + 2) / 3


def test_from_pipelines_with_variant_groups() -> None:
    @pipefunc(output_name="x")
    def f(a, b):
        return a + b

    @pipefunc(output_name="y")
    def g(x, c):
        return x * c

    pipeline1 = Pipeline([f, g])
    pipeline2 = Pipeline([f.copy(func=lambda a, b: a - b), g.copy(func=lambda x, c: x / c)])
    variant_pipeline = VariantPipeline.from_pipelines(
        ("group", "add_mul", pipeline1),
        ("group", "sub_div", pipeline2),
    )
    add_mul_pipeline = variant_pipeline.with_variant(select={"group": "add_mul"})
    sub_div_pipeline = variant_pipeline.with_variant(select={"group": "sub_div"})
    assert isinstance(add_mul_pipeline, Pipeline)
    assert isinstance(sub_div_pipeline, Pipeline)
    assert add_mul_pipeline(a=1, b=2, c=3) == 9  # (1 + 2) * 3
    assert sub_div_pipeline(a=1, b=2, c=3) == -1 / 3  # (1 - 2) / 3


def test_from_pipelines_with_common_function_different_variant() -> None:
    @pipefunc(output_name="x")
    def f(a, b):
        return a + b

    @pipefunc(output_name="y")
    def g(x, c):
        return x * c

    pipeline1 = Pipeline([f, g])
    pipeline2 = Pipeline([f, g.copy(func=lambda x, c: x / c)])

    # f is common, but in pipeline2 it will be given a variant group and name
    variant_pipeline = VariantPipeline.from_pipelines(
        ("group1", "add_mul", pipeline1),
        ("group1", "add_div", pipeline2),
    )
    assert len(variant_pipeline.functions) == 3
    assert is_identical_pipefunc(variant_pipeline.functions[0], f)  # The common function f
    assert variant_pipeline.functions[1].variant == "add_mul"
    assert variant_pipeline.functions[2].variant == "add_div"


def test_exception_no_pipelines() -> None:
    with pytest.raises(ValueError, match="At least 2 pipelines must be provided"):
        VariantPipeline.from_pipelines()
