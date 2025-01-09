from pipefunc import Pipeline, VariantPipeline, pipefunc


def test_variant_pipeline_single_group():
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
    pipeline_sub = pipeline.with_variant(select="sub")
    assert pipeline_add(a=1, b=2) == 3
    assert pipeline_sub(a=1, b=2) == -1


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
