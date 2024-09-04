import pytest

from pipefunc import Pipeline, pipefunc


@pytest.fixture
def pipeline() -> Pipeline:
    @pipefunc(output_name="out1", defaults={"c": 1, "d": 1, "e": 1})
    def f1(a, b, c, d, e):
        return a + b + c + d + e

    @pipefunc(output_name="out2", defaults={"f": 1, "g": 1})
    def f2(c, d, e, f, g, out1=1):
        return c + d + e + f + g + out1

    @pipefunc(output_name="out3", defaults={"h": 1, "i": 1})
    def f3(e, f, g, h, i, out1=1, out2=1):
        return e + f + g + h + i + out1 + out2

    @pipefunc(output_name="out4", defaults={"j": 1})
    def f4(g, h, i, j, out1=1, out2=1, out3=1):
        return g + h + i + j + out1 + out2 + out3

    @pipefunc(output_name="out5", bound={"k": 1})
    def f5(h, i, j, k, out1=1, out2=1, out3=1, out4=1):
        return h + i + j + k + out1 + out2 + out3 + out4

    return Pipeline([f1, f2, f3, f4, f5])


@pytest.fixture
def pipeline_mapspec(pipeline: Pipeline) -> Pipeline:
    pipeline = pipeline.copy()
    pipeline.add_mapspec_axis("a", axis="i")
    pipeline.add_mapspec_axis("b", axis="j")
    return pipeline


@pytest.fixture
def pipeline_with_cache(pipeline: Pipeline) -> Pipeline:
    pipeline = pipeline.copy(cache_type="simple")
    for func in pipeline.functions:
        func.cache = True
    return pipeline


@pytest.mark.benchmark
def test_calling_pipeline_directly(pipeline: Pipeline) -> None:
    for a in range(10):
        for b in range(10):
            pipeline(a=a, b=b)


@pytest.mark.benchmark
def test_calling_pipeline_directly_with_cache(pipeline_with_cache: Pipeline) -> None:
    for a in range(10):
        for b in range(10):
            pipeline_with_cache(a=a, b=b)


@pytest.mark.benchmark
def test_map_sequential_with_dict_storage(pipeline_mapspec: Pipeline) -> None:
    a = list(range(10))
    b = list(range(10))

    pipeline_mapspec.map(
        inputs={"a": a, "b": b},
        parallel=False,
        storage="dict",
    )
