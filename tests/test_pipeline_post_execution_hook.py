from pipefunc import Pipeline, pipefunc


def test_post_execution_hook() -> None:
    hook_calls = []

    def hook(func, result, kwargs):
        hook_calls.append((func.output_name, result, kwargs))

    @pipefunc(output_name="c", post_execution_hook=hook)
    def f(a, b):
        return a + b

    # Test direct function call
    result = f(a=1, b=2)
    assert result == 3
    assert len(hook_calls) == 1
    assert hook_calls[0] == ("c", 3, {"a": 1, "b": 2})

    # Test in pipeline
    @pipefunc(output_name="d", post_execution_hook=hook)
    def g(c):
        return c * 2

    pipeline = Pipeline([f, g])
    result = pipeline(a=1, b=2)
    assert result == 6
    assert len(hook_calls) == 3  # Two more calls from pipeline execution
    assert hook_calls[1] == ("c", 3, {"a": 1, "b": 2})
    assert hook_calls[2] == ("d", 6, {"c": 3})


def test_post_execution_hook_with_map() -> None:
    hook_calls = []

    def hook(func, result, kwargs):
        hook_calls.append((func.output_name, result, kwargs))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]", post_execution_hook=hook)
    def f(x):
        return x * 2

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3]}
    pipeline.map(inputs, parallel=False, storage="dict")

    assert len(hook_calls) == 3
    assert hook_calls[0] == ("y", 2, {"x": 1})
    assert hook_calls[1] == ("y", 4, {"x": 2})
    assert hook_calls[2] == ("y", 6, {"x": 3})
