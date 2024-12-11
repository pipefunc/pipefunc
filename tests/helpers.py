from pipefunc import pipefunc
from pipefunc.pipeline import Pipeline


@pipefunc(output_name="test_function")
def test_function(arg1: str, arg2: str) -> str:
    return f"{arg1} {arg2}"


pipeline_test_function = Pipeline([test_function])
