import random
from typing import Any

import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc.testing import patch


@pipefunc(output_name="c")
def f() -> Any:
    msg = "test"
    raise ValueError(msg)


def test_pipeline_patch():
    """Test patching a function in the pipeline successfully."""
    pipeline = Pipeline([f])

    # Use with pytest raises to expect no error and validate the patching
    with patch(pipeline, "f") as mock:
        # Set the mock return value
        mock.return_value = 1

        # Verify the patched behavior of the pipeline
        result = pipeline()
        assert result == 1

        # Ensure the mock was called
        mock.assert_called_once()


def test_multiple():
    my_first = PipeFunc(random.randint, output_name="rnd", defaults={"a": 0, "b": 10})

    @pipefunc(output_name="result")
    def my_second(rnd):
        msg = "This function should be mocked"
        raise RuntimeError(msg)

    pipeline = Pipeline([my_first, my_second])

    # Patch a single function
    with patch(pipeline, "my_second") as mock:
        mock.return_value = 5
        pipeline()

    # Patch multiple functions
    with patch(pipeline, "random.randint") as mock1, patch(pipeline, "my_second") as mock2:
        mock1.return_value = 3
        mock2.return_value = 5
        assert pipeline() == 5

    with pytest.raises(ValueError, match="No function named 'my_third' found in the pipeline."):  # noqa: SIM117
        with patch(pipeline, "my_third"):
            pass
