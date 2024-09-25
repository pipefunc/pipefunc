from typing import Any

from pipefunc import Pipeline, pipefunc
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
