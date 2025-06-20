from __future__ import annotations

import pytest

from pipefunc._pipefunc_utils import handle_error


def function_that_raises_empty_args() -> None:
    # Some exceptions can be raised with no arguments
    raise ValueError


def test_handle_error_empty_args() -> None:
    # We use a context manager to ensure the exception is raised
    with pytest.raises(ValueError) as exc_info:  # noqa: PT011, PT012
        try:
            function_that_raises_empty_args()
        except Exception as e:  # noqa: BLE001
            handle_error(e, function_that_raises_empty_args, {})

    # Get the actual error message from the exception
    error_message = str(exc_info.value)
    # The message should contain our added text even if original exception was empty
    msg = "Error occurred while executing function"
    assert msg in error_message or msg in exc_info.value.__notes__[0]
    func_name = "function_that_raises_empty_args"
    assert func_name in error_message or func_name in exc_info.value.__notes__[0]


def test_handle_error_with_args() -> None:
    original_message = "Original error message"
    with pytest.raises(ValueError) as exc_info:  # noqa: PT011, PT012
        try:
            raise ValueError(original_message)  # noqa: TRY301
        except Exception as e:  # noqa: BLE001
            handle_error(e, function_that_raises_empty_args, {})

    error_message = str(exc_info.value)
    assert original_message in error_message
    msg = "Error occurred while executing function"
    assert msg in error_message or msg in exc_info.value.__notes__[0]
