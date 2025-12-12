"""Tests for the resume_validation parameter in pipeline.map()."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc

if TYPE_CHECKING:
    from pathlib import Path


class BrokenEqualityInput:  # noqa: PLW1641
    """Input class with broken __eq__ that always returns True."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        # Broken equality - always returns True
        return True


class NoEqualityInput:  # noqa: PLW1641
    """Input class that raises an exception when comparing."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        msg = "Comparison not supported"
        raise NotImplementedError(msg)


def test_resume_validation_auto_with_broken_eq(tmp_path: Path, capsys) -> None:
    """Test that 'auto' mode warns but proceeds when equality comparison fails."""

    @pipefunc(output_name="y")
    def double_it(x: NoEqualityInput) -> int:
        return 2 * x.value

    pipeline = Pipeline([(double_it, "x[i] -> y[i]")])

    inputs1 = {"x": [NoEqualityInput(1), NoEqualityInput(2)]}

    # First run - create the run folder
    results1 = pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )
    np.testing.assert_array_equal(results1["y"].output, [2, 4])

    # Second run with same inputs but resume=True, resume_validation="auto"
    # Should warn but proceed
    inputs2 = {"x": [NoEqualityInput(1), NoEqualityInput(2)]}
    results2 = pipeline.map(
        inputs2,
        run_folder=tmp_path,
        resume=True,
        resume_validation="auto",
        parallel=False,
        storage="dict",
    )
    np.testing.assert_array_equal(results2["y"].output, [2, 4])

    # Check that warning was printed
    captured = capsys.readouterr()
    assert "Could not compare new `inputs` to `inputs` from previous run" in captured.out


def test_resume_validation_strict_with_broken_eq(tmp_path: Path) -> None:
    """Test that 'strict' mode raises error when equality comparison fails."""

    @pipefunc(output_name="y")
    def double_it(x: NoEqualityInput) -> int:
        return 2 * x.value

    pipeline = Pipeline([(double_it, "x[i] -> y[i]")])

    inputs1 = {"x": [NoEqualityInput(1), NoEqualityInput(2)]}

    # First run - create the run folder
    results1 = pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )
    np.testing.assert_array_equal(results1["y"].output, [2, 4])

    # Second run with same inputs but resume=True, resume_validation="strict"
    # Should raise error
    inputs2 = {"x": [NoEqualityInput(1), NoEqualityInput(2)]}
    with pytest.raises(
        ValueError,
        match="Cannot compare inputs for equality.*broken `__eq__` implementations",
    ):
        pipeline.map(
            inputs2,
            run_folder=tmp_path,
            resume=True,
            resume_validation="strict",
            parallel=False,
            storage="dict",
        )


def test_resume_validation_skip(tmp_path: Path) -> None:
    """Test that 'skip' mode bypasses input validation entirely."""

    @pipefunc(output_name="y")
    def double_it(x: BrokenEqualityInput) -> int:
        return 2 * x.value

    pipeline = Pipeline([(double_it, "x[i] -> y[i]")])

    inputs1 = {"x": [BrokenEqualityInput(1), BrokenEqualityInput(2)]}

    # First run - create the run folder
    results1 = pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )
    np.testing.assert_array_equal(results1["y"].output, [2, 4])

    # Second run with DIFFERENT inputs but resume=True, resume_validation="skip"
    # Should succeed because validation is skipped
    inputs2 = {"x": [BrokenEqualityInput(10), BrokenEqualityInput(20)]}
    results2 = pipeline.map(
        inputs2,
        run_folder=tmp_path,
        resume=True,
        resume_validation="skip",
        parallel=False,
        storage="dict",
    )
    # Note: Results are from the first run because we're reusing the run folder
    np.testing.assert_array_equal(results2["y"].output, [2, 4])


def test_resume_validation_auto_with_mismatched_inputs(tmp_path: Path) -> None:
    """Test that 'auto' mode raises error when inputs are genuinely different."""

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([(double_it, "x[i] -> y[i]")])

    inputs1 = {"x": [1, 2, 3]}

    # First run
    results1 = pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )
    np.testing.assert_array_equal(results1["y"].output, [2, 4, 6])

    # Second run with different inputs - should fail
    inputs2 = {"x": [4, 5, 6]}
    with pytest.raises(ValueError, match="`inputs` do not match previous run"):
        pipeline.map(
            inputs2,
            run_folder=tmp_path,
            resume=True,
            resume_validation="auto",
            parallel=False,
            storage="dict",
        )


def test_resume_validation_skip_still_validates_shapes(tmp_path: Path) -> None:
    """Test that 'skip' mode still validates shapes and MapSpecs."""

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([(double_it, "x[i] -> y[i]")])

    inputs1 = {"x": [1, 2, 3]}

    # First run
    pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )

    # Second run with different shape - should fail even with skip
    inputs2 = {"x": [1, 2, 3, 4]}
    with pytest.raises(ValueError, match="Shapes do not match previous run"):
        pipeline.map(
            inputs2,
            run_folder=tmp_path,
            resume=True,
            resume_validation="skip",
            parallel=False,
            storage="dict",
        )


def test_resume_validation_with_defaults(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test that defaults are also validated based on resume_validation mode."""

    @pipefunc(output_name="y")
    def add_default(x: int, offset: NoEqualityInput = NoEqualityInput(10)) -> int:  # noqa: B008
        return x + offset.value

    pipeline = Pipeline([add_default])

    inputs1 = {"x": 5}

    # First run
    results1 = pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )
    assert results1["y"].output == 15

    # Second run with resume=True, resume_validation="auto"
    # Should warn about defaults comparison
    results2 = pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=True,
        resume_validation="auto",
        parallel=False,
        storage="dict",
    )
    assert results2["y"].output == 15

    # Check that warning was printed about defaults
    captured = capsys.readouterr()
    assert "Could not compare new `defaults` to `defaults` from previous run" in captured.out


def test_resume_validation_strict_with_defaults(tmp_path: Path) -> None:
    """Test that strict mode raises error for incomparable defaults."""

    @pipefunc(output_name="y")
    def add_default(x: int, offset: NoEqualityInput = NoEqualityInput(10)) -> int:  # noqa: B008
        return x + offset.value

    pipeline = Pipeline([add_default])

    inputs1 = {"x": 5}

    # First run
    pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )

    # Second run with strict validation - should fail
    with pytest.raises(
        ValueError,
        match="Cannot compare defaults for equality.*broken `__eq__` implementations",
    ):
        pipeline.map(
            inputs1,
            run_folder=tmp_path,
            resume=True,
            resume_validation="strict",
            parallel=False,
            storage="dict",
        )


def test_resume_validation_ignored_when_resume_true(tmp_path: Path) -> None:
    """Test that resume_validation is ignored when resume=False."""

    @pipefunc(output_name="y")
    def double_it(x: NoEqualityInput) -> int:
        return 2 * x.value

    pipeline = Pipeline([(double_it, "x[i] -> y[i]")])

    inputs1 = {"x": [NoEqualityInput(1), NoEqualityInput(2)]}

    # First run
    results1 = pipeline.map(
        inputs1,
        run_folder=tmp_path,
        resume=False,
        parallel=False,
        storage="dict",
    )
    np.testing.assert_array_equal(results1["y"].output, [2, 4])

    # Second run with resume=False - resume_validation should be ignored
    # This should work fine regardless of resume_validation value
    inputs2 = {"x": [NoEqualityInput(3), NoEqualityInput(4)]}
    results2 = pipeline.map(
        inputs2,
        run_folder=tmp_path,
        resume=False,
        resume_validation="strict",  # This is ignored
        parallel=False,
        storage="dict",
    )
    np.testing.assert_array_equal(results2["y"].output, [6, 8])


def test_cleanup_parameter_deprecation_warning(tmp_path: Path) -> None:
    """Test that using the deprecated cleanup parameter triggers a deprecation warning."""

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([(double_it, "x[i] -> y[i]")])
    inputs = {"x": [1, 2, 3]}

    # Test cleanup=True triggers warning and behaves like resume=False
    with pytest.warns(
        DeprecationWarning,
        match="The 'cleanup' parameter is deprecated.*Use 'resume' instead",
    ):
        results1 = pipeline.map(
            inputs,
            run_folder=tmp_path,
            cleanup=True,
            parallel=False,
            storage="dict",
        )
    np.testing.assert_array_equal(results1["y"].output, [2, 4, 6])

    # Test cleanup=False triggers warning and behaves like resume=True
    with pytest.warns(
        DeprecationWarning,
        match="The 'cleanup' parameter is deprecated.*Use 'resume' instead",
    ):
        results2 = pipeline.map(
            inputs,
            run_folder=tmp_path,
            cleanup=False,
            parallel=False,
            storage="dict",
        )
    np.testing.assert_array_equal(results2["y"].output, [2, 4, 6])

    # Test that cleanup takes priority over resume
    with pytest.warns(
        DeprecationWarning,
        match="The 'cleanup' parameter is deprecated.*Use 'resume' instead",
    ):
        results3 = pipeline.map(
            inputs,
            run_folder=tmp_path,
            cleanup=True,  # This should override resume=True
            resume=True,
            parallel=False,
            storage="dict",
        )
    # Since cleanup=True (resume=False), the run folder should be cleaned
    np.testing.assert_array_equal(results3["y"].output, [2, 4, 6])
