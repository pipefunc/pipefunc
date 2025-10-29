# ruff: noqa: INP001
"""Run `pipeline.map_async` from a regular Python script."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def shift(x: int) -> int:
    """Add one to each element in `x` to create `y`."""
    return x + 1


pipeline = Pipeline([shift])


def main() -> None:
    """Submit the async map and block until it completes, then show sample output."""
    runner = pipeline.map_async(
        inputs={"x": range(100)},
        executor=ThreadPoolExecutor(max_workers=4),
        start=False,
        display_widgets=False,
    )

    result = runner.result()
    print(result["y"].output[:5])


if __name__ == "__main__":
    main()
