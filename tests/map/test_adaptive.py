from pathlib import Path

import adaptive
import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map._adaptive import make_learners


def test_basic(tmp_path: Path) -> None:
    @pipefunc(output_name="z")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="prod")
    def take_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.prod(z)

    pipeline = Pipeline(
        [
            (add, "x[i], y[j] -> z[i, j]"),
            take_sum,
        ],
    )

    inputs = {"x": [1, 2, 3], "y": [1, 2, 3]}
    learners_lists = make_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=True,
    )
    flat_learners = [learner for learners in learners_lists for learner in learners]
    assert len(flat_learners) == 2
    adaptive.runner.simple(flat_learners[0])
    assert flat_learners[0].data == {
        0: [2],
        1: [3],
        2: [4],
        3: [3],
        4: [4],
        5: [5],
        6: [4],
        7: [5],
        8: [6],
    }
    adaptive.runner.simple(flat_learners[1])
    assert flat_learners[1].data == {0: 172800}
