from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import cloudpickle
from adaptive import SequenceLearner
from adaptive.types import Int
from sortedcontainers import SortedSet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


PointType: TypeAlias = tuple[Int, Any]


@dataclass
class LazySequence:
    """A lazy sequence that can be evaluated on demand."""

    callable: Callable[[], Sequence[Any]]
    sequence: Sequence[Any] | None = None

    def evaluate(self) -> Sequence[Any]:
        """Evaluate the lazy sequence."""
        if self.sequence is None:
            self.sequence = self.callable()
        return self.sequence

    def __len__(self) -> int:
        return 0 if self.sequence is None else len(self.sequence)


class LazySequenceLearner(SequenceLearner):
    """A similar learner to `~adaptive.SequenceLearner`, but the sequence is lazy.

    This is useful when the sequence is not yet available when the learner is created.

    Parameters
    ----------
    function
        The function to learn. Must take a single element `sequence`.
    lazy_sequence
        A `~adaptive.LazySequence` that will be evaluated when needed.

    """

    def __init__(self, function: Callable[[Any], Any], lazy_sequence: LazySequence) -> None:
        self.lazy_sequence = lazy_sequence
        super().__init__(function, self.lazy_sequence)

    def ask(
        self,
        n: int,
        tell_pending: bool = True,  # noqa: FBT001, FBT002
    ) -> tuple[list[PointType], list[float]]:
        if self.lazy_sequence.sequence is None:
            self._evaluate_sequence()
        return super().ask(n, tell_pending)

    def _evaluate_sequence(self) -> None:
        assert self.lazy_sequence.sequence is None
        self.sequence = self.lazy_sequence.evaluate()
        self._ntotal = len(self.sequence)
        self._to_do_indices = SortedSet(range(self._ntotal))
        self._sequence_evaluated = True

    def __getstate__(self) -> tuple:
        state = super().__getstate__()
        return (*state, cloudpickle.dumps(self.lazy_sequence))

    def __setstate__(self, state: tuple) -> None:
        *parent_state, lazy_sequence = state
        super().__setstate__(parent_state)
        self.lazy_sequence = cloudpickle.loads(lazy_sequence)

    def new(self) -> LazySequenceLearner:
        """Return a new `~adaptive.LazySequenceLearner` without the data."""
        return LazySequenceLearner(self._original_function, self.lazy_sequence)
