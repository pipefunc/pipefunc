"""Helper functions for irregular storage behaviour."""

from __future__ import annotations

from typing import Any

import numpy as np


def try_getitem(
    array: np.ndarray,
    key: tuple[int, ...],
) -> tuple[Any, bool]:
    """Return the item at ``key`` or a masked sentinel when irregular."""
    try:
        return array[key], False
    except IndexError:
        return np.ma.masked, True


def infer_irregular_length(value: Any) -> int:
    """Return the realised length for a single irregular axis."""
    if value is None or value is np.ma.masked:
        return 0
    if isinstance(value, np.ma.MaskedArray):
        if value.ndim == 0:
            return 0 if np.ma.is_masked(value) else 1
        if value.mask is np.ma.nomask:
            return int(value.shape[0])
        mask = np.atleast_1d(value.mask)
        valid = np.nonzero(~mask)[0]
        return 0 if valid.size == 0 else int(valid[-1]) + 1
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 1
