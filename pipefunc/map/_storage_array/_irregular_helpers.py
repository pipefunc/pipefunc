"""Helper functions for irregular storage behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def try_getitem(
    array: np.ndarray,
    key: tuple[int, ...],
    *,
    irregular: bool,
) -> tuple[Any, bool]:
    """Return the item at ``key`` or a masked sentinel when irregular."""
    try:
        return array[key], False
    except IndexError:
        if irregular:
            return np.ma.masked, True
        raise


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


def ensure_masked_array_for_irregular(
    data: Any,
    *,
    irregular: bool,
    mask_factory: Callable[[np.ndarray], np.ndarray],
) -> Any:
    """Wrap ``data`` in a MaskedArray when irregular storage tracks masked sentinels."""
    if not irregular or not isinstance(data, np.ndarray):
        return data
    mask = mask_factory(data)
    if np.any(mask):
        return np.ma.MaskedArray(data, mask=mask, dtype=object)
    return data


def irregular_extent(
    irregular: bool,
    internal_shape: Sequence[Any],
    cache: dict[tuple[int, ...], tuple[int, ...] | None] | None,
    external_index: tuple[int, ...],
    compute_extent: Callable[[tuple[int, ...]], tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Return the realised extent along irregular axes for ``external_index``."""
    if not irregular or not internal_shape:
        return None
    if cache is None:
        return None
    if external_index in cache:
        return cache[external_index]
    extent = compute_extent(external_index)
    cache[external_index] = extent
    return extent


def clear_irregular_extent_cache(
    cache: dict[tuple[int, ...], tuple[int, ...] | None] | None,
) -> None:
    """Clear the cached irregular extents if present."""
    if cache is not None:
        cache.clear()
