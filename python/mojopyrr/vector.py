"""Common vector operations with the final axis treated as components."""

from __future__ import annotations

import numpy as np

from ._lib import addr, empty, f64, lib


def _shape(value) -> tuple[np.ndarray, tuple[int, ...], int]:
    original = np.asarray(value)
    if original.ndim == 0:
        raise ValueError("a vector must have at least one dimension")
    return f64(original), original.shape, original.shape[-1]


def _float_dtype(value):
    dtype = np.asarray(value).dtype
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float64)


def _finish(value: np.ndarray, shape: tuple[int, ...], dtype):
    result = value.reshape(shape).astype(dtype, copy=False)
    return result[()] if not shape else result


def normalize(vec):
    source, shape, width = _shape(vec)
    if width == 0:
        return np.asarray(vec, dtype=_float_dtype(vec)).copy()
    result = empty(shape)
    rows = source.size // width
    lib().mpr_vector_normalize(
        addr(source), addr(result), rows, width, 1.0
    )
    return _finish(result, shape, _float_dtype(vec))


def normalise(vec):
    return normalize(vec)


def squared_length(vec):
    source, shape, width = _shape(vec)
    if width == 0:
        return np.zeros(shape[:-1], dtype=_float_dtype(vec))
    result = empty(shape[:-1])
    rows = source.size // width
    lib().mpr_vector_lengths(addr(source), addr(result), rows, width, 1)
    target = _float_dtype(vec)
    return _finish(result, shape[:-1], target)


def length(vec):
    source, shape, width = _shape(vec)
    if width == 0:
        return np.zeros(shape[:-1], dtype=_float_dtype(vec))
    result = empty(shape[:-1])
    rows = source.size // width
    lib().mpr_vector_lengths(addr(source), addr(result), rows, width, 0)
    return _finish(result, shape[:-1], _float_dtype(vec))


def set_length(vec, len):
    unit = normalize(vec)
    return (unit.T * len).T


def dot(v1, v2):
    left, right = np.broadcast_arrays(np.asarray(v1), np.asarray(v2))
    if left.ndim == 0:
        raise ValueError("a vector must have at least one dimension")
    shape = left.shape[:-1]
    width = left.shape[-1]
    if width == 0:
        return _finish(
            np.zeros(shape, dtype=np.float64),
            shape,
            np.result_type(left.dtype, right.dtype),
        )
    a = f64(left)
    b = f64(right)
    result = empty(shape)
    lib().mpr_vector_dot(
        addr(a), addr(b), addr(result), a.size // width, width
    )
    target = np.result_type(left.dtype, right.dtype)
    return _finish(result, shape, target)


def interpolate(v1, v2, delta):
    left, right = np.broadcast_arrays(np.asarray(v1), np.asarray(v2))
    target = np.result_type(left.dtype, right.dtype, np.asarray(delta).dtype)
    if np.ndim(delta):
        return left + ((right - left) * delta)
    a = f64(left)
    b = f64(right)
    result = empty(left.shape)
    lib().mpr_vector_interpolate(
        addr(a), addr(b), addr(result), a.size, float(delta)
    )
    return result.astype(target, copy=False)
