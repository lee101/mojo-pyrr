"""Shared fixed-size matrix dispatch."""

from __future__ import annotations

import numpy as np

from ._lib import addr, empty, f64, lib


def multiply(lhs, rhs, size):
    left, right = np.asarray(lhs), np.asarray(rhs)
    if left.shape == right.shape == (size, size):
        return np.dot(left, right)
    left, right = np.broadcast_arrays(left, right)
    if left.shape[-2:] != (size, size):
        raise ValueError(f"matrices must be {size}x{size}")
    a, b = f64(left), f64(right)
    result = empty(left.shape)
    lib().mpr_matmul(addr(a), addr(b), addr(result), a.size // (size * size), size)
    return result.astype(np.result_type(left.dtype, right.dtype), copy=False)


def inverse(value, size):
    original = np.asarray(value)
    if original.shape[-2:] != (size, size):
        raise ValueError(f"matrices must be {size}x{size}")
    source = f64(original)
    result = empty(original.shape)
    scratch = empty(original.shape)
    failed = lib().mpr_inverse(
        addr(source),
        addr(result),
        addr(scratch),
        source.size // (size * size),
        size,
    )
    if failed:
        raise np.linalg.LinAlgError("Singular matrix")
    dtype = (
        original.dtype
        if np.issubdtype(original.dtype, np.floating)
        else np.dtype(np.float64)
    )
    return result.astype(dtype, copy=False)


def apply(value, vec, size):
    matrix = np.asarray(value)
    vector = np.asarray(vec)
    if matrix.shape[-2:] != (size, size) or vector.shape[-1] != size:
        raise ValueError("Vector size unsupported")
    batch = np.broadcast_shapes(matrix.shape[:-2], vector.shape[:-1])
    matrices = f64(np.broadcast_to(matrix, batch + (size, size)))
    vectors = f64(np.broadcast_to(vector, batch + (size,)))
    result = empty(vectors.shape)
    lib().mpr_matvec(
        addr(matrices),
        addr(vectors),
        addr(result),
        vectors.size // size,
        size,
        size * size,
    )
    return result.astype(np.result_type(matrix.dtype, vector.dtype), copy=False)


def apply_points44(value, vec):
    matrix = np.asarray(value)
    vector = np.asarray(vec)
    if matrix.shape[-2:] != (4, 4) or vector.shape[-1] != 3:
        raise ValueError("Vector size unsupported")
    batch = np.broadcast_shapes(matrix.shape[:-2], vector.shape[:-1])
    matrices = f64(np.broadcast_to(matrix, batch + (4, 4)))
    vectors = f64(np.broadcast_to(vector, batch + (3,)))
    result = empty(vectors.shape)
    lib().mpr_transform_points3(
        addr(matrices),
        addr(vectors),
        addr(result),
        vectors.size // 3,
        16,
    )
    return result.astype(np.result_type(matrix.dtype, vector.dtype), copy=False)
