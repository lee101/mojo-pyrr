"""Row-major 3x3 matrices compatible with pyrr."""

from __future__ import annotations

import numpy as np

from . import quaternion, vector
from ._lib import addr, empty, f64, lib
from ._matrix import apply, inverse as _inverse, multiply as _multiply


def create_identity(dtype=None):
    return np.identity(3, dtype=dtype)


def create_from_matrix44(mat, dtype=None):
    return np.array(np.asarray(mat)[0:3, 0:3], dtype=dtype)


def create_from_eulers(eulers, dtype=None):
    value = np.asarray(eulers)
    dtype = dtype or value.dtype
    roll, pitch, yaw = value
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, -cy * sp * cr + sy * sr, cy * sp * sr + sy * cr],
            [sp, cp * cr, -cp * sr],
            [-sy * cp, sy * sp * cr + cy * sr, -sy * sp * sr + cy * cr],
        ],
        dtype=dtype,
    )


def create_from_axis_rotation(axis, theta, dtype=None):
    value = np.asarray(axis)
    dtype = dtype or value.dtype
    x, y, z = vector.normalize(value)
    sine, cosine = np.sin(theta), np.cos(theta)
    t = 1 - cosine
    return np.array(
        [
            [x * x * t + cosine, y * x * t + z * sine, z * x * t - y * sine],
            [x * y * t - z * sine, y * y * t + cosine, z * y * t + x * sine],
            [x * z * t + y * sine, y * z * t - x * sine, z * z * t + cosine],
        ],
        dtype=dtype,
    )


def create_from_quaternion(quat, dtype=None):
    original = np.asarray(quat)
    source = f64(original)
    result = empty(original.shape[:-1] + (3, 3))
    lib().mpr_quat_to_mat33(addr(source), addr(result), source.size // 4)
    return result.astype(dtype or original.dtype, copy=False)


def create_from_inverse_of_quaternion(quat, dtype=None):
    value = np.asarray(quat)
    dtype = dtype or value.dtype
    x, y, z, w = value
    x2, y2, z2 = x**2, y**2, z**2
    wx, wy, xy = w * x, w * y, x * y
    wz, xz, yz = w * z, x * z, y * z
    return np.array(
        [
            [1 - 2 * (y2 + z2), 2 * (xy + wz), 2 * (xz - wy)],
            [2 * (xy - wz), 1 - 2 * (x2 + z2), 2 * (yz + wx)],
            [2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (x2 + y2)],
        ],
        dtype=dtype,
    )


def create_from_scale(scale, dtype=None):
    result = np.diagflat(scale)
    return result.astype(dtype) if dtype else result


def create_from_x_rotation(theta, dtype=None):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=dtype,
    )


def create_from_y_rotation(theta, dtype=None):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=dtype,
    )


def create_from_z_rotation(theta, dtype=None):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )


def apply_to_vector(mat, vec):
    return apply(mat, vec, 3)


def multiply(m1, m2):
    try:
        if m1.ndim == 2 == m2.ndim:
            return m1.dot(m2)
    except AttributeError:
        pass
    return _multiply(m1, m2, 3)


def inverse(mat):
    return _inverse(mat, 3)


def create_direction_scale(direction, scale):
    value = np.asarray(direction)
    if not np.isclose(np.linalg.norm(value), 1.0):
        value = vector.normalize(value)
    x, y, z = value
    sm1 = scale - 1.0
    return np.array(
        [
            [1.0 + sm1 * x**2, sm1 * x * y**2, sm1 * x * z],
            [sm1 * x * y, 1.0 + sm1 * y, sm1 * y * z],
            [sm1 * x * z, sm1 * y * z, 1.0 + sm1 * z**2],
        ]
    )
