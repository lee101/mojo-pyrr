"""Quaternion creation and manipulation in pyrr's ``[x, y, z, w]`` order."""

from __future__ import annotations

import numpy as np

from . import vector, vector4
from ._lib import addr, empty, f64, lib


def create(x=0.0, y=0.0, z=0.0, w=1.0, dtype=None):
    return np.array([x, y, z, w], dtype=dtype)


def create_from_x_rotation(theta, dtype=None):
    half = theta * 0.5
    return np.array([np.sin(half), 0.0, 0.0, np.cos(half)], dtype=dtype)


def create_from_y_rotation(theta, dtype=None):
    half = theta * 0.5
    return np.array([0.0, np.sin(half), 0.0, np.cos(half)], dtype=dtype)


def create_from_z_rotation(theta, dtype=None):
    half = theta * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=dtype)


def create_from_axis_rotation(axis, theta, dtype=None):
    original = np.asarray(axis)
    if original.shape[-1] != 3:
        raise ValueError("axis must end in three components")
    source = f64(original)
    shape = original.shape[:-1] + (4,)
    result = empty(shape)
    lib().mpr_axis_angle_quat(
        addr(source), addr(result), source.size // 3, float(theta)
    )
    return result.astype(dtype or original.dtype, copy=False)


def create_from_axis(axis, dtype=None):
    value = np.asarray(axis)
    return create_from_axis_rotation(value, np.linalg.norm(value), dtype)


def create_from_matrix(mat, dtype=None):
    original = np.asarray(mat)
    if original.shape[-2:] == (4, 4):
        original = original[..., :3, :3]
    if original.shape[-2:] != (3, 3):
        raise ValueError("matrix must be 3x3 or 4x4")
    source = f64(original)
    result = empty(original.shape[:-2] + (4,))
    lib().mpr_mat33_to_quat(
        addr(source), addr(result), source.size // 9
    )
    return result.astype(dtype or original.dtype, copy=False)


def create_from_eulers(eulers, dtype=None):
    value = np.asarray(eulers)
    dtype = dtype or value.dtype
    roll, pitch, yaw = value
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5
    sr, cr = np.sin(half_roll), np.cos(half_roll)
    sp, cp = np.sin(half_pitch), np.cos(half_pitch)
    sy, cy = np.sin(half_yaw), np.cos(half_yaw)
    return np.array(
        [
            sr * cp * cy + cr * sp * sy,
            cr * sp * cy - sr * cp * sy,
            cr * cp * sy + sr * sp * cy,
            cr * cp * cy - sr * sp * sy,
        ],
        dtype=dtype,
    )


def create_from_inverse_of_eulers(eulers, dtype=None):
    value = np.asarray(eulers)
    dtype = dtype or value.dtype
    roll, pitch, yaw = value
    sr, cr = np.sin(roll * 0.5), np.cos(roll * 0.5)
    sp, cp = np.sin(pitch * 0.5), np.cos(pitch * 0.5)
    sy, cy = np.sin(yaw * 0.5), np.cos(yaw * 0.5)
    return np.array(
        [
            cy * sp * cr + sy * cp * sr,
            -cy * sp * sr + sy * cp * cr,
            -sy * sp * cr + cy * cp * sr,
            cy * cp * cr + sy * sp * sr,
        ],
        dtype=dtype,
    )


def cross(quat1, quat2):
    left, right = np.broadcast_arrays(np.asarray(quat1), np.asarray(quat2))
    if left.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    a, b = f64(left), f64(right)
    result = empty(left.shape)
    lib().mpr_quat_cross(addr(a), addr(b), addr(result), a.size // 4)
    return result.astype(left.dtype, copy=False)


def lerp(quat1, quat2, t):
    t = np.clip(t, 0, 1)
    return normalize(np.asarray(quat1) * (1 - t) + np.asarray(quat2) * t)


def slerp(quat1, quat2, t):
    left, right = np.broadcast_arrays(np.asarray(quat1), np.asarray(quat2))
    if left.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    a, b = f64(left), f64(right)
    result = empty(left.shape)
    lib().mpr_quat_slerp(
        addr(a), addr(b), addr(result), a.size // 4, float(t)
    )
    return result.astype(np.result_type(left.dtype, right.dtype), copy=False)


def is_zero_length(quat):
    return bool(np.all(np.asarray(quat) == 0.0))


def is_non_zero_length(quat):
    return not is_zero_length(quat)


def squared_length(quat):
    return vector4.squared_length(quat)


def length(quat):
    return vector4.length(quat)


def normalize(quat):
    return vector4.normalize(quat)


def normalise(quat):
    return normalize(quat)


def rotation_angle(quat):
    return np.arccos(np.asarray(quat)[..., 3]) * 2.0


def rotation_axis(quat):
    value = np.asarray(quat)
    scale_squared = 1.0 - value[..., 3] ** 2
    result = np.empty(value.shape[:-1] + (3,), dtype=value.dtype)
    valid = scale_squared > 0.0
    if result.ndim == 1:
        if not valid:
            return np.array([0.0, 0.0, -1.0], dtype=value.dtype)
        return value[:3] / np.sqrt(scale_squared)
    result[~valid] = [0.0, 0.0, -1.0]
    result[valid] = value[valid, :3] / np.sqrt(scale_squared[valid, None])
    return result


def dot(quat1, quat2):
    return vector4.dot(quat1, quat2)


def conjugate(quat):
    value = np.asarray(quat)
    result = np.array(value, copy=True)
    result[..., :3] *= -1
    return result


def exp(quat):
    original = np.asarray(quat)
    if original.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    source = f64(original)
    result = empty(original.shape)
    lib().mpr_quat_exp(addr(source), addr(result), source.size // 4)
    return result.astype(original.dtype, copy=False)


def power(quat, exponent):
    value = np.asarray(quat)
    if np.fabs(value[3]) > 0.9999:
        raise AssertionError
    alpha = np.arccos(value[3])
    new_alpha = alpha * exponent
    multiple = np.sin(new_alpha) / np.sin(alpha)
    return np.array(
        [
            value[0] * multiple,
            value[1] * multiple,
            value[2] * multiple,
            np.cos(new_alpha),
        ],
        dtype=value.dtype,
    )


def inverse(quat):
    value = conjugate(quat)
    return (value.T / length(quat)).T


def negate(quat):
    return np.asarray(quat) * -1.0


def is_identity(quat):
    return np.allclose(quat, [0.0, 0.0, 0.0, 1.0])


def apply_to_vector(quat, vec):
    q = np.asarray(quat)
    value = np.asarray(vec)
    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError("quaternions must end in four components")
    if value.ndim == 0 or value.shape[-1] not in (3, 4):
        raise ValueError("Vector size unsupported")
    batch_shape = np.broadcast_shapes(q.shape[:-1], value.shape[:-1])
    q_view = np.broadcast_to(q, batch_shape + (4,))
    v_view = np.broadcast_to(value, batch_shape + (value.shape[-1],))
    q_source = f64(q_view)
    v_source = f64(v_view)
    result = empty(v_view.shape)
    lib().mpr_quat_apply(
        addr(q_source),
        addr(v_source),
        addr(result),
        v_source.size // value.shape[-1],
        value.shape[-1],
        4,
    )
    return result.astype(np.result_type(q.dtype, value.dtype), copy=False)


class index:
    x = 0
    y = 1
    z = 2
    w = 3
