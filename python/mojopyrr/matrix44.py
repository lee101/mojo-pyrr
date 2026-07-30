"""Row-major 4x4 transformation and projection matrices."""

from __future__ import annotations

import numpy as np

from . import matrix33, quaternion, vector
from ._matrix import apply, apply_points44, inverse as _inverse, multiply as _multiply


def create_identity(dtype=None):
    return np.identity(4, dtype=dtype)


def create_from_matrix33(mat, dtype=None):
    result = np.identity(4, dtype=dtype)
    result[0:3, 0:3] = mat
    return result


def create_matrix33_view(mat):
    return mat[0:3, 0:3]


def create_from_eulers(eulers, dtype=None):
    value = np.asarray(eulers)
    dtype = dtype or value.dtype
    result = create_identity(dtype)
    result[:3, :3] = matrix33.create_from_eulers(value, dtype)
    return result


def create_from_axis_rotation(axis, theta, dtype=None):
    value = np.asarray(axis)
    dtype = dtype or value.dtype
    result = create_identity(dtype)
    result[:3, :3] = matrix33.create_from_axis_rotation(value, theta, dtype)
    return result


def create_from_quaternion(quat, dtype=None):
    value = np.asarray(quat)
    dtype = dtype or value.dtype
    result = create_identity(dtype)
    result[:3, :3] = matrix33.create_from_quaternion(value, dtype)
    return result


def create_from_inverse_of_quaternion(quat, dtype=None):
    value = np.asarray(quat)
    dtype = dtype or value.dtype
    result = create_identity(dtype)
    result[:3, :3] = matrix33.create_from_inverse_of_quaternion(value, dtype)
    return result


def create_from_translation(vec, dtype=None):
    value = np.asarray(vec)
    dtype = dtype or value.dtype
    result = create_identity(dtype)
    result[3, :3] = value[:3]
    return result


def create_from_scale(scale, dtype=None):
    result = np.diagflat([scale[0], scale[1], scale[2], 1.0])
    return result.astype(dtype) if dtype else result


def create_from_x_rotation(theta, dtype=None):
    result = create_identity(dtype)
    result[:3, :3] = matrix33.create_from_x_rotation(theta, dtype)
    return result


def create_from_y_rotation(theta, dtype=None):
    result = create_identity(dtype)
    result[:3, :3] = matrix33.create_from_y_rotation(theta, dtype)
    return result


def create_from_z_rotation(theta, dtype=None):
    result = create_identity(dtype)
    result[:3, :3] = matrix33.create_from_z_rotation(theta, dtype)
    return result


def apply_to_vector(mat, vec):
    value = np.asarray(vec)
    if value.shape[-1] == 3:
        return apply_points44(mat, value)
    if value.shape[-1] == 4:
        return apply(mat, value, 4)
    raise ValueError("Vector size unsupported")


def multiply(m1, m2):
    try:
        if m1.ndim == 2 == m2.ndim:
            return m1.dot(m2)
    except AttributeError:
        pass
    return _multiply(m1, m2, 4)


def create_perspective_projection(fovy, aspect, near, far, dtype=None):
    ymax = near * np.tan(fovy * np.pi / 360.0)
    xmax = ymax * aspect
    return create_perspective_projection_from_bounds(
        -xmax, xmax, -ymax, ymax, near, far, dtype=dtype
    )


def create_perspective_projection_matrix(fovy, aspect, near, far, dtype=None):
    return create_perspective_projection(fovy, aspect, near, far, dtype)


def create_perspective_projection_from_bounds(
    left, right, bottom, top, near, far, dtype=None
):
    a = (right + left) / (right - left)
    b = (top + bottom) / (top - bottom)
    c = -(far + near) / (far - near)
    d = -2.0 * far * near / (far - near)
    e = 2.0 * near / (right - left)
    f = 2.0 * near / (top - bottom)
    return np.array(
        ((e, 0.0, 0.0, 0.0), (0.0, f, 0.0, 0.0), (a, b, c, -1.0), (0.0, 0.0, d, 0.0)),
        dtype=dtype,
    )


def create_perspective_projection_matrix_from_bounds(
    left, right, bottom, top, near, far, dtype=None
):
    return create_perspective_projection_from_bounds(
        left, right, bottom, top, near, far, dtype
    )


def create_orthogonal_projection(
    left, right, bottom, top, near, far, dtype=None
):
    rml = right - left
    tmb = top - bottom
    fmn = far - near
    a, b, c = 2.0 / rml, 2.0 / tmb, -2.0 / fmn
    tx = -(right + left) / rml
    ty = -(top + bottom) / tmb
    tz = -(far + near) / fmn
    return np.array(
        ((a, 0.0, 0.0, 0.0), (0.0, b, 0.0, 0.0), (0.0, 0.0, c, 0.0), (tx, ty, tz, 1.0)),
        dtype=dtype,
    )


def create_orthogonal_projection_matrix(
    left, right, bottom, top, near, far, dtype=None
):
    return create_orthogonal_projection(
        left, right, bottom, top, near, far, dtype
    )


def create_look_at(eye, target, up, dtype=None):
    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)
    forward = vector.normalize(target - eye)
    side = vector.normalize(np.cross(forward, up))
    up = vector.normalize(np.cross(side, forward))
    return np.array(
        (
            (side[0], up[0], -forward[0], 0.0),
            (side[1], up[1], -forward[1], 0.0),
            (side[2], up[2], -forward[2], 0.0),
            (-np.dot(side, eye), -np.dot(up, eye), np.dot(forward, eye), 1.0),
        ),
        dtype=dtype,
    )


def inverse(m):
    return _inverse(m, 4)


def decompose(m):
    value = np.asarray(m)
    scale = np.linalg.norm(value[:3, :3], axis=1)
    if np.linalg.det(value) < 0:
        scale[0] *= -1
    position = value[3, :3]
    rotation = value[:3, :3] * (1 / scale)[:, None]
    return scale, quaternion.create_from_matrix(rotation), position
