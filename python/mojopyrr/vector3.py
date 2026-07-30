"""Three-component vectors."""

from __future__ import annotations

import numpy as np

from . import vector
from ._lib import addr, empty, f64, lib
from .vector import dot, interpolate, length, normalise, normalize, set_length, squared_length


def create(x=0.0, y=0.0, z=0.0, dtype=None):
    if isinstance(x, (list, np.ndarray)):
        raise ValueError("Function requires non-list arguments")
    return np.array([x, y, z], dtype=dtype)


def create_unit_length_x(dtype=None):
    return np.array([1.0, 0.0, 0.0], dtype=dtype)


def create_unit_length_y(dtype=None):
    return np.array([0.0, 1.0, 0.0], dtype=dtype)


def create_unit_length_z(dtype=None):
    return np.array([0.0, 0.0, 1.0], dtype=dtype)


def create_from_vector4(vector, dtype=None):
    value = np.asarray(vector)
    dtype = dtype or value.dtype
    return np.array(value[:3], dtype=dtype), value[3]


def create_from_matrix44_translation(mat, dtype=None):
    return np.array(np.asarray(mat)[3, :3], dtype=dtype)


def cross(v1, v2):
    left, right = np.broadcast_arrays(np.asarray(v1), np.asarray(v2))
    if left.shape[-1] != 3:
        raise ValueError("incompatible dimensions for cross product")
    a, b = f64(left), f64(right)
    result = empty(left.shape)
    lib().mpr_vector_cross3(addr(a), addr(b), addr(result), a.size // 3)
    return result.astype(np.result_type(left.dtype, right.dtype), copy=False)


def generate_normals(v1, v2, v3, normalize_result=True):
    normal = cross(np.asarray(v3) - np.asarray(v2), np.asarray(v1) - np.asarray(v2))
    return vector.normalize(normal) if normalize_result else normal


def generate_vertex_normals(vertices, index, normalize_result=True):
    vertices = np.asarray(vertices)
    index = np.asarray(index)
    v1, v2, v3 = np.rollaxis(vertices[index], axis=-2)
    face_normals = generate_normals(v1, v2, v3, normalize_result=False)
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, index[..., i], face_normals)
    return vector.normalize(vertex_normals) if normalize_result else vertex_normals


class index:
    x = 0
    y = 1
    z = 2


class unit:
    x = create_unit_length_x()
    y = create_unit_length_y()
    z = create_unit_length_z()
