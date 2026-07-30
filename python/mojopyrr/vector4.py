"""Four-component vectors."""

from __future__ import annotations

import numpy as np

from .vector import dot, interpolate, length, normalise, normalize, set_length, squared_length


def create(x=0.0, y=0.0, z=0.0, w=0.0, dtype=None):
    if isinstance(x, (list, np.ndarray)):
        raise ValueError("Function requires non-list arguments")
    return np.array([x, y, z, w], dtype=dtype)


def create_unit_length_x(dtype=None):
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)


def create_unit_length_y(dtype=None):
    return np.array([0.0, 1.0, 0.0, 0.0], dtype=dtype)


def create_unit_length_z(dtype=None):
    return np.array([0.0, 0.0, 1.0, 0.0], dtype=dtype)


def create_unit_length_w(dtype=None):
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=dtype)


def create_from_vector3(vector, w=0.0, dtype=None):
    value = np.asarray(vector)
    dtype = dtype or value.dtype
    return np.array([value[0], value[1], value[2], w], dtype=dtype)


def create_from_matrix44_translation(mat, dtype=None):
    return np.array(np.asarray(mat)[3, :4], dtype=dtype)


class index:
    x = 0
    y = 1
    z = 2
    w = 3


class unit:
    x = create_unit_length_x()
    y = create_unit_length_y()
    z = create_unit_length_z()
    w = create_unit_length_w()
