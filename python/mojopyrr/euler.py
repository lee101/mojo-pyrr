"""Euler angle helpers used by pyrr, ordered as roll, pitch, yaw."""

import numpy as np


class index:
    roll = 0
    pitch = 1
    yaw = 2


def create(roll=0.0, pitch=0.0, yaw=0.0, dtype=None):
    return np.array((roll, pitch, yaw), dtype=dtype)


def create_from_x_rotation(theta, dtype=None):
    return np.array([theta, 0.0, 0.0], dtype=dtype)


def create_from_y_rotation(theta, dtype=None):
    return np.array([0.0, theta, 0.0], dtype=dtype)


def create_from_z_rotation(theta, dtype=None):
    return np.array([0.0, 0.0, theta], dtype=dtype)


def roll(eulers):
    return eulers[0]


def pitch(eulers):
    return eulers[1]


def yaw(eulers):
    return eulers[2]
