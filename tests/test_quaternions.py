import numpy as np
import pytest
import pyrr

from mojopyrr import quaternion


@pytest.fixture
def rotations():
    rng = np.random.default_rng(8)
    axes = rng.normal(size=(64, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    angles = rng.uniform(-2.5, 2.5, size=64)
    return axes, angles


@pytest.mark.parametrize(
    "name", ["create_from_x_rotation", "create_from_y_rotation", "create_from_z_rotation"]
)
def test_principal_rotation_constructors(name):
    np.testing.assert_allclose(
        getattr(quaternion, name)(0.73), getattr(pyrr.quaternion, name)(0.73)
    )


def test_axis_rotation_constructor(rotations):
    axes, angles = rotations
    for axis, angle in zip(axes, angles):
        np.testing.assert_allclose(
            quaternion.create_from_axis_rotation(axis, angle),
            pyrr.quaternion.create_from_axis_rotation(axis, angle),
            rtol=1e-13,
            atol=1e-13,
        )


@pytest.mark.parametrize(
    "name", ["create_from_eulers", "create_from_inverse_of_eulers"]
)
def test_euler_constructors(name):
    eulers = np.array([0.31, -0.72, 1.19])
    np.testing.assert_allclose(
        getattr(quaternion, name)(eulers), getattr(pyrr.quaternion, name)(eulers)
    )


def test_quaternion_cross_batch(rotations):
    axes, angles = rotations
    left = np.stack(
        [pyrr.quaternion.create_from_axis_rotation(a, t) for a, t in zip(axes, angles)]
    )
    right = np.roll(left, 1, axis=0)
    expected = np.stack(
        [pyrr.quaternion.cross(a, b) for a, b in zip(left, right)]
    )
    np.testing.assert_allclose(quaternion.cross(left, right), expected)


@pytest.mark.parametrize("t", [-0.2, 0.0, 0.25, 0.7, 1.0, 1.2])
def test_slerp_parity(t):
    left = pyrr.quaternion.create_from_eulers([0.2, -0.4, 0.9])
    right = pyrr.quaternion.create_from_eulers([-0.7, 0.3, 0.1])
    np.testing.assert_allclose(
        quaternion.slerp(left, right, t),
        pyrr.quaternion.slerp(left, right, t),
        rtol=1e-13,
        atol=1e-13,
    )


def test_lerp_parity():
    left = pyrr.quaternion.create_from_eulers([0.2, -0.4, 0.9])
    right = pyrr.quaternion.create_from_eulers([-0.7, 0.3, 0.1])
    np.testing.assert_allclose(
        quaternion.lerp(left, right, 0.35), pyrr.quaternion.lerp(left, right, 0.35)
    )


@pytest.mark.parametrize(
    "name",
    [
        "length",
        "squared_length",
        "normalize",
        "normalise",
        "rotation_angle",
        "rotation_axis",
        "conjugate",
        "negate",
        "inverse",
        "exp",
    ],
)
def test_quaternion_unary_parity(name):
    value = pyrr.quaternion.create_from_eulers([0.2, -0.4, 0.9])
    np.testing.assert_allclose(
        getattr(quaternion, name)(value),
        getattr(pyrr.quaternion, name)(value),
        rtol=5e-13,
        atol=5e-13,
    )


def test_quaternion_predicates_and_dot():
    identity = quaternion.create()
    assert quaternion.is_identity(identity)
    assert not quaternion.is_zero_length(identity)
    assert quaternion.is_non_zero_length(identity)
    assert quaternion.is_zero_length([0, 0, 0, 0])
    assert quaternion.dot(identity, identity) == pyrr.quaternion.dot(identity, identity)


@pytest.mark.parametrize("size", [3, 4])
def test_apply_to_vector_parity(size):
    quat = pyrr.quaternion.create_from_eulers([0.2, -0.4, 0.9])
    value = np.arange(1, size + 1, dtype=np.float64)
    np.testing.assert_allclose(
        quaternion.apply_to_vector(quat, value),
        pyrr.quaternion.apply_to_vector(quat, value),
    )


def test_apply_to_integer_vector_does_not_narrow():
    quat = np.array([0.1, 0.2, 0.3, 0.9])
    value = np.array([1, 2, 3])
    actual = quaternion.apply_to_vector(quat, value)
    # Upstream creates an integer intermediate and silently truncates here.
    # Compare with its floating-point path, which is the intended arithmetic.
    expected = pyrr.quaternion.apply_to_vector(quat, value.astype(np.float64))
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected)


def test_power_parity():
    value = pyrr.quaternion.create_from_eulers([0.2, -0.4, 0.9])
    np.testing.assert_allclose(
        quaternion.power(value, 1.7), pyrr.quaternion.power(value, 1.7)
    )


def test_matrix_round_trip():
    quat = pyrr.quaternion.create_from_eulers([0.2, -0.4, 0.9])
    matrix = pyrr.matrix33.create_from_quaternion(quat)
    np.testing.assert_allclose(
        quaternion.create_from_matrix(matrix),
        pyrr.quaternion.create_from_matrix(matrix),
    )
