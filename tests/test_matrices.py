import numpy as np
import pytest
import pyrr

from mojopyrr import matrix33, matrix44


EULERS = np.array([0.27, -0.63, 1.11])
AXIS = np.array([1.0, -2.0, 0.5])
QUAT = pyrr.quaternion.create_from_eulers(EULERS)


@pytest.mark.parametrize(
    "name,args",
    [
        ("create_identity", ()),
        ("create_from_eulers", (EULERS,)),
        ("create_from_axis_rotation", (AXIS, 0.81)),
        ("create_from_quaternion", (QUAT,)),
        ("create_from_inverse_of_quaternion", (QUAT,)),
        ("create_from_scale", ([2.0, 3.0, 4.0],)),
        ("create_from_x_rotation", (0.4,)),
        ("create_from_y_rotation", (0.4,)),
        ("create_from_z_rotation", (0.4,)),
    ],
)
def test_matrix33_constructors(name, args):
    np.testing.assert_allclose(
        getattr(matrix33, name)(*args), getattr(pyrr.matrix33, name)(*args)
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("create_identity", ()),
        ("create_from_eulers", (EULERS,)),
        ("create_from_axis_rotation", (AXIS, 0.81)),
        ("create_from_quaternion", (QUAT,)),
        ("create_from_inverse_of_quaternion", (QUAT,)),
        ("create_from_translation", ([2.0, 3.0, 4.0],)),
        ("create_from_scale", ([2.0, 3.0, 4.0],)),
        ("create_from_x_rotation", (0.4,)),
        ("create_from_y_rotation", (0.4,)),
        ("create_from_z_rotation", (0.4,)),
        ("create_perspective_projection", (60.0, 16 / 9, 0.1, 100.0)),
        (
            "create_perspective_projection_from_bounds",
            (-1.0, 1.0, -0.5, 0.5, 0.1, 100.0),
        ),
        (
            "create_orthogonal_projection",
            (-2.0, 2.0, -1.0, 1.0, 0.1, 100.0),
        ),
    ],
)
def test_matrix44_constructors(name, args):
    np.testing.assert_allclose(
        getattr(matrix44, name)(*args), getattr(pyrr.matrix44, name)(*args)
    )


@pytest.mark.parametrize("module,size", [(matrix33, 3), (matrix44, 4)])
def test_matrix_multiply_apply_inverse(module, size):
    reference = pyrr.matrix33 if size == 3 else pyrr.matrix44
    left = getattr(reference, "create_from_eulers")(EULERS)
    right = getattr(reference, "create_from_scale")([2.0, 3.0, 4.0])
    combined = reference.multiply(left, right)
    np.testing.assert_allclose(module.multiply(left, right), combined)
    vector = np.arange(1, size + 1, dtype=np.float64)
    np.testing.assert_allclose(
        module.apply_to_vector(combined, vector),
        reference.apply_to_vector(combined, vector),
    )
    np.testing.assert_allclose(module.inverse(combined), reference.inverse(combined))


@pytest.mark.parametrize("module,size", [(matrix33, 3), (matrix44, 4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_scalar_matrix_multiply_dtype_and_list_paths(module, size, dtype):
    reference = pyrr.matrix33 if size == 3 else pyrr.matrix44
    left = np.arange(size * size, dtype=dtype).reshape(size, size)
    right = np.eye(size, dtype=dtype) * 2
    expected = reference.multiply(left, right)
    actual = module.multiply(left, right)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        module.multiply(left.tolist(), right.tolist()),
        reference.multiply(left.tolist(), right.tolist()),
    )


def test_matrix44_point_transform_parity():
    matrix = pyrr.matrix44.multiply(
        pyrr.matrix44.create_from_translation([4.0, -2.0, 7.0]),
        pyrr.matrix44.create_from_eulers(EULERS),
    )
    point = np.array([2.0, 3.0, -1.0])
    np.testing.assert_allclose(
        matrix44.apply_to_vector(matrix, point),
        pyrr.matrix44.apply_to_vector(matrix, point),
    )


def test_matrix44_point_at_infinity_parity():
    matrix = np.zeros((4, 4))
    point = np.array([2.0, 3.0, -1.0])
    np.testing.assert_array_equal(
        matrix44.apply_to_vector(matrix, point),
        pyrr.matrix44.apply_to_vector(matrix, point),
    )


def test_look_at_parity():
    args = ([3.0, 4.0, 8.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        matrix44.create_look_at(*args),
        pyrr.matrix44.create_look_at(*args),
        atol=1e-14,
    )


def test_decompose_parity():
    matrix = pyrr.matrix44.multiply(
        pyrr.matrix44.create_from_scale([2.0, 3.0, 4.0]),
        pyrr.matrix44.create_from_quaternion(QUAT),
    )
    matrix[3, :3] = [7.0, -3.0, 2.0]
    actual = matrix44.decompose(matrix)
    expected = pyrr.matrix44.decompose(matrix)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right)


@pytest.mark.parametrize("module,size", [(matrix33, 3), (matrix44, 4)])
def test_singular_inverse_raises(module, size):
    with pytest.raises(np.linalg.LinAlgError):
        module.inverse(np.zeros((size, size)))


def test_batched_extension_matches_scalar_reference():
    rng = np.random.default_rng(99)
    matrices = np.stack(
        [
            pyrr.matrix44.create_from_axis_rotation(axis, angle)
            for axis, angle in zip(rng.normal(size=(32, 3)), rng.normal(size=32))
        ]
    )
    vectors = rng.normal(size=(32, 4))
    expected = np.stack(
        [pyrr.matrix44.apply_to_vector(m, v) for m, v in zip(matrices, vectors)]
    )
    np.testing.assert_allclose(matrix44.apply_to_vector(matrices, vectors), expected)


@pytest.mark.parametrize("module,size", [(matrix33, 3), (matrix44, 4)])
def test_batched_matrix_multiply_simd_and_tail(module, size):
    rng = np.random.default_rng(811 + size)
    left = rng.normal(size=(7, size, size))
    right = rng.normal(size=(7, size, size))
    reference = pyrr.matrix33 if size == 3 else pyrr.matrix44
    expected = np.stack(
        [reference.multiply(a, b) for a, b in zip(left, right)]
    )
    np.testing.assert_allclose(module.multiply(left, right), expected)


def test_remaining_matrix33_constructors():
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    np.testing.assert_array_equal(
        matrix33.create_from_matrix44(matrix),
        pyrr.matrix33.create_from_matrix44(matrix),
    )
    args = ([1.0, 2.0, -3.0], 0.25)
    np.testing.assert_allclose(
        matrix33.create_direction_scale(*args),
        pyrr.matrix33.create_direction_scale(*args),
    )


def test_matrix44_views_and_constructor_aliases():
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    np.testing.assert_array_equal(
        matrix44.create_matrix33_view(matrix),
        pyrr.matrix44.create_matrix33_view(matrix),
    )
    matrix33_value = np.arange(9, dtype=np.float64).reshape(3, 3)
    np.testing.assert_array_equal(
        matrix44.create_from_matrix33(matrix33_value),
        pyrr.matrix44.create_from_matrix33(matrix33_value),
    )
    cases = [
        ("create_perspective_projection_matrix", (60.0, 1.5, 0.1, 100.0)),
        (
            "create_perspective_projection_matrix_from_bounds",
            (-1.0, 1.0, -0.5, 0.5, 0.1, 100.0),
        ),
        (
            "create_orthogonal_projection_matrix",
            (-2.0, 2.0, -1.0, 1.0, 0.1, 100.0),
        ),
    ]
    for name, args in cases:
        np.testing.assert_allclose(
            getattr(matrix44, name)(*args), getattr(pyrr.matrix44, name)(*args)
        )
