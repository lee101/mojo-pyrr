import numpy as np
import pytest
import pyrr

from mojopyrr import vector, vector3, vector4


@pytest.fixture
def vectors():
    return np.random.default_rng(42).normal(size=(257, 3))


@pytest.mark.parametrize("name", ["length", "squared_length", "normalize", "normalise"])
def test_vector_batch_parity(name, vectors):
    expected = getattr(pyrr.vector, name)(vectors)
    actual = getattr(vector, name)(vectors)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_vector_dtype_and_values(dtype):
    value = np.array([[3, 4, 0], [1, -2, 2]], dtype=dtype)
    for name in ("length", "squared_length", "normalize"):
        expected = getattr(pyrr.vector, name)(value)
        actual = getattr(vector, name)(value)
        assert np.asarray(actual).dtype == np.asarray(expected).dtype
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_dot_batch_parity(vectors):
    other = np.random.default_rng(3).normal(size=vectors.shape)
    np.testing.assert_allclose(vector.dot(vectors, other), pyrr.vector.dot(vectors, other))


def test_interpolate_parity(vectors):
    other = np.random.default_rng(3).normal(size=vectors.shape)
    np.testing.assert_allclose(
        vector.interpolate(vectors, other, 0.37),
        pyrr.vector.interpolate(vectors, other, 0.37),
    )


def test_set_length_parity(vectors):
    np.testing.assert_allclose(
        vector.set_length(vectors, 7.5),
        pyrr.vector.set_length(vectors, 7.5),
    )


def test_cross_batch_parity(vectors):
    other = np.random.default_rng(3).normal(size=vectors.shape)
    np.testing.assert_allclose(
        vector3.cross(vectors, other), pyrr.vector3.cross(vectors, other)
    )


@pytest.mark.parametrize("normalize_result", [False, True])
def test_generate_normals_parity(vectors, normalize_result):
    actual = vector3.generate_normals(
        vectors[:-2], vectors[1:-1], vectors[2:], normalize_result
    )
    expected = pyrr.vector3.generate_normals(
        vectors[:-2], vectors[1:-1], vectors[2:], normalize_result
    )
    np.testing.assert_allclose(actual, expected)


def test_generate_vertex_normals_parity():
    vertices = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    indices = np.array([[0, 1, 2], [0, 2, 3]])
    np.testing.assert_allclose(
        vector3.generate_vertex_normals(vertices, indices),
        pyrr.vector3.generate_vertex_normals(vertices, indices),
    )


@pytest.mark.parametrize(
    "module,name,args",
    [
        (vector3, "create", (1, 2, 3)),
        (vector4, "create", (1, 2, 3, 4)),
        (vector3, "create_unit_length_x", ()),
        (vector3, "create_unit_length_y", ()),
        (vector3, "create_unit_length_z", ()),
        (vector4, "create_unit_length_w", ()),
    ],
)
def test_vector_constructors(module, name, args):
    reference = pyrr.vector3 if module is vector3 else pyrr.vector4
    np.testing.assert_array_equal(
        getattr(module, name)(*args), getattr(reference, name)(*args)
    )


def test_vector_conversions():
    value = np.array([1.0, 2.0, 3.0, 4.0])
    ours, ours_w = vector3.create_from_vector4(value)
    theirs, theirs_w = pyrr.vector3.create_from_vector4(value)
    np.testing.assert_array_equal(ours, theirs)
    assert ours_w == theirs_w
    np.testing.assert_array_equal(
        vector4.create_from_vector3(ours, ours_w),
        pyrr.vector4.create_from_vector3(theirs, theirs_w),
    )


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3)])
def test_empty_vector_batches_are_safe(shape):
    value = np.empty(shape)
    for name in ("normalize", "length", "squared_length"):
        np.testing.assert_array_equal(
            getattr(vector, name)(value), getattr(pyrr.vector, name)(value)
        )
    np.testing.assert_array_equal(vector.dot(value, value), pyrr.vector.dot(value, value))


@pytest.mark.parametrize("module", [vector3, vector4])
def test_translation_conversion_parity(module):
    matrix = np.arange(16, dtype=np.float64).reshape(4, 4)
    reference = pyrr.vector3 if module is vector3 else pyrr.vector4
    np.testing.assert_array_equal(
        module.create_from_matrix44_translation(matrix),
        reference.create_from_matrix44_translation(matrix),
    )
