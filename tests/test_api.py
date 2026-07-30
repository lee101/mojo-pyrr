import inspect

import numpy as np
import mojopyrr
import pyrr
import pytest

from mojopyrr import euler, quaternion, vector3, vector4
from mojopyrr._lib import addr


def test_covered_module_exports():
    assert mojopyrr.__all__ == [
        "euler",
        "matrix33",
        "matrix44",
        "quaternion",
        "vector",
        "vector3",
        "vector4",
    ]


def test_public_signatures_match_upstream():
    for module_name in mojopyrr.__all__:
        ours = getattr(mojopyrr, module_name)
        theirs = getattr(pyrr, module_name)
        for name in dir(ours):
            if name.startswith("_") or not inspect.isfunction(getattr(ours, name)):
                continue
            if hasattr(theirs, name):
                assert inspect.signature(getattr(ours, name)) == inspect.signature(
                    getattr(theirs, name)
                ), f"{module_name}.{name}"


@pytest.mark.parametrize(
    "name,args",
    [
        ("create", (0.1, 0.2, 0.3)),
        ("create_from_x_rotation", (0.1,)),
        ("create_from_y_rotation", (0.2,)),
        ("create_from_z_rotation", (0.3,)),
    ],
)
def test_euler_helpers(name, args):
    actual = getattr(euler, name)(*args)
    expected = getattr(pyrr.euler, name)(*args)
    np.testing.assert_array_equal(actual, expected)
    assert euler.roll(actual) == pyrr.euler.roll(expected)
    assert euler.pitch(actual) == pyrr.euler.pitch(expected)
    assert euler.yaw(actual) == pyrr.euler.yaw(expected)


def test_remaining_quaternion_constructors():
    np.testing.assert_array_equal(quaternion.create(), pyrr.quaternion.create())
    axis = np.array([0.2, -0.3, 0.4])
    np.testing.assert_allclose(
        quaternion.create_from_axis(axis), pyrr.quaternion.create_from_axis(axis)
    )


@pytest.mark.parametrize("module", [vector3, vector4])
def test_all_unit_vector_constructors(module):
    reference = pyrr.vector3 if module is vector3 else pyrr.vector4
    names = ["create_unit_length_x", "create_unit_length_y", "create_unit_length_z"]
    if module is vector4:
        names.append("create_unit_length_w")
    for name in names:
        np.testing.assert_array_equal(
            getattr(module, name)(), getattr(reference, name)()
        )


def test_ffi_address_contract():
    with pytest.raises(TypeError):
        addr(np.ones(3, dtype=np.float32))
    with pytest.raises(TypeError):
        addr(np.ones((2, 3), dtype=np.float64)[:, ::2])
