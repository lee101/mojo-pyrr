"""ctypes access to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "pyrr.mojo")
LIB = os.environ.get("MOJOPYRR_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyrr.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpr_vector_normalize": ([I, I, I, I, F], None),
    "mpr_vector_lengths": ([I, I, I, I, I], None),
    "mpr_vector_dot": ([I, I, I, I, I], None),
    "mpr_vector_interpolate": ([I, I, I, I, F], None),
    "mpr_vector_cross3": ([I, I, I, I], None),
    "mpr_matmul": ([I, I, I, I, I], None),
    "mpr_matvec": ([I, I, I, I, I, I], None),
    "mpr_transform_points3": ([I, I, I, I, I], None),
    "mpr_inverse": ([I, I, I, I, I], I),
    "mpr_quat_cross": ([I, I, I, I], None),
    "mpr_quat_slerp": ([I, I, I, I, F], None),
    "mpr_quat_apply": ([I, I, I, I, I, I], None),
    "mpr_axis_angle_quat": ([I, I, I, F], None),
    "mpr_quat_to_mat33": ([I, I, I], None),
    "mpr_mat33_to_quat": ([I, I, I], None),
    "mpr_quat_exp": ([I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJOPYRR_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJOPYRR_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJOPYRR_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        SRC,
        "-o",
        LIB,
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=1800
    )
    if process.returncode or not os.path.exists(LIB):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB


_loaded = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def f64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64)


def empty(shape) -> np.ndarray:
    return np.empty(shape, dtype=np.float64, order="C")


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.dtype != np.float64 or not value.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous float64 arrays")
    address = int(value.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return address


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
