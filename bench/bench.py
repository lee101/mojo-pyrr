"""Benchmark mojopyrr against pyrr 0.10.3 on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojopyrr as mojo  # noqa: E402
import pyrr as reference  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(2026)
    vectors = np.ascontiguousarray(rng.normal(size=(1_000_000, 3)))
    other = np.ascontiguousarray(rng.normal(size=vectors.shape))

    raw4 = rng.normal(scale=0.15, size=(100_000, 4, 4))
    matrices4 = np.ascontiguousarray(
        raw4 @ np.swapaxes(raw4, -1, -2) + np.eye(4)
    )
    raw3 = rng.normal(scale=0.15, size=(200_000, 3, 3))
    matrices3 = np.ascontiguousarray(
        raw3 @ np.swapaxes(raw3, -1, -2) + np.eye(3)
    )
    scalar_left = matrices4[0]
    scalar_right = matrices4[1]

    def mojo_scalar_multiply():
        result = None
        for _ in range(50_000):
            result = mojo.matrix44.multiply(scalar_left, scalar_right)
        return result

    def pyrr_scalar_multiply():
        result = None
        for _ in range(50_000):
            result = reference.matrix44.multiply(scalar_left, scalar_right)
        return result

    cases = [
        (
            "vector.normalize",
            "1,000,000 x 3",
            lambda: mojo.vector.normalize(vectors),
            lambda: reference.vector.normalize(vectors),
        ),
        (
            "vector.dot",
            "1,000,000 x 3",
            lambda: mojo.vector.dot(vectors, other),
            lambda: reference.vector.dot(vectors, other),
        ),
        (
            "vector3.cross",
            "1,000,000 x 3",
            lambda: mojo.vector3.cross(vectors, other),
            lambda: reference.vector3.cross(vectors, other),
        ),
        (
            "matrix33.inverse",
            "200,000 matrices",
            lambda: mojo.matrix33.inverse(matrices3),
            lambda: reference.matrix33.inverse(matrices3),
        ),
        (
            "matrix44.inverse",
            "100,000 matrices",
            lambda: mojo.matrix44.inverse(matrices4),
            lambda: reference.matrix44.inverse(matrices4),
        ),
        (
            "matrix44.multiply",
            "50,000 scalar calls",
            mojo_scalar_multiply,
            pyrr_scalar_multiply,
        ),
    ]

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, pyrr {reference.__version__}")
    print()
    print("| operation | input | Mojo | pyrr | pyrr / Mojo | outcome |")
    print("|---|---:|---:|---:|---:|---|")
    for name, shape, ours, theirs in cases:
        np.testing.assert_allclose(ours(), theirs(), rtol=2e-12, atol=2e-12)
        ours_time = timeit(ours)
        their_time = timeit(theirs)
        ratio = their_time / ours_time
        outcome = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {shape} | {ours_time * 1e3:.2f} ms | "
            f"{their_time * 1e3:.2f} ms | {ratio:.2f}x | {outcome} |"
        )


if __name__ == "__main__":
    main()
