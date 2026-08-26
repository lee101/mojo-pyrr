# mojo-pyrr

`mojo-pyrr` is a standalone Mojo port of the compute-heavy vector, matrix, and
quaternion operations in [pyrr](https://github.com/adamlwgriffiths/Pyrr). It
provides NumPy-facing Python modules with pyrr's function names and signatures,
while batching the arithmetic into one compiled Mojo shared library.

Use it as `import mojopyrr as pyrr` when migrating functional pyrr code. The
reference version used by this repository is pyrr 0.10.3.

## Coverage

The functional APIs of these modules are covered:

- `vector`: normalize/normalise, length, squared length, set length, dot, and
  interpolation over a single vector or an arbitrary batch.
- `vector3` and `vector4`: constructors and conversions, plus 3D cross products,
  face normals, and indexed vertex normals.
- `matrix33`: all constructors, multiplication, row-vector application,
  directional scale, and inversion.
- `matrix44`: transformation, rotation, scale, view, perspective and orthogonal
  projection constructors; multiplication, vector/point application, inversion,
  look-at, and decomposition.
- `quaternion`: all constructors and conversions, multiplication, lerp/slerp,
  length operations, conjugate/inverse/negate, exponential and power, axis/angle
  access, predicates, and vector application.
- The small `euler` helpers needed by the rotation APIs.

Matrix and quaternion kernels also accept matching or broadcast batches where
the upstream implementation only handles one object. Batched matrix
multiplication is pairwise over the leading dimensions; this is intentionally
more useful than NumPy `dot`'s higher-dimensional product shape.

Not covered are pyrr's ndarray object façade (`Matrix33`, `Matrix44`,
`Quaternion`, `Vector3`, and `Vector4`) and unrelated geometry modules such as
rays, planes, spheres, rectangles, and bounding boxes. This package covers the
functional 3D-math subset, so it is not a wholesale replacement for every pyrr
import.

## Install

The checked-in Pixi environment pins the Mojo nightly used by the source and
installs NumPy, pytest, and pyrr 0.10.3:

```sh
pixi install
pixi run build
```

The build writes `dist/libmojo-pyrr.so`. Tests and benchmarks run inside the
same environment:

```sh
pixi run test
pixi run bench
```

## Usage

This example applies a pyrr-layout transform to a batch of points and rotates a
vector with a quaternion:

```python
import numpy as np
import mojopyrr as pyrr

rotation = pyrr.matrix44.create_from_y_rotation(np.pi / 2)
translation = pyrr.matrix44.create_from_translation([10.0, 0.0, 0.0])
transform = pyrr.matrix44.multiply(rotation, translation)

points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
print(pyrr.matrix44.apply_to_vector(transform, points))

quat = pyrr.quaternion.create_from_axis_rotation([0.0, 1.0, 0.0], np.pi / 2)
print(pyrr.quaternion.apply_to_vector(quat, [1.0, 0.0, 0.0]))
```

Run it from the checkout with `pixi run python example.py`; Pixi sets
`PYTHONPATH=python`.

## Benchmarks

Measured with `pixi run bench`, taking the best of five runs after correctness
checks on identical inputs. The Pixi task holds `/tmp/mojo-bench.lock` to avoid
overlapping factory benchmarks.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux 6.8.0-136-generic.
Python 3.13.14, NumPy 2.5.1, pyrr 0.10.3.

| operation | input | Mojo | pyrr | pyrr / Mojo | outcome |
|---|---:|---:|---:|---:|---|
| `vector.normalize` | 1,000,000 x 3 | 6.87 ms | 44.95 ms | 6.54x | faster |
| `vector.dot` | 1,000,000 x 3 | 6.32 ms | 31.51 ms | 4.99x | faster |
| `vector3.cross` | 1,000,000 x 3 | 7.44 ms | 384.85 ms | 51.72x | faster |
| `matrix33.inverse` | 200,000 matrices | 15.77 ms | 165.56 ms | 10.50x | faster |
| `matrix44.inverse` | 100,000 matrices | 15.17 ms | 91.01 ms | 6.00x | faster |
| `matrix44.multiply` | 50,000 scalar calls | 64.54 ms | 71.96 ms | 1.12x | faster |

Scalar matrix multiplication stays on NumPy's zero-copy `dot` path, avoiding a
ctypes round trip for a small fixed amount of work. Batched extensions use
the Mojo kernel, whose native-width column blocks are SIMD-vectorized with a scalar
tail for 3x3 matrices.

No parallel or GPU path is provided. The only near-parity benchmark is a scalar
4x4 multiplication, where thread launch, FFI, and device-transfer overhead would
dominate the work. The batched fixed-size and elementwise kernels are already at
least about 5x faster and remain below the roughly 2-flop-per-byte threshold for
GPU offload.

## How it works

`src/pyrr.mojo` is one compilation unit exporting a flat C ABI. NumPy owns every
input, output, and inversion scratch allocation. Python passes each contiguous
float64 buffer as a 64-bit integer address through ctypes; Mojo reconstructs it
as `UnsafePointer[Float64, AnyOrigin[mut=True]]`. No object ownership crosses
the ABI and the shared library performs no heap allocation.

Matrices remain C-contiguous, row-major arrays with pyrr's row-vector
convention: translation is in the final row, and a vector is evaluated as
`vector @ matrix`. Public results are cast back to pyrr-compatible input dtypes
where its NumPy behavior preserves them.

## License

MIT
