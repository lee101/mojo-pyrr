"""Float64 vector, matrix, and quaternion kernels exposed through a C ABI."""

from std.math import acos, cos, exp, sin, sqrt
from std.sys import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def p(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


@export("mpr_vector_normalize")
def mpr_vector_normalize(src: Int, dst: Int, n: Int, d: Int, scale: Float64) abi("C"):
    var a = p(src)
    var z = p(dst)
    for r in range(n):
        var total = 0.0
        for j in range(d):
            var value = a[r * d + j]
            total += value * value
        var factor = scale / sqrt(total)
        for j in range(d):
            z[r * d + j] = a[r * d + j] * factor


@export("mpr_vector_lengths")
def mpr_vector_lengths(src: Int, dst: Int, n: Int, d: Int, squared: Int) abi("C"):
    var a = p(src)
    var z = p(dst)
    for r in range(n):
        var total = 0.0
        for j in range(d):
            var value = a[r * d + j]
            total += value * value
        z[r] = total if squared != 0 else sqrt(total)


@export("mpr_vector_dot")
def mpr_vector_dot(lhs: Int, rhs: Int, dst: Int, n: Int, d: Int) abi("C"):
    var a = p(lhs)
    var b = p(rhs)
    var z = p(dst)
    for r in range(n):
        var acc = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= d:
            acc += a.load[width=W](r * d + j) * b.load[width=W](r * d + j)
            j += W
        var total = acc.reduce_add()
        while j < d:
            total += a[r * d + j] * b[r * d + j]
            j += 1
        z[r] = total


@export("mpr_vector_interpolate")
def mpr_vector_interpolate(
    lhs: Int, rhs: Int, dst: Int, count: Int, delta: Float64
) abi("C"):
    var a = p(lhs)
    var b = p(rhs)
    var z = p(dst)
    var vd = SIMD[DType.float64, W](delta)
    var i = 0
    while i + W <= count:
        var av = a.load[width=W](i)
        z.store(i, av + (b.load[width=W](i) - av) * vd)
        i += W
    while i < count:
        z[i] = a[i] + (b[i] - a[i]) * delta
        i += 1


@export("mpr_vector_cross3")
def mpr_vector_cross3(lhs: Int, rhs: Int, dst: Int, n: Int) abi("C"):
    var a = p(lhs)
    var b = p(rhs)
    var z = p(dst)
    for r in range(n):
        var i = r * 3
        z[i] = a[i + 1] * b[i + 2] - a[i + 2] * b[i + 1]
        z[i + 1] = a[i + 2] * b[i] - a[i] * b[i + 2]
        z[i + 2] = a[i] * b[i + 1] - a[i + 1] * b[i]


@export("mpr_matmul")
def mpr_matmul(lhs: Int, rhs: Int, dst: Int, n: Int, size: Int) abi("C"):
    var a = p(lhs)
    var b = p(rhs)
    var z = p(dst)
    var width = size * size
    for batch in range(n):
        var base = batch * width
        for row in range(size):
            var col = 0
            while col + W <= size:
                var acc = SIMD[DType.float64, W](0.0)
                for k in range(size):
                    acc += SIMD[DType.float64, W](
                        a[base + row * size + k]
                    ) * b.load[width=W](base + k * size + col)
                z.store(base + row * size + col, acc)
                col += W
            while col < size:
                var total = 0.0
                for k in range(size):
                    total += a[base + row * size + k] * b[base + k * size + col]
                z[base + row * size + col] = total
                col += 1


@export("mpr_matvec")
def mpr_matvec(
    matrix: Int, vectors: Int, dst: Int, n: Int, size: Int, matrix_stride: Int
) abi("C"):
    var m = p(matrix)
    var v = p(vectors)
    var z = p(dst)
    for batch in range(n):
        var mb = batch * matrix_stride
        var vb = batch * size
        for col in range(size):
            var total = 0.0
            for row in range(size):
                total += v[vb + row] * m[mb + row * size + col]
            z[vb + col] = total


@export("mpr_transform_points3")
def mpr_transform_points3(
    matrix: Int, vectors: Int, dst: Int, n: Int, matrix_stride: Int
) abi("C"):
    var m = p(matrix)
    var v = p(vectors)
    var z = p(dst)
    for batch in range(n):
        var mb = batch * matrix_stride
        var vb = batch * 3
        var x = v[vb]
        var y = v[vb + 1]
        var zz = v[vb + 2]
        var w = x * m[mb + 3] + y * m[mb + 7] + zz * m[mb + 11] + m[mb + 15]
        if abs(w) <= 1.0e-8:
            var zero = abs(w) - abs(w)
            var infinity = 1.0 / zero
            z[vb] = infinity
            z[vb + 1] = infinity
            z[vb + 2] = infinity
        else:
            z[vb] = (x * m[mb] + y * m[mb + 4] + zz * m[mb + 8] + m[mb + 12]) / w
            z[vb + 1] = (x * m[mb + 1] + y * m[mb + 5] + zz * m[mb + 9] + m[mb + 13]) / w
            z[vb + 2] = (x * m[mb + 2] + y * m[mb + 6] + zz * m[mb + 10] + m[mb + 14]) / w


@export("mpr_inverse")
def mpr_inverse(
    src: Int, dst: Int, scratch: Int, n: Int, size: Int
) abi("C") -> Int:
    var a = p(src)
    var z = p(dst)
    var inv = p(scratch)
    var width = size * size
    for batch in range(n):
        var base = batch * width
        for i in range(width):
            z[base + i] = a[base + i]
            inv[base + i] = 0.0
        for i in range(size):
            inv[base + i * size + i] = 1.0
        for col in range(size):
            var pivot = col
            var pivot_abs = abs(z[base + col * size + col])
            for row in range(col + 1, size):
                var candidate = abs(z[base + row * size + col])
                if candidate > pivot_abs:
                    pivot_abs = candidate
                    pivot = row
            if pivot_abs == 0.0:
                return batch + 1
            if pivot != col:
                for j in range(size):
                    var tmp = z[base + col * size + j]
                    z[base + col * size + j] = z[base + pivot * size + j]
                    z[base + pivot * size + j] = tmp
                    tmp = inv[base + col * size + j]
                    inv[base + col * size + j] = inv[base + pivot * size + j]
                    inv[base + pivot * size + j] = tmp
            var divisor = z[base + col * size + col]
            for j in range(size):
                z[base + col * size + j] /= divisor
                inv[base + col * size + j] /= divisor
            for row in range(size):
                if row == col:
                    continue
                var factor = z[base + row * size + col]
                for j in range(size):
                    z[base + row * size + j] -= factor * z[base + col * size + j]
                    inv[base + row * size + j] -= factor * inv[base + col * size + j]
        for i in range(width):
            z[base + i] = inv[base + i]
    return 0


@export("mpr_quat_cross")
def mpr_quat_cross(lhs: Int, rhs: Int, dst: Int, n: Int) abi("C"):
    var a = p(lhs)
    var b = p(rhs)
    var z = p(dst)
    for batch in range(n):
        var i = batch * 4
        var ax = a[i]
        var ay = a[i + 1]
        var az = a[i + 2]
        var aw = a[i + 3]
        var bx = b[i]
        var by = b[i + 1]
        var bz = b[i + 2]
        var bw = b[i + 3]
        z[i] = ax * bw + ay * bz - az * by + aw * bx
        z[i + 1] = -ax * bz + ay * bw + az * bx + aw * by
        z[i + 2] = ax * by - ay * bx + az * bw + aw * bz
        z[i + 3] = -ax * bx - ay * by - az * bz + aw * bw


@export("mpr_quat_slerp")
def mpr_quat_slerp(lhs: Int, rhs: Int, dst: Int, n: Int, t_in: Float64) abi("C"):
    var a = p(lhs)
    var b = p(rhs)
    var z = p(dst)
    var t = max(0.0, min(1.0, t_in))
    for batch in range(n):
        var i = batch * 4
        var dot = a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3]
        var sign = 1.0
        if dot < 0.0:
            dot = -dot
            sign = -1.0
        if dot < 0.95:
            var angle = acos(dot)
            var denom = sin(angle)
            var wa = sin(angle * (1.0 - t)) / denom
            var wb = sign * sin(angle * t) / denom
            for j in range(4):
                z[i + j] = a[i + j] * wa + b[i + j] * wb
        else:
            var total = 0.0
            for j in range(4):
                var value = a[i + j] * (1.0 - t) + b[i + j] * t
                z[i + j] = value
                total += value * value
            var factor = 1.0 / sqrt(total)
            for j in range(4):
                z[i + j] *= factor


@export("mpr_quat_apply")
def mpr_quat_apply(
    quats: Int, vectors: Int, dst: Int, n: Int, vector_size: Int, quat_stride: Int
) abi("C"):
    var q = p(quats)
    var v = p(vectors)
    var z = p(dst)
    for batch in range(n):
        var qi = batch * quat_stride
        var vi = batch * vector_size
        var qx = q[qi]
        var qy = q[qi + 1]
        var qz = q[qi + 2]
        var qw = q[qi + 3]
        var vx = v[vi]
        var vy = v[vi + 1]
        var vz = v[vi + 2]
        var norm_vector = qx * qx + qy * qy + qz * qz
        var scale = qw * qw - norm_vector
        var twice_dot = 2.0 * (qx * vx + qy * vy + qz * vz)
        z[vi] = scale * vx + twice_dot * qx + 2.0 * qw * (qy * vz - qz * vy)
        z[vi + 1] = scale * vy + twice_dot * qy + 2.0 * qw * (qz * vx - qx * vz)
        z[vi + 2] = scale * vz + twice_dot * qz + 2.0 * qw * (qx * vy - qy * vx)
        if vector_size == 4:
            z[vi + 3] = v[vi + 3] * (norm_vector + qw * qw)


@export("mpr_axis_angle_quat")
def mpr_axis_angle_quat(
    axes: Int, dst: Int, n: Int, theta: Float64
) abi("C"):
    var a = p(axes)
    var z = p(dst)
    var half = theta * 0.5
    var sine = sin(half)
    var cosine = cos(half)
    for batch in range(n):
        var ai = batch * 3
        var qi = batch * 4
        var length = sqrt(a[ai] * a[ai] + a[ai + 1] * a[ai + 1] + a[ai + 2] * a[ai + 2])
        var factor = sine / length
        z[qi] = a[ai] * factor
        z[qi + 1] = a[ai + 1] * factor
        z[qi + 2] = a[ai + 2] * factor
        z[qi + 3] = cosine


@export("mpr_quat_to_mat33")
def mpr_quat_to_mat33(quats: Int, dst: Int, n: Int) abi("C"):
    var q = p(quats)
    var z = p(dst)
    for batch in range(n):
        var qi = batch * 4
        var mi = batch * 9
        var x = q[qi]
        var y = q[qi + 1]
        var zz = q[qi + 2]
        var w = q[qi + 3]
        var invs = 1.0 / (x * x + y * y + zz * zz + w * w)
        z[mi] = (x * x - y * y - zz * zz + w * w) * invs
        z[mi + 4] = (-x * x + y * y - zz * zz + w * w) * invs
        z[mi + 8] = (-x * x - y * y + zz * zz + w * w) * invs
        z[mi + 3] = 2.0 * (x * y + zz * w) * invs
        z[mi + 1] = 2.0 * (x * y - zz * w) * invs
        z[mi + 6] = 2.0 * (x * zz - y * w) * invs
        z[mi + 2] = 2.0 * (x * zz + y * w) * invs
        z[mi + 7] = 2.0 * (y * zz + x * w) * invs
        z[mi + 5] = 2.0 * (y * zz - x * w) * invs


@export("mpr_mat33_to_quat")
def mpr_mat33_to_quat(matrices: Int, dst: Int, n: Int) abi("C"):
    var m = p(matrices)
    var z = p(dst)
    for batch in range(n):
        var mi = batch * 9
        var qi = batch * 4
        var trace = m[mi] + m[mi + 4] + m[mi + 8]
        if trace > 0.0:
            var s = 0.5 / sqrt(trace + 1.0)
            z[qi] = (m[mi + 7] - m[mi + 5]) * s
            z[qi + 1] = (m[mi + 2] - m[mi + 6]) * s
            z[qi + 2] = (m[mi + 3] - m[mi + 1]) * s
            z[qi + 3] = 0.25 / s
        elif m[mi] > m[mi + 4] and m[mi] > m[mi + 8]:
            var s = 2.0 * sqrt(1.0 + m[mi] - m[mi + 4] - m[mi + 8])
            z[qi] = 0.25 * s
            z[qi + 1] = (m[mi + 1] + m[mi + 3]) / s
            z[qi + 2] = (m[mi + 2] + m[mi + 6]) / s
            z[qi + 3] = (m[mi + 7] - m[mi + 5]) / s
        elif m[mi + 4] > m[mi + 8]:
            var s = 2.0 * sqrt(1.0 + m[mi + 4] - m[mi] - m[mi + 8])
            z[qi] = (m[mi + 1] + m[mi + 3]) / s
            z[qi + 1] = 0.25 * s
            z[qi + 2] = (m[mi + 5] + m[mi + 7]) / s
            z[qi + 3] = (m[mi + 2] - m[mi + 6]) / s
        else:
            var s = 2.0 * sqrt(1.0 + m[mi + 8] - m[mi] - m[mi + 4])
            z[qi] = (m[mi + 2] + m[mi + 6]) / s
            z[qi + 1] = (m[mi + 5] + m[mi + 7]) / s
            z[qi + 2] = 0.25 * s
            z[qi + 3] = (m[mi + 3] - m[mi + 1]) / s


@export("mpr_quat_exp")
def mpr_quat_exp(quats: Int, dst: Int, n: Int) abi("C"):
    var q = p(quats)
    var z = p(dst)
    for batch in range(n):
        var i = batch * 4
        var vector_norm = sqrt(q[i] * q[i] + q[i + 1] * q[i + 1] + q[i + 2] * q[i + 2])
        var e = exp(q[i + 3])
        if abs(vector_norm) <= 1.0e-8:
            z[i] = 0.0
            z[i + 1] = 0.0
            z[i + 2] = 0.0
            z[i + 3] = e
        else:
            var factor = e * sin(vector_norm) / vector_norm
            z[i] = q[i] * factor
            z[i + 1] = q[i + 1] * factor
            z[i + 2] = q[i + 2] * factor
            z[i + 3] = e * cos(vector_norm)
