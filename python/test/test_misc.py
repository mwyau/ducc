# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2025 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras


import gc

import ducc0
import numpy as np
import pytest
from ducc0.misc import special_add_at
from numpy.testing import assert_allclose
from _longdouble_support import HAS_NATIVE_LONGDOUBLE

pmp = pytest.mark.parametrize

@pytest.mark.skipif(not HAS_NATIVE_LONGDOUBLE,
                    reason="native long-double buffers unavailable or PPC safeguard")
def test_native_longdouble_misc():
    real = np.array([1, 2, 3], dtype=np.longdouble)
    real_other = np.array([2, 4, 6], dtype=np.longdouble)
    complex_ = np.array([1+2j, 2-1j, 4+0.5j], dtype=np.clongdouble)
    complex_other = np.array([2+1j, 1+0.5j, 3-2j], dtype=np.clongdouble)

    assert_allclose(ducc0.misc.vdot(real, real_other), np.vdot(real, real_other))
    assert_allclose(ducc0.misc.vdot(complex_, complex_other),
                    np.vdot(complex_, complex_other))
    real64 = real_other.astype(np.float64)
    assert_allclose(ducc0.misc.vdot(real, real64), np.vdot(real, real64))
    assert_allclose(ducc0.misc.vdot(real64, real), np.vdot(real64, real))
    assert_allclose(ducc0.misc.l2error(real, real_other), 0.5)
    assert_allclose(ducc0.misc.l2error(real, real64), 0.5)
    assert_allclose(ducc0.misc.l2error(real64, real), 0.5)

    complex128 = complex_other.astype(np.complex128)
    assert_allclose(ducc0.misc.vdot(complex_, complex128),
                    np.vdot(complex_, complex128))
    assert_allclose(ducc0.misc.vdot(complex128, complex_),
                    np.vdot(complex128, complex_))
    mixed_complex_l2 = np.sqrt(
        np.sum(np.abs(complex_ - complex128)**2) /
        max(np.sum(np.abs(complex_)**2), np.sum(np.abs(complex128)**2)))
    assert_allclose(ducc0.misc.l2error(complex_, complex128), mixed_complex_l2)
    assert_allclose(ducc0.misc.l2error(complex128, complex_), mixed_complex_l2)
    assert_allclose(ducc0.misc.l2error(complex_, complex_other),
                    np.sqrt(np.sum(np.abs(complex_-complex_other)**2) /
                            max(np.sum(np.abs(complex_)**2),
                                np.sum(np.abs(complex_other)**2))))

    for dtype, columns in ((np.longdouble, 512), (np.clongdouble, 256)):
        arr = np.arange(2*columns, dtype=np.longdouble).reshape(2, columns)
        if dtype is np.clongdouble:
            arr = arr.astype(np.clongdouble) * (1+2j)
        source = arr.copy()
        copied = ducc0.misc.make_noncritical(source, nthreads=2)
        assert type(copied) is np.ndarray
        assert copied.dtype == arr.dtype
        assert memoryview(copied).format == memoryview(arr).format
        assert np.array_equal(copied, source)
        assert not np.shares_memory(copied, source)
        assert copied.strides[0] % 4096 != 0
        del source
        gc.collect()
        assert np.array_equal(copied, arr)


@pmp("dtype", (np.float32, np.float64, np.complex64, np.complex128))
def test_native_longdouble_fallback_keeps_ordinary_misc_overloads(dtype):
    values = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * values[::-1]
    a = values.astype(dtype)
    b = (values[::-1] * 0.5).astype(dtype)
    assert_allclose(ducc0.misc.vdot(a, b), np.vdot(a, b))
    expected = np.sqrt(
        np.sum(np.abs(a-b)**2) / max(np.sum(np.abs(a)**2), np.sum(np.abs(b)**2)))
    assert_allclose(ducc0.misc.l2error(a, b), expected)
    if ducc0.__wrapper__ == "nanobind":
        assert_allclose(ducc0.misc.vdot(memoryview(a), memoryview(b)), np.vdot(a, b))
        assert_allclose(ducc0.misc.l2error(memoryview(a), memoryview(b)), expected)


def test_empty_noncritical_dtype_and_layout():
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    if HAS_NATIVE_LONGDOUBLE:
        dtypes.extend((np.longdouble, np.clongdouble))

    for dtype in dtypes:
        columns = 4096 // np.dtype(dtype).itemsize
        shape = (2, columns)
        arr = ducc0.misc.empty_noncritical(shape, dtype, nthreads=2)
        assert type(arr) is np.ndarray
        assert arr.shape == shape
        assert arr.dtype == np.dtype(dtype)
        assert memoryview(arr).format == memoryview(np.empty(1, dtype=dtype)).format
        assert arr.strides[0] % 4096 != 0


@pmp("shape", ([43], [654, 23], [32, 3, 11]))
@pmp("dtype_cov", (np.float32, np.float64))
@pmp("cplx", (False, True))
@pmp("broadcast", (False, True))
@pmp("nthreads", (1, 2))
def test_gaussenergy(shape, dtype_cov, cplx, broadcast, nthreads):
    rng = np.random.default_rng(42)

    a = rng.uniform(-.5, .5, shape).astype(dtype_cov)
    b = rng.uniform(-.5, .5, shape).astype(dtype_cov)
    c = rng.uniform(-.5, .5, shape).astype(dtype_cov)
    if cplx:
        a = a + 1j*rng.uniform(-.5, .5, shape).astype(dtype_cov)
        b = b + 1j*rng.uniform(-.5, .5, shape).astype(dtype_cov)
    if broadcast:
        a = np.broadcast_to(a[2:3], b.shape)
    res = ducc0.misc.experimental.LogUnnormalizedGaussProbability(a, b, c, nthreads)
    ref = 0.5*ducc0.misc.vdot((a-b)*c, a-b).real
    rtol = 1e-5 if dtype_cov == np.float32 else 1e-12
    assert_allclose(res, ref, rtol=rtol)

    res, deriv = ducc0.misc.experimental.LogUnnormalizedGaussProbabilityWithDeriv(a, b, c, nthreads=nthreads)
    assert_allclose(res, ref, rtol=rtol)
    assert_allclose(deriv, (a-b)*c, rtol=rtol)


@pmp("a_shape, axis, index, b, expected",
     [
        # Repeated index: accumulate at same position
        ((3,), 0, np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0]), np.array([0.0, 6.0, 0.0])),
        # Index in reversed order
        ((3,), 0, np.array([2, 1, 0]), np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])),
        # All zeros in index (accumulate to one bin)
        ((3,), 0, np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]), np.array([6.0, 0.0, 0.0])),
        # All same index in 2D along axis 0
        ((3, 2), 0, np.array([1, 1, 1]), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[0, 0], [9, 12], [0, 0]])),
        # Mixed and repeated index in 2D
        ((4, 2), 0, np.array([1, 0, 1]), np.array([[10, 20], [30, 40], [50, 60]]), np.array([[30, 40], [60, 80], [0, 0], [0, 0]])),
        # Index skipping bins
        ((5,), 0, np.array([0, 2, 4]), np.array([5.0, 10.0, 15.0]), np.array([5.0, 0.0, 10.0, 0.0, 15.0])),
        # Broadcasting pattern in axis 1
        ((2, 4), 1, np.array([1, 1, 2]), np.array([[1, 2, 3], [4, 5, 6]]), np.array([[0, 3, 3, 0], [0, 9, 6, 0]])),
        # Using every bin more than once (with overlaps)
        ((4,), 0, np.array([1, 2, 1, 3]), np.array([2.0, 4.0, 6.0, 8.0]), np.array([0.0, 8.0, 4.0, 8.0])),
     ]
)
@pmp("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_special_add_at_creative(a_shape, axis, index, b, expected, dtype):
    b, expected = b.astype(dtype), expected.astype(dtype)
    a = np.zeros(a_shape, dtype=b.dtype)
    out = special_add_at(a, axis=axis, index=index, b=b)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_special_add_at_complex(dtype):
    a = np.zeros((3,), dtype=dtype)
    b = np.array([1+2j, 3+4j, 5+6j], dtype=dtype)

    index = np.array([0, 1, 2], dtype=np.int32)
    expected = np.array([1+2j, 3+4j, 5+6j], dtype=dtype)
    out = special_add_at(a.copy(), axis=0, index=index, b=b)
    np.testing.assert_array_almost_equal(out, expected)

    index = np.array([0, 1, 1], dtype=np.int32)
    expected = np.array([1+2j, 3+4j + 5+6j, 0], dtype=dtype)
    out = special_add_at(a.copy(), axis=0, index=index, b=b)
    np.testing.assert_array_almost_equal(out, expected)
