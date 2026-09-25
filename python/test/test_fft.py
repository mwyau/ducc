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
# Copyright(C) 2020-2025 Max-Planck-Society


import ducc0
import ducc0.fft as fft
from ducc0.misc import l2error as l2error
# import pyfftw
import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose
import ctypes
import platform
import sys

pmp = pytest.mark.parametrize

shapes1D = ((10,), (127,))
shapes2D = ((128, 128), (128, 129),
            (1, 129), (2, 127), (3, 127), (6, 127),
            (129, 1), (127, 2), (127, 3), (127, 6))
shapes3D = ((32, 17, 39),(32, 1, 39),(2, 3, 17),(2,8,17),(5,7,5))
shapes = shapes1D+shapes2D+shapes3D
len1D = list(range(1, 256)) + list(range(1700, 2048)) + [137*137]


def _assert_close(a, b, epsilon):
    ext_dtypes = (np.dtype(np.longdouble), np.dtype(np.clongdouble))
    if np.asarray(a).dtype in ext_dtypes or np.asarray(b).dtype in ext_dtypes:
        # misc.l2error currently cannot consume DUCC-private nanobind views.
        # Keep its normalized L2 metric and accumulate in native long double.
        aa, bb = np.asarray(a), np.asarray(b)
        aa2 = np.sum(np.abs(aa)**2, dtype=np.longdouble)
        bb2 = np.sum(np.abs(bb)**2, dtype=np.longdouble)
        delta2 = np.sum(np.abs(aa-bb)**2, dtype=np.longdouble)
        denominator = max(aa2, bb2)
        err = np.longdouble(0) if denominator == 0 else np.sqrt(delta2/denominator)
        assert_allclose(float(err), 0, atol=epsilon)
        return
    assert_allclose(l2error(a, b), 0, atol=epsilon)


def fftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return fft.c2c(a, axes=axes, forward=True, inorm=inorm,
                   out=out, nthreads=nthreads)


def ifftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return fft.c2c(a, axes=axes, forward=False, inorm=inorm,
                   out=out, nthreads=nthreads)


def rfftn(a, axes=None, inorm=0, nthreads=1):
    return fft.r2c(a, axes=axes, forward=True, inorm=inorm,
                   nthreads=nthreads)


def irfftn(a, axes=None, lastsize=0, inorm=0, nthreads=1):
    return fft.c2r(a, axes=axes, lastsize=lastsize, forward=False,
                   inorm=inorm, nthreads=nthreads)


def hc2c_fftpack(inp, otype):
    n = inp.shape[0]
    n2 = (n-1)//2
    out = np.zeros_like(inp, dtype=otype)
    out[0] = inp[0]
    if n % 2 == 0:
        out[n//2] = inp[-1]
    out[1:n2+1] = inp[1:1+2*n2:2] + 1j*inp[2:2+2*n2:2]
    out[-1:-n2-1:-1] = inp[1:1+2*n2:2] - 1j*inp[2:2+2*n2:2]
    return out


def hc2c_fftw(inp, otype):
    n = inp.shape[0]
    n2 = (n-1)//2
    out = np.zeros_like(inp, dtype=otype)
    out[0] = inp[0]
    if n % 2 == 0:
        out[n//2] = inp[n//2]
    out[1:n2+1] = inp[1:n2+1] + 1j*inp[-1:-n2-1:-1]
    out[-1:-n2-1:-1] = inp[1:n2+1] - 1j*inp[-1:-n2-1:-1]
    return out


tol = {np.float32: 6e-7, np.float64: 2e-15, np.longdouble: 1e-18}
ctype = {np.float32: np.complex64,
         np.float64: np.complex128,
         np.longdouble: np.clongdouble}


def _native_buffer_format(dtype):
    fmt = memoryview(np.empty(1, dtype=dtype)).format
    if fmt and fmt[0] in "@=<>!":
        prefix, fmt = fmt[0], fmt[1:]
        little = sys.byteorder == "little"
        if (prefix == "<" and not little) or (prefix in ">!" and little):
            return None
    return fmt


on_ppc64le = "ppc64le" in platform.machine().lower()
true_long_double = (
    np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    and np.dtype(np.longdouble).itemsize == ctypes.sizeof(ctypes.c_longdouble)
    and np.dtype(np.clongdouble).itemsize == 2 * ctypes.sizeof(ctypes.c_longdouble)
    and _native_buffer_format(np.longdouble) == "g"
    and _native_buffer_format(np.clongdouble) == "Zg"
    and not on_ppc64le
)
dtypes = [np.float32, np.float64]
if true_long_double:
    dtypes += [np.longdouble]


@pytest.mark.skipif(not true_long_double,
                    reason="native long-double buffers unavailable or PPC safeguard")
def test_native_longdouble_buffers():
    rng = np.random.default_rng(42)
    real_base = (rng.random(24) - 0.5).astype(np.longdouble)
    real = real_base[::2]
    real.setflags(write=False)

    spectrum = np.empty(real.size // 2 + 1, dtype=np.clongdouble)
    assert fft.r2c(real, out=spectrum) is spectrum
    restored_base = np.empty(real.size * 2, dtype=np.longdouble)
    restored = restored_base[::2]
    assert fft.c2r(spectrum, lastsize=real.size, forward=False, inorm=2,
                   out=restored) is restored
    assert restored.dtype == np.dtype(np.longdouble)
    _assert_close(real, restored, 1e-15)

    inplace_buf = np.empty(real.size // 2 + 1, dtype=np.clongdouble)
    inplace_real = inplace_buf.view(np.longdouble)[:real.size]
    inplace_real[()] = real
    inplace_ref = inplace_real.copy()
    fft.r2c(inplace_real, out=inplace_buf)
    assert fft.c2r(inplace_buf, lastsize=real.size, forward=False, inorm=2,
                    out=inplace_real) is inplace_real
    _assert_close(inplace_ref, inplace_real, 1e-15)

    complex_base = (real_base + 1j * real_base[::-1]).astype(np.clongdouble)
    complex_input = complex_base[::2]
    complex_input.setflags(write=False)
    complex_out = np.empty(complex_input.size, dtype=np.clongdouble)
    assert fft.c2c(complex_input, out=complex_out) is complex_out
    complex_back = fft.c2c(complex_out, forward=False, inorm=2)
    assert complex_back.dtype == np.dtype(np.clongdouble)
    _assert_close(complex_input, complex_back, 1e-15)

    inplace = complex_input.copy()
    assert fft.c2c(inplace, out=inplace) is inplace
    fft.c2c(inplace, forward=False, inorm=2, out=inplace)
    _assert_close(complex_input, inplace, 1e-15)

    c2c_real = fft.c2c(real)
    assert c2c_real.dtype == np.dtype(np.clongdouble)
    _assert_close(fft.genuine_fht(real), c2c_real.real-c2c_real.imag, 1e-15)
    _assert_close(fft.genuine_hartley(real), c2c_real.real+c2c_real.imag, 1e-15)
    dct = fft.dct(real, type=2)
    dst = fft.dst(real, type=2)
    assert dct.dtype == np.dtype(np.longdouble)
    assert dst.dtype == np.dtype(np.longdouble)
    _assert_close(real, fft.dct(dct, type=3, inorm=2), 1e-15)
    _assert_close(real, fft.dst(dst, type=3, inorm=2), 1e-15)
    hartley = fft.genuine_hartley(real)
    assert hartley.dtype == np.dtype(np.longdouble)
    _assert_close(real, fft.genuine_hartley(hartley, inorm=2), 1e-15)

    kernel = (real_base[:12] + 1j * real_base[12:]).astype(np.clongdouble)
    conv_out_base = np.empty(32, dtype=np.clongdouble)
    conv_out = conv_out_base[::2]
    assert fft.convolve_axis(complex_input, conv_out, 0, kernel) is conv_out
    conv_ref = fft.convolve_axis(
        complex_input.astype(np.complex128),
        np.empty(conv_out.shape, dtype=np.complex128), 0,
        kernel.astype(np.complex128))
    _assert_close(conv_ref, conv_out, 1e-15)

    with pytest.raises(TypeError):
        fft.r2c(real.tolist())
    foreign = real_base.astype(np.dtype(np.longdouble).newbyteorder("S"))
    if not foreign.dtype.isnative:
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            fft.r2c(foreign)


@pmp("len", len1D)
@pmp("inorm", [0, 1, 2])
@pmp("dtype", dtypes)
def test1D(len, inorm, dtype):
    rng = np.random.default_rng(42)
    a = rng.random(len)-0.5 + 1j*rng.random(len)-0.5j
    a = a.astype(ctype[dtype])
    eps = tol[dtype]
    _assert_close(a, ifftn(fftn(a, inorm=inorm), inorm=2-inorm), eps)
    _assert_close(a.real, ifftn(fftn(a.real, inorm=inorm), inorm=2-inorm), eps)
    _assert_close(a.real, fftn(ifftn(a.real, inorm=inorm), inorm=2-inorm), eps)
    _assert_close(a.real, irfftn(rfftn(a.real, inorm=inorm),
                                 inorm=2-inorm, lastsize=len), eps)
    _assert_close(fftn(a.real.astype(ctype[dtype])), fftn(a.real), eps)
    tmp = a.copy()
    assert_(ifftn(fftn(tmp, out=tmp, inorm=inorm), out=tmp, inorm=2-inorm)
            is tmp)
    _assert_close(tmp, a, eps)
    tmp = fftn(a.real, inorm=inorm)
    ref = tmp.real-tmp.imag
    _assert_close(fft.separable_fht(a.real,inorm=inorm), ref, eps)
    ref = tmp.real+tmp.imag
    _assert_close(fft.separable_hartley(a.real,inorm=inorm), ref, eps)


@pmp("shp", shapes)
@pmp("nthreads", (0, 1, 2))
@pmp("inorm", [0, 1, 2])
def test_fftn(shp, nthreads, inorm):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    _assert_close(a, ifftn(fftn(a, nthreads=nthreads, inorm=inorm),
                           nthreads=nthreads, inorm=2-inorm), 1e-15)
    a = a.astype(np.complex64)
    _assert_close(a, ifftn(fftn(a, nthreads=nthreads, inorm=inorm),
                           nthreads=nthreads, inorm=2-inorm), 5e-7)


@pmp("shp", shapes2D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
@pmp("inorm", [0, 1, 2])
def test_fftn2D(shp, axes, inorm):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    _assert_close(a, ifftn(fftn(a, axes=axes, inorm=inorm),
                           axes=axes, inorm=2-inorm), 1e-15)
    a = a.astype(np.complex64)
    _assert_close(a, ifftn(fftn(a, axes=axes, inorm=inorm),
                           axes=axes, inorm=2-inorm), 5e-7)


@pmp("shp", shapes)
def test_rfftn(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    tmp1 = rfftn(a)
    tmp2 = fftn(a)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    _assert_close(tmp1, tmp2[part], 1e-15)
    a = a.astype(np.float32)
    tmp1 = rfftn(a)
    tmp2 = fftn(a)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    _assert_close(tmp1, tmp2[part], 5e-7)


@pmp("shp", shapes2D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
def test_rfftn2D(shp, axes):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    tmp1 = rfftn(a, axes=axes)
    tmp2 = fftn(a, axes=axes)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    _assert_close(tmp1, tmp2[part], 1e-15)
    a = a.astype(np.float32)
    tmp1 = rfftn(a, axes=axes)
    tmp2 = fftn(a, axes=axes)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    _assert_close(tmp1, tmp2[part], 5e-7)


@pmp("shp", shapes)
def test_identity(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    _assert_close(ifftn(fftn(a), inorm=2), a, 1.5e-15)
    _assert_close(ifftn(fftn(a.real), inorm=2), a.real, 1.5e-15)
    _assert_close(fftn(ifftn(a.real), inorm=2), a.real, 1.5e-15)
    tmp = a.copy()
    assert_(ifftn(fftn(tmp, out=tmp), inorm=2, out=tmp) is tmp)
    _assert_close(tmp, a, 1.5e-15)
    a = a.astype(np.complex64)
    _assert_close(ifftn(fftn(a), inorm=2), a, 6e-7)


@pmp("shp", shapes)
def test_identity_r(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    b = a.astype(np.float32)
    for ax in range(a.ndim):
        n = a.shape[ax]
        _assert_close(irfftn(rfftn(a, (ax,)), (ax,), lastsize=n, inorm=2),
                      a, 1e-15)
        _assert_close(irfftn(rfftn(b, (ax,)), (ax,), lastsize=n, inorm=2),
                      b, 5e-7)


@pmp("shp", shapes)
def test_identity_r2(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    a = rfftn(irfftn(a))
    _assert_close(rfftn(irfftn(a), inorm=2), a, 1e-15)


@pmp("shp", shapes2D+shapes3D)
@pmp("nthreads", [1,2,11])
def test_genuine_hartley(shp, nthreads):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    ref = fftn(a.astype(np.complex128))
    v1 = fft.genuine_fht(a, nthreads=nthreads)
    _assert_close(v1, ref.real-ref.imag, 1e-15)
    v1 = fft.genuine_hartley(a, nthreads=nthreads)
    _assert_close(v1, ref.real+ref.imag, 1e-15)


@pmp("shp", shapes)
def test_hartley_identity(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    v1 = fft.separable_fht(fft.separable_fht(a))/a.size
    _assert_close(a, v1, 1e-15)
    v1 = fft.separable_hartley(fft.separable_hartley(a))/a.size
    _assert_close(a, v1, 1e-15)


@pmp("shp", shapes)
@pmp("nthreads", [1,2,11])
def test_genuine_hartley_identity(shp, nthreads):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    v1 = fft.genuine_fht(fft.genuine_fht(a), nthreads=nthreads)/a.size
    _assert_close(a, v1, 1e-15)
    v1 = fft.genuine_hartley(fft.genuine_hartley(a), nthreads=nthreads)/a.size
    _assert_close(a, v1, 1e-15)
    v1 = a.copy()
    assert_(fft.genuine_fht(
        fft.genuine_fht(v1, out=v1), inorm=2, out=v1, nthreads=nthreads) is v1)
    _assert_close(a, v1, 1e-15)
    v1 = a.copy()
    assert_(fft.genuine_hartley(
        fft.genuine_hartley(v1, out=v1), inorm=2, out=v1, nthreads=nthreads) is v1)
    _assert_close(a, v1, 1e-15)


@pmp("shp", shapes2D+shapes3D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
def test_genuine_hartley_2D(shp, axes):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    _assert_close(fft.genuine_fht(fft.genuine_fht(
        a, axes=axes), axes=axes, inorm=2), a, 1e-15)
    _assert_close(fft.genuine_hartley(fft.genuine_hartley(
        a, axes=axes), axes=axes, inorm=2), a, 1e-15)


def test_hartley_multiD():
    rng = np.random.default_rng(42)
    for i in range(1000):
        ndim = rng.integers(1, 6)
        axlen = int(10)
        shape = rng.integers(1, axlen+1, ndim)
        axes = np.arange(ndim)
        rng.shuffle(axes)
        nax = rng.integers(1, ndim+1)
        axes = axes[:nax]
        a = rng.random(shape)-0.5
        nthreads=rng.integers(1, 8)
        c = fft.c2c(a.astype(np.complex128),axes=axes, nthreads=nthreads)
        b = fft.genuine_fht(a,axes=axes, nthreads=nthreads)
        _assert_close(b, c.real-c.imag, 1e-10)
        b = fft.genuine_hartley(a,axes=axes, nthreads=nthreads)
        _assert_close(b, c.real+c.imag, 1e-10)


@pmp("len", len1D)
@pmp("inorm", [0, 1])  # inorm==2 not needed, tested via inverse
@pmp("type", [1, 2, 3, 4])
@pmp("dtype", dtypes)
def testdcst1D(len, inorm, type, dtype):
    rng = np.random.default_rng(42)
    a = (rng.random(len)-0.5).astype(dtype)
    eps = tol[dtype]
    itp = (0, 1, 3, 2, 4)
    itype = itp[type]
    if type != 1 or len > 1:  # there are no length-1 type 1 DCTs
        _assert_close(a, fft.dct(fft.dct(a, inorm=inorm, type=type),
                      inorm=2-inorm, type=itype), eps)
    _assert_close(a, fft.dst(fft.dst(a, inorm=inorm, type=type), inorm=2-inorm,
                  type=itype), eps)


@pmp("len", (3, 4, 5, 6, 7, 8, 9, 10))
@pmp("dtype", dtypes)
def test_r2r_extra(len, dtype):
    rng = np.random.default_rng(42)
    a = (rng.random(len)-0.5).astype(dtype)
    eps = tol[dtype]
    ref = fft.c2c(a, forward=False)
    test = fft.r2r_fftpack(a, (0,), real2hermitian=True, forward=False)
    testc = hc2c_fftpack(test, ctype[dtype])
    _assert_close(ref, testc, eps)
    ref = fft.c2c(ref, forward=True)
    test = fft.r2r_fftpack(test, (0,), real2hermitian=False, forward=True)
    _assert_close(ref, test, eps)

    ref = fft.c2c(a, forward=True)
    test = fft.r2r_fftw(a, (0,), forward=True)
    testc = hc2c_fftw(test, ctype[dtype])
    _assert_close(ref, testc, eps)
    ref = fft.c2c(ref, forward=False)
    test = fft.r2r_fftw(test, (0,), forward=False)
    _assert_close(ref, test, eps)


def refconv(a, newlen, axis, k):
    try:
        import scipy.ndimage
        import scipy.signal
        import scipy.fft
    except:
        pytest.skip()
    k = scipy.fft.fftshift(k)
    tmp=scipy.ndimage.convolve1d(a,k,axis,mode='wrap')
    tmp=scipy.signal.resample(tmp,newlen,axis=axis)
    return tmp


@pmp("L1", tuple(range(3,30)))
@pmp("L2", tuple(range(3,30)))
@pmp("dtype", (np.float32, np.float64, np.complex64, np.complex128))
def test_conv(L1,L2,dtype):
    if issubclass(dtype, np.complexfloating):
        a = (np.random.random(L1) + 1j*np.random.random(L1)).astype(dtype)
        k = (np.random.random(L1) + 1j*np.random.random(L1)).astype(dtype)
    else:
        a = np.random.random(L1).astype(dtype)
        k = np.random.random(L1).astype(dtype)
    b = np.zeros(L2).astype(dtype)
    x = fft.convolve_axis(a,b,0,k)
    x2 = refconv(a,L2,0,k)
    eps = tol[x2.real.dtype.type]
    _assert_close(x, x2, eps)


@pmp("L1", tuple(range(3,10)))
@pmp("L2", tuple(range(3,10)))
@pmp("dtype", (np.float32, np.float64, np.complex64, np.complex128))
def test_conv2(L1,L2,dtype):
    shp = (5,L1,20)
    shp2 = (5,L2,20)
    if issubclass(dtype, np.complexfloating):
        a = (np.random.random(shp) + 1j*np.random.random(shp)).astype(dtype)
        k = (np.random.random(L1) + 1j*np.random.random(L1)).astype(dtype)
    else:
        a = np.random.random(shp).astype(dtype)
        k = np.random.random(L1).astype(dtype)
    b = np.zeros(shp2).astype(dtype)
    x = fft.convolve_axis(a,b,1,k)
    x2 = refconv(a,L2,1,k)
    eps = tol[x2.real.dtype.type]
    _assert_close(x, x2, eps)


def test_multi_iter_bug():
    a=np.zeros((128000,),dtype=np.complex128)
    # this used to raise an exception
    fft.c2c(a[::2],axes=(0,),nthreads=8)


@pmp("shp", shapes2D+shapes3D)
def test_inplace_r2c_c2r(shp):
    shpc = list(shp)
    shpc[-1] = shp[-1]//2 + 1
    buf = np.empty(shpc, dtype=np.complex128)
    slc_r = tuple(slice(0,lim) for lim in shp)
    bufr = buf.view(np.float64)[slc_r]
    bufr[()] = np.random.random(shp)
    ref = bufr.copy()
    ducc0.fft.r2c(bufr, out=buf)
    ducc0.fft.c2r(buf, out=bufr,inorm=2, lastsize=shp[-1], forward=False)
    _assert_close(ref, bufr, 1e-12)
