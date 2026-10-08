#!/usr/bin/env python3
"""Emulated, fixed-width SVE smoke test for a single-target DUCC build.

Run with qemu-aarch64 -cpu max,sveN=on,sve-default-vector-length=N/8.
Never benchmark performance under this emulator.
"""
import argparse
import ctypes
import math

import numpy as np

import ducc0


def verify_vector_length(isa, bits):
    assert ducc0.misc.cpu_info()["architecture"] == "aarch64"
    info = ducc0.misc.cpu_info()
    assert "sve" in info["features"], info
    # QEMU user-mode max exposes SVE2 even for an SVE1-targeted binary.
    # The compile-time ISA is checked separately by the C++ probe.
    if isa == "sve2":
        assert "sve2" in info["features"], info

    # Linux PR_SVE_GET_VL returns a vector length in bytes in its low 16 bits.
    libc = ctypes.CDLL(None, use_errno=True)
    vl = libc.prctl(51, 0, 0, 0, 0)
    if vl == -1:
        raise OSError(ctypes.get_errno(), "PR_SVE_GET_VL failed")
    observed = (vl & 0xFFFF) * 8
    assert observed == bits, f"QEMU SVE vector length {observed} != {bits}"
    print(f"ISA={isa} SVE vector length={observed} bits; CPU={info}", flush=True)


def verify_fft():
    rng = np.random.default_rng(42)
    # Batched 1D transforms exercise SIMD-across-transforms, while a single
    # transform exercises DUCC's separate scalar/within-transform path.
    for dtype in (np.complex64, np.complex128):
        for shape in ((127,), (6, 128), (6, 127)):
            print(f"START FFT complex dtype={dtype.__name__} shape={shape}", flush=True)
            x = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)
            y = ducc0.fft.c2c(x, axes=(-1,), forward=True, nthreads=1)
            ref = np.fft.fft(x, axis=-1)
            np.testing.assert_allclose(y, ref,
                rtol=4e-5 if dtype == np.complex64 else 2e-12,
                atol=4e-5 if dtype == np.complex64 else 2e-12)
    for dtype in (np.float32, np.float64):
        print(f"START FFT real dtype={dtype.__name__} shape=(5, 96)", flush=True)
        x = rng.normal(size=(5, 96)).astype(dtype)
        spectrum = ducc0.fft.r2c(x, axes=(-1,), forward=True, nthreads=1)
        recovered = ducc0.fft.c2r(
            spectrum, axes=(-1,), lastsize=x.shape[-1],
            forward=False, inorm=2, nthreads=1)
        np.testing.assert_allclose(
            recovered, x,
            rtol=4e-5 if dtype == np.float32 else 2e-12,
            atol=4e-5 if dtype == np.float32 else 2e-12)
    print("PASS FFT complex/real, odd/even, batched and both precisions", flush=True)


def verify_sht():
    lmax = 11
    rng = np.random.default_rng(17)
    nalm = (lmax + 1)*(lmax + 2)//2
    alm = (rng.normal(size=(1, nalm)) +
           1j*rng.normal(size=(1, nalm))).astype(np.complex128)
    alm[:, :lmax + 1].imag = 0
    kwargs = dict(lmax=lmax, mmax=lmax, spin=0,
                  geometry="CC", nthreads=1)
    sky = ducc0.sht.synthesis_2d(alm=alm, ntheta=lmax+2,
                                 nphi=2*lmax+2, **kwargs)
    reconstructed = ducc0.sht.analysis_2d(map=sky, **kwargs)
    np.testing.assert_allclose(reconstructed, alm, rtol=1e-11, atol=1e-11)
    print("PASS SHT synthesis/analysis", flush=True)


def verify_nufft():
    rng = np.random.default_rng(22)
    n = 48
    npoints = 12
    coord = rng.uniform(-math.pi, math.pi, size=(npoints, 1))
    vals = rng.normal(size=npoints) + 1j*rng.normal(size=npoints)
    out = np.empty((n,), dtype=np.complex128)
    actual = ducc0.nufft.nu2u(
        points=vals, coord=coord, out=out, forward=True,
        epsilon=1e-9, nthreads=1, periodicity=2*math.pi,
        fft_order=False)
    wave_numbers = np.arange(n) - n//2
    reference = np.sum(vals[:, None] *
        np.exp(-1j*coord[:, 0, None]*wave_numbers[None, :]), axis=0)
    np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-7)
    print("PASS NUFFT nonuniform-to-uniform against direct DFT", flush=True)


def verify_wgridder():
    rng = np.random.default_rng(31)
    nrow, nchan, nx, ny = 5, 2, 16, 16
    pixsize = math.pi / (180*60*nx)
    freq = 1e9 + np.arange(nchan)*1e8
    uvw = (rng.random((nrow, 3))-0.5)/(pixsize*freq[0]/299792458.)
    vis = (rng.normal(size=(nrow, nchan)) +
           1j*rng.normal(size=(nrow, nchan))).astype(np.complex128)
    dirty = rng.normal(size=(nx, ny))
    kwargs = dict(uvw=uvw, freq=freq, pixsize_x=pixsize,
                  pixsize_y=pixsize, epsilon=1e-6,
                  do_wgridding=False, nthreads=1, verbosity=0)
    image = ducc0.wgridder.vis2dirty(
        vis=vis, npix_x=nx, npix_y=ny, **kwargs)
    outvis = ducc0.wgridder.dirty2vis(dirty=dirty, **kwargs)
    left = np.vdot(vis, outvis).real
    right = np.vdot(image, dirty).real
    np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-7)
    print("PASS wgridder forward/adjoint", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isa", choices=("sve", "sve2"), required=True)
    parser.add_argument("--bits", type=int, choices=(256, 512), required=True)
    args = parser.parse_args()
    verify_vector_length(args.isa, args.bits)
    verify_fft()
    print("START SHT", flush=True)
    verify_sht()
    print("START NUFFT", flush=True)
    verify_nufft()
    print("START wgridder", flush=True)
    verify_wgridder()
    print("PASS fixed-width SVE QEMU smoke suite", flush=True)


if __name__ == "__main__":
    main()
