/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

#ifndef DUCC0_FFT_KERNELS_H
#define DUCC0_FFT_KERNELS_H

#include <complex>
#include <cstddef>
#include <vector>

#include "ducc0/infra/mav.h"

namespace ducc0 {

namespace detail_fft_dispatch {

template<typename T> struct FftKernels
  {
  using Shape = std::vector<std::size_t>;
  using C2C = void (*)(const cfmav<std::complex<T>> &,
    const vfmav<std::complex<T>> &, const Shape &, bool, T, std::size_t);
  using Dct = void (*)(const cfmav<T> &, const vfmav<T> &,
    const Shape &, int, T, bool, std::size_t);
  using Dst = Dct;
  using R2cAxis = void (*)(const cfmav<T> &,
    const vfmav<std::complex<T>> &, std::size_t, bool, T, std::size_t);
  using R2cAxes = void (*)(const cfmav<T> &,
    const vfmav<std::complex<T>> &, const Shape &, bool, T, std::size_t);
  using C2rAxis = void (*)(const cfmav<std::complex<T>> &,
    const vfmav<T> &, std::size_t, bool, T, std::size_t);
  using C2rAxes = void (*)(const cfmav<std::complex<T>> &,
    const vfmav<T> &, const Shape &, bool, T, std::size_t);
  using C2rMut = void (*)(const vfmav<std::complex<T>> &,
    const vfmav<T> &, const Shape &, bool, T, std::size_t);
  using R2rFftpack = void (*)(const cfmav<T> &, const vfmav<T> &,
    const Shape &, bool, bool, T, std::size_t);
  using R2rFftw = void (*)(const cfmav<T> &, const vfmav<T> &,
    const Shape &, bool, T, std::size_t);
  using R2r = void (*)(const cfmav<T> &, const vfmav<T> &,
    const Shape &, T, std::size_t);
  using ConvolveReal = void (*)(const cfmav<T> &, const vfmav<T> &,
    std::size_t, const cmav<T,1> &, std::size_t);
  using ConvolveComplex = void (*)(const cfmav<std::complex<T>> &,
    const vfmav<std::complex<T>> &, std::size_t,
    const cmav<std::complex<T>,1> &, std::size_t);
  C2C c2c;
  Dct dct;
  Dst dst;
  R2cAxis r2c_axis;
  R2cAxes r2c_axes;
  C2rAxis c2r_axis;
  C2rAxes c2r_axes;
  C2rMut c2r_mut;
  R2rFftpack r2r_fftpack;
  R2rFftw r2r_fftw;
  R2r r2r_separable_hartley;
  R2r r2r_separable_fht;
  R2r r2r_genuine_hartley;
  R2r r2r_genuine_fht;
  ConvolveReal convolve_real;
  ConvolveComplex convolve_complex;
  };

#define DUCC0_DECLARE_FFT_PROFILE(suffix) \
  extern const FftKernels<float> fft_kernels_##suffix##_f32; \
  extern const FftKernels<double> fft_kernels_##suffix##_f64

DUCC0_DECLARE_FFT_PROFILE(x86_64);
DUCC0_DECLARE_FFT_PROFILE(x86_64_v2);
DUCC0_DECLARE_FFT_PROFILE(x86_64_v3);
DUCC0_DECLARE_FFT_PROFILE(x86_64_v4);

#undef DUCC0_DECLARE_FFT_PROFILE

} // namespace detail_fft_dispatch
} // namespace ducc0

#endif
