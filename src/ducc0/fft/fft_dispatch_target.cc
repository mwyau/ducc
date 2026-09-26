/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

#include "ducc0/fft/fft_target_profile.h"

#define DUCC0_FFT_PRIVATE_NAMESPACE 1
#include "ducc0/fft/fftnd_impl.h"
#include "ducc0/fft/fft_kernels.h"

namespace ducc0 {
namespace detail_fft_dispatch {
#define DUCC0_FFT_KERNEL_INITIALIZER(T) \
  { \
    &DUCC0_FFT_NAMESPACE::c2c<T>, \
    &DUCC0_FFT_NAMESPACE::dct<T>, \
    &DUCC0_FFT_NAMESPACE::dst<T>, \
    static_cast<typename FftKernels<T>::R2cAxis>(&DUCC0_FFT_NAMESPACE::r2c<T>), \
    static_cast<typename FftKernels<T>::R2cAxes>(&DUCC0_FFT_NAMESPACE::r2c<T>), \
    static_cast<typename FftKernels<T>::C2rAxis>(&DUCC0_FFT_NAMESPACE::c2r<T>), \
    static_cast<typename FftKernels<T>::C2rAxes>(&DUCC0_FFT_NAMESPACE::c2r<T>), \
    &DUCC0_FFT_NAMESPACE::c2r_mut<T>, \
    &DUCC0_FFT_NAMESPACE::r2r_fftpack<T>, \
    &DUCC0_FFT_NAMESPACE::r2r_fftw<T>, \
    &DUCC0_FFT_NAMESPACE::r2r_separable_hartley<T>, \
    &DUCC0_FFT_NAMESPACE::r2r_separable_fht<T>, \
    &DUCC0_FFT_NAMESPACE::r2r_genuine_hartley<T>, \
    &DUCC0_FFT_NAMESPACE::r2r_genuine_fht<T>, \
    static_cast<typename FftKernels<T>::ConvolveReal>( \
      &DUCC0_FFT_NAMESPACE::convolve_axis<T>), \
    static_cast<typename FftKernels<T>::ConvolveComplex>( \
      &DUCC0_FFT_NAMESPACE::convolve_axis<T>) \
  }

extern const FftKernels<float> DUCC0_FFT_KERNELS_F32=
  DUCC0_FFT_KERNEL_INITIALIZER(float);
extern const FftKernels<double> DUCC0_FFT_KERNELS_F64=
  DUCC0_FFT_KERNEL_INITIALIZER(double);

#undef DUCC0_FFT_KERNEL_INITIALIZER

} // namespace detail_fft_dispatch
} // namespace ducc0
