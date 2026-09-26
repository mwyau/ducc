/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

#ifndef DUCC0_FFT_DISPATCH_H
#define DUCC0_FFT_DISPATCH_H

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "ducc0/infra/cpu_dispatch.h"
#include "ducc0/fft/fft_kernels.h"

namespace ducc0 {
namespace detail_fft_dispatch {

using cpu_dispatch::TargetBinding;
using cpu_dispatch::Level;
using cpu_dispatch::ResolvedTarget;

inline constexpr std::size_t fft_optimized_count =
  0
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V2)
  + 1
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V3)
  + 1
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V4)
  + 1
#endif
  ;

template<typename T> inline const std::array<TargetBinding<FftKernels<T>>,
  fft_optimized_count> &fft_optimized();

template<> inline const std::array<TargetBinding<FftKernels<float>>,
  fft_optimized_count> &fft_optimized<float>()
  {
  static constexpr std::array<TargetBinding<FftKernels<float>>,
    fft_optimized_count> result = {{
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V2)
      {Level::x86_64_v2, &fft_kernels_x86_64_v2_f32},
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V3)
      {Level::x86_64_v3, &fft_kernels_x86_64_v3_f32},
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V4)
      {Level::x86_64_v4, &fft_kernels_x86_64_v4_f32},
#endif
      }};
  return result;
  }

template<> inline const std::array<TargetBinding<FftKernels<double>>,
  fft_optimized_count> &fft_optimized<double>()
  {
  static constexpr std::array<TargetBinding<FftKernels<double>>,
    fft_optimized_count> result = {{
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V2)
      {Level::x86_64_v2, &fft_kernels_x86_64_v2_f64},
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V3)
      {Level::x86_64_v3, &fft_kernels_x86_64_v3_f64},
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V4)
      {Level::x86_64_v4, &fft_kernels_x86_64_v4_f64},
#endif
      }};
  return result;
  }

inline constexpr std::array<Level,3> fft_preference = {
  Level::x86_64_v4, Level::x86_64_v3, Level::x86_64_v2};

template<typename T> inline const FftKernels<T> &fft_baseline();
template<> inline const FftKernels<float> &fft_baseline<float>()
  { return fft_kernels_x86_64_f32; }
template<> inline const FftKernels<double> &fft_baseline<double>()
  { return fft_kernels_x86_64_f64; }

template<typename T> inline const ResolvedTarget<FftKernels<T>> &selected_fft_target()
  {
  static const auto selected=cpu_dispatch::highest_compiled(
    cpu_dispatch::cpu_info(), fft_baseline<T>(),
    fft_optimized<T>(), fft_preference);
  return selected;
  }

inline std::vector<Level> fft_compiled_targets()
  {
  std::vector<Level> result = {Level::x86_64};
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V2)
  result.push_back(Level::x86_64_v2);
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V3)
  result.push_back(Level::x86_64_v3);
#endif
#if defined(DUCC0_DISPATCH_FFT_HAS_X86_64_V4)
  result.push_back(Level::x86_64_v4);
#endif
  return result;
  }

template<typename T> inline const FftKernels<T> &selected_fft_kernels()
  { return selected_fft_target<T>().implementation; }


} // namespace detail_fft_dispatch
} // namespace ducc0

#endif
