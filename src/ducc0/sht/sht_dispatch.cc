/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 */

/*! \file sht_dispatch.cc
 *  Lazy runtime selection of target-private SHT inner loops.
 */

#include "ducc0/sht/sht_dispatch.h"

namespace ducc0 {

namespace detail_sht {

namespace target_sht_sse2 {
void a2m_float(SHT_mode, const vmav<std::complex<double>,2> &,
  const vmav<std::complex<float>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void a2m_double(SHT_mode, const vmav<std::complex<double>,2> &,
  const vmav<std::complex<double>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void m2a_float(SHT_mode, const vmav<std::complex<double>,2> &,
  const cmav<std::complex<float>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void m2a_double(SHT_mode, const vmav<std::complex<double>,2> &,
  const cmav<std::complex<double>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
}

#if defined(DUCC0_DISPATCH_HAS_AVX)
namespace target_sht_avx {
void a2m_float(SHT_mode, const vmav<std::complex<double>,2> &,
  const vmav<std::complex<float>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void a2m_double(SHT_mode, const vmav<std::complex<double>,2> &,
  const vmav<std::complex<double>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void m2a_float(SHT_mode, const vmav<std::complex<double>,2> &,
  const cmav<std::complex<float>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void m2a_double(SHT_mode, const vmav<std::complex<double>,2> &,
  const cmav<std::complex<double>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
}
#endif

#if defined(DUCC0_DISPATCH_HAS_AVX512)
namespace target_sht_avx512 {
void a2m_float(SHT_mode, const vmav<std::complex<double>,2> &,
  const vmav<std::complex<float>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void a2m_double(SHT_mode, const vmav<std::complex<double>,2> &,
  const vmav<std::complex<double>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void m2a_float(SHT_mode, const vmav<std::complex<double>,2> &,
  const cmav<std::complex<float>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
void m2a_double(SHT_mode, const vmav<std::complex<double>,2> &,
  const cmav<std::complex<double>,3> &, const std::vector<dispatch_ringdata> &,
  dispatch_ylmgen &, size_t);
}
#endif

namespace {

ShtKernels choose_sht_kernels()
  {
  size_t ncompiled = 0;
  const auto *compiled = cpu_dispatch::compiled_targets(ncompiled);
  constexpr cpu_dispatch::TargetId preference[] =
    {cpu_dispatch::TargetId::avx512, cpu_dispatch::TargetId::avx,
     cpu_dispatch::TargetId::sse2};
  const auto &state = cpu_dispatch::runtime_state();
  const auto target = cpu_dispatch::select_target(state.targets,
    compiled, ncompiled, preference,
    sizeof(preference)/sizeof(preference[0]), state.max_target);

  switch (target)
    {
#if defined(DUCC0_DISPATCH_HAS_AVX512)
    case cpu_dispatch::TargetId::avx512:
      return {target, target_sht_avx512::a2m_float,
        target_sht_avx512::a2m_double, target_sht_avx512::m2a_float,
        target_sht_avx512::m2a_double};
#endif
#if defined(DUCC0_DISPATCH_HAS_AVX)
    case cpu_dispatch::TargetId::avx:
      return {target, target_sht_avx::a2m_float, target_sht_avx::a2m_double,
        target_sht_avx::m2a_float, target_sht_avx::m2a_double};
#endif
    case cpu_dispatch::TargetId::sse2:
      return {cpu_dispatch::TargetId::sse2,
        target_sht_sse2::a2m_float, target_sht_sse2::a2m_double,
        target_sht_sse2::m2a_float, target_sht_sse2::m2a_double};
    }
  return {cpu_dispatch::TargetId::sse2,
    target_sht_sse2::a2m_float, target_sht_sse2::a2m_double,
    target_sht_sse2::m2a_float, target_sht_sse2::m2a_double};
  }

} // namespace

const ShtKernels &sht_kernels()
  {
  static const ShtKernels selected = choose_sht_kernels();
  return selected;
  }

const char *selected_sht_target()
  { return cpu_dispatch::target_name(sht_kernels().target); }

} // namespace detail_sht

} // namespace ducc0
