/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 */

/*! \file sht_dispatch_target.cc
 *  Target-private wrappers for the SHT inner loops.
 */

#include "ducc0/sht/sht_dispatch.h"
#include "ducc0/infra/simd.h"
#include "ducc0/math/constants.h"
#include "ducc0/sht/sht_inner_loop.h"

namespace ducc0 {

namespace detail_sht {

#if DUCC0_DISPATCH_SHT_TARGET == 0
namespace target_sht_sse2 {
#elif DUCC0_DISPATCH_SHT_TARGET == 1
namespace target_sht_avx {
#elif DUCC0_DISPATCH_SHT_TARGET == 2
namespace target_sht_avx512 {
#else
#error "unsupported DUCC0_DISPATCH_SHT_TARGET"
#endif

DUCC0_NOINLINE void a2m_float(SHT_mode mode,
  const vmav<std::complex<double>,2> &almtmp,
  const vmav<std::complex<float>,3> &phase,
  const std::vector<ringdata> &rdata, Ylmgen &gen, size_t mi)
  { detail_sht_inner_loop::inner_loop_a2m<float>(mode, almtmp, phase, rdata, gen, mi); }

DUCC0_NOINLINE void a2m_double(SHT_mode mode,
  const vmav<std::complex<double>,2> &almtmp,
  const vmav<std::complex<double>,3> &phase,
  const std::vector<ringdata> &rdata, Ylmgen &gen, size_t mi)
  { detail_sht_inner_loop::inner_loop_a2m<double>(mode, almtmp, phase, rdata, gen, mi); }

DUCC0_NOINLINE void m2a_float(SHT_mode mode,
  const vmav<std::complex<double>,2> &almtmp,
  const cmav<std::complex<float>,3> &phase,
  const std::vector<ringdata> &rdata, Ylmgen &gen, size_t mi)
  { detail_sht_inner_loop::inner_loop_m2a<float>(mode, almtmp, phase, rdata, gen, mi); }

DUCC0_NOINLINE void m2a_double(SHT_mode mode,
  const vmav<std::complex<double>,2> &almtmp,
  const cmav<std::complex<double>,3> &phase,
  const std::vector<ringdata> &rdata, Ylmgen &gen, size_t mi)
  { detail_sht_inner_loop::inner_loop_m2a<double>(mode, almtmp, phase, rdata, gen, mi); }

} // target namespace

} // namespace detail_sht

} // namespace ducc0
