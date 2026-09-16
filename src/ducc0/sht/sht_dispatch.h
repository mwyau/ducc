/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 */

/*! \file sht_dispatch.h
 *  Neutral ABI for the target-private SHT inner loops.
 */

#ifndef DUCC0_SHT_DISPATCH_H
#define DUCC0_SHT_DISPATCH_H

#include <complex>
#include <cstddef>
#include <vector>

#include "ducc0/infra/cpu_dispatch.h"
#include "ducc0/infra/mav.h"
#include "ducc0/sht/sht.h"

namespace ducc0 {

namespace detail_sht {

namespace detail_sht_inner_loop {
struct ringdata;
class Ylmgen;
}

using dispatch_ringdata = detail_sht_inner_loop::ringdata;
using dispatch_ylmgen = detail_sht_inner_loop::Ylmgen;

// Types crossing this boundary must stay independent of target SIMD types and
// vector width; the target-private implementations receive only neutral state.

using ShtInnerA2MFloat = void (*)(SHT_mode,
  const vmav<std::complex<double>,2> &,
  const vmav<std::complex<float>,3> &,
  const std::vector<dispatch_ringdata> &, dispatch_ylmgen &, size_t);
using ShtInnerA2MDouble = void (*)(SHT_mode,
  const vmav<std::complex<double>,2> &,
  const vmav<std::complex<double>,3> &,
  const std::vector<dispatch_ringdata> &, dispatch_ylmgen &, size_t);
using ShtInnerM2AFloat = void (*)(SHT_mode,
  const vmav<std::complex<double>,2> &,
  const cmav<std::complex<float>,3> &,
  const std::vector<dispatch_ringdata> &, dispatch_ylmgen &, size_t);
using ShtInnerM2ADouble = void (*)(SHT_mode,
  const vmav<std::complex<double>,2> &,
  const cmav<std::complex<double>,3> &,
  const std::vector<dispatch_ringdata> &, dispatch_ylmgen &, size_t);

struct ShtKernels
  {
  cpu_dispatch::TargetId target;
  ShtInnerA2MFloat a2m_float;
  ShtInnerA2MDouble a2m_double;
  ShtInnerM2AFloat m2a_float;
  ShtInnerM2ADouble m2a_double;
  };

const ShtKernels &sht_kernels();
const char *selected_sht_target();

} // namespace detail_sht

} // namespace ducc0

#endif
