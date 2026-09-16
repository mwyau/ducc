/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 */

/*! \file cpu_dispatch.cc
 *  Runtime CPU target detection and selection.
 */

#include "ducc0/infra/cpu_dispatch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#define DUCC0_CPU_DISPATCH_X86_64 1
#else
#define DUCC0_CPU_DISPATCH_X86_64 0
#endif

#if DUCC0_CPU_DISPATCH_X86_64 && defined(_MSC_VER)
#include <intrin.h>
#elif DUCC0_CPU_DISPATCH_X86_64 && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

namespace ducc0 {

namespace cpu_dispatch {

namespace {

#if DUCC0_CPU_DISPATCH_X86_64
constexpr bool compiled_for_x86_64 = true;
#else
constexpr bool compiled_for_x86_64 = false;
#endif

// Raw CPUID details are intentionally private. They describe compiler
// profiles, not a public CPU capability API.
struct RawCpuState
  {
  bool avx = false;
  bool xsave = false;
  bool osxsave = false;
  bool avx2 = false;
  bool fma = false;
  bool f16c = false;
  bool avx512f = false;
  };

void cpuid(unsigned int leaf, unsigned int subleaf,
  unsigned int &eax, unsigned int &ebx, unsigned int &ecx,
  unsigned int &edx) noexcept
  {
#if DUCC0_CPU_DISPATCH_X86_64 && defined(_MSC_VER)
  int values[4];
  __cpuidex(values, int(leaf), int(subleaf));
  eax = unsigned(values[0]);
  ebx = unsigned(values[1]);
  ecx = unsigned(values[2]);
  edx = unsigned(values[3]);
#elif DUCC0_CPU_DISPATCH_X86_64 && (defined(__GNUC__) || defined(__clang__))
  unsigned int values[4];
  __cpuid_count(leaf, subleaf, values[0], values[1], values[2], values[3]);
  eax = values[0];
  ebx = values[1];
  ecx = values[2];
  edx = values[3];
#else
  (void)leaf;
  (void)subleaf;
  eax = ebx = ecx = edx = 0;
#endif
  }

unsigned int max_cpuid_leaf() noexcept
  {
#if DUCC0_CPU_DISPATCH_X86_64 && defined(_MSC_VER)
  int values[4];
  __cpuid(values, 0);
  return unsigned(values[0]);
#elif DUCC0_CPU_DISPATCH_X86_64 && (defined(__GNUC__) || defined(__clang__))
  return __get_cpuid_max(0, nullptr);
#else
  return 0;
#endif
  }

uint64_t read_xcr0() noexcept
  {
#if DUCC0_CPU_DISPATCH_X86_64 && defined(_MSC_VER)
  return uint64_t(_xgetbv(0));
#elif DUCC0_CPU_DISPATCH_X86_64 && (defined(__GNUC__) || defined(__clang__))
  unsigned int eax, edx;
  __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (uint64_t(edx)<<32) | eax;
#else
  return 0;
#endif
  }

bool os_avx_state_enabled(const RawCpuState &raw, uint64_t &xcr0) noexcept
  {
  if (!raw.avx || !raw.xsave || !raw.osxsave)
    return false;

  // XGETBV is executed only after CPUID has advertised the architectural
  // XSTATE interface and the OS has enabled it.
  xcr0 = read_xcr0();
  return (xcr0 & 0x6) == 0x6;  // XMM and YMM state
  }

bool os_avx512_state_enabled(uint64_t xcr0) noexcept
  {
  // XMM, YMM, opmask, ZMM_Hi256, and Hi16_ZMM state.
  return (xcr0 & 0xe6) == 0xe6;
  }

bool compiler_avx512_profile_usable(const RawCpuState &raw) noexcept
  {
  // The target is compiled with -mavx512f. GCC's profile permits AVX2 and
  // AVX512F; it does not require the optional FMA or F16C instructions.
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
  // LLVM's X86 target model requires AVX2, FMA, F16C, and AVX512F here.
  return raw.avx2 && raw.fma && raw.f16c && raw.avx512f;
#elif defined(_MSC_VER)
  // /arch:AVX512's minimum profile is kept private to this implementation.
  // The selected wrapper uses AVX, AVX2/FMA, and AVX512F instructions.
  return raw.avx2 && raw.fma && raw.avx512f;
#elif defined(__GNUC__)
  (void)raw.fma;
  (void)raw.f16c;
  return raw.avx2 && raw.avx512f;
#else
  return false;
#endif
  }

RawCpuState detect_raw_cpu_state() noexcept
  {
  RawCpuState result;
  if (!compiled_for_x86_64)
    return result;

  const unsigned int max_leaf = max_cpuid_leaf();
  if (max_leaf < 1)
    return result;

  unsigned int eax, ebx, ecx, edx;
  cpuid(1, 0, eax, ebx, ecx, edx);
  result.avx = (ecx & (1u<<28)) != 0;
  result.xsave = (ecx & (1u<<26)) != 0;
  result.osxsave = (ecx & (1u<<27)) != 0;
  result.fma = (ecx & (1u<<12)) != 0;
  result.f16c = (ecx & (1u<<29)) != 0;

  if (max_leaf >= 7)
    {
    cpuid(7, 0, eax, ebx, ecx, edx);
    result.avx2 = (ebx & (1u<<5)) != 0;
    result.avx512f = (ebx & (1u<<16)) != 0;
    }
  return result;
  }

CpuTargets detect_cpu_targets_impl() noexcept
  {
  CpuTargets result;
  result.x86 = compiled_for_x86_64;
  if (!compiled_for_x86_64)
    return result;

  const auto raw = detect_raw_cpu_state();
  uint64_t xcr0 = 0;
  result.avx = os_avx_state_enabled(raw, xcr0);
  result.avx512 = result.avx && os_avx512_state_enabled(xcr0) &&
    compiler_avx512_profile_usable(raw);
  return result;
  }

constexpr TargetDescriptor sse2_target = {TargetId::sse2, "sse2"};

#if defined(DUCC0_CPU_DISPATCH)
constexpr TargetDescriptor compiled_profile_list[] =
  {
  sse2_target,
#if defined(DUCC0_DISPATCH_HAS_AVX)
  {TargetId::avx, "avx"},
#endif
#if defined(DUCC0_DISPATCH_HAS_AVX512)
  {TargetId::avx512, "avx512"},
#endif
  };
#else
constexpr TargetDescriptor compiled_profile_list[] = {sse2_target};
#endif

RuntimeState make_runtime_state() noexcept
  {
  RuntimeState result;
  result.targets = detect_cpu_targets_impl();
#if defined(DUCC0_CPU_DISPATCH)
  const char *value = std::getenv("DUCC0_CPU_MAX");
  if (value && *value)
    {
    result.max_target = parse_cpu_max(value);
    if (result.max_target == CpuMax::invalid)
      {
      std::fprintf(stderr,
        "ducc0: ignoring invalid DUCC0_CPU_MAX='%s' "
        "(expected sse2, avx, or avx512)\n", value);
      result.max_target = CpuMax::none;
      }
    }
#endif
  return result;
  }

} // namespace

CpuTargets detect_cpu_targets() noexcept
  { return detect_cpu_targets_impl(); }

const RuntimeState &runtime_state() noexcept
  {
  static const RuntimeState result = make_runtime_state();
  return result;
  }

CpuMax parse_cpu_max(const char *value) noexcept
  {
  if (!value || !*value)
    return CpuMax::none;
  if (std::strcmp(value, "sse2")==0)
    return CpuMax::sse2;
  if (std::strcmp(value, "avx")==0)
    return CpuMax::avx;
  if (std::strcmp(value, "avx512")==0)
    return CpuMax::avx512;
  return CpuMax::invalid;
  }

const char *cpu_max_name(CpuMax value) noexcept
  {
  switch (value)
    {
    case CpuMax::none: return "none";
    case CpuMax::sse2: return "sse2";
    case CpuMax::avx: return "avx";
    case CpuMax::avx512: return "avx512";
    case CpuMax::invalid: return "invalid";
    }
  return "invalid";
  }

const char *target_name(TargetId value) noexcept
  {
  switch (value)
    {
    case TargetId::sse2: return "sse2";
    case TargetId::avx: return "avx";
    case TargetId::avx512: return "avx512";
    }
  return "unknown";
  }

bool target_is_allowed(TargetId target, CpuMax max_target) noexcept
  {
  switch (max_target)
    {
    case CpuMax::none:
    case CpuMax::invalid:
      return true;
    case CpuMax::sse2:
      return target==TargetId::sse2;
    case CpuMax::avx:
      return target==TargetId::sse2 || target==TargetId::avx;
    case CpuMax::avx512:
      return target==TargetId::sse2 || target==TargetId::avx ||
        target==TargetId::avx512;
    }
  return false;
  }

bool target_is_usable(TargetId target, const CpuTargets &targets) noexcept
  {
  switch (target)
    {
    case TargetId::sse2: return targets.x86;
    case TargetId::avx: return targets.avx;
    case TargetId::avx512: return targets.avx512;
    }
  return false;
  }

TargetId select_target(const CpuTargets &targets,
  const TargetDescriptor *compiled, size_t ncompiled,
  const TargetId *preference, size_t npreference,
  CpuMax max_target) noexcept
  {
  for (size_t ip=0; ip<npreference; ++ip)
    for (size_t it=0; it<ncompiled; ++it)
      if (compiled[it].id==preference[ip] &&
          target_is_allowed(compiled[it].id, max_target) &&
          target_is_usable(compiled[it].id, targets))
        return compiled[it].id;

  // Dispatch is configured only for x86-64, whose lowest target is SSE2.
  return TargetId::sse2;
  }

const TargetDescriptor *compiled_targets(size_t &count) noexcept
  {
  count = sizeof(compiled_profile_list)/sizeof(compiled_profile_list[0]);
  return compiled_profile_list;
  }

const char *mode_name() noexcept
  {
#if defined(DUCC0_CPU_DISPATCH)
  return "dispatch";
#else
  return "compile-time";
#endif
  }

} // namespace cpu_dispatch

} // namespace ducc0

#undef DUCC0_CPU_DISPATCH_X86_64
