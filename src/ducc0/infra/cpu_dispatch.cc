/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

#include "ducc0/infra/cpu_dispatch.h"

#include <cstdio>
#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <cpuid.h>
#else
#error "DUCC x86-64 dispatch requires an x86-64 compiler target"
#endif

namespace ducc0 {
namespace cpu_dispatch {
namespace {

struct Cpuid { unsigned eax=0, ebx=0, ecx=0, edx=0; };

Level classify(std::uint32_t features, std::uint64_t xcr0,
  bool avx512_os_override=false) noexcept
  {
  const auto has=[&](Feature f) { return (features & feature_bit(f)) != 0; };
  const bool v2 = has(Feature::sse3) && has(Feature::ssse3)
    && has(Feature::sse41) && has(Feature::sse42)
    && has(Feature::popcnt) && has(Feature::cx16) && has(Feature::lahf);
  if (!v2) return Level::x86_64;

  const bool ymm_state=(xcr0&0x6)==0x6;
  const bool v3 = has(Feature::avx) && has(Feature::avx2) && has(Feature::fma)
    && has(Feature::bmi1) && has(Feature::bmi2) && has(Feature::lzcnt)
    && has(Feature::f16c) && has(Feature::movbe) && has(Feature::xsave)
    && has(Feature::osxsave) && ymm_state;
  if (!v3) return Level::x86_64_v2;

  const bool zmm_state=(xcr0&0xe6)==0xe6;
  const bool v4 = has(Feature::avx512f) && has(Feature::avx512cd)
    && has(Feature::avx512vl) && has(Feature::avx512bw)
    && has(Feature::avx512dq) && (zmm_state || avx512_os_override);
  return v4 ? Level::x86_64_v4 : Level::x86_64_v3;
  }

bool cpuid(unsigned leaf, unsigned subleaf, Cpuid &out) noexcept
  {
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  out={unsigned(regs[0]),unsigned(regs[1]),unsigned(regs[2]),unsigned(regs[3])};
  return true;
#else
  return __get_cpuid_count(leaf,subleaf,&out.eax,&out.ebx,&out.ecx,&out.edx)!=0;
#endif
  }

unsigned max_basic_leaf() noexcept
  {
#if defined(_MSC_VER)
  Cpuid regs;
  cpuid(0,0,regs);
  return regs.eax;
#else
  return __get_cpuid_max(0,nullptr);
#endif
  }

unsigned max_extended_leaf() noexcept
  {
#if defined(_MSC_VER)
  Cpuid regs;
  cpuid(0x80000000u,0,regs);
  return regs.eax;
#else
  return __get_cpuid_max(0x80000000u,nullptr);
#endif
  }

std::uint64_t xgetbv0() noexcept
  {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned eax, edx;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (std::uint64_t(edx)<<32) | eax;
#endif
  }

CpuInfo detect_cpu() noexcept
  {
  CpuInfo result;
  const auto maxbasic=max_basic_leaf();
  if (maxbasic<1) return result;

  Cpuid leaf1;
  cpuid(1,0,leaf1);
  Cpuid ext1;
  if (max_extended_leaf()>=0x80000001u)
    cpuid(0x80000001u,0,ext1);
  Cpuid leaf7;
  const bool have_leaf7=maxbasic>=7 && cpuid(7,0,leaf7);

  auto set=[&](Feature f, bool present)
    { if (present) result.features|=feature_bit(f); };
  set(Feature::sse3, (leaf1.ecx&(1u<<0))!=0);
  set(Feature::ssse3, (leaf1.ecx&(1u<<9))!=0);
  set(Feature::cx16, (leaf1.ecx&(1u<<13))!=0);
  set(Feature::sse41, (leaf1.ecx&(1u<<19))!=0);
  set(Feature::sse42, (leaf1.ecx&(1u<<20))!=0);
  const bool lzcnt=(ext1.ecx&(1u<<5))!=0;
  set(Feature::popcnt, (leaf1.ecx&(1u<<23))!=0);
  set(Feature::movbe, (leaf1.ecx&(1u<<22))!=0);
  set(Feature::fma, (leaf1.ecx&(1u<<12))!=0);
  set(Feature::xsave, (leaf1.ecx&(1u<<26))!=0);
  set(Feature::osxsave, (leaf1.ecx&(1u<<27))!=0);
  set(Feature::avx, (leaf1.ecx&(1u<<28))!=0);
  set(Feature::f16c, (leaf1.ecx&(1u<<29))!=0);
  set(Feature::lahf, (ext1.ecx&(1u<<0))!=0);
  set(Feature::lzcnt, lzcnt);
  if (have_leaf7)
    {
    set(Feature::bmi1, (leaf7.ebx&(1u<<3))!=0);
    set(Feature::avx2, (leaf7.ebx&(1u<<5))!=0);
    set(Feature::bmi2, (leaf7.ebx&(1u<<8))!=0);
    set(Feature::avx512f, (leaf7.ebx&(1u<<16))!=0);
    set(Feature::avx512dq, (leaf7.ebx&(1u<<17))!=0);
    set(Feature::avx512cd, (leaf7.ebx&(1u<<28))!=0);
    set(Feature::avx512bw, (leaf7.ebx&(1u<<30))!=0);
    set(Feature::avx512vl, (leaf7.ebx&(1u<<31))!=0);
    }

  // XGETBV is only valid after CPUID confirms XSAVE and OSXSAVE support.
  if (has_feature(result,Feature::xsave) && has_feature(result,Feature::osxsave)
      && has_feature(result,Feature::avx))
    result.xcr0=xgetbv0();

  bool avx512_os_override=false;
#if defined(__APPLE__) && defined(__x86_64__)
  // Match NumPy's current Darwin check when the kernel masks ZMM state from
  // XCR0 but advertises AVX-512 support through the x86-64 commpage.
  if ((result.xcr0&0x6)==0x6 && (result.xcr0&0xe6)!=0xe6)
    {
    constexpr std::uintptr_t commpage=0x00007fffffe00000ULL;
    const auto version=*reinterpret_cast<const std::uint16_t *>(commpage+0x01e);
    if (version>12)
      {
      const auto caps=*reinterpret_cast<const std::uint64_t *>(commpage+0x010);
      avx512_os_override=(caps&0x0000004000000000ULL)!=0;
      }
    }
#endif
  result.usable=classify(result.features,result.xcr0,avx512_os_override);

  Level cap;
  if (const char *env=std::getenv("DUCC0_CPU_MAX"))
    {
    if (parse_level(env,cap)) result.max_allowed=cap;
    else
      {
      result.max_allowed=Level::x86_64;
      std::fputs("DUCC0_CPU_MAX must be x86-64, x86-64-v2, x86-64-v3, or x86-64-v4; using x86-64.\n",stderr);
      }
    }
  return result;
  }

} // namespace

const CpuInfo &cpu_info() noexcept
  {
  static const CpuInfo result=detect_cpu();
  return result;
  }

const char *level_name(Level level) noexcept
  {
  switch (level)
    {
    case Level::x86_64: return "x86-64";
    case Level::x86_64_v2: return "x86-64-v2";
    case Level::x86_64_v3: return "x86-64-v3";
    case Level::x86_64_v4: return "x86-64-v4";
    }
  return "x86-64";
  }

const char *feature_name(Feature feature) noexcept
  {
  switch (feature)
    {
    case Feature::sse3: return "sse3";
    case Feature::ssse3: return "ssse3";
    case Feature::sse41: return "sse4.1";
    case Feature::sse42: return "sse4.2";
    case Feature::popcnt: return "popcnt";
    case Feature::cx16: return "cx16";
    case Feature::lahf: return "lahf";
    case Feature::avx: return "avx";
    case Feature::avx2: return "avx2";
    case Feature::fma: return "fma";
    case Feature::bmi1: return "bmi1";
    case Feature::bmi2: return "bmi2";
    case Feature::lzcnt: return "lzcnt";
    case Feature::f16c: return "f16c";
    case Feature::movbe: return "movbe";
    case Feature::xsave: return "xsave";
    case Feature::osxsave: return "osxsave";
    case Feature::avx512f: return "avx512f";
    case Feature::avx512cd: return "avx512cd";
    case Feature::avx512vl: return "avx512vl";
    case Feature::avx512bw: return "avx512bw";
    case Feature::avx512dq: return "avx512dq";
    }
  return "unknown";
  }

bool parse_level(const char *name, Level &result) noexcept
  {
  if (!name) return false;
  for (auto level : all_levels)
    {
    const char *a=name, *b=level_name(level);
    while (*a && *b && *a==*b) { ++a; ++b; }
    if (!*a && !*b) { result=level; return true; }
    }
  return false;
  }

} // namespace cpu_dispatch
} // namespace ducc0
