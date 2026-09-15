#include <cstdint>
#include <iomanip>
#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace {

struct cpuid_result
  {
  int eax, ebx, ecx, edx;
  };

cpuid_result cpuid(unsigned leaf, unsigned subleaf)
  {
#if defined(_MSC_VER)
  int r[4];
  __cpuidex(r, static_cast<int>(leaf), static_cast<int>(subleaf));
  return {r[0], r[1], r[2], r[3]};
#else
  unsigned a, b, c, d;
  __cpuid_count(leaf, subleaf, a, b, c, d);
  return {static_cast<int>(a), static_cast<int>(b), static_cast<int>(c),
    static_cast<int>(d)};
#endif
  }

std::uint64_t xcr0()
  {
#if defined(_MSC_VER)
  return static_cast<std::uint64_t>(_xgetbv(0));
#else
  unsigned eax, edx;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<std::uint64_t>(edx)<<32) | eax;
#endif
  }

}  // namespace

int main()
  {
  const auto leaf0 = cpuid(0, 0);
  const auto leaf1 = cpuid(1, 0);
  const auto max_leaf = static_cast<unsigned>(leaf0.eax);
  const bool osxsave = (leaf1.ecx & (1<<27)) != 0;
  const bool avx_cpu = (leaf1.ecx & (1<<28)) != 0;
  const std::uint64_t state = osxsave ? xcr0() : 0;
  const bool avx_os = (state & 0x6) == 0x6;

  bool avx2_cpu = false;
  bool avx512f_cpu = false;
  bool avx512dq_cpu = false;
  bool avx512cd_cpu = false;
  bool avx512bw_cpu = false;
  bool avx512vl_cpu = false;
  if (max_leaf >= 7)
    {
    const auto leaf7 = cpuid(7, 0);
    avx2_cpu = (leaf7.ebx & (1<<5)) != 0;
    avx512f_cpu = (leaf7.ebx & (1<<16)) != 0;
    avx512dq_cpu = (leaf7.ebx & (1<<17)) != 0;
    avx512cd_cpu = (leaf7.ebx & (1<<28)) != 0;
    avx512bw_cpu = (leaf7.ebx & (1<<30)) != 0;
    avx512vl_cpu = (leaf7.ebx & (1<<31)) != 0;
    }

  const bool avx_usable = avx_cpu && osxsave && avx_os;
  const bool avx512_os = (state & 0xe6) == 0xe6;
  const bool avx512_target_cpu = avx512f_cpu && avx512dq_cpu && avx512cd_cpu
    && avx512bw_cpu && avx512vl_cpu;
  const bool avx512_usable = avx512_target_cpu && avx_usable && avx512_os;

  std::cout << "sse2=" << ((leaf1.edx & (1<<26)) != 0)
            << " sse3=" << ((leaf1.ecx & 1) != 0)
            << " avx_cpu=" << avx_cpu
            << " avx2_cpu=" << avx2_cpu
            << " avx512f_cpu=" << avx512f_cpu
            << " avx512dq_cpu=" << avx512dq_cpu
            << " avx512cd_cpu=" << avx512cd_cpu
            << " avx512bw_cpu=" << avx512bw_cpu
            << " avx512vl_cpu=" << avx512vl_cpu
            << " osxsave=" << osxsave
            << " xcr0=0x" << std::hex << state << std::dec
            << " avx_usable=" << avx_usable
            << " avx512_usable=" << avx512_usable << '\n';
  return 0;
  }
