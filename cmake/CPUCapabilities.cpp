/*
 * Small, standalone CPU capability diagnostic for CI and developer use.
 *
 * This file is intentionally not part of the DUCC0 library or its build.
 */

#include <cstdint>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64)
#define DUCC0_CPU_X86
#define DUCC0_CPU_X86_64
#elif defined(__i386__) || defined(_M_IX86)
#define DUCC0_CPU_X86
#endif

#if defined(DUCC0_CPU_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace {

#if defined(DUCC0_CPU_X86)

struct cpuid_result
  {
  std::uint32_t eax, ebx, ecx, edx;
  };

cpuid_result cpuid(std::uint32_t leaf, std::uint32_t subleaf)
  {
#if defined(_MSC_VER)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  return {static_cast<std::uint32_t>(regs[0]),
          static_cast<std::uint32_t>(regs[1]),
          static_cast<std::uint32_t>(regs[2]),
          static_cast<std::uint32_t>(regs[3])};
#else
  unsigned int eax, ebx, ecx, edx;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  return {eax, ebx, ecx, edx};
#endif
  }

bool has_bit(std::uint32_t value, unsigned bit)
  { return (value & (std::uint32_t(1) << bit)) != 0; }

std::uint64_t read_xcr(std::uint32_t index)
  {
#if defined(_MSC_VER)
  return static_cast<std::uint64_t>(_xgetbv(index));
#else
  std::uint32_t eax, edx;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0"
    : "=a"(eax), "=d"(edx) : "c"(index));
  return (std::uint64_t(edx)<<32) | eax;
#endif
  }

#endif

}

int main()
  {
#if defined(DUCC0_CPU_X86)
  const auto leaf0 = cpuid(0, 0);
  const auto leaf1 = (leaf0.eax>=1) ? cpuid(1, 0) : cpuid_result{0,0,0,0};
  const auto leaf7 = (leaf0.eax>=7) ? cpuid(7, 0) : cpuid_result{0,0,0,0};

  const bool sse2 = has_bit(leaf1.edx, 26);
  const bool sse3 = has_bit(leaf1.ecx, 0);
  const bool avx_cpu = has_bit(leaf1.ecx, 28);
  const bool osxsave = has_bit(leaf1.ecx, 27);
  std::uint64_t xcr0 = 0;
  if (osxsave)
    xcr0 = read_xcr(0);

  const bool avx2_cpu = has_bit(leaf7.ebx, 5);
  const bool avx512f_cpu = has_bit(leaf7.ebx, 16);
  const bool avx512dq_cpu = has_bit(leaf7.ebx, 17);
  const bool avx512cd_cpu = has_bit(leaf7.ebx, 28);
  const bool avx512bw_cpu = has_bit(leaf7.ebx, 30);
  const bool avx512vl_cpu = has_bit(leaf7.ebx, 31);

  const bool avx_usable = avx_cpu && osxsave && ((xcr0 & 0x6)==0x6);
  const bool avx512_usable = avx_usable && avx512f_cpu && avx512dq_cpu
    && avx512cd_cpu && avx512bw_cpu && avx512vl_cpu
    && ((xcr0 & 0xe6)==0xe6);

  std::cout << "arch="
#if defined(DUCC0_CPU_X86_64)
    << "x86_64";
#else
    << "x86";
#endif
  std::cout << " sse2=" << (sse2 ? 1 : 0)
    << " sse3=" << (sse3 ? 1 : 0)
    << " avx_cpu=" << (avx_cpu ? 1 : 0)
    << " avx2_cpu=" << (avx2_cpu ? 1 : 0)
    << " avx512f_cpu=" << (avx512f_cpu ? 1 : 0)
    << " avx512dq_cpu=" << (avx512dq_cpu ? 1 : 0)
    << " avx512cd_cpu=" << (avx512cd_cpu ? 1 : 0)
    << " avx512bw_cpu=" << (avx512bw_cpu ? 1 : 0)
    << " avx512vl_cpu=" << (avx512vl_cpu ? 1 : 0)
    << " osxsave=" << (osxsave ? 1 : 0)
    << " xcr0=0x" << std::hex << xcr0 << std::dec
    << " avx_usable=" << (avx_usable ? 1 : 0)
    << " avx512_usable=" << (avx512_usable ? 1 : 0)
    << '\n';
#elif defined(__aarch64__) || defined(_M_ARM64)
  std::cout << "arch=arm64"
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64)
    << " neon=1"
#else
    << " neon=0"
#endif
#if defined(__ARM_FEATURE_SVE) && __ARM_FEATURE_SVE
    << " sve=1"
#else
    << " sve=0"
#endif
    << '\n';
#elif defined(__arm__) || defined(_M_ARM)
  std::cout << "arch=arm"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    << " neon=1"
#else
    << " neon=0"
#endif
    << '\n';
#elif defined(__powerpc64__)
  std::cout << "arch=ppc64 detailed_simd_detection=not_implemented\n";
#elif defined(__powerpc__)
  std::cout << "arch=ppc detailed_simd_detection=not_implemented\n";
#elif defined(__riscv)
  std::cout << "arch=riscv detailed_simd_detection=not_implemented\n";
#elif defined(__s390x__)
  std::cout << "arch=s390x detailed_simd_detection=not_implemented\n";
#elif defined(__wasm64__)
  std::cout << "arch=wasm64 detailed_simd_detection=not_implemented\n";
#elif defined(__wasm32__)
  std::cout << "arch=wasm32 detailed_simd_detection=not_implemented\n";
#elif defined(__loongarch64)
  std::cout << "arch=loongarch64 detailed_simd_detection=not_implemented\n";
#else
  std::cout << "arch=unknown detailed_simd_detection=not_implemented\n";
#endif
  return 0;
  }
