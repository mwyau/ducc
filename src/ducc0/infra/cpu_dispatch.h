/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

/*! \file cpu_dispatch.h
 *  Runtime selection for the x86-64 psABI levels used by FFT dispatch.
 */

#ifndef DUCC0_CPU_DISPATCH_H
#define DUCC0_CPU_DISPATCH_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace ducc0 {
namespace cpu_dispatch {

enum class Level : std::uint8_t { x86_64, x86_64_v2, x86_64_v3, x86_64_v4 };

inline constexpr std::array<Level,4> all_levels = {
  Level::x86_64, Level::x86_64_v2, Level::x86_64_v3, Level::x86_64_v4};

enum class Feature : std::uint8_t
  { sse3, ssse3, sse41, sse42, popcnt, cx16, lahf, avx, avx2, fma, bmi1,
    bmi2, lzcnt, f16c, movbe, xsave, osxsave, avx512f, avx512cd,
    avx512vl, avx512bw, avx512dq };

inline constexpr std::array<Feature,6> reported_features = {
  Feature::sse3, Feature::ssse3, Feature::sse41, Feature::sse42,
  Feature::avx, Feature::avx2};

constexpr std::uint32_t feature_bit(Feature feature) noexcept
  { return std::uint32_t(1u) << static_cast<unsigned>(feature); }

struct CpuInfo
  {
  std::uint32_t features = 0;
  std::uint64_t xcr0 = 0;
  Level usable = Level::x86_64;
  Level max_allowed = Level::x86_64_v4;
  };

const CpuInfo &cpu_info() noexcept;
const char *level_name(Level level) noexcept;
const char *feature_name(Feature feature) noexcept;
bool parse_level(const char *name, Level &result) noexcept;

inline bool has_feature(const CpuInfo &cpu, Feature feature) noexcept
  { return (cpu.features & feature_bit(feature)) != 0; }

template<typename Table> struct TargetBinding
  {
  Level level;
  const Table *implementation;
  };

template<typename Table> struct ResolvedTarget
  {
  Level level;
  const Table &implementation;
  };

inline bool usable(Level level, const CpuInfo &cpu) noexcept
  {
  return static_cast<unsigned>(level)<=static_cast<unsigned>(cpu.usable)
      && static_cast<unsigned>(level)<=static_cast<unsigned>(cpu.max_allowed);
  }

// FFT's float and double kernel tables share this typed profile selector.
template<typename Table, std::size_t NT, std::size_t NP>
ResolvedTarget<Table> highest_compiled(const CpuInfo &cpu,
  const Table &baseline,
  const std::array<TargetBinding<Table>,NT> &compiled,
  const std::array<Level,NP> &preference) noexcept
  {
  for (auto level : preference)
    if (level!=Level::x86_64 && usable(level,cpu))
      for (const auto &binding : compiled)
        if (binding.level==level && binding.implementation)
          return {level,*binding.implementation};
  return {Level::x86_64,baseline};
  }

} // namespace cpu_dispatch
} // namespace ducc0

#endif
