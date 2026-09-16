/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 */

/*! \file cpu_dispatch.h
 *  Runtime CPU target detection and selection.
 */

#ifndef DUCC0_CPU_DISPATCH_H
#define DUCC0_CPU_DISPATCH_H

#include <cstddef>
#include <cstdint>

namespace ducc0 {

namespace cpu_dispatch {

enum class TargetId : uint8_t
  {
  sse2,
  avx,
  avx512
  };

enum class CpuMax : uint8_t
  {
  none,
  sse2,
  avx,
  avx512,
  invalid
  };

struct TargetDescriptor
  {
  TargetId id;
  const char *name;
  };

// This is deliberately limited to the implementations DUCC can select.
// Compiler-specific CPUID prerequisites remain private to cpu_dispatch.cc.
struct CpuTargets
  {
  bool x86 = false;
  bool avx = false;
  bool avx512 = false;
  };

struct RuntimeState
  {
  CpuTargets targets;
  CpuMax max_target = CpuMax::none;
  };

CpuTargets detect_cpu_targets() noexcept;
const RuntimeState &runtime_state() noexcept;

CpuMax parse_cpu_max(const char *value) noexcept;
const char *cpu_max_name(CpuMax value) noexcept;
const char *target_name(TargetId value) noexcept;

bool target_is_allowed(TargetId target, CpuMax max_target) noexcept;
bool target_is_usable(TargetId target, const CpuTargets &targets) noexcept;

TargetId select_target(const CpuTargets &targets,
  const TargetDescriptor *compiled_targets, size_t ncompiled,
  const TargetId *preference, size_t npreference,
  CpuMax max_target=CpuMax::none) noexcept;

const TargetDescriptor *compiled_targets(size_t &count) noexcept;
const char *mode_name() noexcept;

} // namespace cpu_dispatch

} // namespace ducc0

#endif
