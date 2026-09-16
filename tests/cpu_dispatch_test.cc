#include "ducc0/infra/cpu_dispatch.h"

#include <cassert>
#include <cstring>

using namespace ducc0::cpu_dispatch;

namespace {

constexpr TargetDescriptor all_targets[] =
  {
  {TargetId::sse2, "sse2"},
  {TargetId::avx, "avx"},
  {TargetId::avx512, "avx512"}
  };
constexpr TargetId preference[] =
  {TargetId::avx512, TargetId::avx, TargetId::sse2};

TargetId select(CpuTargets targets, CpuMax cap=CpuMax::none,
  const TargetDescriptor *compiled=all_targets, size_t ncompiled=3)
  {
  return select_target(targets, compiled, ncompiled, preference,
    sizeof(preference)/sizeof(preference[0]), cap);
  }

void test_selection()
  {
  const CpuTargets only_sse2{true, false, false};
  const CpuTargets avx_cpu{true, true, false};
  const CpuTargets avx512_cpu{true, true, true};

  // Synthetic target states exercise the ordered fallback without touching
  // the host's actual CPUID state.
  assert(select(only_sse2)==TargetId::sse2);
  assert(select(avx_cpu)==TargetId::avx);
  assert(select(avx512_cpu)==TargetId::avx512);

  // A target absent from the binary cannot be selected.
  assert(select(avx512_cpu, CpuMax::none, all_targets, 2)==TargetId::avx);
  assert(select(avx512_cpu, CpuMax::none, all_targets, 1)==TargetId::sse2);

  // Caps are upper bounds and never force an unsupported target.
  assert(select(avx512_cpu, CpuMax::sse2)==TargetId::sse2);
  assert(select(avx512_cpu, CpuMax::avx)==TargetId::avx);
  assert(select(avx512_cpu, CpuMax::avx512)==TargetId::avx512);
  assert(select(avx_cpu, CpuMax::avx512)==TargetId::avx);
  assert(select(only_sse2, CpuMax::avx)==TargetId::sse2);

  assert(target_is_usable(TargetId::sse2, only_sse2));
  assert(!target_is_usable(TargetId::avx, only_sse2));
  assert(target_is_usable(TargetId::avx, avx_cpu));
  assert(!target_is_usable(TargetId::avx512, avx_cpu));
  assert(target_is_usable(TargetId::avx512, avx512_cpu));

  // Invalid caps are ignored by the pure selector; runtime parsing reports
  // them and converts them to an uncapped state.
  assert(select(avx512_cpu, CpuMax::invalid)==TargetId::avx512);
  assert(parse_cpu_max(nullptr)==CpuMax::none);
  assert(parse_cpu_max("")==CpuMax::none);
  assert(parse_cpu_max("sse2")==CpuMax::sse2);
  assert(parse_cpu_max("avx")==CpuMax::avx);
  assert(parse_cpu_max("avx512")==CpuMax::avx512);
  assert(parse_cpu_max("none")==CpuMax::invalid);
  assert(parse_cpu_max("not-a-target")==CpuMax::invalid);
  }

void test_compiled_targets()
  {
  size_t ncompiled = 0;
  const auto *compiled = compiled_targets(ncompiled);
  assert(ncompiled >= 1);
  assert(compiled[0].id==TargetId::sse2);
  assert(std::strcmp(compiled[0].name, "sse2")==0);

  size_t expected = 1;
#if defined(DUCC0_DISPATCH_HAS_AVX)
  assert(ncompiled > expected && compiled[expected].id==TargetId::avx);
  assert(std::strcmp(compiled[expected].name, "avx")==0);
  ++expected;
#endif
#if defined(DUCC0_DISPATCH_HAS_AVX512)
  assert(ncompiled > expected && compiled[expected].id==TargetId::avx512);
  assert(std::strcmp(compiled[expected].name, "avx512")==0);
  ++expected;
#endif
  assert(ncompiled==expected);

  const auto actual = detect_cpu_targets();
  assert(!actual.avx512 || actual.avx);
  }

} // namespace

int main()
  {
  test_selection();
  test_compiled_targets();
  assert(std::strcmp(target_name(TargetId::sse2), "sse2")==0);
  assert(std::strcmp(cpu_max_name(CpuMax::sse2), "sse2")==0);
  return 0;
  }
