#ifndef DUCC0_MULTIARCH_H
#define DUCC0_MULTIARCH_H

#include <cstdint>
#include <string>
#include <vector>

namespace ducc0_multiarch {

using profile_mask = std::uint32_t;

constexpr profile_mask profile_bit(int psabi_level)
  { return (psabi_level >= 1 && psabi_level <= 4) ? (1u << psabi_level) : 0; }

using arm_profile_mask = std::uint32_t;

constexpr arm_profile_mask arm_profile_bit(int profile)
  { return (profile >= 1 && profile <= 3) ? (1u << profile) : 0; }

// No v2 build is compiled; x86-64-v2 hosts use the v1 build.
inline constexpr profile_mask ducc_compiled_profiles_mask =
    profile_bit(1) | profile_bit(3) | profile_bit(4);

struct profile_state
  {
  int host_psabi_level;
  int configured_limit;
  int active_profile;
  };

namespace detail {

struct arm_features
  {
  bool neon = false;
  bool sve = false;
  bool sve2 = false;
  };

constexpr bool usable_arm_sve(const arm_features &features)
  { return features.neon && features.sve; }

constexpr bool usable_arm_sve2(const arm_features &features)
  { return usable_arm_sve(features) && features.sve2; }

constexpr bool supports_arm_profile(const arm_features &features,
                                    int profile)
  {
  switch (profile)
    {
    case 1: return true; // NEON is the Linux AArch64 baseline.
    case 2: return usable_arm_sve(features);
    case 3: return usable_arm_sve2(features);
    default: return false;
    }
  }

constexpr int arm_host_profile(const arm_features &features)
  {
  if (usable_arm_sve2(features)) return 3;
  if (usable_arm_sve(features)) return 2;
  return 1;
  }

struct x86_features
  {
  bool sse3 = false;
  bool ssse3 = false;
  bool sse41 = false;
  bool sse42 = false;
  bool cx16 = false;
  bool lahf_sahf = false;
  bool popcnt = false;
  bool avx = false;
  bool xsave = false;
  bool osxsave = false;
  bool ymm_state = false;
  bool avx2 = false;
  bool bmi1 = false;
  bool bmi2 = false;
  bool f16c = false;
  bool fma = false;
  bool lzcnt = false;
  bool movbe = false;
  bool avx512f = false;
  bool avx512dq = false;
  bool avx512cd = false;
  bool avx512bw = false;
  bool avx512vl = false;
  bool zmm_state = false;
  };

constexpr bool usable_avx(const x86_features &features)
  {
  return features.avx && features.xsave && features.osxsave
      && features.ymm_state;
  }

constexpr bool usable_avx2(const x86_features &features)
  { return usable_avx(features) && features.avx2; }

constexpr bool usable_avx512(const x86_features &features)
  {
  return usable_avx(features) && features.avx512f && features.avx512dq
      && features.avx512cd && features.avx512bw && features.avx512vl
      && features.zmm_state;
  }

constexpr bool usable_psabi_v2(const x86_features &features)
  {
  return features.cx16 && features.lahf_sahf && features.popcnt
      && features.sse3 && features.ssse3 && features.sse41 && features.sse42;
  }

constexpr bool usable_psabi_v3(const x86_features &features)
  {
  return usable_psabi_v2(features) && usable_avx2(features)
      && features.bmi1 && features.bmi2 && features.f16c && features.fma
      && features.lzcnt && features.movbe;
  }

constexpr bool usable_psabi_v4(const x86_features &features)
  { return usable_psabi_v3(features) && usable_avx512(features); }

constexpr int psabi_level(const x86_features &features)
  {
  if (usable_psabi_v4(features)) return 4;
  if (usable_psabi_v3(features)) return 3;
  if (usable_psabi_v2(features)) return 2;
  return 1; // Linux x86-64 guarantees the x86-64-v1 ABI baseline.
  }

} // namespace detail

struct cpu_capabilities
  {
  int x86_psabi_level = 0;
  detail::arm_features arm;
  std::vector<std::string> features;
  };

struct arm_profile_state
  {
  int configured_limit;
  int active_profile;
  };

const char *architecture_name();
cpu_capabilities detect_cpu_capabilities();
int configured_psabi_limit();
std::vector<int> compiled_profiles(
  profile_mask profiles = ducc_compiled_profiles_mask);
std::vector<int> available_profiles(profile_mask profiles,
                                    int host_psabi_level);
int select_profile(int host_psabi_level, int configured_limit,
                   profile_mask profiles);
const char *profile_name(int psabi_level);
profile_state current_profile_state(int host_psabi_level,
  profile_mask profiles = ducc_compiled_profiles_mask);

std::vector<int> compiled_arm_profiles(arm_profile_mask profiles);
std::vector<int> available_arm_profiles(arm_profile_mask profiles,
                                        const detail::arm_features &features);
int select_arm_profile(const detail::arm_features &features,
                       int configured_limit, arm_profile_mask profiles);
int configured_arm_profile_limit();
const char *arm_profile_name(int profile);
arm_profile_state current_arm_profile_state(
  const detail::arm_features &features, arm_profile_mask profiles);

} // namespace ducc0_multiarch

#endif
