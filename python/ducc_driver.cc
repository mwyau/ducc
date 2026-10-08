#ifdef DUCC0_USE_NANOBIND
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
namespace py = nanobind;
#else
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#include <stdexcept>
#include <utility>

#include "multiarch.h"

namespace {

void add_cpu_info(py::module_ &m, bool multiarch,
                  ducc0_multiarch::cpu_capabilities cpu,
#ifdef DUCC0_MULTIARCH_ARM64
                  ducc0_multiarch::arm_profile_state state
#else
                  ducc0_multiarch::profile_state state
#endif
                  )
  {
  const auto features = std::move(cpu.features);
  m.attr("misc").attr("cpu_info") = py::cpp_function(
    [multiarch, state, features, cpu]()
    {
    py::dict result;
    result["architecture"] = ducc0_multiarch::architecture_name();
    py::list feature_names;
    for (const auto &feature : features) feature_names.append(feature);
    result["features"] = feature_names;
    result["multiarch"] = multiarch;
    if (multiarch)
      {
      py::list compiled;
#ifdef DUCC0_MULTIARCH_ARM64
      for (int profile : ducc0_multiarch::compiled_arm_profiles(
             DUCC0_ARM_COMPILED_PROFILES_MASK))
        compiled.append(ducc0_multiarch::arm_profile_name(profile));
#else
      for (int profile : ducc0_multiarch::compiled_profiles())
        compiled.append(ducc0_multiarch::profile_name(profile));
#endif
      result["compiled_profiles"] = compiled;

      py::list available;
#ifdef DUCC0_MULTIARCH_ARM64
      for (int profile : ducc0_multiarch::available_arm_profiles(
             DUCC0_ARM_COMPILED_PROFILES_MASK, cpu.arm))
        available.append(ducc0_multiarch::arm_profile_name(profile));
      result["configured_limit"] =
        ducc0_multiarch::arm_profile_name(state.configured_limit);
      result["active_profile"] =
        ducc0_multiarch::arm_profile_name(state.active_profile);
#else
      for (int profile : ducc0_multiarch::available_profiles(
             ducc0_multiarch::ducc_compiled_profiles_mask,
             state.host_psabi_level))
        available.append(ducc0_multiarch::profile_name(profile));
      result["configured_limit"] =
        ducc0_multiarch::profile_name(state.configured_limit);
      result["active_profile"] =
        ducc0_multiarch::profile_name(state.active_profile);
#endif
      result["available_profiles"] = available;
      }
    return result;
    },
    R"doc(Return information about the current CPU and DUCC dispatch configuration.

The ``features`` entry lists SIMD/vector features recognized by DUCC that are
available to the current process.

For multiarch builds, the result also reports the compiled and available
architecture profiles, the configured profile limit, and the profile selected
for the current process.

Returns
-------
dict
    CPU and dispatch information.)doc");
  }

}

#ifdef DUCC0_MULTIARCH
#ifdef DUCC0_MULTIARCH_ARM64
namespace ducc0_neon { void add_ducc0(py::module_ &m); }
#ifdef DUCC0_ARM_BUILD_SVE
namespace ducc0_sve { void add_ducc0(py::module_ &m); }
#endif
#ifdef DUCC0_ARM_BUILD_SVE2
namespace ducc0_sve2 { void add_ducc0(py::module_ &m); }
#endif
#else
namespace ducc0_v1 { void add_ducc0(py::module_ &m); }
namespace ducc0_v3 { void add_ducc0(py::module_ &m); }
namespace ducc0_v4 { void add_ducc0(py::module_ &m); }
#endif
#else
namespace ducc0 { void add_ducc0(py::module_ &m); }
#endif

#ifdef DUCC0_USE_NANOBIND
NB_MODULE(PKGNAME, m)
#else
PYBIND11_MODULE(PKGNAME, m, py::mod_gil_not_used())
#endif
  {
#define DUCC0_XSTRINGIFY(s) DUCC0_STRINGIFY(s)
#define DUCC0_STRINGIFY(s) #s
  m.attr("__version__") = DUCC0_XSTRINGIFY(PKGVERSION);
#undef DUCC0_STRINGIFY
#undef DUCC0_XSTRINGIFY
#ifdef DUCC0_USE_NANOBIND
  m.attr("__wrapper__") = "nanobind";
#else
  m.attr("__wrapper__") = "pybind11";
#endif

#ifdef DUCC0_MULTIARCH
  const auto cpu = ducc0_multiarch::detect_cpu_capabilities();
#ifdef DUCC0_MULTIARCH_ARM64
  const auto state = ducc0_multiarch::current_arm_profile_state(
    cpu.arm, DUCC0_ARM_COMPILED_PROFILES_MASK);
  switch (state.active_profile)
    {
    case 3:
#ifdef DUCC0_ARM_BUILD_SVE2
      ducc0_sve2::add_ducc0(m);
      break;
#endif
    case 2:
#ifdef DUCC0_ARM_BUILD_SVE
      ducc0_sve::add_ducc0(m);
      break;
#endif
    case 1:
      ducc0_neon::add_ducc0(m);
      break;
    default:
      throw std::runtime_error("no compatible DUCC0 ARM64 multiarch profile");
    }
#else
  const auto state = ducc0_multiarch::current_profile_state(
    cpu.x86_psabi_level);
  switch (state.active_profile)
    {
    case 4:
      ducc0_v4::add_ducc0(m);
      break;
    case 3:
      ducc0_v3::add_ducc0(m);
      break;
    case 1:
      ducc0_v1::add_ducc0(m);
      break;
    default:
      throw std::runtime_error("no compatible DUCC0 multiarch profile");
    }
#endif
  add_cpu_info(m, true, cpu, state);
#else
  ducc0::add_ducc0(m);
  const auto cpu = ducc0_multiarch::detect_cpu_capabilities();
  add_cpu_info(m, false, cpu, {});
#endif
  }
