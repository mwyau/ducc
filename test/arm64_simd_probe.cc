// Standalone diagnostic for the compiler's native std::experimental::simd ABI.
// Build it separately with each ARM64 profile's flags; it is not part of DUCC.

#include <cstdio>
#include <experimental/simd>

namespace stdx = std::experimental;
using native_double = stdx::native_simd<double>;

template<typename T> constexpr const char *type_description()
  {
  return __PRETTY_FUNCTION__;
  }

extern "C" __attribute__((noinline)) void ducc_simd_probe(
  double *out, const double *a, const double *b)
  {
  native_double x(a, stdx::element_aligned_tag{});
  native_double y(b, stdx::element_aligned_tag{});
  (x*y+x).copy_to(out, stdx::element_aligned_tag{});
  }

int main()
  {
  const char *backend = "scalar";
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) \
    && (__ARM_FEATURE_SVE_BITS > 0)
  backend = "SVE";
#elif defined(__ARM_NEON)
  if (native_double::size() > 1) backend = "NEON";
#else
  if (native_double::size() > 1) backend = "native-vector";
#endif

  int sve_bits = -1;
#ifdef __ARM_FEATURE_SVE_BITS
  sve_bits = __ARM_FEATURE_SVE_BITS;
#endif
  std::printf("backend=%s lanes=%zu sve_target=%d sve_bits=%d sve2_target=%d\n",
    backend, native_double::size(),
#ifdef __ARM_FEATURE_SVE
    1,
#else
    0,
#endif
    sve_bits,
#ifdef __ARM_FEATURE_SVE2
    1
#else
    0
#endif
  );
  std::puts(type_description<native_double>());
  }
