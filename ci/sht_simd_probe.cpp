#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "ducc0/infra/simd.h"
#include "ducc0/sht/sht.h"
#include "ducc0/sht/sphere_interpol.h"
#include "ducc0/fft/fft.h"
#include "ducc0/nufft/nufft.h"
#include "ducc0/math/math_utils.h"
#include "ducc0/math/gl_integrator.h"
#include "ducc0/math/constants.h"
#include "ducc0/math/solvers.h"
#include "ducc0/sht/sht_utils.h"
#include "ducc0/infra/timers.h"
#include "ducc0/sht/sht_inner_loop.h"

#if defined(_MSC_VER)
#define DUCC_SHT_PROBE_NOINLINE __declspec(noinline)
#else
#define DUCC_SHT_PROBE_NOINLINE __attribute__((noinline, used))
#endif

#if !defined(__AVX512F__)
#error "compile this probe for AVX-512F"
#endif

using Tv = ducc0::detail_sht::detail_sht_inner_loop::Tv;
static_assert(Tv::size()==8, "the SHT probe must use eight doubles");

extern "C" DUCC_SHT_PROBE_NOINLINE void ducc_sht_simd_probe(
  const double *a, const double *b, const double *c, const double *d,
  std::complex<double> *cc)
  {
  const Tv va = ducc0::loadu<Tv>(a);
  const Tv vb = ducc0::loadu<Tv>(b);
  const Tv vc = ducc0::loadu<Tv>(c);
  const Tv vd = ducc0::loadu<Tv>(d);
  ducc0::detail_sht::detail_sht_inner_loop::vhsum_cmplx_special(
    va, vb, vc, vd, cc);
  }

int main()
  {
  alignas(64) double a[8] = {}, b[8] = {}, c[8] = {}, d[8] = {};
  std::complex<double> cc[2] = {};
  ducc_sht_simd_probe(a, b, c, d, cc);
  return (cc[0] == std::complex<double>(0., 0.)
    && cc[1] == std::complex<double>(0., 0.)) ? 0 : 1;
  }
