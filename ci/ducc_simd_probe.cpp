#include <iostream>

#include "ducc0/math/gridding_kernel.h"

#ifdef DUCC0_NO_SIMD
#error "DUCC0_NO_SIMD must not be active for this probe"
#endif

#if defined(DUCC_EXPECT_AVX)
#ifndef __AVX__
#error "the AVX target did not define __AVX__"
#endif
#ifdef __AVX512F__
#error "the AVX target unexpectedly defined __AVX512F__"
#endif
static_assert(ducc0::native_simd<float>::size()==8, "unexpected AVX float width");
static_assert(ducc0::native_simd<double>::size()==4, "unexpected AVX double width");
static_assert(ducc0::bounded_simd<float,8>::size()==8,
  "bounded float SIMD must remain capped at 8");
#elif defined(DUCC_EXPECT_AVX512)
#ifndef __AVX512F__
#error "the AVX-512 target did not define __AVX512F__"
#endif
static_assert(ducc0::native_simd<float>::size()==16,
  "unexpected AVX-512 float width");
static_assert(ducc0::native_simd<double>::size()==8,
  "unexpected AVX-512 double width");
static_assert(ducc0::bounded_simd<float,8>::size()==8,
  "bounded float SIMD must remain capped at 8");
#else
#error "compile with DUCC_EXPECT_AVX or DUCC_EXPECT_AVX512"
#endif

int main()
  {
  alignas(64) float real[16] = {}, imag[16] = {};
  using bounded_float = ducc0::bounded_simd<float,8>;
  const auto z = ducc0::detail_gridding_kernel::hsum_cmplx<float>(
    ducc0::loadu<bounded_float>(real), ducc0::loadu<bounded_float>(imag));
  std::cout << "DUCC0_NO_SIMD=0"
            << " __SSE2__="
#ifdef __SSE2__
            << 1
#else
            << 0
#endif
            << " __SSE3__="
#ifdef __SSE3__
            << 1
#else
            << 0
#endif
            << " __AVX__="
#ifdef __AVX__
            << 1
#else
            << 0
#endif
            << " __AVX512F__="
#ifdef __AVX512F__
            << 1
#else
            << 0
#endif
            << " native_float=" << ducc0::native_simd<float>::size()
            << " native_double=" << ducc0::native_simd<double>::size()
            << " bounded_float8=" << ducc0::bounded_simd<float,8>::size()
            << " hsum=" << z.real() << "," << z.imag()
            << '\n';
  }
